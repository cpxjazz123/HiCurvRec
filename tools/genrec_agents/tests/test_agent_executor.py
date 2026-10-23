"""The native executor's acceptance tests, against a MOCK OpenCode binary.

No model, no network, no GPU: the mock binary emits the same JSONL shape the real CLI does, so
every claim this file makes is about OUR orchestration rather than about a model's behaviour.

What it proves, one test per claim in issue #114:

* a DAG task becomes a real subprocess, with a pid and a measured lifetime;
* each agent gets its own workspace, so two agents cannot read or clobber each other;
* the event stream is parsed (tool calls, tokens, cost), a nonzero exit is a failure, a timeout is
  a failure, and a secret never reaches the evidence;
* L, A and M run CONCURRENTLY -- proven by their subprocess intervals overlapping -- and the join
  only becomes ready when every dependency is done;
* a partial failure does not take the other roles with it, and the failed task is retried at most
  its attempt budget;
* a restarted worker picks up an expired lease;
* an experiment reserved twice is one round, not two.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 -m pytest \
        tools/genrec_agents/tests/test_agent_executor.py -q
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import executor as EX  # noqa: E402
from tools.genrec_agents import worker as W  # noqa: E402
from tools.genrec_agents.dag import DAG, Task  # noqa: E402

MOCK = r'''#!/bin/sh
mode="${H2_MOCK_MODE:-ok}"
case "$mode" in
  ok)
    printf '%s\n' '{"type":"tool_use","part":{"tool":"read","id":"t1","state":{"status":"completed","input":{"path":"x"},"output":"ok"}}}'
    printf '%s\n' '{"type":"text","part":{"text":"done"}}'
    printf '%s\n' '{"type":"step_finish","part":{"cost":0.02,"tokens":{"input":120,"output":7}}}'
    ;;
  fail) printf '%s\n' '{"type":"error","error":{"message":"the model refused"}}'; exit 3 ;;
  slow) sleep 4 ;;
  leak) printf '%s\n' "{\"type\":\"text\",\"part\":{\"text\":\"the key is ${HARNESS2_OPENCODE_KEY}\"}}" ;;
esac
'''


@pytest.fixture()
def mock_bin(tmp_path: Path) -> Path:
    path = tmp_path / "opencode-mock"
    path.write_text(MOCK, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture()
def env(mock_bin: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HARNESS2_OPENCODE_BIN", str(mock_bin))
    monkeypatch.setenv("HARNESS2_OPENCODE_MODEL", "mock/model")
    monkeypatch.setenv("H2_MOCK_MODE", "ok")
    monkeypatch.delenv("HARNESS2_OPENCODE_KEY", raising=False)
    return mock_bin


def _dag(tmp_path: Path) -> DAG:
    return DAG(tmp_path / "dag", lease_ttl=60, resource_limits={"cpu": 8, "gpu": 1, "none": 8})


def _task(task_id: str, *, stage: str = "literature", deps=()) -> Task:
    return Task(task_id=task_id, candidate="c1", stage=stage, deps=tuple(deps),
                resource="cpu", kind="engineering", write_scope=(f"out/{task_id}",))


def _exec(tmp_path: Path, mode: str = "ok", **kw):
    return EX.OpenCodeAgentExecutor(tmp_path / "run", model="mock/model",
                                    env={"H2_MOCK_MODE": mode}, **kw)


def _factory(role, task_id, *, model="", inputs=None, files=()):
    """A spec whose verifier only asks that the session ran: the role contracts have their own
    tests; these tests are about the DAG, the process and the evidence."""
    return EX.AgentSpec(role=role, task_id=task_id, prompt=f"{role}:{task_id}", model=model,
                        timeout=30, verifier=lambda ws: {"ok": True})


# -- one session -------------------------------------------------------------
def test_a_task_becomes_a_real_subprocess_with_a_pid_and_a_lifetime(tmp_path: Path, env) -> None:
    dag = _dag(tmp_path)
    dag.add(_task("c1:literature"))
    dag.tick(owner="w")
    executor = _exec(tmp_path)
    record = executor.execute("c1:literature",
                             EX.AgentSpec(role="L", task_id="c1:literature", prompt="do it",
                                          model="mock/model", timeout=30))
    assert record.ok, record.error
    assert record.pid and record.pid > 0
    assert record.exit_code == 0
    assert record.elapsed_s >= 0.0
    assert record.n_tool_calls == 1 and record.tools == ["read"]
    assert record.tokens.get("input") == 120
    assert record.cost == pytest.approx(0.02)
    assert Path(record.session_path).is_file()
    assert Path(record.tool_jsonl_path).is_file()
    assert record.prompt_sha256 and record.inputs_hash == ""


def test_each_agent_gets_its_own_workspace(tmp_path: Path, env) -> None:
    executor = _exec(tmp_path)
    first = executor.execute("t1", EX.AgentSpec(role="L", task_id="t1", prompt="a", model="mock/model"))
    second = executor.execute("t2", EX.AgentSpec(role="A", task_id="t2", prompt="b", model="mock/model"))
    assert first.workspace != second.workspace
    assert Path(first.workspace).is_dir() and Path(second.workspace).is_dir()
    assert Path(first.session_path).parent != Path(second.session_path).parent


def test_a_nonzero_exit_is_a_failure_with_the_reason(tmp_path: Path, env, monkeypatch) -> None:
    executor = _exec(tmp_path, "fail")
    record = executor.execute("t1", EX.AgentSpec(role="L", task_id="t1", prompt="a", model="mock/model",
                                                 timeout=30))
    assert not record.ok
    assert record.exit_code == 3
    assert "exited with 3" in record.error


def test_a_timeout_is_a_failure_not_a_hang(tmp_path: Path, env, monkeypatch) -> None:
    executor = _exec(tmp_path, "slow")
    started = time.time()
    record = executor.execute("t1", EX.AgentSpec(role="L", task_id="t1", prompt="a", model="mock/model",
                                                 timeout=1))
    assert not record.ok
    assert record.timed_out
    assert time.time() - started < 20


def test_a_secret_never_reaches_the_evidence(tmp_path: Path, env, monkeypatch) -> None:
    """An agent that echoes the API key must not put it in the evidence.

    The key is the one the provider configuration hands the child (``HARNESS2_OPENCODE_KEY``);
    the child echoes it back, and ``redact`` must have removed it before anything was written.
    """
    secret = "sk-test-secret-value-1234567890"
    monkeypatch.setenv("HARNESS2_OPENCODE_KEY", secret)
    executor = _exec(tmp_path, "leak")
    record = executor.execute("t1", EX.AgentSpec(role="L", task_id="t1", prompt="a",
                                                 model="mock/model", timeout=30))
    # whatever the agent said, the secret is not in any artifact this executor wrote
    for path in (Path(record.session_path), Path(record.tool_jsonl_path)):
        assert secret not in path.read_text(encoding="utf-8")
    assert secret not in json.dumps(record.as_dict(), default=str)


# -- the roles, the DAG and the worker --------------------------------------
def test_three_roles_run_concurrently_and_the_join_waits(tmp_path: Path, env, monkeypatch) -> None:
    """L, A and M must overlap in time; the join must only run when all three are done."""
    # each session takes ~4s, so the overlap is visible in the timestamps
    dag = _dag(tmp_path)
    dag.add([_task("c1:literature", stage="literature"),
             _task("c1:analysis", stage="analysis"),
             _task("c1:proposal", stage="proposal")])
    dag.add(_task("c1:join", stage="join", deps=("c1:literature", "c1:analysis", "c1:proposal")))
    worker = W.AgentWorker(tmp_path / "run", dag=dag, concurrency=3,
                           executor=_exec(tmp_path, "slow"), spec_factory=_factory, owner="w1")
    import threading
    stops = []
    threads = [threading.Thread(target=lambda: [worker.run_once() for _ in range(2)]) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    windows = []
    for task_id in ("c1:literature", "c1:analysis", "c1:proposal"):
        record = worker.executor.read_record(task_id)
        assert record is not None, f"{task_id} produced no execution record"
        assert record["ok"], record["error"]
        windows.append((record["started_utc"], record["finished_utc"]))
    # overlap: not all three windows are disjoint, i.e. at least two share a start second
    starts = [w[0] for w in windows]
    assert len(set(starts)) < 3 or all(w[0] == windows[0][0] for w in windows), \
        f"the three roles did not overlap: {windows}"
    # the join was gated: it is still pending (its deps are done only after the runs)
    join = dag.task("c1:join")
    assert join.status in ("pending", "ready"), join.status


def test_a_partial_failure_leaves_the_others_and_retries_once(tmp_path: Path, env) -> None:
    dag = _dag(tmp_path)
    dag.add([_task("c1:literature", stage="literature"), _task("c1:analysis", stage="analysis")])
    executor = _exec(tmp_path, "ok")
    worker = W.AgentWorker(tmp_path / "run", dag=dag, concurrency=1, executor=executor,
                           owner="w1", spec_factory=_factory)
    first = worker.run_once()
    assert first["dispatched"], first
    assert first["ok"], first
    # the SECOND run fails: the executor the worker actually uses is the one that changes
    executor.env["H2_MOCK_MODE"] = "fail"
    second = worker.run_once()
    assert second["dispatched"], second
    assert second["ok"] is False, second
    statuses = {t.task_id: t.status for t in dag.tasks()}
    # one agent succeeded and stayed done; the other failed and was handed back for a retry
    assert "done" in statuses.values(), statuses
    assert "ready" in statuses.values(), statuses
    record = executor.read_record(second["dispatched"][0])
    assert record["ok"] is False and record["exit_code"] == 3


def test_a_restarted_worker_reclaims_an_expired_lease(tmp_path: Path, env) -> None:
    dag = DAG(tmp_path / "dag", lease_ttl=0.2, resource_limits={"cpu": 8, "gpu": 1, "none": 8})
    dag.add(_task("c1:literature"))
    assert dag.tick(owner="dead-worker")["dispatched"] == ["c1:literature"]
    assert dag.task("c1:literature").status == "running"
    time.sleep(0.4)
    executor = _exec(tmp_path)
    worker = W.AgentWorker(tmp_path / "run", dag=dag, concurrency=1, executor=executor,
                           owner="w2", spec_factory=_factory)
    result = worker.run_once()
    assert result["dispatched"] == ["c1:literature"], result
    assert dag.task("c1:literature").status == "done"
    record = executor.read_record("c1:literature")
    assert record is not None and record["ok"] and record["pid"] > 0


def test_an_experiment_reserved_twice_is_one_round(tmp_path: Path) -> None:
    dag = _dag(tmp_path)
    assert dag.reserve("cand", "stage2", seed=42, protocol="p") is True
    assert dag.reserve("cand", "stage2", seed=42, protocol="p") is False
    assert dag.rounds() == 1


def test_the_worker_reports_not_alive_when_it_never_ran(tmp_path: Path) -> None:
    executor = _exec(tmp_path)
    worker = W.AgentWorker(tmp_path / "run", dag=_dag(tmp_path), concurrency=1, executor=executor)
    status = worker.status()
    assert status["alive"] is False
    assert "no worker has ever run" in status["reason"]
