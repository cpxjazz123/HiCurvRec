"""The campaign DAG scheduler's tests (issue #112, P0).

The issue asks for regression tests proving: same-mechanism parallel overlap, correct join
ordering, restart and lease safety, zero duplicate experiments under races, and 72 idle ticks
producing zero false rounds. Each of those is one test below, plus the write-scope rule that
turns "no concurrent file corruption" into a mechanical refusal.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/harness2_genrec/tests/test_dag.py -q
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.harness2_genrec import dag as D  # noqa: E402


class Clock:
    """A clock the test moves by hand, so lease expiry is deterministic."""

    def __init__(self) -> None:
        self.value = 1_000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _task(task_id: str, *, deps=(), stage="L", resource="cpu", kind="engineering",
          scope=(), candidate="arm") -> D.Task:
    return D.Task(task_id=task_id, candidate=candidate, stage=stage, deps=tuple(deps),
                  resource=resource, kind=kind, write_scope=tuple(scope) or (f"out/{task_id}",))


# --------------------------------------------------------------------------------------
# join ordering
# --------------------------------------------------------------------------------------
def test_a_join_is_not_ready_until_every_dependency_is_done(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60)
    graph.add([_task("L"), _task("A"), _task("M")])
    graph.add(_task("join_R", deps=("L", "A", "M"), stage=D.JOIN_R))
    assert [task.task_id for task in graph.ready()] == ["A", "L", "M"]

    graph.tick(owner="w1")
    for name in ("L", "A"):
        graph.complete(name, "w1")
    assert graph.task("join_R").status == D.PENDING, "the join must wait for M"

    graph.tick(owner="w1")
    graph.complete("M", "w1")
    assert graph.task("join_R").status == D.READY


# --------------------------------------------------------------------------------------
# same candidate, concurrent workstreams
# --------------------------------------------------------------------------------------
def test_two_workstreams_of_one_candidate_run_concurrently(tmp_path):
    """The DAG's whole point: L/A/M and implement/oracle are concurrent, not sequential."""
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60, resource_limits={"cpu": 4, "gpu": 1,
                                                                       "none": 8})
    graph.add([_task("implement", stage="implement", scope=("out/impl",)),
               _task("oracle", stage="oracle", scope=("out/oracle",))])
    first = graph.tick(owner="worker-a")
    second = graph.tick(owner="worker-b")
    ran = set(first["dispatched"]) | set(second["dispatched"])
    assert ran == {"implement", "oracle"}
    assert graph.task("implement").status == D.RUNNING
    assert graph.task("oracle").status == D.RUNNING
    windows = [(task.started_at, task.lease_expires) for task in graph.running()]
    assert all(start == windows[0][0] for start, _ in windows), "they started in the same window"


def test_the_resource_limit_bounds_the_concurrency(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60,
                  resource_limits={"cpu": 1, "gpu": 1, "none": 8})
    graph.add([_task("one"), _task("two")])
    graph.tick(owner="w")
    assert sum(1 for task in graph.tasks() if task.status == D.RUNNING) == 1


def test_two_tasks_may_not_share_a_write_scope(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60)
    graph.add([_task("writer_a", scope=("results/shared",)),
               _task("writer_b", scope=("results/shared/sub",))])
    graph.tick(owner="w")
    running = {task.task_id for task in graph.running()}
    assert len(running) == 1, "overlapping scopes must not both be dispatched"
    # the deferred one is still ready and runs once the first releases the scope
    graph.complete(next(iter(running)), "w")
    graph.tick(owner="w")
    assert len(graph.running()) == 1


# --------------------------------------------------------------------------------------
# leases and restart
# --------------------------------------------------------------------------------------
def test_an_expired_lease_is_reclaimed_and_a_spent_attempt_budget_blocks(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=30)
    graph.add(D.Task(task_id="t", candidate="arm", stage="L", write_scope=("out/t",),
                     max_attempts=2))
    graph.tick(owner="dead-worker")
    assert graph.task("t").status == D.RUNNING
    clock.advance(31)
    result = graph.tick(owner="live-worker")
    assert result["reclaimed"] == ["t"]
    assert graph.task("t").status == D.RUNNING, "the live worker re-claimed it"
    assert graph.task("t").attempts == 2
    clock.advance(31)
    graph.tick(owner="live-worker")
    assert graph.task("t").status == D.BLOCKED
    assert graph.task("t").error


def test_a_restarted_scheduler_sees_the_same_graph_and_leases(tmp_path):
    clock = Clock()
    first = D.DAG(tmp_path, clock=clock, lease_ttl=30)
    first.add(_task("step"))
    first.tick(owner="worker-1")
    # a new process (a restart) reads the same directory
    second = D.DAG(tmp_path, clock=clock, lease_ttl=30)
    task = second.task("step")
    assert task.status == D.RUNNING and task.lease_owner == "worker-1"
    assert second.heartbeat("step", "worker-1") is True
    assert second.heartbeat("step", "someone-else") is False
    clock.advance(31)
    assert second.tick(owner="worker-2")["reclaimed"] == ["step"]


def test_only_the_lease_holder_can_complete_or_fail_a_task(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60)
    graph.add(_task("step"))
    graph.tick(owner="owner")
    with pytest.raises(D.SchedulerError):
        graph.complete("step", "intruder")
    with pytest.raises(D.SchedulerError):
        graph.fail("step", "intruder", "nope")
    graph.complete("step", "owner")
    assert graph.task("step").status == D.DONE


def test_bounded_retries_then_blocked(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60)
    graph.add(D.Task(task_id="t", candidate="arm", stage="implement", write_scope=("out/t",),
                     max_attempts=2))
    graph.tick(owner="w")
    graph.fail("t", "w", "flaky")
    assert graph.task("t").status == D.READY
    graph.tick(owner="w")
    graph.fail("t", "w", "flaky again")
    assert graph.task("t").status == D.BLOCKED
    graph.tick(owner="w")
    assert graph.task("t").status == D.BLOCKED, "a blocked task is never re-dispatched"


# --------------------------------------------------------------------------------------
# authorisation is a state
# --------------------------------------------------------------------------------------
def test_a_gpu_task_waits_for_authorisation_and_is_never_dispatched_without_it(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60)
    graph.add(_task("stage2", stage="stage2", resource="gpu"))
    result = graph.tick(owner="w")
    assert result["dispatched"] == []
    assert graph.task("stage2").status == D.WAITING_AUTHORISATION
    assert result["deferred"]["stage2"] == "no operator authorisation for gpu"
    graph.authorise("gpu", "operator-2026-10-23", "single-seed run approved")
    graph.tick(owner="w")
    assert graph.task("stage2").status == D.RUNNING


# --------------------------------------------------------------------------------------
# idle ticks produce no rounds
# --------------------------------------------------------------------------------------
def test_seventy_two_idle_ticks_produce_zero_rounds_and_zero_training(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60)
    graph.add(D.Task(task_id="stage2", candidate="arm", stage="stage2", resource="gpu",
                     kind="experiment", write_scope=("out/stage2",)))
    for _ in range(72):
        clock.advance(5)
        graph.tick(owner="coordinator")
    assert graph.rounds() == 0
    assert graph.idle_ticks() == 72
    assert graph.state()["dispatched"] == 0
    assert all(task.status != D.RUNNING for task in graph.tasks())
    # and the event log says why, instead of pretending to have done research
    reasons = [event for event in graph.events() if event["event"] in ("idle_tick", "deferred")]
    assert reasons, "an idle campaign must record why it is idle"


def test_an_engineering_dispatch_is_not_a_round(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60)
    graph.add(_task("fixtures", stage="oracle", kind="engineering"))
    graph.tick(owner="w")
    graph.complete("fixtures", "w")
    assert graph.rounds() == 0, "a dry run or a fixture build is not a scientific round"


# --------------------------------------------------------------------------------------
# experiments: exactly once, even under a race
# --------------------------------------------------------------------------------------
def test_no_duplicate_experiment_under_a_race(tmp_path):
    clock = Clock()
    results: list[bool] = []
    barrier = threading.Barrier(8)

    def contender():
        graph = D.DAG(tmp_path, clock=clock)   # each worker has its own handle
        barrier.wait(timeout=30)
        results.append(graph.reserve("arm_x", "stage2", seed=42, protocol="frozen"))

    threads = [threading.Thread(target=contender) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 1, f"exactly one reservation, got {results}"
    graph = D.DAG(tmp_path, clock=clock)
    assert len(graph.reservations()) == 1
    assert graph.rounds() == 1
    assert graph.reserve("arm_x", "stage2", seed=42, protocol="frozen") is False
    assert graph.rounds() == 1


def test_a_different_seed_or_protocol_is_a_different_reservation(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock)
    assert graph.reserve("arm_x", "stage2", seed=42) is True
    assert graph.reserve("arm_x", "stage2", seed=43) is True
    assert graph.reserve("arm_x", "stage3", seed=42) is True
    assert graph.rounds() == 3


# --------------------------------------------------------------------------------------
# selective invalidation, content-addressed artifacts
# --------------------------------------------------------------------------------------
def test_invalidation_requeues_only_the_affected_subgraph(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60)
    graph.add([_task("L"), _task("A"), _task("join_R", deps=("L", "A")),
               _task("implement", deps=("join_R",), scope=("out/impl",)),
               _task("unrelated")])
    # drive the whole graph to DONE, one dispatch at a time
    for _ in range(10):
        graph.tick(owner="w", limit=1)
        for task in graph.running():
            graph.complete(task.task_id, "w")
        if all(task.status == D.DONE for task in graph.tasks()):
            break
    assert all(task.status == D.DONE for task in graph.tasks())

    affected = graph.invalidate("join_R", reason="the M sketch changed")
    assert set(affected) == {"join_R", "implement"}
    assert graph.task("implement").status == D.PENDING
    assert graph.task("L").status == D.DONE, "an unaffected ancestor stays done"
    # re-running the join makes the implementation ready again, nothing else
    graph.tick(owner="w", limit=1)
    for task in graph.running():
        graph.complete(task.task_id, "w")
    assert graph.task("implement").status == D.READY


def test_the_same_bytes_are_stored_once_under_their_digest(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock)
    first = graph.store_artifact(b"proposal")
    second = graph.store_artifact(b"proposal")
    assert first == second
    assert Path(first["path"]).read_bytes() == b"proposal"
    files = [path for path in (tmp_path / "artifacts").rglob("*") if path.is_file()]
    assert len(files) == 1


def test_completing_a_task_records_its_artifact_hashes(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60)
    payload = tmp_path / "report.json"
    payload.write_text('{"ok": true}\n', encoding="utf-8")
    graph.add(_task("oracle"))
    graph.tick(owner="w")
    graph.complete("oracle", "w", artifacts=[payload], detail="fixtures built")
    task = graph.task("oracle")
    assert task.artifacts and len(task.artifacts[0]["sha256"]) == 64
    assert task.detail == "fixtures built"


def test_the_report_counts_statuses_without_inventing_progress(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60)
    graph.add([_task("a"), _task("b", kind="experiment", resource="gpu")])
    graph.tick(owner="w")
    report = graph.report()
    assert report["rounds"] == 0
    assert report["status_counts"].get(D.WAITING_AUTHORISATION) == 1
    assert report["experiment_tasks"] == 1
    assert report["experiment_done"] == 0


def test_a_task_that_depends_on_itself_is_refused(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock)
    with pytest.raises(D.SchedulerError):
        graph.add(D.Task(task_id="loop", candidate="arm", stage="L", deps=("loop",)))


def test_naming_an_unknown_dependency_is_refused(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock)
    with pytest.raises(D.SchedulerError):
        graph.add(_task("join_E", deps=("missing_task",)))


def test_tasks_are_immutable_once_added(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock)
    graph.add(_task("t"))
    with pytest.raises(D.SchedulerError):
        graph.add(_task("t"))


# --------------------------------------------------------------------------------------
# the supervised worker
# --------------------------------------------------------------------------------------
def test_a_supervised_worker_keeps_its_lease_alive_while_the_task_runs(tmp_path):
    """A task that outlives the TTL must not be reclaimed while its worker is heartbeating."""
    # the REAL clock here: a lease is a wall-clock claim, and a fake clock advanced inside the
    # task would test the fixture instead of the heartbeat
    graph = D.DAG(tmp_path, lease_ttl=0.3)
    graph.add(_task("long"))

    seen: list[list[str]] = []

    def run(task):
        time.sleep(0.9)                     # three lease TTLs
        seen.append(graph.tick(owner="other")["reclaimed"])

    graph.serve(run, owner="worker", max_ticks=1)
    assert seen == [[]], "the lease must hold while the worker heartbeats"
    assert graph.task("long").status == D.DONE


def test_a_supervised_worker_marks_a_raising_task_failed_and_retries(tmp_path):
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60)
    graph.add(D.Task(task_id="flaky", candidate="arm", stage="implement", write_scope=("out/f",),
                     max_attempts=2))
    attempts: list[int] = []

    def run(task):
        attempts.append(task.attempts)
        raise RuntimeError("boom")

    graph.serve(run, owner="worker", max_ticks=2)
    assert attempts == [1, 2]
    assert graph.task("flaky").status == D.BLOCKED
    assert "boom" in graph.task("flaky").error


# --------------------------------------------------------------------------------------
# the gate: the dependency graph itself refuses to release what follows an unmet gate
# --------------------------------------------------------------------------------------
def test_a_gated_task_refuses_to_complete_without_a_satisfied_measurement(tmp_path):
    """The ER gate before Stage-3 lives in the GRAPH, not only in the evaluation plan."""
    graph = D.DAG(tmp_path, lease_ttl=60)
    graph.add(D.Task(task_id="er_gate", candidate="arm", stage="er_gate",
                     requires={"metric": "hard_l1_er", "op": ">", "threshold": 16.38953942232631},
                     write_scope=("out/gate",)))
    graph.add(D.Task(task_id="stage3", candidate="arm", stage="stage3", deps=("er_gate",),
                     resource="gpu", kind="experiment", write_scope=("out/stage3",)))
    graph.tick(owner="w")
    assert graph.task("er_gate").status == D.RUNNING
    assert graph.task("stage3").status == D.PENDING, "Stage-3 waits for the gate"

    with pytest.raises(D.SchedulerError):
        graph.complete("er_gate", "w")                       # no measurement at all
    with pytest.raises(D.SchedulerError):
        graph.complete("er_gate", "w", metrics={"hard_l1_er": 16.2})
    with pytest.raises(D.SchedulerError):
        # strict: a value exactly AT the threshold does not clear a ">" gate
        graph.complete("er_gate", "w", metrics={"hard_l1_er": 16.38953942232631})
    assert graph.task("er_gate").status == D.RUNNING, "a refused gate is not completed"

    graph.complete("er_gate", "w", metrics={"hard_l1_er": 16.9})
    assert graph.task("er_gate").status == D.DONE
    assert graph.task("stage3").status == D.READY, "the gate releases what depends on it"
    recorded = graph.task("er_gate").as_dict().get("gate_measurement")
    assert recorded and recorded["metric"] == "hard_l1_er" and recorded["value"] == 16.9


def test_a_task_with_a_malformed_requirement_is_refused(tmp_path):
    graph = D.DAG(tmp_path, lease_ttl=60)
    with pytest.raises(D.SchedulerError):
        graph.add(D.Task(task_id="bad_gate", candidate="arm", stage="er_gate",
                         requires={"op": ">"}))


def test_a_ready_task_whose_dependency_was_closed_is_demoted_and_never_dispatched(tmp_path):
    """A promotion that outlives its dependency must not stay dispatchable.

    The campaign's queue held a stage2 that had been promoted while its join was DONE and
    that was still READY after the join was closed by a refutation. A tick then leased the
    stage2 of a family that had already been refuted twice. The rule: a READY task whose
    dependencies are not all DONE is demoted, and a closed dependency closes the task.
    """
    clock = Clock()
    # the GPU is at capacity, so the promoted stage2 stays READY across ticks -- the
    # situation the campaign was in when the join was closed under it
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60,
                  resource_limits={"cpu": 4, "gpu": 0, "none": 8})
    graph.add([_task("join"), _task("stage2", deps=("join",), stage="stage2", resource="gpu",
                                   kind="experiment")])
    graph.authorise("gpu", "operator", "the operator authorised the GPU work")

    graph.tick(owner="w1")
    graph.complete("join", "w1")
    graph.tick(owner="w1")
    assert graph.task("stage2").status == D.READY, "the stage2 is promoted once its join is done"

    # the join is closed afterwards: the family was refuted
    graph.set_status("join", D.BLOCKED, "refuted on its own measurements")
    dispatched = graph.tick(owner="w1")
    assert dispatched["dispatched"] == [], "a closed dependency must stop the dispatch"
    assert graph.task("stage2").status == D.BLOCKED
    assert graph.task("stage2").status in D.NOT_DISPATCHABLE


def test_a_ready_task_waits_for_an_unfinished_dependency_instead_of_running(tmp_path):
    """A dependency that is merely unfinished demotes the task to PENDING, not BLOCKED."""
    clock = Clock()
    graph = D.DAG(tmp_path, clock=clock, lease_ttl=60,
                  resource_limits={"cpu": 4, "gpu": 0, "none": 8})
    graph.add([_task("join"), _task("stage2", deps=("join",), stage="stage2", resource="gpu",
                                   kind="experiment")])
    graph.authorise("gpu", "operator", "the operator authorised the GPU work")
    graph.tick(owner="w1")
    graph.complete("join", "w1")
    graph.tick(owner="w1")
    assert graph.task("stage2").status == D.READY

    # the join is re-opened (its own dependency was closed), so it is unfinished again
    graph.set_status("join", D.PENDING, "re-opened")
    graph.tick(owner="w1")
    assert graph.task("stage2").status == D.PENDING
    assert graph.task("stage2").status != D.READY, "PENDING is not dispatchable"
    assert graph.task("stage2").status in D.NOT_DISPATCHABLE
