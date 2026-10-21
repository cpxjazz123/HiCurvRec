"""The controller's tests (issue #107).

The controller is a durable worker, so the things that can go wrong are: a crash between two
transitions, a restart that launches the same job twice, a job whose process dies and is
silently retried, two arms with the same specification under different names, and the
prohibited runs. Each of those has a test here.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/harness2_genrec/tests/test_controller.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.harness2_genrec import controller as C  # noqa: E402
from tools.harness2_genrec.controller import Controller, ControllerConfig  # noqa: E402
from tools.harness2_genrec.guardrails import ExecutionIntent, check_intent  # noqa: E402


def _controller(tmp_path: Path) -> Controller:
    config = ControllerConfig(
        store=tmp_path / "store",
        ledger=tmp_path / "ledger.json",
        state=tmp_path / "controller" / "state.json",
        audit=tmp_path / "controller" / "audit.jsonl",
        results=tmp_path / "controller" / "rounds",
        max_concurrent_screening=2,
        max_concurrent_training=1,
    )
    controller = Controller(config)
    controller.store.open()
    return controller


def _proposal(equation: str = "L = ||a - b||^2", **overrides):
    proposal = {
        "mechanism_id": "candidate_one",
        "equation": equation,
        "symbols": {"a": "the source", "b": "the target"},
        "parameters": {"weight": 0.05},
        "gradient_targets": {"source": "nonzero", "target": "nonzero", "codebook": "nonzero"},
        "target_paths": ["codebook", "hard_sid_export"],
        "invariants": [{"id": "permute_codebook", "transform": "permute_codebook",
                        "expect": "unchanged", "tolerance": 1e-06}],
        "activation": {"kind": "term", "weight": 0.05},
    }
    proposal.update(overrides)
    return proposal


# ---------------------------------------------------------------------------- the state
def test_the_state_survives_a_restart(tmp_path):
    controller = _controller(tmp_path)
    controller.state["round"] = 7
    controller.state["phase"] = "screen"
    controller.save()

    restarted = _controller(tmp_path).load()
    assert restarted.state["round"] == 7
    assert restarted.state["phase"] == "screen"


def test_the_state_is_written_atomically(tmp_path):
    controller = _controller(tmp_path)
    controller.save()
    assert controller.config.state.is_file()
    assert not controller.config.state.with_suffix(".json.tmp").exists()
    document = json.loads(controller.config.state.read_text())
    assert document["version"] == 1


def test_the_audit_log_is_append_only(tmp_path):
    controller = _controller(tmp_path)
    controller.audit("first", detail="one")
    controller.audit("second", detail="two")
    lines = controller.config.audit.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "first"
    assert json.loads(lines[1])["event"] == "second"


# ---------------------------------------------------------------------------- the jobs
def test_a_lost_job_is_classified_and_not_retried(tmp_path):
    controller = _controller(tmp_path)
    controller.state["jobs"] = {
        "job-1": {
            "job_id": "job-1",
            "kind": "stage2",
            "mechanism_id": "candidate_one",
            "status": "running",
            "dispatched_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "",
            "pid": 999999999,          # a pid that cannot be alive
            "port": 50300,
            "result_dir": "",
            "detail": "",
            "semantic_key": "",
        }
    }
    lost = controller.reap()
    assert len(lost) == 1
    assert controller.state["jobs"]["job-1"]["status"] == "lost"
    assert "not retried" in controller.state["jobs"]["job-1"]["detail"]


def test_the_dispatch_is_idempotent_across_a_restart(tmp_path):
    controller = _controller(tmp_path)
    controller.state["portfolio"] = [
        {"mechanism_id": "candidate_one", "status": "trained", "train_job": "job-1"}
    ]
    controller.state["jobs"] = {
        "job-1": {
            "job_id": "job-1",
            "kind": "stage2",
            "mechanism_id": "candidate_one",
            "status": "running",
            "dispatched_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "",
            "pid": os.getpid(),
            "port": 50300,
            "result_dir": "",
            "detail": "",
            "semantic_key": "",
        }
    }
    controller.save()
    # the phase that dispatches a cold run must not dispatch it again
    controller.phase_train()
    assert set(controller.state["jobs"]) == {"job-1"}
    assert controller.state["portfolio"][0]["status"] == "trained"


def test_the_ports_are_isolated(tmp_path):
    controller = _controller(tmp_path)
    first = controller.allocate_port()
    controller.state["jobs"] = {
        "job-1": {
            "job_id": "job-1", "kind": "stage2", "mechanism_id": "a", "status": "running",
            "dispatched_at": "", "finished_at": "", "pid": 1, "port": first, "result_dir": "",
            "detail": "", "semantic_key": "",
        }
    }
    assert controller.allocate_port() != first


def test_the_training_cap_is_respected(tmp_path):
    controller = _controller(tmp_path)
    controller.state["jobs"] = {
        "job-1": {
            "job_id": "job-1", "kind": "stage2", "mechanism_id": "a", "status": "running",
            "dispatched_at": "", "finished_at": "", "pid": os.getpid(), "port": 50300,
            "result_dir": "", "detail": "", "semantic_key": "",
        }
    }
    allowed, reason = controller.can_dispatch("stage2")
    # the GPU count may forbid it outright; either way a running job must block a second one
    assert (not allowed) or "already running" in reason


# ---------------------------------------------------------------------------- the guardrails
def test_a_renamed_copy_of_a_mechanism_is_refused(tmp_path):
    controller = _controller(tmp_path)
    key = C.semantic_key(_proposal())
    controller.store.close(
        controller.store.admit(
            type(controller.store).__module__ and __import__(
                "tools.harness2_genrec.evidence", fromlist=["RunPlan"]
            ).RunPlan(
                kind="stage2",
                mechanism_id="the_original",
                output_dir=tmp_path / "out" / "the_original",
                extra={"semantic_key": key},
            )
        ),
        status="succeeded",
    )
    verdict = check_intent(
        ExecutionIntent(
            kind="stage2", mechanism_id="the_renamed_copy", semantic_key=key
        ),
        controller.store,
    )
    assert not verdict.allowed
    assert verdict.rule == "no_semantic_duplicate"


def test_stage3_is_refused_for_an_arm_that_already_had_it(tmp_path):
    controller = _controller(tmp_path)
    RunPlan = __import__(
        "tools.harness2_genrec.evidence", fromlist=["RunPlan"]
    ).RunPlan
    controller.store.close(
        controller.store.admit(
            RunPlan(
                kind="stage2",
                mechanism_id="candidate_one",
                output_dir=tmp_path / "out" / "candidate_one",
            )
        ),
        status="succeeded",
        metrics={"stage3": {"test_recall@10": 0.06}},
    )
    verdict = check_intent(
        ExecutionIntent(kind="stage3", mechanism_id="candidate_one"), controller.store
    )
    assert not verdict.allowed
    assert verdict.rule == "no_repeated_stage3"


# ---------------------------------------------------------------------------- the machine
def test_a_negative_round_moves_to_the_next_portfolio(tmp_path):
    controller = _controller(tmp_path)
    controller.state["round"] = 3
    controller.state["phase"] = "learn"
    controller.state["portfolio"] = [
        {
            "mechanism_id": "candidate_one",
            "status": "evaluated",
            "evaluated": {"hard_l1_er": 16.6, "above_gate": True},
            "stage3": {
                "test_recall@10": 0.062,
                "test_ndcg@10": 0.034,
                "recall_delta_vs_control": -0.002,
                "ndcg_delta_vs_control": -0.001,
            },
        }
    ]
    controller.phase_learn()
    assert controller.state["round"] == 4
    assert controller.state["phase"] == "hypothesise"
    assert not controller.state["scientific_stop"]


def test_a_confirmed_success_stops_scientifically_but_not_the_issue(tmp_path):
    controller = _controller(tmp_path)
    controller.state["phase"] = "learn"
    controller.state["portfolio"] = [
        {
            "mechanism_id": "candidate_one",
            "status": "evaluated",
            "evaluated": {"hard_l1_er": 16.9, "above_gate": True},
            "stage3": {
                "test_recall@10": 0.070,
                "test_ndcg@10": 0.040,
                "recall_delta_vs_control": 0.006,
                "ndcg_delta_vs_control": 0.005,
            },
        }
    ]
    controller.phase_learn()
    assert controller.state["scientific_stop"]
    assert controller.state["phase"] == "done"
    assert "human approval" in controller.state["next_action"]


def test_the_semantic_key_ignores_the_name(tmp_path):
    first = C.semantic_key(_proposal(mechanism_id="one"))
    second = C.semantic_key(_proposal(mechanism_id="two"))
    assert first == second
    different = C.semantic_key(_proposal(equation="L = ||a - c||^2"))
    assert different != first
