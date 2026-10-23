"""The guardrail tests (issue #106, acceptance).

Each test attempts a PROHIBITED execution and asserts that the controller refuses it
before any process is launched and any artifact is written.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_guardrails.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import guardrails as G  # noqa: E402
from tools.genrec_agents.evidence import EvidenceStore, RunPlan  # noqa: E402


def _store_with(tmp_path: Path, *, mechanism_id: str, status: str) -> EvidenceStore:
    store = EvidenceStore(tmp_path / "store").open()
    plan = RunPlan(
        kind="stage2",
        mechanism_id=mechanism_id,
        output_dir=tmp_path / "out" / mechanism_id,
    )
    run = store.admit(plan)
    store.close(run, status=status)
    return store


def test_a_second_seed_is_refused(tmp_path):
    report = G.check_intent(G.ExecutionIntent(kind="stage2", seed=43))
    assert not report.allowed
    assert report.rule == "single_seed"
    with pytest.raises(G.GuardrailRefusal) as raised:
        G.refuse_if_prohibited(G.ExecutionIntent(kind="stage2", seed=43))
    assert "frozen seed" in str(raised.value)


def test_the_frozen_seed_is_allowed(tmp_path):
    report = G.check_intent(G.ExecutionIntent(kind="stage2", seed=G.FROZEN_SEED))
    assert report.allowed, report.detail


def test_retraining_the_control_is_refused(tmp_path):
    report = G.check_intent(G.ExecutionIntent(kind="stage2", mechanism_id="r8_control"))
    assert not report.allowed
    assert report.rule == "no_control_retrain"
    assert "archived control" in report.detail


def test_running_an_archived_policy_is_refused(tmp_path):
    report = G.check_intent(
        G.ExecutionIntent(kind="policy", mechanism_id="x", policy="old")
    )
    assert not report.allowed
    assert report.rule == "no_archived_policy_execution"


def test_rerunning_a_failed_arm_is_refused(tmp_path):
    store = _store_with(tmp_path, mechanism_id="failed_arm", status="failed")
    report = G.check_intent(
        G.ExecutionIntent(kind="stage2", mechanism_id="failed_arm"), store
    )
    assert not report.allowed
    assert report.rule == "no_failed_arm_rerun"
    assert "already has a failed run" in report.detail


def test_a_refused_arm_may_not_be_re_evaluated_either(tmp_path):
    store = _store_with(tmp_path, mechanism_id="refused_arm", status="refused")
    report = G.check_intent(
        G.ExecutionIntent(kind="stage3", mechanism_id="refused_arm"), store
    )
    assert not report.allowed
    assert report.rule == "no_failed_arm_rerun"


def test_a_new_candidate_is_allowed(tmp_path):
    store = _store_with(tmp_path, mechanism_id="old_arm", status="succeeded")
    report = G.check_intent(
        G.ExecutionIntent(kind="stage2", mechanism_id="brand_new_arm"), store
    )
    assert report.allowed, report.detail
    assert "single_seed" in report.checked
    assert "no_control_retrain" in report.checked
    assert "no_archived_policy_execution" in report.checked
    assert "no_failed_arm_rerun" in report.checked


def test_the_frozen_evaluation_may_not_be_consumed_twice(tmp_path):
    store = EvidenceStore(tmp_path / "store").open()
    plan = RunPlan(
        kind="stage2", mechanism_id="candidate", output_dir=tmp_path / "out" / "candidate"
    )
    run = store.admit(plan)
    store.close(
        run,
        status="succeeded",
        metrics={"frozen_evaluation": {"hard_l1_er": 16.5}},
    )
    report = G.check_intent(
        G.ExecutionIntent(
            kind="stage2", mechanism_id="candidate", consumes_frozen_evaluation=True
        ),
        store,
    )
    assert not report.allowed
    assert report.rule == "no_repeated_frozen_evaluation"
    assert "not a search oracle" in report.detail


def test_the_refusal_names_the_rule_and_the_reason(tmp_path):
    with pytest.raises(G.GuardrailRefusal) as raised:
        G.refuse_if_prohibited(G.ExecutionIntent(kind="stage2", mechanism_id="r8_control"))
    assert raised.value.rule == "no_control_retrain"
    assert raised.value.detail
