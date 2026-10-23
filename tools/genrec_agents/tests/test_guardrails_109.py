"""The guardrails' end-to-end acceptance tests (issue #109, section 7.2).

The issue names the prohibitions that have to be enforced before a process is launched: a
seed other than 42 for a Stage-2, a Stage-3 or a G_H1 probe; a repeated probe, cold run,
frozen evaluation or Stage-3; a control or an archived policy; and a mechanism that
repeats one that already exists.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_guardrails_109.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import historical_fingerprints as HF  # noqa: E402
from tools.genrec_agents.evidence import EvidenceStore, RunPlan  # noqa: E402
from tools.genrec_agents.guardrails import (  # noqa: E402
    ExecutionIntent,
    GuardrailRefusal,
    check_intent,
    refuse_if_prohibited,
)


def _store(tmp_path: Path) -> EvidenceStore:
    return EvidenceStore(tmp_path / "store").open()


def _admit(store: EvidenceStore, tmp_path: Path, **overrides) -> None:
    """Record a finished run in the store, as the controller does."""
    plan = {
        "kind": "stage2",
        "mechanism_id": "candidate_one",
        "stage": "cold_train",
        "output_dir": tmp_path / "out" / "candidate_one",
        "extra": {},
    }
    plan.update(overrides)
    run = store.admit(
        RunPlan(
            kind=plan["kind"],
            mechanism_id=plan["mechanism_id"],
            stage=plan["stage"],
            output_dir=plan["output_dir"],
            extra=plan["extra"],
        ),
        allow_existing_output=True,
    )
    store.close(run, status=overrides.get("status", "succeeded"), metrics=overrides.get("metrics"))


@pytest.mark.parametrize("kind", ["stage2", "stage3", "probe"])
def test_a_seed_other_than_42_is_refused_for_every_kind(tmp_path, kind):
    """One stochastic realisation per mechanism: no seed but 42, for any kind of run."""
    report = check_intent(ExecutionIntent(kind=kind, mechanism_id="candidate_one", seed=43))
    assert not report.allowed
    assert report.rule == "single_seed"
    assert report.checked[-1] == "single_seed"


@pytest.mark.parametrize("seed", [43, 44, 0, 12345])
def test_every_other_seed_is_refused(tmp_path, seed):
    report = check_intent(ExecutionIntent(kind="stage2", mechanism_id="candidate_one", seed=seed))
    assert not report.allowed
    assert report.rule == "single_seed"


def test_a_second_cold_run_of_the_same_arm_is_refused(tmp_path):
    """Exactly one official cold Stage-2 run per novel arm."""
    store = _store(tmp_path)
    _admit(store, tmp_path, kind="stage2", mechanism_id="candidate_one")
    report = check_intent(
        ExecutionIntent(kind="stage2", mechanism_id="candidate_one"), store
    )
    assert not report.allowed
    assert report.rule == "no_repeated_cold_run"


def test_a_second_probe_of_the_same_export_is_refused_even_under_another_name(tmp_path):
    """A probe is refused by the export it reads, not by the arm's name."""
    store = _store(tmp_path)
    _admit(
        store,
        tmp_path,
        kind="probe",
        mechanism_id="the_first_name",
        output_dir=tmp_path / "out" / "probe_one",
        extra={"sid_sha256": "abc123"},
    )
    report = check_intent(
        ExecutionIntent(
            kind="probe",
            mechanism_id="a_different_name",
            sid_sha256="abc123",
        ),
        store,
    )
    assert not report.allowed
    assert report.rule == "no_repeated_probe"
    assert "renamed arm" in report.detail


def test_a_repeated_frozen_evaluation_is_refused(tmp_path):
    store = _store(tmp_path)
    _admit(
        store,
        tmp_path,
        kind="stage2",
        mechanism_id="candidate_one",
        metrics={"frozen_evaluation": {"hard_l1_er": 16.9}},
    )
    report = check_intent(
        ExecutionIntent(
            kind="stage2", mechanism_id="candidate_one", consumes_frozen_evaluation=True
        ),
        store,
    )
    assert not report.allowed
    assert report.rule == "no_repeated_frozen_evaluation"


def test_a_repeated_stage3_is_refused(tmp_path):
    store = _store(tmp_path)
    _admit(
        store,
        tmp_path,
        kind="stage3",
        mechanism_id="candidate_one",
        output_dir=tmp_path / "out" / "s3",
        metrics={"stage3": {"test_recall@10": 0.06}},
    )
    report = check_intent(
        ExecutionIntent(kind="stage3", mechanism_id="candidate_one"), store
    )
    assert not report.allowed
    assert report.rule == "no_repeated_stage3"


def test_the_control_is_refused_for_every_kind(tmp_path):
    """The control is evidence, never a thing to run."""
    for kind in ("stage2", "stage3", "probe"):
        report = check_intent(ExecutionIntent(kind=kind, mechanism_id="r8_control"))
        assert not report.allowed
        assert report.rule == "no_control_retrain"


def test_an_archived_policy_is_refused(tmp_path):
    report = check_intent(
        ExecutionIntent(kind="policy", mechanism_id="candidate_one", policy="learned")
    )
    assert not report.allowed
    assert report.rule == "no_archived_policy_execution"


def test_a_mechanism_that_repeats_a_registered_one_is_refused(tmp_path, monkeypatch):
    """The mathematics is checked against the whole historical record."""
    import tools.genrec_agents.guardrails as guardrails

    proposal = {
        "equation": "L = ||a - b||^2 + 0.05 * ||c_i - c_j||^2",
        "parameters": {"weight": 0.05},
        "gradient_targets": {"source": "nonzero", "target": "nonzero", "codebook": "nonzero"},
        "target_paths": ["codebook"],
        "codebook_interaction": "the gradient pushes the two codewords apart",
    }
    fingerprint = HF.fingerprint(proposal)
    registry = {
        "version": 1,
        "sources": {},
        "entries": [
            {
                "kind": "campaign_proposal",
                "id": "an_old_arm",
                "primary": fingerprint,
                "sid_effect": "",
            }
        ],
        "historical_models": 0,
        "campaign_runs": 0,
    }
    registry["registry_sha256"] = HF._registry_digest(registry)
    path = tmp_path / "registry.json"
    HF.save_registry(registry, path)
    monkeypatch.setattr(HF, "DEFAULT_REGISTRY", path)

    report = check_intent(
        ExecutionIntent(
            kind="stage2",
            mechanism_id="a_new_name",
            fingerprint=fingerprint,
            sid_effect_fingerprint=HF.fingerprint_components(proposal)["sid_effect"],
        )
    )
    assert not report.allowed
    assert report.rule == "no_historical_repetition"
    assert "an_old_arm" in report.detail


def test_a_genuinely_new_mechanism_is_allowed(tmp_path, monkeypatch):
    """The newness rule must not refuse a mechanism that is actually new."""
    import tools.genrec_agents.guardrails as guardrails

    registry = {
        "version": 1,
        "sources": {},
        "entries": [],
        "historical_models": 0,
        "campaign_runs": 0,
    }
    registry["registry_sha256"] = HF._registry_digest(registry)
    path = tmp_path / "registry.json"
    HF.save_registry(registry, path)
    monkeypatch.setattr(HF, "DEFAULT_REGISTRY", path)

    report = check_intent(
        ExecutionIntent(
            kind="stage2",
            mechanism_id="a_new_name",
            fingerprint=HF.fingerprint({"equation": "L = trace(S^T C)"}),
        )
    )
    assert report.allowed
    assert "no_historical_repetition" in report.checked


def test_the_enforcing_wrapper_raises_before_anything_starts(tmp_path):
    """A prohibited intent raises at the guardrail, not at the launch."""
    with pytest.raises(GuardrailRefusal) as raised:
        refuse_if_prohibited(ExecutionIntent(kind="stage2", mechanism_id="r8_control"))
    assert raised.value.rule == "no_control_retrain"
