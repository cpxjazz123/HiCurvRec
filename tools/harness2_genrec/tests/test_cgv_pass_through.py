"""The CGV pass-through's tests (issue #110, section 7).

The declaration is caller-supplied data. A proposal that does not carry all seven groups is
blocked, a group that is present but empty is blocked, and the generic template is never
substituted for a contract that was not supplied -- a verifier that falls back to a template
verifies the template instead of the mechanism.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/harness2_genrec/tests/test_cgv_pass_through.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.harness2_genrec import candidates as CAND  # noqa: E402
from tools.harness2_genrec import mini_campaign  # noqa: E402


def _contract() -> dict:
    return {
        "mechanism": {"id": "arm", "type": "auxiliary_term", "entrypoints": ["mechanism_loss"]},
        "model": {"adapter": "rqvae", "parameter_groups": {"codebook": "codebook"}},
        "forward_contract": {"observables": ["loss"], "invariants": []},
        "gradient_contract": {"mode": "invariants", "target": "codebook", "invariants": []},
        "update_contract": {"optimizer": "adam", "lr": 1e-3},
        "distributed_contract": {"world_size": 4, "aggregation": "mean"},
        "runtime_contract": {"entrypoint": "train_rqvae.py", "observe": ["loss"]},
    }


def _candidate(**overrides):
    candidate = {
        "id": "arm_one",
        "title": "a hyperbolic term",
        "equation": "L = d_c(x, y)^2",
        "symbols": {"x": "the source"},
        "parameters": {"weight": 0.05},
        "gradient_targets": {"source": "zero", "target": "zero", "codebook": "nonzero"},
        "target_paths": ["codebook"],
        "behaviour_signal": "the transition pair",
        "codebook_interaction": "the codeword moves along the geodesic",
        "encoder_interaction": "none",
        "risk": "it may not change the export",
        "falsifier": "the export does not change",
        "why_not_the_measured_failures": "it is a geodesic term, not a pairwise pull",
        "hyperbolic_model": "poincare_ball",
        "hyperbolic_operator": "geodesic distance",
        "curvature": -1.0,
        "literature_ids": [],
        "cgv_contract": _contract(),
    }
    candidate.update(overrides)
    return candidate


def test_a_complete_contract_passes():
    assert CAND.validate_candidate(_candidate()) == []


def test_a_missing_group_is_blocking():
    contract = _contract()
    del contract["update_contract"]
    problems = CAND.validate_candidate(_candidate(cgv_contract=contract))
    assert problems
    assert any("update_contract" in problem for problem in problems)


def test_an_empty_group_is_blocking():
    contract = _contract()
    contract["gradient_contract"] = {}
    problems = CAND.validate_candidate(_candidate(cgv_contract=contract))
    assert problems
    assert any("gradient_contract" in problem for problem in problems)


def test_a_missing_contract_is_blocking():
    candidate = _candidate()
    del candidate["cgv_contract"]
    problems = CAND.validate_candidate(candidate)
    assert problems
    assert any("cgv_contract" in problem for problem in problems)


def test_a_contract_that_is_not_an_object_is_blocking():
    problems = CAND.validate_candidate(_candidate(cgv_contract="seven groups"))
    assert problems
    assert any("not an object" in problem for problem in problems)


def test_a_missing_hyperbolic_declaration_is_blocking():
    candidate = _candidate()
    del candidate["hyperbolic_operator"]
    problems = CAND.validate_candidate(candidate)
    assert problems
    assert any("hyperbolic" in problem for problem in problems)


def test_the_generic_template_is_never_substituted(tmp_path):
    """A proposal with no contract must block the launch, not fall back to the template."""
    with pytest.raises(mini_campaign.MissingContractError):
        mini_campaign._write_spec(
            tmp_path / "spec.json", "arm_one", {"weight": 0.05}, 0.05, contract=None
        )


def test_the_callers_contract_is_what_is_written(tmp_path):
    """The spec that reaches the verifier carries the caller's groups."""
    contract = _contract()
    contract["gradient_contract"]["mode"] = "invariants"
    contract["gradient_contract"]["target"] = "the_callers_target"
    path = mini_campaign._write_spec(
        tmp_path / "spec.json", "arm_one", {"weight": 0.05}, 0.05, contract=contract
    )
    spec = json.loads(path.read_text())
    for group in CAND.CGV_CONTRACT_GROUPS:
        assert group in spec
    # the caller's declaration is what is verified
    assert spec["gradient_contract"]["mode"] == "invariants"
    assert spec["gradient_contract"]["target"] == "the_callers_target"
    assert spec["update_contract"]["optimizer"] == "adam"
    # and the arm's own identity is the launch's truth, not the declaration's
    assert spec["mechanism"]["id"] == "arm_one"
