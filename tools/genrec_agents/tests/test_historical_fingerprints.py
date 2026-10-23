"""The historical fingerprint registry's tests (issue #109, section 2.2).

The registry is what stops an old mechanism from being resubmitted under a new name, so the
tests are about the behaviours the issue names: a rename is not novelty, a weight-only
change is not novelty, an old arm's resubmission is not novelty, and a genuinely new
mechanism is.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_historical_fingerprints.py -q
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


def _proposal(**overrides):
    proposal = {
        "mechanism_id": "candidate_one",
        "equation": "L = ||a - b||^2 + 0.05 * ||c_i - c_j||^2",
        "symbols": {"a": "the source", "b": "the target"},
        "parameters": {"weight": 0.05, "rho": 0.1},
        "gradient_targets": {"source": "nonzero", "target": "nonzero", "codebook": "nonzero"},
        "target_paths": ["codebook", "hard_sid_export"],
        "behaviour_signal": "the transition pair",
        "codebook_interaction": "the gradient pushes the two codewords apart",
        "encoder_interaction": "the encoder is untouched",
    }
    proposal.update(overrides)
    return proposal


def _registry(entries=None):
    return {
        "version": 1,
        "sources": {},
        "entries": entries or [],
        "historical_models": 0,
        "campaign_runs": 0,
    }


def test_renaming_a_mechanism_does_not_make_it_new():
    """A new name is not novelty."""
    proposal = _proposal()
    registry = _registry(
        [
            {
                "kind": "campaign_run",
                "id": "the_old_name",
                "primary": HF.fingerprint(proposal),
                "sid_effect": HF.fingerprint_components(proposal)["sid_effect"],
            }
        ]
    )
    verdict = HF.check_novelty(
        _proposal(mechanism_id="a_brand_new_name"), registry, mechanism_id="a_brand_new_name"
    )
    assert not verdict.novel
    assert verdict.matched_component == "primary"
    assert verdict.matched_id == "the_old_name"


def test_a_weight_only_change_is_not_novel():
    """The declared parameter values are not part of the mathematics."""
    original = _proposal()
    changed = _proposal(parameters={"weight": 0.5, "rho": 0.1})
    changed["equation"] = "L = ||a - b||^2 + 0.5 * ||c_i - c_j||^2"
    assert HF.fingerprint(original) == HF.fingerprint(changed)
    registry = _registry(
        [
            {
                "kind": "campaign_run",
                "id": "the_old_arm",
                "primary": HF.fingerprint(original),
                "sid_effect": HF.fingerprint_components(original)["sid_effect"],
            }
        ]
    )
    verdict = HF.check_novelty(changed, registry)
    assert not verdict.novel


def test_a_change_of_the_mathematics_is_novel():
    """A different objective is a different mechanism."""
    original = _proposal()
    different = _proposal(equation="L = ||a - b||^3 + 0.05 * ||c_i - c_j||^2")
    assert HF.fingerprint(original) != HF.fingerprint(different)
    registry = _registry(
        [
            {
                "kind": "campaign_run",
                "id": "the_old_arm",
                "primary": HF.fingerprint(original),
                "sid_effect": HF.fingerprint_components(original)["sid_effect"],
            }
        ]
    )
    verdict = HF.check_novelty(different, registry)
    assert verdict.novel


def test_whitespace_and_number_formatting_do_not_change_the_fingerprint():
    """The same mathematics written differently is the same mathematics."""
    first = _proposal(equation="L = ||a - b||^2 + 0.05*||c_i - c_j||^2")
    second = _proposal(equation="L=||a-b||^2  +  0.05 * ||c_i - c_j||^2")
    assert HF.fingerprint(first) == HF.fingerprint(second)
    assert HF.fingerprint(_proposal(equation="L = ||a - b||^2 + 5e-2 * ||c_i - c_j||^2")) == HF.fingerprint(
        _proposal(equation="L = ||a - b||^2 + 0.05 * ||c_i - c_j||^2")
    )


def test_the_same_sid_effect_is_reported_but_not_refused():
    """A substantive mathematical change is novelty, whatever the prose about the effect says."""
    original = _proposal()
    different_maths = _proposal(equation="L = ||a - b||^3 + 0.05 * ||c_i - c_j||^2")
    registry = _registry(
        [
            {
                "kind": "campaign_run",
                "id": "the_old_arm",
                "primary": HF.fingerprint(original),
                "sid_effect": HF.fingerprint_components(original)["sid_effect"],
            }
        ]
    )
    verdict = HF.check_novelty(different_maths, registry)
    assert verdict.novel
    assert verdict.matched_component == "sid_effect"
    assert "admitted with that recorded" in verdict.detail


def test_a_genuinely_new_mechanism_is_admitted():
    """A new objective with a new effect on the codes is novel."""
    registry = _registry(
        [
            {
                "kind": "campaign_run",
                "id": "the_old_arm",
                "primary": HF.fingerprint(_proposal()),
                "sid_effect": HF.fingerprint_components(_proposal())["sid_effect"],
            }
        ]
    )
    new = _proposal(
        equation="L = trace(S^T C) with C the behaviour covariance",
        codebook_interaction="the codebook rows rotate towards the behaviour principal axes",
        behaviour_signal="the behaviour covariance of the pair",
    )
    verdict = HF.check_novelty(new, registry)
    assert verdict.novel
    assert verdict.matched_component == ""


def test_an_archived_model_id_is_refused():
    """Resubmitting under an archived model's name is refused."""
    registry = _registry(
        [
            {
                "kind": "historical_model",
                "id": "TIGER_RQ-VAE",
                "primary": HF._sha256_text("historical_model:TIGER_RQ-VAE"),
                "sid_effect": HF._sha256_text("historical_model:TIGER_RQ-VAE"),
            }
        ]
    )
    verdict = HF.check_novelty(_proposal(), registry, mechanism_id="TIGER_RQ-VAE")
    assert not verdict.novel
    assert verdict.matched_component == "id"


def test_a_registry_whose_sources_changed_is_refused(tmp_path):
    """The registry is immutable: it cannot be used when its sources moved under it."""
    registry = HF.build_registry(REPO_ROOT)
    path = tmp_path / "registry.json"
    HF.save_registry(registry, path)
    loaded = HF.load_registry(path)
    assert loaded["registry_sha256"] == registry["registry_sha256"]
    # an edited registry is refused
    document = json.loads(path.read_text(encoding="utf-8"))
    document["entries"].append({"kind": "campaign_run", "id": "sneaky", "primary": "x"})
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        HF.load_registry(path)


def test_the_registry_covers_the_real_history_and_is_reproducible():
    """The real registry: the 38 archived models and the campaign's runs, built twice."""
    first = HF.build_registry(REPO_ROOT)
    second = HF.build_registry(REPO_ROOT)
    assert first["registry_sha256"] == second["registry_sha256"]
    assert first["historical_models"] == 38
    assert first["campaign_runs"] > 0
    summary = HF.summarise(first)
    assert summary["entries"] == len(first["entries"])
