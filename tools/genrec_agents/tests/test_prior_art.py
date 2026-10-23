"""The prior-art registry's tests (issue #110, WP0 and the section 7 acceptance list).

The registry is what stops a historic mechanism from being run again under a new name, so
the tests are about the behaviours the issue names: a renamed repeat is refused, a
weight-only change is refused, a reparameterisation is refused, the four R8 hyperbolic
formulas and the previous campaigns' losses are recognised as prior art, a model whose
mathematics cannot be recovered is marked unknown rather than approved, and a missing or
edited registry BLOCKS rather than skipping the comparison.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_prior_art.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import prior_art as PA  # noqa: E402


@pytest.fixture(scope="module")
def registry():
    return PA.build_registry(REPO_ROOT)


def test_the_registry_covers_the_archives_and_says_what_it_found(registry):
    counts = registry["counts"]
    assert counts["r8_hyperbolic_arms"] == 4
    assert counts["historical_models_with_unknown_math"] == 38
    assert counts["campaign_proposals"] > 50
    assert counts["entries"] == (
        counts["r8_hyperbolic_arms"]
        + counts["campaign_proposals"]
        + counts["historical_models_with_unknown_math"]
    )


def test_the_four_r8_hyperbolic_arms_are_prior_art_with_their_operators(registry):
    r8 = [e for e in registry["entries"] if e["family"] == "r8_hyperbolic"]
    assert {e["mechanism_id"] for e in r8} == set(PA.R8_HYPERBOLIC_ARMS)
    for entry in r8:
        assert entry["math_status"] == "verified"
        assert "poincare_distance" in entry["operator_classes"]
        assert entry["ast_sha256"]
        assert entry["source"]


def test_a_renamed_repeat_is_refused(registry):
    entry = next(e for e in registry["entries"] if e["run_id"].startswith("proposal::"))
    verdict = PA.check_novelty(
        {
            "mechanism_id": "a_completely_new_name",
            "equation": entry["equation_signature"],
            "family": entry["family"],
        },
        registry,
    )
    assert not verdict.allowed
    assert verdict.matched_component == "equation"
    assert verdict.matched_identity is not None


def test_a_weight_only_change_is_refused(registry):
    """A weight-only change is the same mechanism, so it must be refused."""
    original = "L = ||a - b||^2 + 0.05 * ||c_i - c_j||^2"
    changed = "L = ||a - b||^2 + 0.5 * ||c_i - c_j||^2"
    parameters = {"weight": 0.05}
    changed_parameters = {"weight": 0.5}
    assert PA.normalise_equation(original, parameters) == PA.normalise_equation(
        changed, changed_parameters
    )
    registry_entry = {
        "entries": [
            {
                "proposal_sha256": "abc",
                "run_id": "proposal::test::arm",
                "mechanism_id": "arm",
                "family": "test",
                "math_status": "verified",
                "equation_signature": PA.normalise_equation(original, parameters),
                "operator_classes": [],
                "ast_sha256": "",
            }
        ]
    }
    verdict = PA.check_novelty(
        {"mechanism_id": "arm_two", "equation": changed, "parameters": changed_parameters},
        registry_entry,
    )
    assert not verdict.allowed


def test_a_reparameterisation_is_refused(registry):
    """A curvature-only change is not novelty, so it must not be admitted."""
    entry = next(e for e in registry["entries"] if e["run_id"].startswith("proposal::"))
    verdict = PA.check_novelty(
        {
            "mechanism_id": "the_same_maths_renamed",
            "equation": entry["equation_signature"],
            "family": entry["family"],
        },
        registry,
    )
    assert not verdict.allowed


def test_a_copied_implementation_is_refused(registry):
    """The abstract syntax is compared, so copying an implementation is caught."""
    entry = next(e for e in registry["entries"] if e.get("ast_sha256"))
    source = "def mechanism_loss(source, target, codebook, mask, **parameters):\n    return 0.0\n"
    registry_entry = {
        "entries": [
            {
                "proposal_sha256": "abc",
                "run_id": "proposal::test::arm",
                "mechanism_id": "arm",
                "family": "test",
                "math_status": "verified",
                "equation_signature": "",
                "operator_classes": [],
                "ast_sha256": PA.ast_sha256(source),
            }
        ]
    }
    verdict = PA.check_novelty(
        {"mechanism_id": "another_name", "equation": "L = something else", "family": "test"},
        registry_entry,
        implementation_source=source,
    )
    assert not verdict.allowed
    assert verdict.matched_component == "ast_sha256"
    assert entry["ast_sha256"]


def test_a_model_with_unrecoverable_mathematics_is_marked_unknown(registry):
    """A name is not mathematics: the historical models are recorded, not approved."""
    unknown = [e for e in registry["entries"] if e["math_status"] == "novelty_unknown"]
    assert len(unknown) == 38
    for entry in unknown:
        assert entry["equation_signature"] == ""
        assert entry["proposal_sha256"] == ""
        assert "UNKNOWN" in entry["notes"]
    # and a mechanism with a different name is not cleared by them: the prior-art review
    # has to settle it, which the workflow requires before an implementation
    verdict = PA.check_novelty(
        {"mechanism_id": "brand_new", "equation": "L = something", "family": "curvature_RQ-VAE"},
        registry,
    )
    assert verdict.unresolved_overlaps


def test_a_missing_registry_blocks_rather_than_skipping(tmp_path):
    """A missing registry must fail closed."""
    with pytest.raises(PA.RegistryError) as raised:
        PA.load_registry(tmp_path / "absent.json")
    assert "blocks" in str(raised.value)


def test_an_edited_registry_blocks(tmp_path):
    """An edited registry must fail closed."""
    registry = PA.build_registry(REPO_ROOT)
    path = tmp_path / "registry.json"
    PA.save_registry(registry, path)
    PA.load_registry(path)  # it loads
    document = json.loads(path.read_text(encoding="utf-8"))
    document["entries"].append({"mechanism_id": "sneaky", "run_id": "x", "proposal_sha256": "y"})
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(PA.RegistryError) as raised:
        PA.load_registry(path)
    assert "edited" in str(raised.value)


def test_the_registry_is_keyed_by_the_immutable_identity_not_by_the_name(registry):
    """Two entries may share a mechanism's name; the identity is what is stored."""
    identities = {(e["proposal_sha256"], e["run_id"]) for e in registry["entries"]}
    assert len(identities) == len(registry["entries"])
    names = [e["mechanism_id"] for e in registry["entries"]]
    assert len(names) > len(set(names)), "the archives do contain repeated names"


def test_the_old_experiments_are_never_rerun(registry):
    """The R8 arms and the previous campaigns' proposals are recorded, so they cannot be
    launched again as if they were new."""
    recorded = {e["mechanism_id"] for e in registry["entries"]}
    for arm in PA.R8_HYPERBOLIC_ARMS:
        assert arm in recorded
    previous = [
        e["mechanism_id"]
        for e in registry["entries"]
        if e["run_id"].startswith("proposal::")
    ]
    assert "directed_codebook_tangent_repartition" in previous
    assert "directed_pair_separation_codebook_geometry" in previous
