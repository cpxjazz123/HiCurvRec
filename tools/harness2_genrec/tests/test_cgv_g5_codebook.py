"""The G5 codebook repair's regression tests (issue #112, P0).

The distributed gate used to set the codebook deviation to ``NaN`` in the non-trace branch, so
the half of the check that a codebook-targeting mechanism actually needs was silently waived.
The repair pools the per-rank codebook gradients INDEPENDENTLY -- with the declared aggregation
oracle, on gradients the child computed without DDP -- and compares that pooling with what the
distributed path produced.

The tests cover both directions and the fail-closed rule:

* a matching payload passes (positive control),
* a perturbed production fails (failing control),
* a contract that targets the codebook but delivers no codebook gradients FAILS rather than
  passing by omission,
* and the R10 arm, run for real through the multi-rank child, passes the codebook half while its
  encoder half still fails -- because the arm's declared executor does not assemble the host's
  behaviour routing. That is the number the issue asks us to compare against (4.800e-02).

These tests need torch (the engine imports it), so run them under the project interpreter:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m pytest \
        tools/harness2_genrec/tests/test_cgv_g5_codebook.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CGV_DIR = REPO_ROOT / "tools" / "cgv"
if str(CGV_DIR) not in sys.path:
    sys.path.insert(0, str(CGV_DIR))

torch = pytest.importorskip("torch")

import engine  # noqa: E402
import oracles  # noqa: E402

AGGREGATION = "ddp_mean"
TOLERANCE = 1e-05


# --------------------------------------------------------------------------------------
# the declaration decides whether the codebook must be covered
# --------------------------------------------------------------------------------------
def _contract(target: str = "codebook", flag: bool | None = None) -> dict:
    distributed = {"world_size": 4, "aggregation": AGGREGATION, "atol": TOLERANCE,
                   "oracle": "oracles:aggregate"}
    if flag is not None:
        distributed["verify_encoder_and_codebook_gradients"] = flag
    return {"gradient_contract": {"target": target}, "distributed_contract": distributed}


def test_a_codebook_target_requires_the_codebook_check():
    assert engine.codebook_check_required(_contract("codebook")) is True


def test_an_encoder_target_with_the_flag_off_does_not_require_it():
    assert engine.codebook_check_required(_contract("encoder")) is False


def test_the_flag_requires_it_even_for_an_encoder_target():
    assert engine.codebook_check_required(_contract("encoder", flag=True)) is True


# --------------------------------------------------------------------------------------
# the reference the pooling is compared against
# --------------------------------------------------------------------------------------
def test_the_codebook_reference_is_the_base_plus_the_mechanism():
    """The production path backwards the base onto the codebook and runs the mechanism's own
    assembly; the behaviour terms reach the encoder, not the codebook."""
    base = [torch.ones(3)]
    mechanism = [torch.full((3,), 2.0)]
    behaviour = [torch.full((3,), 100.0)]
    reference = oracles.codebook_reference({"base_loss": base, "mechanism": mechanism,
                                            "l1_loss": behaviour})
    assert torch.equal(reference[0], torch.full((3,), 3.0))


def test_a_reference_without_a_base_gradient_is_refused():
    with pytest.raises(ValueError):
        oracles.codebook_reference({"mechanism": [torch.ones(2)]})


# --------------------------------------------------------------------------------------
# the pooling comparison: positive and failing controls
# --------------------------------------------------------------------------------------
def _local_codebook(world_size: int = 4, base: float = 1.0, mechanism: float = 0.5):
    """Per-rank local codebook gradients, as the child computes them without DDP."""
    local = []
    for rank in range(world_size):
        local.append({
            # the ranks differ, so a wrong pooling (e.g. taking rank 0) is visible
            "base_loss": [torch.full((4,), base + 0.1 * rank)],
            "mechanism": [torch.full((4,), mechanism + 0.2 * rank)],
        })
    return local


def test_a_matching_pooling_passes():
    """POSITIVE CONTROL: the production is the mean of what the ranks actually accumulated."""
    local = _local_codebook()
    mean_base = sum(1.0 + 0.1 * rank for rank in range(4)) / 4
    mean_mechanism = sum(0.5 + 0.2 * rank for rank in range(4)) / 4
    production = [torch.full((4,), mean_base + mean_mechanism)]
    deviation, names = engine.compare_pooled_codebook(
        local, production, oracles.aggregate, AGGREGATION, None)
    assert names == ["base_loss", "mechanism"]
    assert deviation <= TOLERANCE
    assert deviation == pytest.approx(0.0, abs=1e-6)  # float32 reduction noise


def test_a_production_that_dropped_the_mechanism_fails():
    """FAILING CONTROL: the mechanism's contribution is missing from the distributed path."""
    local = _local_codebook()
    mean_base = sum(1.0 + 0.1 * rank for rank in range(4)) / 4
    production = [torch.full((4,), mean_base)]
    deviation, _ = engine.compare_pooled_codebook(
        local, production, oracles.aggregate, AGGREGATION, None)
    assert deviation > TOLERANCE
    assert deviation == pytest.approx(sum(0.5 + 0.2 * rank for rank in range(4)) / 4, abs=1e-6)


def test_a_rank_that_double_counted_its_term_fails():
    """FAILING CONTROL: one rank's local production is not what its local gradient says."""
    local = _local_codebook()
    mean_base = sum(1.0 + 0.1 * rank for rank in range(4)) / 4
    mean_mechanism = sum(0.5 + 0.2 * rank for rank in range(4)) / 4
    production = [torch.full((4,), mean_base + 2.0 * mean_mechanism)]
    deviation, _ = engine.compare_pooled_codebook(
        local, production, oracles.aggregate, AGGREGATION, None)
    assert deviation > TOLERANCE


def test_the_weights_of_a_rank_sizeaware_aggregation_are_honoured():
    """The declared aggregation is honoured: with uneven rank sizes a plain mean is wrong."""
    local = _local_codebook(world_size=2)
    weights = [0.25, 0.75]
    # 0.25*(1.0 + 0.5) + 0.75*(1.1 + 0.7) = 1.725
    production = [torch.full((4,), 0.25 * (1.0 + 0.5) + 0.75 * (1.1 + 0.7))]
    deviation, _ = engine.compare_pooled_codebook(
        local, production, oracles.aggregate, AGGREGATION, weights)
    assert deviation <= TOLERANCE


# --------------------------------------------------------------------------------------
# the real multi-rank run on the R10 arm
# --------------------------------------------------------------------------------------
def _r10_spec() -> dict:
    path = (REPO_ROOT / "results/hyperbolic_research_runs/proposals"
            / "r10_horosphere_busemann.json")
    return json.loads(path.read_text(encoding="utf-8"))["cgv_contract"]


@pytest.mark.skipif(
    not (REPO_ROOT / "stage2_RQ-VAE/TIGER_r10_hyp_busemann_horosphere/train_rqvae.py").is_file(),
    reason="the R10 arm tree is not present")
def test_the_r10_arm_passes_the_codebook_half_and_still_fails_the_encoder_half():
    """The end-to-end check on a real codebook-targeting arm.

    The codebook half must now PASS (the repair's whole point), and the encoder half must still
    FAIL with the previously reported 4.800e-02: the arm's declared executor adds the horosphere
    term to the codebook but does not assemble the host's behaviour routing, so the declared
    ``unprojected`` rule does not describe what the executor does. G5 is not waived for it.
    """
    tree = REPO_ROOT / "stage2_RQ-VAE/TIGER_r10_hyp_busemann_horosphere"
    result = engine.run_cgv(tree, _r10_spec(), gates=("G5",))
    gate = result["gates"]["G5"]
    measurements = gate["measurements"]

    assert measurements["codebook_pooled_deviation"] <= TOLERANCE, measurements
    assert {"base_loss", "mechanism"} <= set(measurements["codebook_terms"])
    assert measurements["pooled_deviation"] > TOLERANCE, (
        "the encoder half should still fail: the declared rule does not match the executor")
    assert measurements["pooled_deviation"] == pytest.approx(4.800e-02, abs=2e-03), measurements
    assert gate["state"] == "failed"
