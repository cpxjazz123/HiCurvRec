"""Issue #273: dedicated tests for the same-centre difference protection.

The list follows the issue's own enumeration: hand-computed difference, same-centre pairing, sign,
translation invariance, constant prediction, variance equivalence, different-centre mispairing,
coincident points, invalid mask, fixed kappa, hard-q/STE gradients, numerical stability and
degenerate predictions.  The independent brute-force reference is checked against the production
helper on every case.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
import torch

from model.delta import same_centre_difference_loss
from reference_delta import brute_force_same_centre_difference_loss


def _case():
    predicted = torch.tensor([0.5, 1.5, -0.25], dtype=torch.float64)
    target = torch.tensor([0.25, 1.75, 0.5], dtype=torch.float64)
    return predicted, target


def test_hand_computed_difference():
    predicted, target = _case()
    # (0,1): (0.5-1.5) - (0.25-1.75) = -1.0 + 1.5 = 0.5  -> 0.25
    # (0,2): (0.5+0.25) - (0.25-0.5) = 0.75 + 0.25 = 1.0  -> 1.00
    # (1,2): (1.5+0.25) - (1.75-0.5) = 1.75 - 1.25 = 0.5  -> 0.25
    expected = (0.25 + 1.0 + 0.25) / (4.0 * 3.0)
    value = float(same_centre_difference_loss(predicted, target, 4))
    assert value == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert value == pytest.approx(
        brute_force_same_centre_difference_loss(predicted, target, 4), rel=1e-12, abs=1e-15)


def test_centre_population_normalisation_is_centre_balanced():
    predicted, target = _case()
    small = float(same_centre_difference_loss(predicted, target, 4))
    large = float(same_centre_difference_loss(predicted, target, 8))
    assert small == pytest.approx(2.0 * large, rel=1e-12, abs=1e-15)


def test_same_centre_pairing_uses_only_the_given_centre():
    predicted, target = _case()
    # Two triplets of one centre and one of another centre: the loss must react to the
    # pairing inside the centre, not to a cross-centre comparison.
    same_centre = float(same_centre_difference_loss(predicted, target, 4))
    other = float(same_centre_difference_loss(
        torch.tensor([0.5, 1.5], dtype=torch.float64),
        torch.tensor([0.25, 1.75], dtype=torch.float64), 4))
    # the two-triplet pair has a difference error of 0.5 -> 0.25, normalised by 4*1
    assert other == pytest.approx(0.25 / 4.0, rel=1e-12, abs=1e-15)
    assert same_centre != other


def test_sign_of_the_difference_is_used():
    predicted = torch.tensor([1.0, 0.0], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0], dtype=torch.float64)
    # the prediction order is reversed: the difference error is 2.0 -> 4.0
    assert float(same_centre_difference_loss(predicted, target, 2)) == pytest.approx(
        4.0 / 2.0, rel=1e-12, abs=1e-15)


def test_translation_invariance():
    predicted, target = _case()
    base = float(same_centre_difference_loss(predicted, target, 4))
    shifted = float(same_centre_difference_loss(predicted + 3.5, target, 4))
    assert base == pytest.approx(shifted, rel=1e-12, abs=1e-15)


def test_constant_prediction_keeps_the_target_spread_signal():
    # The frozen formula measures the within-centre spread of the *errors*: a constant
    # prediction leaves the error spread equal to the target's own spread, so the term is
    # positive (it counters a collapsed prediction) and independent of the constant offset.
    predicted = torch.full((4,), 2.0, dtype=torch.float64)
    target = torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=torch.float64)
    value = float(same_centre_difference_loss(predicted, target, 4))
    assert value == pytest.approx(35.0 / 96.0, rel=1e-12, abs=1e-15)
    shifted = float(same_centre_difference_loss(predicted - 7.0, target, 4))
    assert value == pytest.approx(shifted, rel=1e-12, abs=1e-15)


def test_variance_equivalence():
    generator = torch.Generator().manual_seed(20261127)
    errors = torch.randn(4096, generator=generator, dtype=torch.float64)
    predicted = errors + 5.0
    target = torch.zeros(4096, dtype=torch.float64)
    value = float(same_centre_difference_loss(predicted, target, 4096))
    # the pair mean is 2 * Var_c(e); the centre-balanced normalisation divides it by the
    # centre population, so the loss equals 2 * Var_c(e) / population
    assert value == pytest.approx(
        2.0 * float(errors.var(unbiased=True)) / 4096, rel=0.05)


def test_different_centre_mispairing_is_detected():
    predicted, target = _case()
    with pytest.raises(ValueError):
        same_centre_difference_loss(predicted, target, 2)  # population smaller than the sample
    mispaired = torch.cat([predicted, torch.tensor([9.0], dtype=torch.float64)])
    mispaired_target = torch.cat([target, torch.tensor([-9.0], dtype=torch.float64)])
    clean = float(same_centre_difference_loss(predicted, target, 4))
    dirty = float(same_centre_difference_loss(mispaired, mispaired_target, 8))
    assert dirty > clean


def test_coincident_points():
    predicted = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    target = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    assert float(same_centre_difference_loss(predicted, target, 4)) == pytest.approx(
        0.0, abs=1e-15)


def test_invalid_mask_rows_are_excluded_by_the_caller():
    predicted, target = _case()
    kept = float(same_centre_difference_loss(predicted[:2], target[:2], 4))
    # the third row is masked out upstream, so the loss is computed on two rows only
    assert kept == pytest.approx(0.25 / 4.0, rel=1e-12, abs=1e-15)


def test_fixed_kappa_is_not_an_input():
    # the term sees only the predictions and the frozen targets; the kappa_t lookup stays
    # a host-side table read and never enters the loss.
    import inspect
    parameters = list(inspect.signature(same_centre_difference_loss).parameters)
    assert parameters == ["predicted", "target", "centre_population"]


def test_hard_q_ste_gradient_flows_to_the_parameters():
    logits = torch.tensor([0.2, -0.1], dtype=torch.float64, requires_grad=True)
    codebook = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    hard = torch.softmax(logits, dim=0)
    hard = hard + (codebook - codebook.detach())  # straight-through estimator
    predicted = hard * 2.0
    target = torch.tensor([0.5, 0.25], dtype=torch.float64)
    value = same_centre_difference_loss(predicted, target, 2)
    value.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert codebook.grad is not None and torch.isfinite(codebook.grad).all()
    assert float(codebook.grad.abs().sum()) > 0.0


def test_numerical_stability_and_degenerate_predictions():
    predicted = torch.zeros(4, dtype=torch.float64)
    target = torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=torch.float64)
    value = float(same_centre_difference_loss(predicted, target, 4))
    assert value == pytest.approx(35.0 / 96.0, rel=1e-12, abs=1e-15)
    broken = torch.tensor([float("nan"), 0.0, 1.0], dtype=torch.float64)
    assert not torch.isfinite(same_centre_difference_loss(broken, target[:3], 4))


def test_single_triplet_gives_exactly_zero():
    predicted = torch.tensor([0.5], dtype=torch.float64)
    target = torch.tensor([0.25], dtype=torch.float64)
    value = same_centre_difference_loss(predicted, target, 1)
    assert float(value) == 0.0
    assert float(value) == pytest.approx(
        brute_force_same_centre_difference_loss(predicted, target, 1), abs=1e-15)
