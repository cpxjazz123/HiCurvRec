"""Issue #267 known-answer and edge tests for the relative ranking term."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from model.relative import relative_rank_loss


def test_known_sign_and_hand_value():
    target = torch.tensor([0.0, 1.0], dtype=torch.float64)
    correct = relative_rank_loss(torch.tensor([0.0, 2.0]), target, [0, 0], tau=1.0)
    wrong = relative_rank_loss(torch.tensor([0.0, -2.0]), target, [0, 0], tau=1.0)
    assert correct["valid_pairs"] == 1
    assert wrong["valid_pairs"] == 1
    assert correct["loss"].item() == pytest.approx(float(F.softplus(torch.tensor(-2.0))), abs=1e-12)
    assert wrong["loss"].item() == pytest.approx(float(F.softplus(torch.tensor(2.0))), abs=1e-12)
    assert correct["loss"] < wrong["loss"]


def test_upper_triangle_pair_count_28_rows():
    n = 28
    target = torch.arange(n, dtype=torch.float64)
    result = relative_rank_loss(target, target, [0] * n)
    assert result["all_pairs"] == n * (n - 1) // 2
    assert result["valid_pairs"] == n * (n - 1) // 2
    assert result["tied_pairs"] == 0
    assert result["cross_centre_pairs"] == 0


def test_ties_are_excluded_not_averaged_in():
    result = relative_rank_loss(torch.tensor([0.0, 100.0, 0.0]),
                                torch.tensor([0.0, 0.0, 1.0]),
                                [0, 0, 0], tau=1.0)
    assert result["all_pairs"] == 3
    assert result["valid_pairs"] == 2
    assert result["tied_pairs"] == 1
    # (0,1) is a target tie and must be excluded; the two valid pairs are
    # softplus(0) and softplus(100), so a mean over all three is detectably wrong.
    assert float(result["loss"]) == pytest.approx((math.log(2.0) + 100.0) / 2.0, rel=1e-6)


def test_cross_centre_pairs_are_never_used():
    result = relative_rank_loss(torch.tensor([0.0, 1.0, 0.0]),
                                torch.tensor([0.0, 2.0, 3.0]),
                                [0, 1, 2], tau=1.0)
    assert result["all_pairs"] == 3
    assert result["valid_pairs"] == 0
    assert result["cross_centre_pairs"] == 3
    assert float(result["loss"]) == 0.0


def test_singleton_and_empty_are_differentiable_zero():
    for target in (torch.tensor([1.0]), torch.tensor([1.0, 2.0])):
        pred = target.detach().clone().requires_grad_(True)
        centres = [0] * len(target)
        result = relative_rank_loss(pred, target, centres, tau=1.0)
        if len(target) == 1:
            assert result["valid_pairs"] == 0
            assert float(result["loss"]) == 0.0
            result["loss"].backward()
            assert pred.grad is not None and float(pred.grad.abs().sum()) == 0.0


def test_gradient_direction_and_finite():
    predicted = torch.tensor([0.2, -0.1, 0.6], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64)
    result = relative_rank_loss(predicted, target, [0, 0, 0], tau=1.0)
    result["loss"].backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert float(predicted.grad.abs().sum()) > 0


def test_invalid_arguments_fail_closed():
    with pytest.raises(ValueError):
        relative_rank_loss(torch.zeros(2), torch.zeros(3), [0, 0])
    with pytest.raises(ValueError):
        relative_rank_loss(torch.zeros(2), torch.zeros(2), [0, 0], tau=0.0)
    with pytest.raises(ValueError):
        relative_rank_loss(torch.tensor([0.0, float("nan")]), torch.zeros(2), [0, 0])
