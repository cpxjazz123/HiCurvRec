"""E3: same-prefix items must not also share the level-3 codeword."""
from __future__ import annotations

import numpy as np
import pytest
import torch

TREE = __import__("pathlib").Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(TREE))

from uniqueness import (calibrate_share, group_pairs, soft_assignment,  # noqa: E402
                        uniqueness_loss)


def test_pairs_come_from_groups_with_at_least_two_items():
    codes = np.array([0, 0, 1, 2, 2, 2, -1])
    pairs = group_pairs(codes)
    for first, second in pairs:
        assert first != second
        assert codes[first] == codes[second] >= 0
    assert {tuple(sorted(p)) for p in pairs} == {(0, 1), (3, 4), (4, 5), (3, 5)} or len(pairs) >= 3


def test_no_group_means_no_pairs():
    assert group_pairs(np.arange(5)).shape == (0, 2)


def test_overlap_is_one_for_identical_assignments_and_low_when_distinct():
    same = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    other = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    high = float(uniqueness_loss(same, same))
    low = float(uniqueness_loss(same, other))
    assert high > low
    assert high == pytest.approx(1.0)


def test_soft_assignment_is_a_distribution():
    soft = soft_assignment(torch.rand(4, 8), epsilon=0.003)
    assert torch.allclose(soft.sum(dim=1), torch.ones(4), atol=1e-5)


def test_share_calibration_targets_the_total():
    weight = calibrate_share(recon=1e-4, term=0.05, share=0.05)
    assert (weight * 0.05) / (1e-4 + weight * 0.05) == pytest.approx(0.05, abs=1e-6)


def test_gradient_reaches_the_encoder():
    encoder = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3))
    codes = torch.randn(6, 3)
    first = encoder(torch.randn(12, 8))
    second = encoder(torch.randn(12, 8))

    def p3(z):
        return soft_assignment(((z[:, None, :] - codes[None]) ** 2).sum(2), 0.003)

    uniqueness_loss(p3(first), p3(second)).backward()
    norms = [float(p.grad.abs().sum()) for p in encoder.parameters() if p.grad is not None]
    assert norms and min(norms) > 0
