"""Issue #274: the deterministic known-answer tests for the pair-ranking loss.

Every expected value is written by hand from the contract's formula; nothing is computed at
runtime by the production code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from pair_rank_loss import MARGIN, pair_ranking_loss, pair_ranking_diagnostics

# A two-dimensional representation so every distance is a hand-computable number.
Q = torch.tensor([
    [0.0, 0.0],   # item 0
    [3.0, 0.0],   # item 1
    [0.0, 4.0],   # item 2
    [1.0, 0.0],   # item 3
    [0.0, 1.0],   # item 4
], dtype=torch.float32)

# group 0: the positive pair (0, 1) -> distance 3 ; the negative pair (0, 4) -> distance 1
# group 1: the positive pair (0, 3) -> distance 1 ; the negative pair (1, 2) -> distance 5
# group 2: the positive pair (2, 3) -> distance sqrt(1 + 16) = sqrt(17) ; the negative pair (0, 3) -> distance 1
POS_B = torch.tensor([0, 0, 2])
POS_C = torch.tensor([1, 3, 3])
NEG_D = torch.tensor([0, 1, 0])
NEG_E = torch.tensor([4, 2, 3])


def test_hand_computed_hinge_mean():
    margin = 1.0
    d_pos = [3.0, 1.0, 17.0 ** 0.5]
    d_neg = [1.0, 5.0, 1.0]
    hinges = [max(0.0, p - n + margin) for p, n in zip(d_pos, d_neg)]
    expected = sum(hinges) / 3.0
    value = pair_ranking_loss(Q, POS_B, POS_C, NEG_D, NEG_E, margin)
    assert float(value) == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_inverted_order_is_penalised():
    # the correct order: the positive pair is the close one, the negative pair the far one -> the
    # hinge is satisfied; the inverted order makes the same four items violate the margin
    q = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 10.0]], dtype=torch.float32)
    correct = float(pair_ranking_loss(q, [0], [1], [2], [3], margin=1.0))
    inverted = float(pair_ranking_loss(q, [2], [3], [0], [1], margin=1.0))
    assert correct == pytest.approx(0.0, abs=1e-6)
    assert inverted == pytest.approx(10.0, rel=1e-6)
    assert inverted > correct


def test_insufficient_margin_is_zero():
    # a satisfied pair: d_pos + m <= d_neg -> the hinge is exactly zero
    q = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 10.0]], dtype=torch.float32)
    value = pair_ranking_loss(q, [0], [1], [0], [2], margin=1.0)
    assert float(value) == pytest.approx(0.0, abs=1e-9)
    # and the margin is what makes it satisfied: a smaller distance gap with a larger margin fails
    # the same pair with a margin that no longer fits: d_pos = 1, d_neg = 1.5
    q2 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.5]], dtype=torch.float32)
    assert float(pair_ranking_loss(q2, [0], [1], [0], [2], margin=0.25)) == pytest.approx(0.0, abs=1e-9)
    assert float(pair_ranking_loss(q2, [0], [1], [0], [2], margin=1.0)) == pytest.approx(0.5, abs=1e-6)
    assert float(pair_ranking_loss(q2, [0], [1], [0], [2], margin=2.0)) == pytest.approx(1.5, abs=1e-6)


def test_all_coincident_pairs_give_the_margin_but_no_gradient():
    q = torch.zeros((4, 2), dtype=torch.float32, requires_grad=True)
    value = pair_ranking_loss(q, [0], [1], [2], [3], margin=0.5)
    assert float(value) == pytest.approx(0.5, rel=1e-6, abs=1e-7)
    value.backward()
    # the hinge is at its flat region: the value is non-zero but the gradient is exactly zero
    assert torch.count_nonzero(q.grad).item() == 0


def test_the_margin_is_fixed_and_positive():
    assert MARGIN > 0
    with pytest.raises(ValueError):
        pair_ranking_loss(Q, POS_B, POS_C, NEG_D, NEG_E, margin=0.0)
    with pytest.raises(ValueError):
        pair_ranking_loss(Q, POS_B, POS_C, NEG_D, NEG_E, margin=-0.5)


def test_the_diagnostics_match_the_hand_computation():
    margin = 1.0
    diagnostics = pair_ranking_diagnostics(Q, POS_B, POS_C, NEG_D, NEG_E, margin)
    d_pos = [3.0, 1.0, 17.0 ** 0.5]
    d_neg = [1.0, 5.0, 1.0]
    assert diagnostics["ranking_accuracy"] == pytest.approx(
        sum(1.0 for p, n in zip(d_pos, d_neg) if p < n) / 3.0)
    assert diagnostics["margin_satisfaction_rate"] == pytest.approx(
        sum(1.0 for p, n in zip(d_pos, d_neg) if p + margin <= n) / 3.0)
    assert diagnostics["mean_distance_gap"] == pytest.approx(
        sum(n - p for p, n in zip(d_pos, d_neg)) / 3.0, rel=1e-6, abs=1e-6)
