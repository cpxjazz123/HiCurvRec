"""Issue #287 round 6: the deterministic known-answer tests for the hard-negative ranking."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from hard_negative_loss import (LEVEL, MARGIN, HardNegativeIndex, encoder_distances,
                                hard_negative_diagnostics, hard_negative_loss, mine_hard_negatives)


# a two-dimensional representation so every distance is hand-computable
Z = torch.tensor([
    [0.0, 0.0],   # 0 anchor
    [3.0, 0.0],   # 1 positive (distance 3)
    [0.0, 4.0],   # 2 negative (distance 4)
    [1.0, 0.0],   # 3
], dtype=torch.float32)
ANCHOR = torch.tensor([0, 0, 3])
POS = torch.tensor([1, 3, 2])
NEG = torch.tensor([2, 2, 0])


def test_hand_computed_hinge_mean():
    margin = 1.0
    # group 0: anchor (0,0), pos (3,0) -> 3, neg (0,4) -> 4      -> hinge 0
    # group 1: anchor (0,0), pos (1,0) -> 1, neg (0,4) -> 4      -> hinge 0
    # group 2: anchor (1,0), pos (0,4) -> sqrt(17), neg (0,0) -> 1 -> hinge sqrt(17)
    d_pos = [3.0, 1.0, 17.0 ** 0.5]
    d_neg = [4.0, 4.0, 1.0]
    hinges = [max(0.0, p - n + margin) for p, n in zip(d_pos, d_neg)]
    expected = sum(hinges) / 3.0
    value = hard_negative_loss(Z, ANCHOR, POS, NEG, margin)
    assert float(value) == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_inverted_order_is_penalised():
    # the negative is closer than the positive: the hinge must be strictly positive
    inverted = hard_negative_loss(Z, ANCHOR, NEG, POS, 0.0 + 1e-6)
    assert float(inverted) > 0.0


def test_insufficient_margin_is_zero():
    # a satisfied pair contributes nothing
    z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 5.0]], dtype=torch.float32)
    value = hard_negative_loss(z, [0], [1], [2], margin=1.0)
    assert float(value) == pytest.approx(0.0, abs=1e-7)


def test_the_margin_must_be_positive_and_is_frozen():
    with pytest.raises(ValueError):
        hard_negative_loss(Z, ANCHOR, POS, NEG, margin=0.0)
    with pytest.raises(ValueError):
        hard_negative_loss(Z, ANCHOR, POS, NEG, margin=-1.0)
    assert MARGIN > 0.0
    assert MARGIN == 0.04035949200819747


def test_the_gradient_reaches_the_representation_and_pushes_the_negative_away():
    z = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 1.0]], dtype=torch.float32, requires_grad=True)
    loss = hard_negative_loss(z, [0], [1], [2], margin=0.5)
    loss.backward()
    assert z.grad is not None
    assert float(torch.linalg.vector_norm(z.grad)) > 0.0
    # The stored gradient is the direction of the loss increase, so the optimiser moves against it:
    # the negative (item 2) must end up further from the anchor (its gradient points back towards it)
    negative_grad = z.grad[2]
    assert float((negative_grad * Z[2]).sum()) < 0.0
    # the positive (item 1) must end up closer to the anchor (its gradient points away from it)
    positive_grad = z.grad[1]
    assert float((positive_grad * Z[1]).sum()) > 0.0


def test_the_term_does_not_touch_the_codebooks():
    # the loss is a function of the representation only: a codebook parameter that is not part of
    # the graph receives no gradient from the term
    codebook = torch.nn.Parameter(torch.zeros(4, 2))
    z = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 1.0]], dtype=torch.float32, requires_grad=True)
    loss = hard_negative_loss(z, [0], [1], [2], margin=0.5)
    loss.backward()
    assert codebook.grad is None
    assert z.grad is not None


def test_the_index_sampling_is_deterministic_and_uses_only_the_training_groups():
    index = HardNegativeIndex(anchor=[0, 1, 2, 3], pos=[4, 5, 6, 7], neg=[8, 9, 10, 11],
                               is_holdout=[False, True, False, True],
                               nbr_ptr=[0, 0, 0, 0, 0], nbr_items=np.zeros(0, dtype=np.int64))
    rng = np.random.default_rng(0)
    first = index.sample(rng, 10)
    rng = np.random.default_rng(0)
    second = index.sample(rng, 10)
    assert np.array_equal(first, second)
    assert set(first.tolist()).issubset({0, 2})


def test_the_index_rejects_a_group_without_a_negative():
    with pytest.raises(ValueError):
        HardNegativeIndex(anchor=[0], pos=[1], neg=[-1], is_holdout=[False],
                           nbr_ptr=[0, 0], nbr_items=np.zeros(0, dtype=np.int64))


def test_the_rows_and_items_mapping():
    index = HardNegativeIndex(anchor=[0, 1], pos=[2, 3], neg=[4, 5], is_holdout=[False, False],
                               nbr_ptr=[0, 0, 0], nbr_items=np.zeros(0, dtype=np.int64))
    groups = np.array([0, 1])
    items = index.items_of(groups)
    assert items.tolist() == [0, 1, 2, 3, 4, 5]
    anchor, pos, neg = index.rows_of(groups, items)
    assert anchor.tolist() == [0, 1]
    assert pos.tolist() == [2, 3]
    assert neg.tolist() == [4, 5]


def _index_with_neighbours():
    # item 0's direct neighbours are 1 and 2
    nbr_ptr = np.array([0, 2, 2, 2], dtype=np.int64)
    nbr_items = np.array([1, 2], dtype=np.int64)
    return HardNegativeIndex(anchor=[0], pos=[1], neg=[5], is_holdout=[False],
                              nbr_ptr=nbr_ptr, nbr_items=nbr_items)


def test_the_mining_excludes_the_direct_neighbours():
    index = _index_with_neighbours()
    # the items 0, 1, 2 share the code 7; 1 and 2 are the anchor's direct neighbours
    codes = np.array([7, 7, 7, 8, 8, 9], dtype=np.int64)
    negatives, bucket, stats = mine_hard_negatives(index, codes, np.random.default_rng(0))
    assert negatives.tolist() == [5] or negatives.tolist() == [0]
    assert stats['refreshed'] == 0  # every co-sharer is a direct neighbour
    assert negatives[0] == 5  # the previous negative is kept


def test_the_mining_picks_a_measured_false_positive():
    index = _index_with_neighbours()
    # the items 0 and 3 share the code 7; 3 is not a direct neighbour of 0
    codes = np.array([7, 7, 7, 7, 8, 9], dtype=np.int64)
    negatives, bucket, stats = mine_hard_negatives(index, codes, np.random.default_rng(0))
    assert negatives.tolist() == [3]
    assert stats['refreshed'] == 1
    assert stats['groups'] == 1


def test_the_mining_is_deterministic_for_a_frozen_seed():
    index = _index_with_neighbours()
    codes = np.array([7, 7, 7, 7, 7, 9], dtype=np.int64)
    first, _, _ = mine_hard_negatives(index, codes, np.random.default_rng(42))
    second, _, _ = mine_hard_negatives(index, codes, np.random.default_rng(42))
    assert first.tolist() == second.tolist()


def test_the_diagnostics_report_the_ranking_state():
    stats = hard_negative_diagnostics(Z, ANCHOR, POS, NEG, margin=1.0)
    assert stats["pair_loss"] >= 0.0
    assert 0.0 <= stats["ranking_accuracy"] <= 1.0
    assert stats["mean_positive_distance"] == pytest.approx((3.0 + 1.0 + 17.0 ** 0.5) / 3.0, rel=1e-5)


def test_the_level_is_the_l1_code():
    assert LEVEL == 0


def test_the_distances_reject_an_index_outside_the_representation():
    with pytest.raises(IndexError):
        encoder_distances(Z, [0], [99], [2])


def test_the_weighted_reduction_is_hand_computable():
    # group 0: d_pos 3, d_neg 4 -> hinge 0 with margin 1; weight 1
    # group 1: d_pos 1, d_neg 4 -> hinge 0; weight 2
    # group 2: d_pos sqrt(17), d_neg 1 -> hinge sqrt(17); weight 7
    hinges = [0.0, 0.0, 17.0 ** 0.5]
    weights = [1, 2, 7]
    expected = sum(h * w for h, w in zip(hinges, weights)) / sum(weights)
    value = hard_negative_loss(Z, ANCHOR, POS, NEG, 1.0, torch.tensor(weights))
    assert float(value) == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_the_weighted_reduction_normalises_by_the_weight_sum():
    # scaling every weight by a constant does not change the loss
    base = hard_negative_loss(Z, ANCHOR, POS, NEG, 1.0, torch.tensor([1, 2, 7]))
    scaled = hard_negative_loss(Z, ANCHOR, POS, NEG, 1.0, torch.tensor([10, 20, 70]))
    assert float(base) == pytest.approx(float(scaled), rel=1e-6)


def test_the_weights_must_match_the_groups():
    with pytest.raises(ValueError):
        hard_negative_loss(Z, ANCHOR, POS, NEG, 1.0, torch.tensor([1, 2]))
    with pytest.raises(ValueError):
        hard_negative_loss(Z, ANCHOR, POS, NEG, 1.0, torch.tensor([0, 0, 0]))


def test_the_mining_reports_the_bucket_sizes():
    index = _index_with_neighbours()
    # items 0, 1, 2 share the code 7; 4 and 5 share the code 8
    codes = np.array([7, 7, 7, 8, 9, 9], dtype=np.int64)
    negatives, bucket, stats = mine_hard_negatives(index, codes, np.random.default_rng(0))
    assert bucket.tolist() == [3]          # the anchor 0's bucket holds the items 0, 1, 2
    assert stats['mean_bucket_size'] == pytest.approx(3.0)
    assert stats['max_bucket_size'] == 3


def test_the_bucket_size_refresh_validates():
    index = _index_with_neighbours()
    index.set_bucket_size(np.array([5], dtype=np.int64))
    assert index.bucket_size.tolist() == [5]
    with pytest.raises(ValueError):
        index.set_bucket_size(np.array([0], dtype=np.int64))
    with pytest.raises(ValueError):
        index.set_bucket_size(np.array([1, 2], dtype=np.int64))
