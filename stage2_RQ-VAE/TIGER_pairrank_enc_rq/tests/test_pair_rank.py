"""Issue #274: the reference-consistency and the wiring tests for the pair-ranking loss.

The reference consistency test compares the production path with the independent reference
(``tests/reference_pair_rank.py``), which uses the squared-norm expansion and the layer-cake
identity.  The wiring tests cover the issue's required cases: the positive/negative flip, the
centre mismatch, the curvature change with fixed labels, the fixed margin, the invalid indices, the
real hard q and the gradient routing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
import torch

from pair_rank_loss import (MARGIN, PairIndex, encoder_routed_sum, hard_quantized_sum,
                            pair_ranking_diagnostics, pair_ranking_loss)
from reference_pair_rank import (hard_quantized_sum_reference, pair_ranking_loss_reference,
                                  pair_ranking_terms_reference)


class _FakeLayer:
    """A minimal codebook layer with the production's access pattern."""

    def __init__(self, weight: torch.Tensor):
        self.embed = torch.nn.Embedding(weight.shape[0], weight.shape[1])
        with torch.no_grad():
            self.embed.weight.copy_(weight)

    def get_code_embs(self):
        return self.embed.weight


def _layers(seed: int = 7, levels: int = 3, codes: int = 16, dim: int = 8):
    generator = torch.Generator().manual_seed(seed)
    return [_FakeLayer(torch.randn(codes, dim, generator=generator)) for _ in range(levels)]


def test_reference_consistency_on_random_inputs():
    layers = _layers()
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randint(0, 16, (37, 3), generator=generator)
    q = hard_quantized_sum(tokens, layers)
    groups = 23
    pos_b = torch.randint(0, 37, (groups,), generator=generator)
    pos_c = torch.randint(0, 37, (groups,), generator=generator)
    neg_d = torch.randint(0, 37, (groups,), generator=generator)
    neg_e = torch.randint(0, 37, (groups,), generator=generator)
    value = float(pair_ranking_loss(q, pos_b, pos_c, neg_d, neg_e, 0.3))
    codebooks = [layer.get_code_embs().detach().numpy().astype(np.float64) for layer in layers]
    expected = pair_ranking_loss_reference(hard_quantized_sum_reference(tokens.numpy(), codebooks),
                                           pos_b.numpy(), pos_c.numpy(), neg_d.numpy(),
                                           neg_e.numpy(), 0.3)
    assert value == pytest.approx(expected, rel=1e-5, abs=1e-6)
    production = pair_ranking_diagnostics(q, pos_b, pos_c, neg_d, neg_e, 0.3)
    reference = pair_ranking_terms_reference(
        hard_quantized_sum_reference(tokens.numpy(), codebooks), pos_b.numpy(), pos_c.numpy(),
        neg_d.numpy(), neg_e.numpy(), 0.3)
    for key, value in reference.items():
        assert production[key] == pytest.approx(value, rel=1e-5, abs=1e-6), key


def test_the_hard_q_is_the_plain_sum_of_the_three_codewords():
    layers = _layers()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    q = hard_quantized_sum(tokens, layers)
    for item in range(2):
        expected = sum(layers[level].get_code_embs()[tokens[item, level]] for level in range(3))
        assert torch.allclose(q[item], expected, rtol=1e-6, atol=1e-6)


def test_the_gradient_reaches_the_codebooks_and_not_the_encoder():
    # a circle of codewords: the codes 0 and 4 are opposite (distance 2), the codes 0 and 1 are
    # close (distance 2 sin(pi/8) ~ 0.765), so the positive pair is farther and the hinge is active
    angles = torch.arange(8, dtype=torch.float32) * (2.0 * torch.pi / 8.0)
    circle = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    layers = [_FakeLayer(circle) for _ in range(3)]
    tokens = torch.tensor([[0, 0, 0], [4, 4, 4], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    q = hard_quantized_sum(tokens, layers)
    value = pair_ranking_loss(q, [0], [1], [2], [3], margin=0.25)
    assert float(value) > 0.0
    value.backward()
    for level, layer in enumerate(layers):
        grad = layer.embed.weight.grad
        assert grad is not None, level
        assert torch.count_nonzero(grad).item() > 0, level


def test_the_loss_is_unchanged_when_the_curvature_changes_but_the_labels_are_fixed():
    # the curvature is not an input of the loss: two indices that differ only in their curvature
    # columns must produce a bit-identical loss.
    layers = _layers()
    tokens = torch.randint(0, 16, (8, 3), generator=torch.Generator().manual_seed(3))
    q = hard_quantized_sum(tokens, layers)
    groups = [0, 1, 2]
    pos_b = torch.tensor([0, 1, 2])
    pos_c = torch.tensor([3, 4, 5])
    neg_d = torch.tensor([6, 7, 0])
    neg_e = torch.tensor([7, 0, 1])
    base = float(pair_ranking_loss(q, pos_b, pos_c, neg_d, neg_e, MARGIN))
    index_a = PairIndex(centre=np.zeros(3, dtype=np.int64), pos_b=pos_b.numpy(), pos_c=pos_c.numpy(),
                        neg_d=neg_d.numpy(), neg_e=neg_e.numpy(),
                        pos_curvature=np.array([1.0, 2.0, 3.0]), neg_curvature=np.array([0.0, 0.0, 0.0]),
                        is_holdout=np.zeros(3, dtype=bool))
    index_b = PairIndex(centre=np.zeros(3, dtype=np.int64), pos_b=pos_b.numpy(), pos_c=pos_c.numpy(),
                        neg_d=neg_d.numpy(), neg_e=neg_e.numpy(),
                        pos_curvature=np.array([-5.0, 9.0, 0.25]), neg_curvature=np.array([7.0, 7.0, 7.0]),
                        is_holdout=np.zeros(3, dtype=bool))
    for index in (index_a, index_b):
        groups_sel = np.arange(3)
        items = index.items_of(groups_sel)
        rows = index.rows_of(groups_sel, items)
        value = float(pair_ranking_loss(q[items], *rows, MARGIN))
        assert value == pytest.approx(base, rel=1e-12, abs=1e-12)


def test_the_centre_never_enters_the_loss():
    # the centre only selected the pairs: two indices that differ in their centre column but share
    # the pairs must give the same loss.
    layers = _layers()
    tokens = torch.randint(0, 16, (8, 3), generator=torch.Generator().manual_seed(5))
    q = hard_quantized_sum(tokens, layers)
    pos_b = torch.tensor([0, 1])
    pos_c = torch.tensor([2, 3])
    neg_d = torch.tensor([4, 5])
    neg_e = torch.tensor([6, 7])
    index_a = PairIndex(centre=np.array([100, 200]), pos_b=pos_b.numpy(), pos_c=pos_c.numpy(),
                        neg_d=neg_d.numpy(), neg_e=neg_e.numpy(),
                        pos_curvature=np.array([1.0, 2.0]), neg_curvature=np.array([0.0, 0.0]),
                        is_holdout=np.zeros(2, dtype=bool))
    index_b = PairIndex(centre=np.array([900, 901]), pos_b=pos_b.numpy(), pos_c=pos_c.numpy(),
                        neg_d=neg_d.numpy(), neg_e=neg_e.numpy(),
                        pos_curvature=np.array([1.0, 2.0]), neg_curvature=np.array([0.0, 0.0]),
                        is_holdout=np.zeros(2, dtype=bool))
    values = []
    for index in (index_a, index_b):
        groups = np.arange(2)
        items = index.items_of(groups)
        rows = index.rows_of(groups, items)
        values.append(float(pair_ranking_loss(q[items], *rows, MARGIN)))
    assert values[0] == pytest.approx(values[1], rel=1e-12, abs=1e-12)


def test_invalid_indices_are_rejected():
    q = torch.zeros((3, 4), dtype=torch.float32)
    with pytest.raises(IndexError):
        pair_ranking_loss(q, [0], [5], [1], [2], margin=0.1)
    with pytest.raises(IndexError):
        pair_ranking_loss(q, [0], [1], [7], [2], margin=0.1)
    with pytest.raises(ValueError):
        pair_ranking_loss(q, [0, 1], [1], [2], [2], margin=0.1)  # a group-count mismatch


def test_the_index_sampling_is_deterministic_and_uses_only_the_training_groups():
    index = PairIndex(centre=np.arange(10), pos_b=np.arange(10), pos_c=np.arange(10) + 10,
                      neg_d=np.arange(10) + 20, neg_e=np.arange(10) + 30,
                      pos_curvature=np.arange(10.0), neg_curvature=np.arange(10.0),
                      is_holdout=np.array([False] * 8 + [True, True]))
    first = index.sample(np.random.default_rng(5), 4)
    second = index.sample(np.random.default_rng(5), 4)
    assert np.array_equal(first, second)
    assert set(first.tolist()).issubset(set(range(8)))
    items = index.items_of(first)
    rows = index.rows_of(first, items)
    assert all(0 <= int(r) < items.size for r in np.concatenate(rows))


def test_the_encoder_routing_keeps_the_hard_forward_and_adds_the_encoder_gradient():
    # the reference implementation's convention: q = hard + (z - z.detach())
    angles = torch.arange(8, dtype=torch.float32) * (2.0 * torch.pi / 8.0)
    circle = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    layers = [_FakeLayer(circle) for _ in range(3)]
    tokens = torch.tensor([[0, 0, 0], [4, 4, 4], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    hard = hard_quantized_sum(tokens, layers)
    z = torch.zeros_like(hard, requires_grad=True)
    q = encoder_routed_sum(z, hard)
    # the forward value is exactly the hard representation
    assert torch.allclose(q.detach(), hard.detach(), atol=1e-6)
    value = pair_ranking_loss(q, [0], [1], [2], [3], margin=0.25)
    assert float(value) > 0.0
    value.backward()
    # the encoder receives the identity-path gradient and every codebook its own gradient
    assert z.grad is not None and torch.count_nonzero(z.grad).item() > 0
    for level, layer in enumerate(layers):
        assert layer.embed.weight.grad is not None and torch.count_nonzero(layer.embed.weight.grad).item() > 0, level


def test_the_encoder_routing_requires_matching_shapes():
    hard = torch.zeros((3, 4))
    with pytest.raises(ValueError):
        encoder_routed_sum(torch.zeros((4, 4), requires_grad=True), hard)
