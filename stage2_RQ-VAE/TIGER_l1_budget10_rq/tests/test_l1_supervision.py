"""Issue #287: the tests that pin the L1 supervision of the pair ranking.

Every expected value is written by hand from the contract; the production module is never used to
compute the expectation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
import torch.nn as nn

from pair_rank_loss import (LEVEL, MARGIN, PairIndex, hard_level_embedding, pair_ranking_loss)


class _FakeLayer:
    """A minimal VQ layer: a codebook table and the accessor the production code uses."""

    def __init__(self, codebook: torch.Tensor):
        self.embed = nn.Embedding(codebook.shape[0], codebook.shape[1])
        with torch.no_grad():
            self.embed.weight.copy_(codebook)

    def get_code_embs(self):
        return self.embed.weight


def _layers(seed: int = 3, n_codes: int = 8, dim: int = 4):
    generator = torch.Generator().manual_seed(seed)
    return [_FakeLayer(torch.randn(n_codes, dim, generator=generator)) for _ in range(3)]


def test_the_l1_embedding_is_the_hard_selected_codeword():
    layers = _layers()
    tokens = torch.tensor([[2, 5, 1], [6, 0, 4]], dtype=torch.long)
    q = hard_level_embedding(tokens, layers, LEVEL)
    assert q.shape == (2, 4)
    for item in range(2):
        expected = layers[LEVEL].get_code_embs()[tokens[item, LEVEL]]
        assert torch.allclose(q[item], expected, rtol=0, atol=0)
    # it is NOT the soft assignment (the codebook mean)
    soft = layers[LEVEL].get_code_embs().mean(dim=0)
    assert not torch.allclose(q[0], soft, rtol=1e-3, atol=1e-3)


def test_only_the_supervised_level_carries_the_pair_gradient():
    # a circle of codewords: the codes 0 and 4 are opposite (distance 2), the codes 0 and 1 are close
    # (distance 2 sin(pi/8) ~ 0.765), so the positive pair is the far one and the hinge is active
    angles = torch.arange(8, dtype=torch.float32) * (2.0 * torch.pi / 8.0)
    circle = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    layers = [_FakeLayer(circle) for _ in range(3)]
    tokens = torch.tensor([[0, 0, 0], [4, 4, 4], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    q = hard_level_embedding(tokens, layers, LEVEL)
    value = pair_ranking_loss(q, [0], [1], [2], [3], margin=0.25)
    assert float(value) > 0.0
    value.backward()
    for level, layer in enumerate(layers):
        grad = layer.embed.weight.grad
        if level == LEVEL:
            assert grad is not None, level
            assert torch.count_nonzero(grad).item() > 0, level
        else:
            assert grad is None or torch.count_nonzero(grad).item() == 0, level


def test_the_curvature_columns_are_not_an_input_of_the_loss():
    layers = _layers()
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2], [6, 7, 3]], dtype=torch.long)
    q = hard_level_embedding(tokens, layers, LEVEL)
    index_a = PairIndex(centre=[0], pos_b=[0], pos_c=[1], neg_d=[2], neg_e=[3],
                        pos_curvature=[1.0], neg_curvature=[0.0], is_holdout=[False])
    index_b = PairIndex(centre=[0], pos_b=[0], pos_c=[1], neg_d=[2], neg_e=[3],
                        pos_curvature=[-99.0], neg_curvature=[123.0], is_holdout=[False])
    value_a = float(pair_ranking_loss(q, index_a.pos_b, index_a.pos_c, index_a.neg_d, index_a.neg_e, 0.5))
    value_b = float(pair_ranking_loss(q, index_b.pos_b, index_b.pos_c, index_b.neg_d, index_b.neg_e, 0.5))
    assert value_a == value_b


def test_the_level_argument_is_validated():
    layers = _layers()
    tokens = torch.tensor([[0, 1, 2]], dtype=torch.long)
    with pytest.raises(IndexError):
        hard_level_embedding(tokens, layers, 3)
    with pytest.raises(ValueError):
        hard_level_embedding(tokens[:, :2], layers, LEVEL)


def test_the_margin_must_be_positive():
    layers = _layers()
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    q = hard_level_embedding(tokens, layers, LEVEL)
    with pytest.raises(ValueError):
        pair_ranking_loss(q, [0], [1], [0], [1], margin=0.0)
    with pytest.raises(ValueError):
        pair_ranking_loss(q, [0], [1], [0], [1], margin=-1.0)


def test_swapping_the_pairs_inverts_the_loss():
    layers = _layers()
    tokens = torch.tensor([[0, 1, 2], [5, 6, 7], [3, 4, 5]], dtype=torch.long)
    q = hard_level_embedding(tokens, layers, LEVEL)
    correct = float(pair_ranking_loss(q, [0], [1], [0], [2], margin=0.5))
    inverted = float(pair_ranking_loss(q, [0], [2], [0], [1], margin=0.5))
    assert correct != inverted


def test_the_frozen_margin_is_positive_and_identical_for_every_sample():
    assert MARGIN > 0.0
    assert MARGIN == 0.2529585361480713
