"""R06 token-residual known answers and switch behaviour."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.token_residual import (identity_projection, level2_target,
                                  masked_token_mean)


def test_masked_token_mean_hand_values():
    tokens = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]],
                           [[7., 8.], [9., 10.], [11., 12.]]])
    mask = torch.tensor([[True, True, False], [False, False, False]])
    out = masked_token_mean(tokens, mask)
    assert torch.equal(out, torch.tensor([[2., 3.], [0., 0.]]))


def test_masked_token_mean_requires_mask_shape():
    try:
        masked_token_mean(torch.zeros(2, 3, 4), torch.zeros(2, 2, dtype=torch.bool))
    except ValueError:
        return
    raise AssertionError("a wrong mask shape must be refused")


def test_identity_projection_starts_as_identity():
    projection = identity_projection(32)
    assert torch.equal(projection.weight, torch.eye(32))
    assert torch.count_nonzero(projection.bias) == 0


def test_scaled_token_offset_uses_the_codebook_resolution():
    from model.token_residual import scaled_token_offset
    projection = identity_projection(2)
    codebook = torch.tensor([[3., 4.], [6., 8.]])
    offset = scaled_token_offset(torch.tensor([[2., 2.]]), projection, codebook)
    # mean codeword norm is (5 + 10) / 2 = 7.5; direction is the unit token mean.
    assert torch.allclose(offset.norm(dim=1), torch.tensor([7.5]), atol=1e-6)
    assert torch.allclose(offset, torch.tensor([[7.5 / 2 ** 0.5, 7.5 / 2 ** 0.5]]), atol=1e-6)


def test_level2_target_switch_and_shape():
    residual = torch.tensor([[1., 2.]])
    shift = torch.tensor([[0.5, -0.5]])
    assert torch.equal(level2_target(residual, None), residual)
    assert torch.equal(level2_target(residual, shift), torch.tensor([[1.5, 1.5]]))
    try:
        level2_target(residual, torch.zeros(1, 3))
    except ValueError:
        return
    raise AssertionError("a mismatched shift must be refused")
