"""Independent numerical checks for the concurrently added A-arm readout.

The auxiliary chain subtracts expected codewords in tangent coordinates. Only
its distance evaluation maps the H half into the ball; it is not a replacement
for the quantizer's native hard-assignment/residual path. All fixtures are tiny
CPU tensors, with no formal training inputs or output artifacts.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TREE))

from model import af3_hierarchy as hierarchy  # noqa: E402


def _expmap_oracle(tangent, curvature):
    if curvature == 0.0:
        return tangent
    radius = tangent.norm(dim=-1, keepdim=True)
    scaled = math.sqrt(curvature) * radius
    denominator = torch.where(scaled == 0, torch.ones_like(scaled), scaled)
    factor = torch.where(scaled == 0, torch.ones_like(scaled),
                         torch.tanh(scaled) / denominator)
    return tangent * factor


def _oracle(latent, codebooks, temperature, curvature):
    """Direct acosh-distance oracle; never calls production mixed_cost."""
    residual = latent
    result = []
    for codebook in codebooks:
        euclidean = (residual[:, None, :16] - codebook[None, :, :16]).square().sum(-1)
        if curvature == 0.0:
            curved = (residual[:, None, 16:] - codebook[None, :, 16:]).square().sum(-1)
        else:
            left = _expmap_oracle(residual[:, 16:], curvature)[:, None, :]
            right = _expmap_oracle(codebook[:, 16:], curvature)[None, :, :]
            squared_gap = (left - right).square().sum(-1)
            denominator = ((1 - curvature * left.square().sum(-1))
                           * (1 - curvature * right.square().sum(-1)))
            argument = 1 + 2 * curvature * squared_gap / denominator
            curved = torch.acosh(argument).square() / (4 * curvature)
        # alpha=1/2 and the outer factor 2 leave E^2 + d_H^2/4.
        assignment = torch.softmax(-(euclidean + curved) / temperature, dim=-1)
        result.append(assignment)
        residual = residual - assignment @ codebook
    return tuple(result)


def _inputs():
    generator = torch.Generator().manual_seed(139)
    latent = torch.randn(3, 32, generator=generator, dtype=torch.float64) * 0.08
    latent[:, 16] += torch.tensor([0.7, 1.1, -0.9], dtype=torch.float64)
    books = [torch.randn(4, 32, generator=generator, dtype=torch.float64) * 0.07
             for _ in range(3)]
    books[0][:, 16] += torch.tensor([0.1, 0.3, -0.3, -0.1], dtype=torch.float64)
    return latent, books


def test_mixed_readout_uses_squared_cost_once_not_fourth_power():
    latent = torch.zeros(1, 32, dtype=torch.float64)
    codebook = torch.zeros(3, 32, dtype=torch.float64)
    codebook[:, 0] = torch.tensor([0.2, 0.6, 1.1], dtype=torch.float64)
    temperature = 0.7
    actual, = hierarchy.soft_assignments(latent, [codebook], temperature, curvature=1.0)
    expected = torch.softmax(-codebook[:, 0].square() / temperature, dim=-1)[None, :]
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


def test_mixed_readout_maps_tangent_residual_at_each_level_and_preserves_chain():
    latent, codebooks = _inputs()
    latent.requires_grad_()
    for book in codebooks:
        book.requires_grad_()
    actual = hierarchy.soft_assignments(latent, codebooks, temperature=0.8, curvature=1.0)
    expected = _oracle(latent, codebooks, temperature=0.8, curvature=1.0)
    for got, want in zip(actual, expected):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, rtol=2e-10, atol=2e-11)
    # Later levels must also backpropagate through the soft tangent residual,
    # not through a native/hard residual or a detached expected codeword.
    probe = torch.tensor([[0.1, 0.5, -0.2, 0.8]], dtype=torch.float64)
    observed_gradients = torch.autograd.grad((actual[-1] * probe).sum(),
                                             [latent, *codebooks], retain_graph=True)
    oracle_gradients = torch.autograd.grad((expected[-1] * probe).sum(), [latent, *codebooks])
    for got, want in zip(observed_gradients, oracle_gradients):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, rtol=2e-8, atol=2e-10)


def test_zero_curvature_retains_original_euclidean_soft_readout():
    latent, codebooks = _inputs()
    actual = hierarchy.soft_assignments(latent, codebooks, temperature=0.8, curvature=0.0)
    expected = _oracle(latent, codebooks, temperature=0.8, curvature=0.0)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-12)


def test_four_term_loss_forwards_curvature_to_its_readout(monkeypatch):
    latent, codebooks = _inputs()
    seen = []
    original = hierarchy.soft_assignments

    def traced(*args, **kwargs):
        seen.append(kwargs.get("curvature"))
        return original(*args, **kwargs)

    monkeypatch.setattr(hierarchy, "soft_assignments", traced)
    weights = {name: 1.0 for name in ("level1", "level2", "level3", "balance")}
    value, _ = hierarchy.hierarchical_loss(latent, codebooks, {}, weights, curvature=1.0)
    assert torch.isfinite(value)
    assert seen == [1.0]


def test_active_adapter_forwards_quantizer_curvature_without_changing_hard_sids(monkeypatch):
    from model import af3_auxiliary as auxiliary
    from model.model import MixedRQVAE

    torch.manual_seed(139)
    config = SimpleNamespace(hidden_sizes=(8,), codebook_dim=32, dropout=0.0,
                             codebook_size=(4, 4, 4), codebook_num=3,
                             beta=0.25, sk_epsilon=-1.0, sk_iters=-1)
    model = MixedRQVAE(config, in_dim=8, curvature=1.0).eval()
    embeddings = torch.randn(4, 8) * 0.1
    previous_sids = model.get_indices(embeddings).clone()
    previous_alpha = model.alpha_table.to_dict()
    pairs = {name: torch.tensor([[0, 1]], dtype=torch.long)
             for name in ("af3", "positive", "negative", "unrelated")}
    pairs["af3_weight"] = torch.ones(1)
    monkeypatch.setattr(auxiliary, "sample_hierarchical_pairs", lambda *args: pairs)
    seen = []

    def traced_loss(latent, codebooks, batch, weights, **kwargs):
        seen.append(kwargs.get("curvature"))
        loss = latent.square().mean() + sum(book.square().mean() for book in codebooks)
        return loss, {name: 1.0 for name in weights}

    monkeypatch.setattr(auxiliary, "hierarchical_loss", traced_loss)
    base = sum(parameter.square().sum() for parameter in model.parameters())
    result, log = auxiliary.augment_loss(base, model, {"all_embeddings": embeddings}, 2250, None)
    assert seen == [1.0], "A must not silently use the legacy Euclidean readout"
    assert log["active"] and float(result.detach()) > float(base.detach())
    torch.testing.assert_close(model.get_indices(embeddings), previous_sids)
    assert model.alpha_table.to_dict() == previous_alpha
