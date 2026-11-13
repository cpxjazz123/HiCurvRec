"""Gradients, the frozen table, the straight-through path, and the export.

    python3 -m pytest stage2_RQ-VAE/TIGER_role_geom_rq/tests/test_gradients_and_export.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.model import MixedRQVAE
from model.role_geometry import build_weights


def build_model(weights=None, seed: int = 3):
    config = SimpleNamespace(hidden_sizes=(32, 16), codebook_num=3, codebook_size=(16, 16, 16),
                             codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
                             fix_code_embs=False, sk_epsilon=0.003, sk_iters=5)
    torch.manual_seed(seed)
    model = MixedRQVAE(config, in_dim=24, curvature=1.0, role_weights=weights)
    for level in model.rq.vq_layers:
        level.embed.weight.data.copy_(torch.randn_like(level.embed.weight) * 0.1)
    return model


def test_the_routing_reaches_the_encoder_and_the_codebook() -> None:
    roles = {item: item % 6 for item in range(9)}
    weights = build_weights("ROLE", 9, roles, list(range(9)))[0]
    model = build_model(weights)
    embeddings = torch.randn(9, 24, generator=torch.Generator().manual_seed(2))
    item_ids = torch.arange(9)
    reconstructed, quant_loss, _, _ = model(embeddings, item_ids)
    loss = reconstructed.pow(2).mean() + quant_loss
    loss.backward()
    encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    codebook_grads = [p.grad for p in model.rq.vq_layers[0].parameters() if p.grad is not None]
    assert encoder_grads, "the encoder received no gradient"
    assert codebook_grads, "the codebook received no gradient"
    for grad in encoder_grads + codebook_grads:
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0


def test_the_role_table_is_not_a_trainable_parameter() -> None:
    roles = {item: item % 6 for item in range(9)}
    weights = build_weights("ROLE", 9, roles, list(range(9)))[0]
    model = build_model(weights)
    names = {name for name, _ in model.named_parameters()}
    assert "role_weights" not in names
    assert not model.role_weights.requires_grad


def test_the_straight_through_readout_is_the_hard_readout() -> None:
    roles = {item: item % 6 for item in range(9)}
    weights = build_weights("ROLE", 9, roles, list(range(9)))[0]
    model = build_model(weights)
    embeddings = torch.randn(9, 24, generator=torch.Generator().manual_seed(4))
    item_ids = torch.arange(9)
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(embeddings)
        readout, _, _, _ = model.rq(encoded, weights=model.routing_for(item_ids))
        quantized = encoded + (readout - encoded).detach()
    assert torch.allclose(quantized, readout, atol=1e-6)


def test_the_export_reads_the_same_codes_the_training_cost_would_pick() -> None:
    roles = {item: item % 6 for item in range(9)}
    weights = build_weights("ROLE", 9, roles, list(range(9)))[0]
    model = build_model(weights)
    embeddings = torch.randn(9, 24, generator=torch.Generator().manual_seed(6))
    item_ids = torch.arange(9)
    model.eval()
    with torch.no_grad():
        tokens = model.get_indices(embeddings, infer_use_sk=False, item_ids=item_ids)
        encoded = model.encoder(embeddings)
        z_e, z_t = encoded[..., :16], encoded[..., 16:]
        from model.poincare_contract import expmap0, mobius_add, project
        residual_e, residual_h = z_e, expmap0(z_t, 1.0)
        row_weights = model.routing_for(item_ids)
        for level, layer in enumerate(model.rq.vq_layers):
            alpha = torch.full((9,), 0.5)
            cost = layer.distances(residual_e, residual_h, alpha, weights=row_weights)
            index = torch.argmin(cost, dim=-1)
            assert torch.equal(index, tokens[:, level])
            codes = layer.get_code_embs()
            picked_e = codes[:, :16][index]
            picked_h = expmap0(codes[:, 16:][index], 1.0)
            residual_e = residual_e - picked_e
            residual_h = project(mobius_add(-picked_h, residual_h, 1.0), 1.0)


def test_the_routing_is_the_same_on_every_rank() -> None:
    roles = {item: item % 6 for item in range(9)}
    first = build_weights("SHUFFLED", 9, roles, list(range(9)))[0]
    second = build_weights("SHUFFLED", 9, roles, list(range(9)))[0]
    assert torch.equal(first, second)
    item_ids = torch.arange(9)
    assert torch.equal(first.index_select(0, item_ids), second.index_select(0, item_ids))
