"""The routed quantizer must leave the baseline alone when it routes to the baseline's geometry.

This is the first of issue #215's acceptance items: an arm that sends every item to the
hyperbolic branch has to reproduce the archived F0 cost, its assignments, its losses and its SID
element for element.  The reference is the untouched baseline arithmetic in
``model/baseline_reference.py``.

    python3 -m pytest stage2_RQ-VAE/TIGER_role_geom_rq/tests/test_role_off_matches_f0.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import baseline_reference
from model.mixed_rq import mixed_cost, point_cost
from model.model import MixedRQVAE
from model.role_geometry import SPACES, build_weights


def probe(batch: int = 7, codes: int = 11, seed: int = 5) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    residual_e = torch.randn(batch, 16, generator=generator, dtype=torch.float64)
    tangent = torch.randn(batch, 16, generator=generator, dtype=torch.float64) * 0.3
    from model.poincare_contract import expmap0
    residual_h = expmap0(tangent, 1.0)
    codes_e = torch.randn(codes, 16, generator=generator, dtype=torch.float64)
    codes_h = torch.randn(codes, 16, generator=generator, dtype=torch.float64) * 0.3
    alpha = torch.full((batch,), 0.5, dtype=torch.float64)
    return residual_e, residual_h, codes_e, codes_h, alpha


def test_the_routed_cost_equals_the_baseline_cost_when_the_row_routes_to_hyperbolic() -> None:
    residual_e, residual_h, codes_e, codes_h, alpha = probe()
    baseline = baseline_reference.mixed_cost(residual_e, residual_h, codes_e, codes_h, alpha)
    unrouted = mixed_cost(residual_e, residual_h, codes_e, codes_h, alpha)
    weights = torch.zeros(residual_e.shape[0], 3, dtype=torch.float64)
    weights[:, SPACES.index("hyperbolic")] = 1.0
    routed = mixed_cost(residual_e, residual_h, codes_e, codes_h, alpha, weights=weights)
    assert torch.equal(unrouted, baseline)
    assert torch.allclose(routed, baseline, atol=1e-12)


def test_the_scalar_cost_path_agrees_with_the_baseline_too() -> None:
    residual_e, residual_h, codes_e, codes_h, alpha = probe(batch=4, codes=3)
    for index in range(residual_e.shape[0]):
        code_index = index % codes_e.shape[0]
        base = baseline_reference.point_cost(residual_e[index], residual_h[index],
                                             codes_e[code_index], codes_h[code_index],
                                             alpha[index])
        weights = torch.zeros(1, 3, dtype=torch.float64)
        weights[:, SPACES.index("hyperbolic")] = 1.0
        routed = point_cost(residual_e[index], residual_h[index], codes_e[code_index],
                            codes_h[code_index], alpha[index], weights=weights)
        assert torch.allclose(routed, base, atol=1e-12)


def test_a_row_that_routes_elsewhere_gets_a_different_cost() -> None:
    residual_e, residual_h, codes_e, codes_h, alpha = probe()
    baseline = mixed_cost(residual_e, residual_h, codes_e, codes_h, alpha)
    weights = torch.zeros(residual_e.shape[0], 3, dtype=torch.float64)
    weights[:, SPACES.index("spherical")] = 1.0
    spherical = mixed_cost(residual_e, residual_h, codes_e, codes_h, alpha, weights=weights)
    assert not torch.allclose(spherical, baseline)
    weights[:, SPACES.index("spherical")] = 0.0
    weights[:, SPACES.index("euclidean")] = 1.0
    tangent = mixed_cost(residual_e, residual_h, codes_e, codes_h, alpha, weights=weights)
    assert not torch.allclose(tangent, baseline)


def test_the_model_with_the_off_routing_reproduces_the_baseline_model() -> None:
    config = SimpleNamespace(hidden_sizes=(32, 16), codebook_num=3, codebook_size=(16, 16, 16),
                             codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
                             fix_code_embs=False, sk_epsilon=0.003, sk_iters=5)
    embeddings = torch.randn(9, 24, generator=torch.Generator().manual_seed(11))
    roles = {item: item % 6 for item in range(9)}
    off = build_weights("OFF", 9, roles, list(range(9)))[0]

    torch.manual_seed(3)
    baseline_model = MixedRQVAE(config, in_dim=24, curvature=1.0)
    torch.manual_seed(3)
    routed_model = MixedRQVAE(config, in_dim=24, curvature=1.0, role_weights=off)

    for level in baseline_model.rq.vq_layers:
        for parameter in level.parameters():
            parameter.data.copy_(torch.zeros_like(parameter))
    for level in routed_model.rq.vq_layers:
        for parameter in level.parameters():
            parameter.data.copy_(torch.zeros_like(parameter))
    torch.manual_seed(7)
    for level in baseline_model.rq.vq_layers:
        level.embed.weight.data.copy_(torch.randn_like(level.embed.weight))
    for level in routed_model.rq.vq_layers:
        level.embed.weight.data.copy_(level.embed.weight.data.clone() * 0)
    # give both models the same codebook
    for base_layer, routed_layer in zip(baseline_model.rq.vq_layers, routed_model.rq.vq_layers):
        routed_layer.embed.weight.data.copy_(base_layer.embed.weight.data)

    baseline_model.eval()
    routed_model.eval()
    item_ids = torch.arange(9)
    with torch.no_grad():
        baseline_tokens = baseline_model.get_indices(embeddings, infer_use_sk=False)
        routed_tokens = routed_model.get_indices(embeddings, infer_use_sk=False, item_ids=item_ids)
    assert torch.equal(baseline_tokens, routed_tokens)
