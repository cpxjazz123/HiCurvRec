"""Issue #180: the same curvature metric on the level-1 assignment cost, with both switches pinned."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))


def _assign():
    from model import r48k2_assign
    return r48k2_assign


def _geometry():
    from model import r48k_geometry
    return r48k_geometry


def test_contrast_only_leaves_the_assignment_cost_exactly_euclidean():
    a = _assign()
    a.set_mode(contrast_only=True)
    latent = torch.randn(4, 8, dtype=torch.float64)
    codes = torch.randn(16, 8, dtype=torch.float64)
    kappa = torch.full((4,), 2.5, dtype=torch.float64)
    got = a.level1_cost(latent, codes, kappa)
    want = (latent.pow(2).sum(1, keepdim=True) - 2 * latent @ codes.t() + codes.pow(2).sum(1, keepdim=True).t())
    assert torch.equal(got, want)


def test_both_on_is_the_pair_curvature_metric_and_pins_the_hand_computation():
    a = _assign()
    g = _geometry()
    a.set_mode(contrast_only=False)
    latent = torch.randn(3, 6, dtype=torch.float64) * 0.1
    codes = torch.randn(5, 6, dtype=torch.float64) * 0.1
    kappa = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    got = a.level1_cost(latent, codes, kappa)
    # hand computation: project each row at its own curvature, then the asinh distance to every code at that curvature
    rows = []
    for i in range(latent.shape[0]):
        k = kappa[i].reshape(1, 1)
        p = g.project_to_ball(latent[i:i + 1], k)
        q = g.project_to_ball(codes, k)
        rows.append(g.distance(p.unsqueeze(1).expand(1, codes.shape[0], p.shape[-1]),
                               q.unsqueeze(0).expand(1, codes.shape[0], q.shape[-1]),
                               k.expand(1, codes.shape[0]).unsqueeze(-1)).reshape(-1))
    assert torch.allclose(got, torch.stack(rows), rtol=1e-12)


def test_the_cost_is_not_constant_and_keeps_sinkhorn_centrable():
    a = _assign()
    a.set_mode(contrast_only=False)
    latent = torch.randn(16, 8, dtype=torch.float64) * 0.05
    codes = torch.randn(32, 8, dtype=torch.float64) * 0.05
    kappa = torch.linspace(0.2, 3.0, 16, dtype=torch.float64)
    cost = a.level1_cost(latent, codes, kappa)
    assert float(cost.std()) > 0.0
    centred = (cost - (cost.max() + cost.min()) / 2) / (cost.max() - (cost.max() + cost.min()) / 2)
    assert bool(torch.isfinite(centred).all())


def test_geometry_off_also_restores_the_euclidean_cost():
    a = _assign()
    g = _geometry()
    g.GEOMETRY_OFF = True
    a.set_mode(contrast_only=False)
    latent = torch.randn(4, 8, dtype=torch.float64)
    codes = torch.randn(16, 8, dtype=torch.float64)
    kappa = torch.full((4,), 2.5, dtype=torch.float64)
    got = a.level1_cost(latent, codes, kappa)
    want = (latent.pow(2).sum(1, keepdim=True) - 2 * latent @ codes.t() + codes.pow(2).sum(1, keepdim=True).t())
    g.GEOMETRY_OFF = False
    assert torch.equal(got, want)


def test_the_layer_hook_uses_the_batch_kappa_when_the_mode_asks_for_it():
    a = _assign()
    from model import layers
    a.set_mode(contrast_only=False)
    layer = layers.VQLayer(8, 4)                      # its codebook is float32, so the rows must match it
    latent = torch.randn(6, 4) * 0.05
    plain = layer._distances(latent)
    a.set_batch_kappa(torch.linspace(0.3, 2.0, 6))
    curved = layer._distances(latent)
    a.set_batch_kappa(None)
    assert not torch.equal(plain, curved)
    assert torch.equal(layer._distances(latent), plain)
