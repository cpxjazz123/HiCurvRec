"""Issue #179's acceptance: coincident points and their gradients must be finite, and the asinh form must agree with the
arccosh form away from the boundary. These are the cases #178's smoke died on."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import r48k_geometry as g  # noqa: E402


def _pair(gap, seed=0, dim=4, kappa=1.0):
    generator = torch.Generator().manual_seed(seed)
    p = (torch.rand(1, dim, generator=generator, dtype=torch.float64) * 2 - 1) * 0.1
    q = p + gap
    k = torch.full((1, 1), kappa, dtype=torch.float64)
    return g.project_to_ball(p, k), g.project_to_ball(q, k), k


def test_coincident_and_near_coincident_distances_and_gradients_are_finite():
    for gap in (0.0, 1e-12, 1e-9, 1e-6, 1e-3):
        p, q, kappa = _pair(gap)
        for tensor in (p, q):
            tensor.requires_grad_(True)
        d = g.distance(p, q, kappa)
        assert bool(torch.isfinite(d).all()), gap
        d.sum().backward()
        for tensor in (p, q):
            assert bool(torch.isfinite(tensor.grad).all()), (gap, tensor.grad)


def test_asinh_form_agrees_with_the_arccosh_form_away_from_the_boundary():
    for seed in (0, 1, 2):
        p, q, kappa = _pair(1e-2, seed=seed)
        mine = g.distance(p, q, kappa)
        # arccosh(1 + x) = 2 asinh(sqrt(x / 2)) with x = 2k|p-q|^2 / ((1-k p^2)(1-k q^2))
        p2 = (p.pow(2).sum(-1, keepdim=True) * kappa)
        q2 = (q.pow(2).sum(-1, keepdim=True) * kappa)
        diff2 = (p - q).pow(2).sum(-1, keepdim=True)
        x = 2 * kappa * diff2 / ((1 - p2) * (1 - q2))
        ref = (2.0 / torch.sqrt(kappa)) * torch.asinh(torch.sqrt(x / 2))
        assert torch.allclose(mine, ref.squeeze(-1), rtol=1e-6, atol=1e-12), seed


def test_gradcheck_passes_at_a_coincident_point():
    p, q, kappa = _pair(0.0)
    p = p.clone().requires_grad_(True)
    q = q.clone().requires_grad_(True)

    def f(a, b):
        return g.distance(a, b, kappa)

    assert torch.autograd.gradcheck(f, (p, q), eps=1e-6, atol=1e-6, rtol=1e-4)
