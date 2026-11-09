"""The reduced prefix surface must be rebuildable from its declared statements,
and the distance gradient must match an independent analytic derivative."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.research.stage2.reference import build_reference

import mixed_geometry as G
import structure as S

D = torch.float64
CONTRACT = json.loads((Path(__file__).resolve().parents[1] / "reference_contract.json").read_text())
CALL = {c["name"]: c for c in CONTRACT["calls"]}["prefix_surface_1d"]


def test_the_reduced_prefix_rebuilds_from_its_statements():
    ref, _ = build_reference(CALL)
    torch.manual_seed(1)
    codes = (torch.rand(2, 18, dtype=D) - 0.5) * 0.2
    h = torch.ones(2, 3, dtype=D)
    h[:, 2] = torch.tensor([1.0, 2.0], dtype=D)
    s = torch.ones(2, 1, dtype=D)
    got = S.reference_prefix_surface_reduced(codes, h, s)
    expected = ref(codes=codes, h=h, s=s)
    assert torch.allclose(got, expected, atol=1e-12)


def test_both_prefix_surfaces_gradcheck():
    ref, _ = build_reference(CALL)
    torch.manual_seed(2)
    codes = ((torch.rand(1, 18, dtype=D) - 0.5) * 0.2).requires_grad_(True)
    h = torch.ones(1, 3, dtype=D)
    h[:, 2] = 2.0
    s = torch.ones(1, 1, dtype=D)
    assert torch.autograd.gradcheck(lambda c: S.reference_prefix_surface_reduced(c, h, s),
                                    (codes,), eps=1e-6, atol=1e-8, rtol=1e-6)
    assert torch.autograd.gradcheck(lambda c: ref(codes=c, h=h, s=s), (codes,),
                                    eps=1e-6, atol=1e-8, rtol=1e-6)


def _ball(t):
    r = t.norm(dim=-1, keepdim=True)
    scale = torch.where(r > 0, torch.tanh(r) / r, torch.ones_like(r))
    return scale * t


def _analytic_phi_distance(a_h, b_h):
    ball_a, ball_b = _ball(a_h), _ball(b_h)
    num = ((ball_a - ball_b) ** 2).sum()
    den = (1 - (ball_a ** 2).sum()) * (1 - (ball_b ** 2).sum())
    return torch.asinh(torch.sqrt(num / den)) ** 2


def test_the_distance_gradient_matches_an_independent_analytic_derivative():
    a, b = torch.zeros(1, 32, dtype=D), torch.zeros(1, 32, dtype=D)
    a[0, 16] = 0.2
    b[0, 17] = 0.1
    a.requires_grad_(True)
    b.requires_grad_(True)
    got = torch.autograd.grad(G.mixed_sq(a, b), a)[0][0, 16:]
    ana = torch.autograd.grad(_analytic_phi_distance(a[0, 16:], b[0, 16:]).sum(), a)[0][0, 16:]
    assert torch.allclose(got, ana, atol=1e-10)

    saved = G.poincare_sq
    G.poincare_sq = lambda x, y: (lambda v: 2 * v - v.detach())(saved(x, y))
    try:
        a2, b2 = torch.zeros(1, 32, dtype=D), torch.zeros(1, 32, dtype=D)
        a2[0, 16] = 0.2
        b2[0, 17] = 0.1
        a2.requires_grad_(True)
        b2.requires_grad_(True)
        doubled = torch.autograd.grad(G.mixed_sq(a2, b2), a2)[0][0, 16:]
    finally:
        G.poincare_sq = saved
    assert not torch.allclose(doubled, ana, atol=1e-10)


if __name__ == "__main__":
    test_the_reduced_prefix_rebuilds_from_its_statements()
    test_both_prefix_surfaces_gradcheck()
    test_the_distance_gradient_matches_an_independent_analytic_derivative()
    print("ok")
