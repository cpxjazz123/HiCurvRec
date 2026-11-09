"""The reference surfaces must carry finite gradients at coincident points,
which is a legal hard-SID collapse, not an excluded input."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mixed_geometry as G
import structure as S


def _check(fn, lead):
    lead = lead.clone().double().requires_grad_(True)
    tail = torch.zeros_like(lead)
    def f(x):
        return fn(torch.cat([lead, x], dim=-1))
    assert torch.autograd.gradcheck(f, (tail,), eps=1e-6, atol=1e-8, rtol=1e-6)


def test_distance_reference_gradcheck_at_the_collision():
    left = torch.zeros(1, 2, dtype=torch.float64).requires_grad_(True)
    right = torch.zeros(1, 2, dtype=torch.float64)
    f = lambda a: G.reference_surface(a, right)
    assert torch.autograd.gradcheck(f, (left,), eps=1e-6, atol=1e-8, rtol=1e-6)
    g = torch.autograd.grad(G.reference_surface(left, right), left)[0]
    assert torch.isfinite(g).all()


def test_structure_reference_gradcheck_at_the_collision():
    points = torch.zeros(1, 6, dtype=torch.float64).requires_grad_(True)
    u = torch.zeros(1, 3, dtype=torch.float64)
    scale = torch.ones(1, 1, dtype=torch.float64)
    f = lambda p: S.reference_objective_surface(p, u, scale)
    assert torch.autograd.gradcheck(f, (points,), eps=1e-6, atol=1e-8, rtol=1e-6)
    g = torch.autograd.grad(S.reference_objective_surface(points, u, scale), points)[0]
    assert torch.isfinite(g).all()


if __name__ == "__main__":
    test_distance_reference_gradcheck_at_the_collision()
    test_structure_reference_gradcheck_at_the_collision()
    print("ok")
