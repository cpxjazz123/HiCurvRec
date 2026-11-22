"""Math checks for the measured-curvature assignment cost (TIGER_kappa_cost_rq).

The mechanism is only allowed to change the Sinkhorn cost matrix (CLAUDE.md 19h), so the
cost itself has to behave like a metric before any epoch is spent on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "stage2_RQ-VAE" / "TIGER_kappa_cost_rq"))

import distances.curved as curved  # noqa: E402


def _ball_points(n: int, dim: int = 6, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, dim)


def test_radius_map_bounds_every_point_inside_the_ball():
    x = _ball_points(64) * 50.0
    tau = torch.tensor(1.0)
    u = curved._to_ball(x, tau)
    assert torch.all(u.norm(dim=-1) < 1.0)


def test_distance_to_itself_is_zero():
    torch.manual_seed(0)
    codes = torch.randn(8, 5)
    tau = codes.norm(dim=-1).mean()
    u = curved._to_ball(codes.clone(), tau)
    value = curved.sq_geodesic(u, u)
    diagonal = torch.diagonal(value)
    assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-5)


def test_cost_is_non_negative_and_finite():
    torch.manual_seed(1)
    latent = torch.randn(32, 5) * 3
    codes = torch.randn(16, 5)
    cost = curved.distances(latent, codes)
    assert cost.shape == (32, 16)
    assert torch.isfinite(cost).all()
    assert torch.all(cost >= 0)


def test_cost_is_monotone_along_a_ray_through_the_code():
    """Along a ray through a code the cost has one minimum, at the code.

    Scale < 1 approaches the code and scale > 1 leaves it, so the assignment needs both
    sides to be ordered: otherwise a latent further out could be assigned while a nearer
    one is not.
    """
    code = torch.tensor([[0.7, 0.0]])
    approaching = [float(curved.distances(code * s, code)[0, 0])
                   for s in (0.2, 0.35, 0.5, 0.75, 1.0)]
    leaving = [float(curved.distances(code * s, code)[0, 0])
               for s in (1.0, 1.25, 1.5, 2.0, 4.0)]
    assert approaching == sorted(approaching, reverse=True), approaching
    assert leaving == sorted(leaving), leaving
    assert approaching[-1] == pytest.approx(leaving[0], abs=1e-5)


def test_cost_is_symmetric_between_two_point_sets():
    torch.manual_seed(2)
    a, b = torch.randn(4, 5), torch.randn(7, 5)
    tau = torch.tensor(1.5)
    forward = curved.sq_geodesic(curved._to_ball(a, tau), curved._to_ball(b, tau))
    backward = curved.sq_geodesic(curved._to_ball(b, tau), curved._to_ball(a, tau))
    assert torch.allclose(forward, backward.t(), atol=1e-5)


def test_it_differs_from_the_euclidean_cost():
    """The mechanism must not silently be the baseline metric."""
    torch.manual_seed(3)
    latent = torch.randn(16, 5) * 2
    codes = torch.randn(8, 5)
    curved_cost = curved.distances(latent, codes)
    euclid = (latent.pow(2).sum(1, keepdim=True)
              - 2 * latent @ codes.t()
              + codes.pow(2).sum(1, keepdim=True).t())
    assert not torch.allclose(curved_cost, euclid, rtol=1e-3, atol=1e-3)


def test_gradients_flow_to_both_arguments():
    torch.manual_seed(4)
    latent = torch.randn(8, 5, requires_grad=True)
    codes = torch.randn(6, 5, requires_grad=True)
    curved.distances(latent, codes).sum().backward()
    for tensor in (latent, codes):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
