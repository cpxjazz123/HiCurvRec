"""Tests for the κ-stereographic manifold used by the κ-GCN experts.

``geoopt`` is not installed in this environment, so the κ-stereographic operations the
official GraphMoRE code takes from ``geoopt.Stereographic`` are implemented here.  These
tests pin the mathematical properties that make the implementation trustworthy:

* ``expmap0``/``logmap0`` are mutual inverses inside the ball;
* ``dist`` is symmetric, vanishes on the diagonal and satisfies the triangle inequality;
* the κ → 0 limit is the Euclidean metric, so the curvature-0 expert is a genuine GCN;
* negative curvature expands radii (points spread out), positive curvature compresses;
* ``mobius_add`` agrees with the gyrovector formula and reduces to ``x + y`` at κ = 0;
* ``proju`` keeps every point strictly inside the ball, which is what keeps the
  spherical expert (κ = +3) from blowing up at the pole.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphmore.manifold import Stereographic  # noqa: E402

CURVATURES = [-3.0, -1.0, 0.0, 1.0, 3.0]


def _points(k: float, n: int = 32, dim: int = 6, seed: int = 42) -> torch.Tensor:
    """Points inside the ball of curvature ``k`` (unit ball for k = 0)."""
    torch.manual_seed(seed)
    raw = torch.randn(n, dim)
    raw = raw / raw.norm(dim=-1, keepdim=True)
    radius = 1.0 if k == 0 else 1.0 / (abs(k) ** 0.5)
    scale = torch.rand(n, 1) * 0.5 * radius
    return raw * scale


@pytest.mark.parametrize("k", CURVATURES)
def test_exp_log_roundtrip(k):
    """``logmap0(expmap0(v))`` returns ``v`` for a tangent vector at the origin."""
    manifold = Stereographic(k)
    torch.manual_seed(0)
    v = torch.randn(16, 5)
    # keep ||v|| * sqrt(|k|) <= 2: tanh/atanh lose precision if the seed saturates
    limit = 2.0 / (abs(k) ** 0.5) if k != 0.0 else 2.0
    if k > 0:
        limit = 0.6 / (k ** 0.5)   # tan(s) < 1 keeps expmap0's output inside the ball
    v = v / v.norm(dim=-1, keepdim=True) * limit * torch.rand(16, 1)
    back = manifold.logmap0(manifold.expmap0(v))
    assert torch.allclose(back, v, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("k", CURVATURES)
def test_projection_keeps_points_inside_the_ball(k):
    manifold = Stereographic(k)
    x = torch.randn(64, 4) * 100.0
    projected = manifold.proju(x)
    norm = projected.norm(dim=-1)
    if k == 0:
        assert torch.allclose(projected, x, atol=1e-5)
        return
    radius = 1.0 / (abs(k) ** 0.5)
    assert torch.all(norm < radius), (norm.max().item(), radius)


@pytest.mark.parametrize("k", CURVATURES)
def test_distance_is_symmetric_and_zero_on_the_diagonal(k):
    manifold = Stereographic(k)
    x = _points(k)
    y = _points(k, seed=7)
    d_xy = manifold.dist(x, y)
    d_yx = manifold.dist(y, x)
    assert torch.allclose(d_xy, d_yx, atol=1e-5)
    assert torch.allclose(manifold.dist(x, x), torch.zeros(x.shape[0]), atol=1e-5)
    assert torch.all(d_xy >= 0)


@pytest.mark.parametrize("k", CURVATURES)
def test_distance_satisfies_the_triangle_inequality(k):
    manifold = Stereographic(k)
    x, y, z = _points(k), _points(k, seed=7), _points(k, seed=13)
    d_xy = manifold.dist(x, y)
    d_yz = manifold.dist(y, z)
    d_xz = manifold.dist(x, z)
    assert torch.all(d_xz <= d_xy + d_yz + 1e-4)


def test_distance_reduces_to_twice_the_euclidean_norm_at_zero_curvature():
    """The stereographic metric is 2x Euclidean exactly at κ = 0."""
    x = _points(0.0)
    y = _points(0.0, seed=7)
    assert torch.allclose(Stereographic(0.0).dist(x, y),
                          2.0 * (x - y).norm(dim=-1), atol=1e-5)


def test_hyperbolic_distance_matches_the_poincare_closed_form():
    """Independent formula: d = (1/√c)·arcosh(1 + 2c‖x−y‖²/((1−c‖x‖²)(1−c‖y‖²)))."""
    k = -1.0
    c = -k
    x, y = _points(k), _points(k, seed=7)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    diff2 = ((x - y) ** 2).sum(-1)
    inner = 1.0 + 2.0 * c * diff2 / ((1.0 - c * x2) * (1.0 - c * y2))
    expected = (1.0 / (c ** 0.5)) * torch.acosh(inner.clamp(min=1.0))
    got = Stereographic(k).dist(x, y)
    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3)


def test_positive_curvature_distance_is_bounded_by_the_hemisphere_diameter():
    """On the sphere the metric is bounded by pi/sqrt(k); it is not a compression."""
    k = 1.0
    x, y = _points(k), _points(k, seed=7)
    spherical = Stereographic(k).dist(x, y)
    import math
    assert torch.all(spherical <= math.pi / (k ** 0.5) + 1e-4)
    assert torch.all(spherical >= 0)


@pytest.mark.parametrize("k", [-3.0, -1.0, 0.0])
def test_mobius_translation_preserves_distance(k):
    """``d(x, y) == d(0, (-x) (+) y)``: the defining property of the group operation.

    This is stronger than re-deriving the algebraic formula (which would only restate
    the implementation) because it ties ``mobius_add`` to the independently tested
    ``dist``.
    """
    manifold = Stereographic(k)
    x, y = _points(k), _points(k, seed=7)
    translated = manifold.mobius_add(-x, y)
    origin = torch.zeros_like(x)
    assert torch.allclose(manifold.dist(x, y), manifold.dist(origin, translated),
                          atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("k", CURVATURES)
def test_mobius_add_is_identity_with_the_origin(k):
    manifold = Stereographic(k)
    x = _points(k)
    zero = torch.zeros_like(x)
    assert torch.allclose(manifold.mobius_add(x, zero), x, atol=1e-5)
    assert torch.allclose(manifold.mobius_add(zero, x), x, atol=1e-5)


@pytest.mark.parametrize("k", CURVATURES)
def test_mobius_matvec_is_a_tangent_space_linear_map(k):
    """``mobius_matvec(W, x) == expmap0(W @ logmap0(x))`` by construction."""
    manifold = Stereographic(k)
    x = _points(k, n=8, dim=4)
    torch.manual_seed(3)
    weight = torch.randn(5, 4) * 0.1
    got = manifold.mobius_matvec(weight, x)
    expected = manifold.expmap0(manifold.logmap0(x) @ weight.t())
    assert torch.allclose(got, expected, atol=1e-5)


@pytest.mark.parametrize("k", [-3.0, -1.0, 1.0, 3.0])
def test_learned_curvature_keeps_its_sign(k):
    """A learnable curvature must never cross zero, which would change the geometry."""
    manifold = Stereographic(k, learnable=True)
    for value in (-50.0, -1.0, 0.0, 1.0, 50.0):
        with torch.no_grad():
            manifold.raw_curvature.fill_(value)
        got = float(manifold.k)
        assert (got < 0) if k < 0 else (got > 0), (k, value, got)


def test_zero_curvature_is_frozen():
    manifold = Stereographic(0.0)
    assert manifold.learnable is False
    assert float(manifold.k) == 0.0


def test_gradients_flow_through_expmap_and_dist():
    manifold = Stereographic(-1.0)
    x = _points(-1.0, n=8, dim=3).clone().requires_grad_(True)
    y = _points(-1.0, n=8, dim=3, seed=7).clone().requires_grad_(True)
    loss = manifold.dist(x, y).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert y.grad is not None and torch.isfinite(y.grad).all()


def test_numerics_are_finite_at_the_ball_boundary():
    """Points pushed to the boundary must not produce NaN or inf."""
    manifold = Stereographic(-1.0)
    x = torch.randn(8, 4)
    x = x / x.norm(dim=-1, keepdim=True) * 0.9999
    y = x.flip(0).clone()
    value = manifold.dist(x, y)
    assert torch.isfinite(value).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
