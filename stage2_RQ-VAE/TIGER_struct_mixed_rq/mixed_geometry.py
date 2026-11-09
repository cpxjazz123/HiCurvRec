"""The fixed E16 x H16 product space, written down once and used everywhere.

Conventions (frozen in math_spec.json):
  * a 32-d latent splits into the first 16 Euclidean coordinates and the last 16
    hyperbolic ones, which are read as a tangent vector at the ball origin;
  * the hyperbolic half is mapped into the Poincare ball of curvature 1 by
    tanh, so its norm is strictly below 1;
  * squared distances are additive on the product:
        d_M^2 = ||e1 - e2||^2 + d_H^2(h1, h2) / 4
    where d_H is the geodesic distance of the ball.  The /4 is the repository
    convention: the native hyperbolic distance is d_0 = 2||.||, so its square is
    4||.||^2 and the product form divides by four.
  * with GEOMETRY_ON = False the same formula is evaluated with the ordinary
    squared difference on the hyperbolic half, which is the Euclidean control
    arm of the geometry switch.
"""
from __future__ import annotations

import torch

E_DIM = 16
H_DIM = 16
CURVATURE = 1.0
GEOMETRY_ON = True


def split(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return z[..., :E_DIM], z[..., E_DIM:E_DIM + H_DIM]


def to_ball(tangent: torch.Tensor, curvature: float = CURVATURE) -> torch.Tensor:
    # Radial projection: a point of the ball is h * tanh(||h||)/||h||, whose norm
    # is tanh(||h||) < 1.  Scaling each coordinate separately would let the norm
    # reach sqrt(16); tanh also saturates to 1.0 in float32, so the last factor
    # keeps the point strictly inside the ball where the geodesic stays finite.
    scale = curvature ** 0.5
    length = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return tangent * (torch.tanh(length * scale) / length) * (1.0 - 1e-6) / scale


TAU = 1e-3   # series threshold: truncation after t^3 is O(t^4) = 1e-12 at this point


def phi(t: torch.Tensor) -> torch.Tensor:
    """asinh(sqrt(t))^2, evaluated so that t = 0 is exact and differentiable.

    phi(0) = 0 and phi'(0) = 1, so two coincident points have exactly zero
    distance with a finite gradient.  For small t the series
    t - t^2/3 + 8 t^3/45 (error O(t^4)) is used; the asymptotic branch never runs
    at t = 0, and the series never runs for large t, so no branch produces a NaN
    in the backward pass.  TAU follows from the truncation bound, not from any
    training result.
    """
    squared = t.clamp_min(0.0)
    series = squared - squared.pow(2) / 3.0 + 8.0 * squared.pow(3) / 45.0
    asymptotic = torch.asinh(squared.clamp_min(TAU).sqrt()).pow(2)
    use_series = squared < TAU
    # where the series is selected the asymptotic branch is still evaluated, so its
    # derivative must be finite there: clamp_min keeps sqrt inside the domain
    return torch.where(use_series, series, asymptotic)


def poincare_sq(a: torch.Tensor, b: torch.Tensor, curvature: float = CURVATURE) -> torch.Tensor:
    """Squared geodesic distance, as 4 * phi(t) with the product form phi(t).

    a and b are points already inside the ball.  t is the standard cross-ratio
    expression ||a-b||^2 / ((1-||a||^2)(1-||b||^2)); the product metric uses
    d_H^2/4 = phi(t), so this function returns 4*phi(t) and mixed_sq divides by 4.
    """
    dtype = a.dtype
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    num = ((a64 - b64) ** 2).sum(-1)
    den = (1.0 - a64.pow(2).sum(-1)) * (1.0 - b64.pow(2).sum(-1))
    t = (num / den.clamp_min(torch.finfo(torch.float64).tiny)).clamp_min(0.0)
    return (4.0 * phi(t) / curvature).to(dtype)


def mixed_sq(left: torch.Tensor, right: torch.Tensor, geometry_on: bool | None = None) -> torch.Tensor:
    """Product-space squared distance; the switch selects the control metric."""
    if geometry_on is None:
        geometry_on = GEOMETRY_ON
    left_e, left_h = split(left)
    right_e, right_h = split(right)
    euclid = ((left_e - right_e) ** 2).sum(-1)
    if not geometry_on:
        return euclid + ((left_h - right_h) ** 2).sum(-1)
    return euclid + poincare_sq(to_ball(left_h), to_ball(right_h)) / 4.0


def pairwise_mixed_sq(points: torch.Tensor, geometry_on: bool | None = None) -> torch.Tensor:
    """(..., n, 32) -> (..., n, n) squared distances."""
    flat = points.reshape(-1, points.shape[-1])
    n = points.shape[-2]
    rows = flat.repeat_interleave(n, dim=0)
    cols = flat.repeat(n, 1)
    distances = mixed_sq(rows, cols, geometry_on).reshape(*points.shape[:-2], n, n)
    return distances


def reference_surface(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The same formula on a 2-coordinate surface (1 Euclidean + 1 radial).

    It exists so the frozen contract can state the surface in closed form and the
    framework can rebuild it independently; `tests/test_reference_surface.py`
    checks that it agrees with `mixed_sq` on full 32-d inputs.
    """
    left_e, left_h = left[..., :1], left[..., 1:2]
    right_e, right_h = right[..., :1], right[..., 1:2]
    guard = 1.0 - 1e-6
    left_ball = torch.tanh(left_h) * guard
    right_ball = torch.tanh(right_h) * guard
    euclid = ((left_e - right_e) ** 2).sum(-1)
    num = 2.0 * ((left_ball - right_ball) ** 2).sum(-1)
    den = ((1.0 - left_ball ** 2).sum(-1)) * ((1.0 - right_ball ** 2).sum(-1))
    argument = 1.0 + (num / den).clamp_min(0.0)
    return euclid + (torch.acosh(argument.clamp_min(1.0)) ** 2) / 4.0
