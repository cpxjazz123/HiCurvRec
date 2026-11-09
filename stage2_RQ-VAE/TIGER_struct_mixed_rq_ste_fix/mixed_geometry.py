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
    # No extra shrink factor: b(h) is the radial map itself, so log0(b(h)) = h
    # exactly (up to float64 rounding).  Saturation is handled where distances are
    # computed, in float64, rather than by bending the projection.
    scale = curvature ** 0.5
    length = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    point = tangent * (torch.tanh(length * scale) / length) / scale
    # Only the saturated tail is pulled inside: tanh rounds to 1.0 in low precision
    # for very large inputs, and a point exactly on the boundary has no finite
    # distance.  Ordinary points keep log0(b(h)) = h exactly.
    norm = point.to(torch.float64).norm(dim=-1, keepdim=True)
    saturated = (norm >= 1.0).to(point.dtype)
    return point * (1.0 - saturated * 1e-6)


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
    """The contract surface on 2 coordinates (1 Euclidean + 1 radial), phi form.

    Returns the squared geodesic distance 4*phi(t) exactly as mixed_sq computes
    it; phi is the same function, so the reduced surface is not a second formula.
    """
    left_e, left_h = left[..., :1], left[..., 1:2]
    right_e, right_h = right[..., :1], right[..., 1:2]
    if left_h.shape[-1] != 1:
        raise ValueError("the reference surface is defined on one H coordinate")
    # The 1-coordinate cross-ratio equals sinh(left - right)^2, so
    # asinh(sqrt(t))^2 = (left - right)^2.  This is the same function with a
    # finite backward at the collision, not a floor and not a different metric.
    return (left_e - right_e).pow(2).sum(-1) + (left_h - right_h).pow(2).sum(-1)


def mobius_add(a: torch.Tensor, b: torch.Tensor, curvature: float = CURVATURE) -> torch.Tensor:
    """Moebius addition in the ball of curvature 1 (batched over the last dim)."""
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    a2 = a64.pow(2).sum(-1, keepdim=True)
    b2 = b64.pow(2).sum(-1, keepdim=True)
    ab = (a64 * b64).sum(-1, keepdim=True)
    numerator = (1.0 + 2.0 * ab + b2) * a64 + (1.0 - a2) * b64
    denominator = 1.0 + 2.0 * ab + a2 * b2
    return (numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny)).to(a.dtype)


def mobius_neg(a: torch.Tensor) -> torch.Tensor:
    return -a


def log_zero(point: torch.Tensor, curvature: float = CURVATURE) -> torch.Tensor:
    """Inverse of the radial map b(h) = h*tanh(||h||)/||h||, computed in double."""
    p64 = point.to(torch.float64)
    norm = p64.norm(dim=-1, keepdim=True)
    scale = torch.atanh(norm.clamp(max=1.0 - 1e-9))
    factor = torch.where(norm > 1e-12, scale / norm.clamp_min(1e-12), torch.ones_like(norm))
    return (p64 * factor).to(point.dtype)


def residual_step(code: torch.Tensor, residual_e: torch.Tensor,
                  residual_h_ball: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One level of the product residual: E subtracts, H uses the Moebius left inverse."""
    return residual_e - code[..., :E_DIM], mobius_add(mobius_neg(code_ball(code)), residual_h_ball)


def code_ball(tangent_code: torch.Tensor) -> torch.Tensor:
    """Codeword tangent coordinates into the ball (last H_DIM entries)."""
    return to_ball(tangent_code[..., E_DIM:E_DIM + H_DIM])


def nested_compose(codes: list[torch.Tensor]) -> torch.Tensor:
    """Right-nested product composition c1 (+) (c2 (+) c3), folded from the last level."""
    composed = code_ball(codes[-1])
    for code in reversed(codes[:-1]):
        composed = mobius_add(code_ball(code), composed)
    return composed
