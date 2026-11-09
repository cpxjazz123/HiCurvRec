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


def poincare_sq(a: torch.Tensor, b: torch.Tensor, curvature: float = CURVATURE) -> torch.Tensor:
    """Squared geodesic distance via the Moebius difference and 2*atanh.

    d^2 = 4 * atanh(||(-a) (+) b||)^2.  The argument is exactly 0 when a == b, so
    two identical points have distance exactly 0; near the diagonal atanh(u) =
    u + u^3/3 keeps the gradient finite, and the only clamp is the open-ball
    bound (u < 1), which no finite point reaches.

    The earlier arcosh(1 + ...) form needed a floor on its argument, which made
    d(x, x) about 5e-7 instead of 0 and turned small distances into a constant.
    """
    # Moebius addition of (-a) and b, curvature 1
    a2 = a.pow(2).sum(-1).unsqueeze(-1)
    b2 = b.pow(2).sum(-1).unsqueeze(-1)
    ab = (a * b).sum(-1, keepdim=True)
    numerator = (1.0 - 2.0 * ab + b2) * (-a) + (1.0 - a2) * b
    denominator = (1.0 - 2.0 * ab + a2 * b2).clamp_min(1e-12)
    difference = numerator / denominator
    # 1 - 1e-12 is not representable in float32 and would clamp to exactly 1,
    # making atanh infinite; 1 - 1e-6 stays below the boundary in both dtypes.
    norm = difference.norm(dim=-1).clamp(max=1.0 - 1e-6)
    return (2.0 * torch.atanh(norm)) ** 2 / curvature


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
