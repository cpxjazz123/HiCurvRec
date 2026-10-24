"""Curvature-parameterised Poincare-ball operators for the transition-residual candidate.

Frozen candidate: issue117-geometry-transition-residual-poincare-v1
(run_d8cb91f13ee040d6, review verdict CPU_ELIGIBLE).

The proposal freezes one implementation whose curvature is a parameter, so the
curved arm and its flat control differ only in the distance formula:

    Exp_0^c(v) = tanh(sqrt(c)||v||) * v / (sqrt(c)||v||),   Exp_0^0(v) = v
    P_R(x)     = x if ||x|| <= R else R * x / ||x||
    D_c^2(x,y) = (1/c) * arcosh(1 + 2c||x-y||^2/((1-c||x||^2)(1-c||y||^2)))^2
    D_0^2(x,y) = 4||x-y||^2                              (the exact c -> 0 limit)

The constants are the proposal's: c = 1.0, tau = 0.1, epsilon = 1e-5,
R = 1 - epsilon, delta = 1e-7. The arcosh argument is clamped to 1 + delta so
the evaluation stays finite, and the stable form is
log1p((u-1) + sqrt((u-1)(u+1))).

Every routine is differentiable except the assignment, and the projection keeps
both arms inside the same ball so their feasible domains are identical.
"""
from __future__ import annotations

import torch

#: Curvature of the curved arm; the flat arm is the c -> 0 limit of the same code.
CURVATURE_C = 1.0
#: Temperature of the transition softmax.
TAU = 0.1
#: The projection radius is 1 - EPSILON for both arms.
EPSILON = 1e-5
RADIUS_R = 1.0 - EPSILON
#: The arcosh argument is clamped to at least 1 + DELTA.
DELTA = 1e-7

__all__ = [
    "CURVATURE_C", "TAU", "EPSILON", "RADIUS_R", "DELTA",
    "exp_0_c", "project_ball", "squared_distance", "ball_points",
]


def _norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)


def exp_0_c(v: torch.Tensor, c: float = CURVATURE_C) -> torch.Tensor:
    """Exponential map at the origin; the identity when the curvature is zero.

    The zero-norm limit and the flat limit are both exact, so no branch of the
    formula is approximated by a special case.
    """
    if c <= 0.0:
        return v
    norm = _norm(v)
    root_c = c ** 0.5
    scaled = root_c * norm
    # tanh(s) / s is 1 at s = 0 and -> 1/sqrt(c) * ... for large s; the clamp
    # keeps the ratio finite without changing the limit.
    ratio = torch.where(
        scaled > 1e-12,
        torch.tanh(scaled) / scaled.clamp(min=1e-12),
        torch.ones_like(scaled),
    )
    return ratio * v


def project_ball(x: torch.Tensor, radius: float = RADIUS_R) -> torch.Tensor:
    """Radial projection into the closed ball of the given radius."""
    norm = _norm(x)
    scale = torch.where(
        norm > radius,
        torch.full_like(norm, radius) / norm.clamp(min=1e-12),
        torch.ones_like(norm),
    )
    return scale * x


def squared_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = CURVATURE_C,
    delta: float = DELTA,
) -> torch.Tensor:
    """Squared geodesic distance in the curvature -c ball, with the flat limit.

    ``x`` and ``y`` are broadcast against each other; the result has the
    broadcast shape. At c = 0 the exact limit 4||x-y||^2 is returned.
    """
    diff = x - y
    diff2 = (diff * diff).sum(dim=-1)
    if c <= 0.0:
        return 4.0 * diff2
    x2 = (x * x).sum(dim=-1)
    y2 = (y * y).sum(dim=-1)
    denominator = (1.0 - c * x2) * (1.0 - c * y2)
    denominator = denominator.clamp(min=1e-12)
    u = 1.0 + 2.0 * c * diff2 / denominator
    u = torch.clamp(u, min=1.0 + delta)
    excess = u - 1.0
    arcosh = torch.log1p(excess + torch.sqrt(excess * (u + 1.0)))
    return (arcosh * arcosh) / c


def ball_points(x: torch.Tensor, c: float = CURVATURE_C,
                radius: float = RADIUS_R) -> torch.Tensor:
    """Map into the ball and project; the representation used by both arms."""
    return project_ball(exp_0_c(x, c=c), radius=radius)
