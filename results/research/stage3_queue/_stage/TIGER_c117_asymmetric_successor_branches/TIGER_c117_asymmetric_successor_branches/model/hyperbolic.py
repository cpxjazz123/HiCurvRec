"""Curvature-parameterised Poincare operators for the successor-branch candidate.

Frozen candidate: geometry_asymmetric_successor_branches_hyp
(run_2d84ba77647b4008, review verdict CPU_ELIGIBLE).

The proposal freezes one implementation whose curvature is a parameter:

    D_c^2(x,y) = (1/c) * arcosh(1 + 2c||x-y||^2 / ((1-c||x||^2)(1-c||y||^2)))^2
    D_0^2(x,y) = 4||x-y||^2                                  (exact c -> 0 limit)

    arcosh(z) is evaluated as log1p(u + sqrt(u*(u+2))) with u = z - 1 >= 0,
    and the argument is clamped to at least 1.0 -- NOT 1 + delta: a fixed floor
    would make the distance constant 2*delta/c near coincidence instead of
    approaching zero.

Both arms project every scored vector into the SAME closed ball ||x|| <= R with
R = 0.9 < 1, so the feasible domain and the representation map are identical and
the only arm-specific rule is the distance formula. The curvature-dependent
projection radius (1-epsilon)/sqrt(c) is not used anywhere: at c = 0 it would be
undefined, and the fixed R applies in both arms by construction.
"""
from __future__ import annotations

import torch

#: Curvature of the curved arm; the flat arm is the c -> 0 limit of this code.
CURVATURE_C = 1.0
#: Shared fixed projection radius for BOTH arms (strictly inside the unit ball).
RADIUS_R = 0.9
#: The arcosh argument is clamped at 1.0, never at 1 + delta.
CLAMP_MIN = 1.0
#: Frozen objective coefficients (proposal: "the frozen Stage2 coefficients").
BETA_COMMIT = 0.25
BETA_CODE = 1.0

__all__ = ["CURVATURE_C", "RADIUS_R", "CLAMP_MIN", "BETA_COMMIT", "BETA_CODE",
           "project_ball", "squared_distance", "arcosh_stable"]


def _norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)


def project_ball(x: torch.Tensor, radius: float = RADIUS_R) -> torch.Tensor:
    """Radial projection into the closed ball of the given radius."""
    norm = _norm(x)
    scale = torch.where(norm > radius,
                        torch.full_like(norm, radius) / norm.clamp(min=1e-12),
                        torch.ones_like(norm))
    return scale * x


def arcosh_stable(u: torch.Tensor) -> torch.Tensor:
    """arcosh(1 + u) for u >= 0, evaluated as log1p(u + sqrt(u*(u+2)))."""
    excess = torch.clamp(u, min=0.0)
    return torch.log1p(excess + torch.sqrt(excess * (excess + 2.0)))


def squared_distance(x: torch.Tensor, y: torch.Tensor,
                     c: float = CURVATURE_C) -> torch.Tensor:
    """Squared geodesic distance, broadcasting, with the exact flat limit."""
    diff = x - y
    diff2 = (diff * diff).sum(dim=-1)
    if c <= 0.0:
        return 4.0 * diff2
    x2 = (x * x).sum(dim=-1)
    y2 = (y * y).sum(dim=-1)
    denominator = ((1.0 - c * x2) * (1.0 - c * y2)).clamp(min=1e-12)
    argument = CLAMP_MIN + 2.0 * c * diff2 / denominator
    # arcosh(z) with z = 1 + u and u clamped to >= 0: the clamp is at 1.0.
    return (arcosh_stable(argument - CLAMP_MIN) ** 2) / c


def squared_distance_matrix(x: torch.Tensor, y: torch.Tensor,
                            c: float = CURVATURE_C) -> torch.Tensor:
    """The same squared distance for every pair, without a 3-D temporary."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1).unsqueeze(0)
    diff2 = (x2 + y2 - 2.0 * (x @ y.t())).clamp(min=0.0)
    if c <= 0.0:
        return 4.0 * diff2
    denominator = ((1.0 - c * x2) * (1.0 - c * y2)).clamp(min=1e-12)
    argument = CLAMP_MIN + 2.0 * c * diff2 / denominator
    return (arcosh_stable(argument - CLAMP_MIN) ** 2) / c
