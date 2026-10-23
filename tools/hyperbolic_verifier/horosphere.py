"""The verifier's own fixtures for the horosphere candidate (issue #112, P1/P2).

Independent of any candidate: this module holds the reference mathematics the candidate's
H1-H7 gates are checked against, and the counterfactual that decides whether the mechanism is
really hyperbolic.

The decisive fixture is the EUCLIDEAN TWIN. The horosphere candidate asks the codeword to sit on
a level set of the Busemann function of a boundary point. In the hyperbolic model that level set
is a sphere TANGENT to the boundary; in the Euclidean model the same construction degenerates to
the half-space through the origin, whose level sets are PLANES and whose level is an unsigned
linear functional. So:

* if the candidate's level is constant on a tangent sphere and varies across it for the Euclidean
  twin, the mechanism cannot be replaced by a Euclidean computation without changing it (H2), and
  the curvature CLASS, not its magnitude, is what the mechanism rests on (H3, per finding F4);
* the guard's skip count and the maximum |level| are the numerical evidence H5 asks for.
"""
from __future__ import annotations

import math

import torch

FLOOR = 1e-30


def euclidean_twin_level(x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """The Euclidean counterpart of the Busemann level: a linear functional of x.

    In the Euclidean model the "horosphere through the origin" is the half-space whose boundary is
    the hyperplane orthogonal to the base point, so the level is ``<x, omega>`` -- unsigned, and
    its level sets are planes.
    """
    omega = omega / omega.norm().clamp_min(FLOOR)
    return (x * omega).sum(-1)


def level_spread_on(points: torch.Tensor, level_fn) -> float:
    """The spread of a level function over a set of points (zero means it is constant there)."""
    levels = level_fn(points)
    return float(levels.max() - levels.min())


def boundary_point(direction: torch.Tensor, c: float) -> torch.Tensor:
    """The boundary point of the curvature -c ball in the given direction."""
    direction = direction / direction.norm().clamp_min(FLOOR)
    return direction / math.sqrt(c)


def guard_levels(levels: torch.Tensor, bound: float) -> torch.Tensor:
    """Which rows stay inside the declared band; the rest are SKIPPED, never clamped."""
    return (levels.abs() < bound).all(dim=-1) if levels.dim() > 1 else (levels.abs() < bound)


__all__ = [
    "boundary_point",
    "euclidean_twin_level",
    "guard_levels",
    "level_spread_on",
]
