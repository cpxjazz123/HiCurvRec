"""Issue #146: a rotationally symmetric warped-product geometry on the 32-d code.

Metric in polar coordinates around the origin of a level's 32-d space:

    ds^2 = dr^2 + f_alpha(r)^2 dOmega^2,   f_alpha(r) = r (1 + alpha r^2)

The radial sectional curvature of this warped product is

    K_rad(r) = -f''(r) / f(r) = -6 alpha / (1 + alpha r^2),

which is the continuous surface this file states and tests.

The scale rule is frozen before training: ``alpha_l = 1 / r50_l^2`` where
``r50_l`` is the median residual norm entering level ``l`` on the TRAIN
embeddings, measured once with the initialized encoder and KMeans codebooks.
It reads no behaviour graph.

The exact geodesic distance is computed here by shooting the geodesic ODE in
arclength (``scipy.integrate.solve_ivp`` + a scalar root). It is differentiable
in principle through the implicit function theorem, but that derivative is NOT
implemented: the pointwise solve already costs milliseconds per pair, while one
training step needs roughly ``batch x 256 x 3`` pairs, so a full 3000-epoch run
is far outside the frozen budget. The block is recorded as
``RESOURCE_BLOCKED`` / ``IMPLEMENTATION_NEEDS_FIX``; no approximation is
substituted for the exact distance.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

__all__ = ["warping", "radial_curvature", "alpha_from_norms", "euclidean_polar",
           "geodesic_distance", "same_ray_distance", "pair_cost_seconds"]


def warping(alpha: float, radius):
    """``f_alpha(r) = r (1 + alpha r^2)``."""
    radius = np.asarray(radius, dtype=np.float64)
    return radius * (1.0 + float(alpha) * radius * radius)


def radial_curvature(alpha: float, radius):
    """``K_rad(r) = -f''(r)/f(r) = -6 alpha / (1 + alpha r^2)``."""
    radius = np.asarray(radius, dtype=np.float64)
    return -6.0 * float(alpha) / (1.0 + float(alpha) * radius * radius)


def alpha_from_norms(norms) -> float:
    """``alpha = 1 / median(norm)^2``; the only input is the TRAIN residual norm."""
    values = np.asarray(norms, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("the residual norms must be a finite non-empty sample")
    median = float(np.median(values))
    if not median > 0.0:
        raise ValueError("the median residual norm must be positive")
    return 1.0 / (median * median)


def euclidean_polar(r1: float, r2: float, dtheta: float) -> float:
    """The flat-space distance, the alpha = 0 limit of the warped product."""
    return math.sqrt(max(r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * math.cos(dtheta), 0.0))


def _geodesic_ode(_s, state, alpha: float):
    radius, _phi, u_r, u_phi = state
    f = float(warping(alpha, radius))
    f_prime = 1.0 + 3.0 * float(alpha) * radius * radius
    return [u_r, u_phi / f,
            f_prime * u_phi * u_phi / f,
            -f_prime * u_r * u_phi / f]


def _shoot(beta: float, alpha: float, r1: float, dtheta: float) -> tuple[float, float]:
    """Integrate from ``r1`` until the polar angle reaches ``dtheta``.

    Returns the radius reached and the arclength. The polar angle increases
    monotonically because ``u_phi = sin(beta) > 0``, so the event is well posed
    even when the geodesic's radius turns around.
    """
    def hit(_s, state, *_args):
        return state[1] - dtheta
    hit.terminal = True
    hit.direction = 1
    solution = solve_ivp(_geodesic_ode, (0.0, 1e4), [r1, 0.0, math.cos(beta), math.sin(beta)],
                         args=(alpha,), events=hit, rtol=1e-9, atol=1e-11, max_step=1.0)
    if solution.t_events[0].size == 0:
        raise RuntimeError("the geodesic did not reach the target angle")
    state = solution.y_events[0][0]
    return float(state[0]), float(solution.t_events[0][0])


def geodesic_distance(alpha: float, r1: float, r2: float, dtheta: float,
                      *, tolerance: float = 1e-8) -> float:
    """The exact geodesic length between ``(r1, 0)`` and ``(r2, dtheta)``.

    The product is rotationally symmetric, so the geodesic lies in the plane of
    the two radial directions. The shooting parameter is the initial angle
    ``beta`` in the orthonormal frame; a scan over beta finds every bracket whose
    final radius crosses ``r2``, and the shortest of the solved roots is the
    geodesic (longer roots are wrapping geodesics).
    """
    r1, r2, dtheta = float(r1), float(r2), float(dtheta)
    if r1 <= 0.0 or r2 <= 0.0:
        raise ValueError("the warped product needs positive radii")
    if dtheta == 0.0:
        return abs(r2 - r1)

    def residual(beta: float) -> float | None:
        try:
            radius, _length = _shoot(beta, alpha, r1, dtheta)
        except RuntimeError:
            return None
        return radius - r2

    grid = np.linspace(1e-3, math.pi - 1e-3, 96)
    values = [residual(beta) for beta in grid]
    roots: list[float] = []
    for left, right, value_left, value_right in zip(grid[:-1], grid[1:],
                                                    values[:-1], values[1:]):
        if value_left is None or value_right is None:
            continue
        if value_left == 0.0:
            roots.append(float(left))
        elif value_left * value_right < 0:
            roots.append(float(brentq(lambda beta: residual(beta), left, right,
                                      xtol=tolerance)))
    if not roots:
        raise RuntimeError("no shooting angle reaches the target radius")
    lengths = [_shoot(beta, alpha, r1, dtheta)[1] for beta in roots]
    return float(min(lengths))


def same_ray_distance(r1: float, r2: float) -> float:
    """``|r1 - r2|``: the exact geodesic for zero angular separation."""
    return abs(float(r2) - float(r1))


def pair_cost_seconds(alpha: float, r1: float, r2: float, dtheta: float,
                      repeats: int = 3) -> float:
    """Seconds per exact distance evaluation, measured, not estimated."""
    start = time.perf_counter()
    for _ in range(repeats):
        geodesic_distance(alpha, r1, r2, dtheta)
    return (time.perf_counter() - start) / repeats
