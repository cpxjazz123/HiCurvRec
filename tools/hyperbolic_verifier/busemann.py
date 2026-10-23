"""The verifier's own Busemann / horosphere mathematics (issue #112, P1).

Independent of any mechanism: this module implements the Busemann function of a point at
infinity, the geometry of its level sets, and the two identities that separate a horosphere from
a geodesic sphere. It exists to answer one question about the R10 arm with proof rather than with
a name: **is the level set its operator produces a horosphere?**

The mathematics, for the unit ball (a ball of curvature -c is the unit ball after x -> sqrt(c) x):

* A Busemann function is defined by a point ``omega`` ON the boundary sphere (``||omega|| = 1``).
  For the unit ball, ``b_omega(x) = log( (1 - ||x||^2) / ||x - omega||^2 )`` represents the
  Busemann class of ``omega``: it is ``-inf``-free, finite everywhere inside the ball, NEGATIVE
  inside the horosphere through the origin and POSITIVE outside it.
* Its level sets are horospheres, which as Euclidean sets are spheres TANGENT to the boundary:
  ``|centre| + radius = 1`` exactly.
* A geodesic sphere -- the set of points at a fixed hyperbolic distance from an INTERIOR point --
  is a Euclidean sphere with ``|centre| + radius < 1``.

So the tangency gap ``|centre| + radius - 1`` is decisive: ``0`` means horosphere, negative means
geodesic sphere, whatever the function is called.

``r10_declared_level`` transcribes the formula the R10 proposal declares, so the comparison is
between two written formulas and never calls the arm's code.
"""
from __future__ import annotations

import math

import torch

FLOOR = 1e-30


# ------------------------------------------------------------------ the model
def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c * c * x2 * y2
    return numerator / denominator.clamp_min(FLOOR)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    nx = (x * x).sum(-1)
    ny = (y * y).sum(-1)
    diff = ((x - y) ** 2).sum(-1)
    arg = 1 + 2 * c * diff / ((1 - c * nx).clamp_min(FLOOR) * (1 - c * ny).clamp_min(FLOOR))
    return torch.acosh(arg.clamp_min(1.0)) / math.sqrt(c)


# ------------------------------------------------------------------ a genuine Busemann function
def busemann_boundary(x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """``b_omega(x)`` for the UNIT ball and a boundary point ``omega`` (``||omega|| = 1``).

    Finite everywhere inside the ball, signed (negative inside the horosphere through the origin,
    positive outside), and its level sets are horospheres tangent at ``omega``.
    """
    omega = omega / omega.norm().clamp_min(FLOOR)
    numerator = 1.0 - (x * x).sum(-1)
    difference = x - omega
    denominator = (difference * difference).sum(-1).clamp_min(FLOOR)
    return torch.log((numerator / denominator).clamp_min(FLOOR))


def horosphere_points(omega: torch.Tensor, radius: float, count: int = 12) -> torch.Tensor:
    """Points of the horosphere tangent at ``omega``, built as a Euclidean sphere by construction.

    The sphere is centred at ``(1 - radius) * omega`` with Euclidean radius ``radius``, so
    ``|centre| + radius = 1`` holds exactly: this is a horosphere, not a fitted approximation.
    """
    omega = omega / omega.norm().clamp_min(FLOOR)
    dim = omega.numel()
    centre = (1.0 - radius) * omega
    # an orthonormal frame in the hyperplane orthogonal to omega
    basis = []
    for index in range(1, dim):
        vector = torch.zeros(dim, dtype=omega.dtype)
        vector[index] = 1.0
        vector = vector - (vector * omega).sum() * omega
        norm = vector.norm()
        if float(norm) > 1e-9:          # skip the direction parallel to omega (it collapses)
            basis.append(vector / norm)
    frame = torch.stack(basis) if basis else torch.zeros(0, dim, dtype=omega.dtype)
    points = []
    for index in range(count):
        phase = 2 * math.pi * index / count
        # a UNIT offset in the orthogonal subspace: the sum of cos(phi_k) over the frame is not
        # a unit vector, so only two coefficients are used (a great circle of the sphere, which
        # is enough to fit the sphere and to test its tangency).
        offset = torch.zeros(dim, dtype=omega.dtype)
        if frame.shape[0] >= 3:
            # a 2-sphere, so the Euclidean sphere through the samples is unique (a circle is not)
            polar = phase
            azimuth = 2.399963 * index          # a golden-angle spiral: spread, not clustered
            offset = (math.sin(polar) * math.cos(azimuth) * frame[0]
                      + math.sin(polar) * math.sin(azimuth) * frame[1]
                      + math.cos(polar) * frame[2])
        elif frame.shape[0] == 2:
            offset = math.cos(phase) * frame[0] + math.sin(phase) * frame[1]
        elif frame.shape[0] == 1:
            offset = math.cos(phase) * frame[0]
        points.append(centre + radius * offset)
    return torch.stack(points)


def fit_euclidean_sphere(points: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Centre and radius of the Euclidean sphere through ``points`` (least squares)."""
    p = points.double()
    matrix = torch.cat([2.0 * p, torch.ones(p.shape[0], 1, dtype=p.dtype)], dim=1)
    target = (p * p).sum(dim=1, keepdim=True)
    solution, *_ = torch.linalg.lstsq(matrix, target)
    centre = solution[: p.shape[1], 0]
    radius = torch.sqrt((solution[p.shape[1], 0] + (centre * centre).sum()).clamp_min(0.0))
    return centre, float(radius)


def tangency_gap(points: torch.Tensor) -> float:
    """``|centre| + radius - 1``: 0 for a horosphere, negative for a geodesic sphere."""
    centre, radius = fit_euclidean_sphere(points)
    return float(centre.norm()) + radius - 1.0


# ------------------------------------------------------------------ geodesic spheres
def geodesic_sphere(xi: torch.Tensor, distance: float, count: int = 12,
                    c: float = 1.0) -> torch.Tensor:
    """Points at hyperbolic distance ``distance`` from the INTERIOR point ``xi``.

    Built with the Mobius exponential map in the unit ball and scaled back to curvature -c, so
    the construction does not depend on any mechanism.
    """
    root = math.sqrt(c)
    unit_xi = xi.double() * root
    step_length = math.tanh(root * distance / 2.0)
    samples = []
    for index in range(count):
        phase = 2 * math.pi * index / count
        direction = torch.zeros_like(unit_xi)
        direction[0] = math.cos(phase)
        if unit_xi.numel() > 1:
            direction[1] = math.sin(phase)
        direction = direction / direction.norm().clamp_min(FLOOR)
        samples.append(mobius_add(unit_xi, direction * step_length, 1.0) / root)
    return torch.stack(samples)


# ------------------------------------------------------------------ the R10 declaration
def r10_declared_level(x: torch.Tensor, xi: torch.Tensor, c: float) -> torch.Tensor:
    """The level function the R10 proposal declares, transcribed from its own equation.

    ``S = -<x, xi>_L`` realised in the ball, ``level = log((S + 1/c) / (S - 1/c)) / sqrt(c)``,
    with the tensor clamp the arm uses. Nothing here calls the arm's code.
    """
    x_root = (1.0 - c * (x * x).sum(-1)).clamp_min(1e-12).sqrt()
    xi_root = (1.0 - c * (xi * xi).sum(-1)).clamp_min(1e-12).sqrt()
    inner = (x * xi).sum(-1)
    numerator = 1.0 / c - inner + x_root * xi_root / c
    denominator = (1.0 / c - inner - x_root * xi_root / c).clamp_min(1e-12)
    return torch.log((numerator / denominator).clamp_min(FLOOR)) / math.sqrt(c)


__all__ = [
    "busemann_boundary",
    "fit_euclidean_sphere",
    "geodesic_sphere",
    "horosphere_points",
    "mobius_add",
    "poincare_distance",
    "r10_declared_level",
    "tangency_gap",
]
