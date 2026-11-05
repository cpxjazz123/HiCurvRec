"""Geodesic capacity cells: behaviour demand sets a per-codeword geodesic radius.

The codebook is the baseline's Poincare codebook, untouched. What the behaviour
changes is the SIZE of the cell each codeword covers: a codeword that serves a
large share of the transition demand gets a larger geodesic radius, and the
radius enters the assignment as an additive offset on the Poincare distance of
that same codeword.

One chart, one codebook, one assignment, one reconstruction: the codeword the
assignment chose is the codeword the decoder receives. That is the property the
previous mechanism lacked - it assigned with transported codewords and summed
the untransported ones, and the codebook collapsed.

The curvature enters twice, in the same formula: the distance is the Poincare
distance ``d_c`` and the cell radius follows the hyperbolic AREA law
``A = (2*pi/c) (cosh(sqrt(c) rho) - 1)``, so the radius of a cell of area ``A``
is ``rho = arccosh(1 + c A / (2 pi)) / sqrt(c)``. At ``c = 0`` the same code
path gives the Euclidean power cell ``rho = sqrt(A / pi)``.
"""
from __future__ import annotations

import math

import torch

from .poincare_contract import distance as poincare_distance
from .poincare_contract import project

#: The two switches of the mechanism. They exist in the code and behave; the
#: experiment policy trains the both-on arm only.
USE_BEHAVIOUR = True
#: The curvature the mechanism uses; ``c = 0`` is the Euclidean limit the
#: geometry-off switch selects.
USE_CURVATURE = True
CURVATURE = 1.0
#: The clip on a cell radius, in units of the ball radius.
RADIUS_FRACTION = 0.95
#: The EMA momentum of the demand: one part in 96, as the proposal fixes it.
DEMAND_MOMENTUM = 1.0 - 1.0 / 96.0


def cell_area(radius: float, curvature: float = CURVATURE) -> float:
    """``A = (2 pi / c) (cosh(sqrt(c) r) - 1)``; ``pi r^2`` at ``c = 0``."""
    if curvature <= 0.0:
        return math.pi * radius * radius
    root = math.sqrt(curvature) * radius
    return 2.0 * math.pi / curvature * (math.cosh(root) - 1.0)


def radius_of_area(area, curvature: float = CURVATURE):
    """The geodesic radius of a cell of area ``area``.

    ``rho = arccosh(1 + c A / (2 pi)) / sqrt(c)``; ``sqrt(A / pi)`` at ``c = 0``.
    """
    if curvature <= 0.0:
        return torch.sqrt(torch.clamp(area, min=0.0) / math.pi)
    argument = 1.0 + curvature * area / (2.0 * math.pi)
    return torch.acosh(argument.clamp_min(1.0)) / math.sqrt(curvature)


def offsets(demand: torch.Tensor, *, base_radius: float, base_area: float,
            curvature: float = CURVATURE) -> torch.Tensor:
    """``delta_k = rho(A0 * demand_k) - h``, the additive offset of a codeword.

    ``demand`` is the per-codeword demand ``bhat`` (mean 1). The offset is a
    geodesic LENGTH, so it enters the score with coefficient 1.
    """
    if not USE_BEHAVIOUR:
        # behaviour off: every cell has the base area, so the offset vanishes
        return torch.zeros_like(demand)
    radius = radius_of_area(base_area * demand.clamp_min(1e-6), curvature)
    ceiling = RADIUS_FRACTION / math.sqrt(curvature) if curvature > 0 else 1e9
    radius = radius.clamp(0.0, ceiling)
    return radius - base_radius


def score(residual: torch.Tensor, codeword: torch.Tensor, offset: torch.Tensor,
          curvature: float = CURVATURE) -> torch.Tensor:
    """``d_c(sigma_c(r), sigma_c(a)) - delta``: the assignment score of a codeword.

    A lower score wins. The offset is per codeword, never per row: a positive
    row scalar would leave every arg-min unchanged.
    """
    effective = curvature if USE_CURVATURE else 0.0
    lifted = project(residual, effective) if effective > 0 else residual
    target = project(codeword, effective) if effective > 0 else codeword
    return poincare_distance(lifted, target, effective) - offset


def assign(residual: torch.Tensor, codebook: torch.Tensor,
           offset: torch.Tensor, curvature: float = CURVATURE) -> tuple:
    """The winning codeword and its score, one row per residual."""
    effective = curvature if USE_CURVATURE else 0.0
    lifted = project(residual, effective) if effective > 0 else residual
    targets = project(codebook, effective) if effective > 0 else codebook
    spacing = poincare_distance(lifted.unsqueeze(1), targets.unsqueeze(0), effective)
    scores = spacing - offset.unsqueeze(0)
    choice = scores.argmin(dim=-1)
    return choice, scores.gather(1, choice.unsqueeze(1)).squeeze(1)


def score_with_demand(residual: torch.Tensor, codeword: torch.Tensor,
                      demand: float, base_radius: float,
                      base_area: float) -> torch.Tensor:
    """The continuous surface of the mechanism, one number per row.

    ``d_c(sigma_c(r), sigma_c(a)) - (rho(A0 * demand) - h)``: the Poincare
    distance of the codeword minus the geodesic offset its demand buys. Both
    halves of the mechanism are in this one number - the curve (the distance and
    the area law) and the behaviour (the demand).
    """
    offset = offsets(torch.as_tensor(demand, dtype=residual.dtype),
                     base_radius=base_radius, base_area=base_area)
    return score(residual, codeword, offset.expand(()))


def demand_from(spacing: torch.Tensor, behaviour: torch.Tensor) -> torch.Tensor:
    """``raw_k = sum_i b_i softmax_k(-D_ik / tau_i)`` with ``tau_i = mean_k D_ik``.

    The soft assignment mass of every codeword, weighted by the behaviour
    salience of the items it attracts. ``spacing`` is ``(B, K)``, ``behaviour``
    is ``(B,)``.
    """
    temperature = spacing.mean(dim=1, keepdim=True).clamp_min(1e-6)
    weights = torch.softmax(-spacing / temperature, dim=1)
    return (weights * behaviour.unsqueeze(1)).sum(dim=0)
