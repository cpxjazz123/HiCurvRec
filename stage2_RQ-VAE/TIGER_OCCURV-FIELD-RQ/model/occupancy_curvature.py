"""A curvature FIELD: every codeword carries its own curvature, set by behaviour.

The mechanism keeps ONE chart - the Poincare ball - and puts the curvature
where the behaviour says it belongs: the reach that the items assigned to a
codeword generate per occupant becomes that codeword's curvature.

* the residual is mapped INTO the ball with the exponential map at the origin,
  ``x = exp_0(z) = tanh(|z|) z / |z|``, after a per-level scale ``s_l`` that
  keeps the points away from the boundary; the same map is applied to the
  codebook, so the residual and the codewords live in the same chart;
* the behaviour reach of an item, ``b_i`` (popularity adjusted, train split
  only), is accumulated per codeword and divided by that codeword's occupancy,
  which gives the reach PER OCCUPANT of every codeword;
* that number, normalised across the codewords, sets the codeword's curvature
  ``kappa_{l,k}``: a codeword few occupants reach strongly gets a larger
  curvature, and the assignment uses the per-codeword-curvature distance.

Two switches, as the proposal requires: the behaviour off replaces ``b_i/pop_i``
by one (a uniform curvature field), and the geometry off replaces the
exponential map and the curved distance by the Euclidean ones.
"""
from __future__ import annotations

import math

import torch

#: The behaviour switch.
USE_BEHAVIOUR = True
#: The geometry switch.
USE_CURVATURE = True

#: The curvature of the field's upper end: ``kappa_max / sqrt(e)``.
KAPPA_MAX = 1.7
#: The clip of the radius of the ball the points live in.
BALL_RADIUS = 0.9
#: The floor of the curvature and of the per-level scale.
EPS = 1e-6
#: The momentum of the per-level residual scale.
SCALE_MOMENTUM = 1.0 / 24.0


def exp0(vector: torch.Tensor) -> torch.Tensor:
    """The exponential map at the origin of the Poincare ball.

    ``R tanh(|v| / R) v / |v|`` with ``R`` the ball radius: a point STRICTLY
    inside the ball for every finite ``v``. The plain ``tanh(|v|)`` saturates
    at the boundary, and a point on the boundary makes the Poincare distance
    between two points blow up - which is how the earlier mechanisms lost their
    assignment. The margin is what keeps the distances finite.
    """
    norm = vector.norm(dim=-1, keepdim=True)
    safe = norm.clamp_min(EPS)
    return (BALL_RADIUS * torch.tanh(norm / BALL_RADIUS)) * vector / safe


def scaled_residual(residual: torch.Tensor, scale: float) -> torch.Tensor:
    """``z = (c / s_l) r`` clipped to the ball radius, then ``exp_0``."""
    z = residual * (BALL_RADIUS / max(scale, EPS))
    norm = z.norm(dim=-1, keepdim=True)
    z = z * torch.clamp(BALL_RADIUS / norm.clamp_min(EPS), max=1.0)
    return exp0(z)


def curvature_field(reach: torch.Tensor, occupancy: torch.Tensor,
                    kappa_max: float = KAPPA_MAX) -> torch.Tensor:
    """``kappa_k = (kappa_max / sqrt(e)) clip(rho_k^{-1/2})`` per codeword.

    ``reach`` is the accumulated behaviour reach of every codeword and
    ``occupancy`` the number of items assigned to it. A codeword many occupants
    reach strongly gets a SMALLER curvature: its cell is wide, and the wide cell
    is what keeps the deeper levels from collapsing onto a few codes.
    """
    if not USE_BEHAVIOUR:
        return torch.full_like(reach, kappa_max / math.sqrt(math.e))
    occupied = occupancy.clamp_min(1.0)
    per_occupant = reach / occupied
    mean = per_occupant.mean().clamp_min(EPS)
    rho = per_occupant / mean
    inverse = rho.clamp_min(EPS).pow(-0.5)
    inverse = inverse.clamp(math.exp(-0.5), math.exp(0.5))
    return (kappa_max / math.sqrt(math.e)) * inverse


def field_distance(point: torch.Tensor, codeword: torch.Tensor,
                   curvature: torch.Tensor) -> torch.Tensor:
    """The distance to a codeword whose curvature is its own.

    ``kappa^{-1/2} acosh(1 + 2 kappa |x - c|^2 / ((1 - kappa |x|^2)(1 - kappa
    |c|^2)))``: the Poincare distance of curvature ``kappa``, scaled. The
    curvature is per codeword and is broadcast over the rows.

    ``point`` is ``(B, D)`` and ``codeword`` is ``(K, D)``; the result is
    ``(B, K)``.
    """
    if not USE_CURVATURE:
        return torch.cdist(point, codeword)
    radius = curvature.reshape(1, -1)
    radius = radius.clamp(0.0, 1.0 / BALL_RADIUS ** 2)
    point_sq = point.pow(2).sum(dim=-1, keepdim=True)
    code_sq = codeword.pow(2).sum(dim=-1).unsqueeze(0)
    difference = (point.unsqueeze(1) - codeword.unsqueeze(0)).pow(2).sum(dim=-1)
    denominator = (1 - radius * point_sq) * (1 - radius * code_sq)
    argument = 1 + 2 * radius * difference / denominator.clamp_min(EPS)
    return radius.clamp_min(EPS).pow(-0.5) * torch.acosh(argument.clamp_min(1.0))


def assign(point: torch.Tensor, codewords: torch.Tensor,
           curvature: torch.Tensor) -> tuple:
    """The winning codeword and its distance, one row per residual.

    ``point`` is ``(B, D)`` inside the ball, ``codewords`` is ``(K, D)`` and
    ``curvature`` is ``(K,)``.
    """
    distances = field_distance(point, codewords, curvature)
    choice = distances.argmin(dim=-1)
    return choice, distances.gather(1, choice.unsqueeze(1)).squeeze(1)


def field_score(point: torch.Tensor, codeword: torch.Tensor,
                curvature: float) -> torch.Tensor:
    """The distance to ONE codeword with its own curvature, one number per row.

    This is the continuous surface of the mechanism: the curve (the distance of
    curvature ``curvature``) and the behaviour (through that curvature) in one
    number.
    """
    radius = float(curvature)
    point_sq = point.pow(2).sum(dim=-1)
    code_sq = codeword.pow(2).sum(dim=-1)
    difference = (point - codeword).pow(2).sum(dim=-1)
    denominator = (1 - radius * point_sq) * (1 - radius * code_sq)
    argument = 1 + 2 * radius * difference / denominator.clamp_min(EPS)
    return radius ** -0.5 * torch.acosh(argument.clamp_min(1.0))


def score_with_curvature(point: torch.Tensor, codeword: torch.Tensor,
                         curvature: float) -> torch.Tensor:
    """The distance to every codeword, with the curvature broadcast over them.

    Kept for the coupling surface: the curvature is per codeword there.
    """
    field = torch.full((codeword.shape[0],), float(curvature), dtype=point.dtype,
                       device=point.device)
    return field_distance(point, codeword, field)
