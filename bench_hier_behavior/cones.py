"""Entailment cones (Ganea, Becigneul & Hofmann 2018) in both geometries.

A cone at apex ``x`` holds the points ``y`` with

    Xi(x, y) <= psi(x),   psi(x) = arcsin(K * g(x))

where ``Xi`` is the angle at ``x`` between the geodesic to the origin and the
geodesic to ``y``, and ``g`` is ``(1 - ||x||^2)/||x||`` in the Poincare ball and
``1/||x||`` in Euclidean space. Both are the closed forms the paper proves
optimal under transitivity; ``K`` is the single capacity knob and is always
fitted on training edges only.

Cones open *outward* from the origin, which is the paper's convention: a general
concept sits near the origin and its more specific children sit further out.
That is exactly the property this benchmark asks the learned representation to
have, so it is never assumed.

Degenerate geometry (a zero-norm apex, or an apex coinciding with the point)
makes the angle undefined rather than large. Those pairs are mapped to
``Xi = pi``, i.e. outside the cone, so a collapsed representation scores as a
collapse instead of poisoning an AUC with NaN.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import bench_config as cfg

_NORM_EPS = 1e-6


def _clamp_cosine(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        value, nan=0.0, posinf=1.0, neginf=-1.0
    ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)


def _outside(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=math.pi, posinf=math.pi, neginf=0.0)


def apex_norm(apex: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(apex, dim=-1).clamp_min(_NORM_EPS)


def euclid_xi_pairs(apex: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Xi for aligned (n, d) pairs: the paper's Eq. (31)."""
    apex_sq = (apex * apex).sum(-1)
    point_sq = (point * point).sum(-1)
    delta_sq = ((apex - point) ** 2).sum(-1)
    denominator = 2.0 * apex_norm(apex) * delta_sq.clamp_min(_NORM_EPS ** 2).sqrt()
    return _outside(torch.arccos(
        _clamp_cosine((point_sq - apex_sq - delta_sq) / denominator)
    ))


def poincare_xi_pairs(apex: torch.Tensor, point: torch.Tensor,
                      curvature: float) -> torch.Tensor:
    """Xi in the Poincare ball, from the paper's Eq. (27)-(28)."""
    apex_sq = (apex * apex).sum(-1)
    point_sq = (point * point).sum(-1)
    dot = (apex * point).sum(-1)
    delta_sq = ((apex - point) ** 2).sum(-1)
    denominator = (
        apex_norm(apex)
        * delta_sq.clamp_min(_NORM_EPS ** 2).sqrt()
        * (1.0 + apex_sq * point_sq - 2.0 * dot).clamp_min(_NORM_EPS ** 2).sqrt()
    )
    numerator = dot * (1.0 + apex_sq) - apex_sq * (1.0 + point_sq)
    angle = torch.arccos(_clamp_cosine(numerator / denominator))
    return _outside(math.pi - angle)


def euclid_aperture_factor(apex: torch.Tensor) -> torch.Tensor:
    return 1.0 / apex_norm(apex)


def poincare_aperture_factor(apex: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(apex, dim=-1).clamp_min(_NORM_EPS)
    return (1.0 - (apex * apex).sum(-1)).clamp_min(0.0) / norm


FACTOR = {
    "euclid": euclid_aperture_factor,
    "poincare": poincare_aperture_factor,
}


def xi_pairs(geometry: str, apex: torch.Tensor,
             point: torch.Tensor) -> torch.Tensor:
    if geometry == "euclid":
        return euclid_xi_pairs(apex, point)
    if geometry == "poincare":
        return poincare_xi_pairs(apex, point, cfg.CURVATURE)
    raise ValueError(f"unknown geometry: {geometry}")


def fit_k(factors: np.ndarray, xi: np.ndarray,
          target: float = cfg.CONE_FIT_TARGET_COVERAGE) -> float:
    """Largest K whose cone covers ``target`` of the fit pairs.

    ``factors`` is the per-pair ``g(x)`` of that pair's apex and ``xi`` the
    per-pair angle. Coverage is monotone in K, so bisection on [0, max factor]
    is exact to float precision.
    """
    factors = np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)
    upper = float(np.max(factors))
    if upper <= 0.0:
        return 0.0
    lo, hi = 0.0, upper
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if coverage_at(factors, xi, mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def coverage_at(factors: np.ndarray, xi: np.ndarray, k: float) -> float:
    if len(xi) == 0:
        return 0.0
    aperture = np.arcsin(np.clip(k * factors, 0.0, 1.0))
    return float(np.mean(xi <= aperture))


def entailment_margin_loss(
    geometry: str,
    model_geometry,
    apex_tangent: torch.Tensor,
    child_tangent: torch.Tensor,
    negative_tangent: torch.Tensor | None,
    *,
    k: float,
    angle_margin: float = 0.05,
    radial_margin: float = 0.02,
    radial_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train positive parent-child containment and reject non-descendants.

    A root-to-coarse edge has no valid negative descendant: every coarse node
    belongs beneath that root. Passing ``None`` therefore trains only positive
    cone containment and outward radial order for that level.
    """
    apex_point = model_geometry.to_point(apex_tangent)
    child_point = model_geometry.to_point(child_tangent)
    factor = FACTOR[geometry](apex_point)
    aperture = torch.asin((float(k) * factor).clamp(0.0, 1.0))
    positive_xi = xi_pairs(geometry, apex_point, child_point)
    positive_loss = F.relu(positive_xi - aperture + angle_margin).mean()
    if negative_tangent is None:
        negative_loss = positive_loss.new_zeros(())
    else:
        negative_point = model_geometry.to_point(negative_tangent)
        negative_xi = xi_pairs(geometry, apex_point, negative_point)
        negative_loss = F.relu(aperture + angle_margin - negative_xi).mean()
    apex_radius = torch.linalg.vector_norm(apex_point, dim=-1)
    child_radius = torch.linalg.vector_norm(child_point, dim=-1)
    radial_loss = F.relu(apex_radius + radial_margin - child_radius).mean()
    total = positive_loss + negative_loss + float(radial_weight) * radial_loss
    return total, positive_loss, negative_loss, radial_loss
