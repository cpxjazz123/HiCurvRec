"""Quantized cone supervision for the residual quantizer.

Implements the entailment-cone energy the paper defines, for both geometries,
on the cumulative quantized representations the RQ-VAE already builds:

    L_cone = E(P_coarse, P_fine) + E(P_coarse, Q2) + E(P_fine, Q3)

The apex angle follows Eqs. 28 (Poincare) and 31 (Euclidean) and the aperture is
psi(x) = arcsin(K * g(x)) with the constant K the reference uses. The angle is
the one the paper defines - not its supplement; taking the supplement inverts
entailment, which ``scripts/cone_supervision_test.py`` pins down.

Category prototypes are independent learnable tangent-space parameters mapped
into the arm's space by the same exp-map the quantizer uses, so both arms carry
exactly the same 89 prototypes of dimension 32 and differ only in the metric.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import _expmap0_tangent, _logmap0_point, _mobius_add


def from_point(
    geometry: str, point: torch.Tensor, curvature: torch.Tensor | float
) -> torch.Tensor:
    """Inverse of :func:`to_point`."""
    if geometry == EUCLID:
        return point
    if geometry == POINCARE:
        return _logmap0_point(point, curvature)
    raise ValueError(f"Unknown geometry: {geometry}")

NORM_EPS = 1e-6
COS_EPS = 1e-7
EUCLID = "euclid"
POINCARE = "poincare"


def to_point(
    geometry: str, tangent: torch.Tensor, curvature: torch.Tensor | float
) -> torch.Tensor:
    """Map origin-tangent vectors into the arm's space."""
    if geometry == EUCLID:
        return tangent
    if geometry == POINCARE:
        return _expmap0_tangent(tangent, curvature)
    raise ValueError(f"Unknown geometry: {geometry}")


def apex_angle(
    geometry: str,
    apex: torch.Tensor,
    point: torch.Tensor,
) -> torch.Tensor:
    """``xi(apex, point)``: the angle at the apex between its two geodesics.

    ``apex`` and ``point`` are points, already in the arm's space, and are
    broadcast on their last dimension.
    """
    apex_sq = apex.square().sum(-1)
    point_sq = point.square().sum(-1)
    dot = (apex * point).sum(-1)
    delta_sq = (apex - point).square().sum(-1).clamp_min(NORM_EPS * NORM_EPS)
    apex_norm = apex.norm(dim=-1).clamp_min(NORM_EPS)
    if geometry == POINCARE:
        numerator = dot * (1.0 + apex_sq) - apex_sq * (1.0 + point_sq)
        denominator = (
            delta_sq.sqrt()
            * apex_norm
            * (1.0 + apex_sq * point_sq - 2.0 * dot).clamp_min(NORM_EPS * NORM_EPS).sqrt()
        )
    elif geometry == EUCLID:
        numerator = point_sq - apex_sq - delta_sq
        denominator = 2.0 * delta_sq.sqrt() * apex_norm
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    cosine = (numerator / denominator).clamp(-1.0 + COS_EPS, 1.0 - COS_EPS)
    return torch.nan_to_num(torch.arccos(cosine), nan=math.pi, posinf=math.pi)


def aperture(
    geometry: str, apex: torch.Tensor, k: float
) -> torch.Tensor:
    """``psi(x)``: the widest cone the capacity constant K admits at x."""
    norm = apex.norm(dim=-1).clamp_min(NORM_EPS)
    if geometry == POINCARE:
        factor = (1.0 - norm.square()).clamp_min(0.0) / norm
    elif geometry == EUCLID:
        factor = 1.0 / norm
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    return torch.asin((float(k) * factor).clamp(0.0, 1.0))


def energy(
    geometry: str, apex: torch.Tensor, point: torch.Tensor, k: float
) -> torch.Tensor:
    """Cone energy: 0 when the point sits inside the apex's cone."""
    return torch.relu(apex_angle(geometry, apex, point) - aperture(geometry, apex, k))


def k_for_aperture(
    geometry: str, apex: torch.Tensor, target_degrees: float
) -> float:
    """K that opens the aperture to ``target_degrees`` at the median apex.

    Fitting K to a coverage target is meaningless on this data: at
    initialisation a prototype sits nowhere near the items it has to contain, so
    no aperture reaches the requested share and the fit pins to its cap, leaving
    a half space that constrains nothing. Specifying the aperture width directly
    puts both arms at the same initial cone width instead.
    """
    factors = aperture_factor(geometry, apex)
    median = float(factors.median())
    if median <= 0.0:
        return 0.0
    return float(math.sin(math.radians(float(target_degrees)))) / median


def coverage_at(
    factors: torch.Tensor, angles: torch.Tensor, k: float
) -> float:
    """Share of pairs the aperture at k contains."""
    psi = torch.asin((float(k) * factors).clamp(0.0, 1.0))
    return float((angles <= psi).to(torch.float32).mean())


def aperture_factor(
    geometry: str, apex: torch.Tensor
) -> torch.Tensor:
    norm = apex.norm(dim=-1).clamp_min(NORM_EPS)
    if geometry == POINCARE:
        return (1.0 - norm.square()).clamp_min(0.0) / norm
    if geometry == EUCLID:
        return 1.0 / norm
    raise ValueError(f"Unknown geometry: {geometry}")


def fit_k(
    geometry: str,
    apex: torch.Tensor,
    point: torch.Tensor,
    target_coverage: float,
    *,
    max_k_fraction: float = 0.5,
) -> float:
    """Largest K whose aperture covers ``target_coverage`` of the pairs.

    Both inputs are already points in the arm's space. Coverage is monotone in
    K, so a bisection is exact to float precision. Run separately per arm: the
    two spaces have different capacity, and sharing one K would hand the flatter
    geometry the wider cone for free.
    """
    factors = aperture_factor(geometry, apex)
    angles = apex_angle(geometry, apex, point)
    upper = float(factors.max())
    if upper <= 0.0:
        return 0.0
    low, high = 0.0, upper * float(max_k_fraction)
    for _ in range(60):
        middle = 0.5 * (low + high)
        if coverage_at(factors, angles, middle) < target_coverage:
            low = middle
        else:
            high = middle
    return high


def containment_loss(
    geometry: str,
    curvature: torch.Tensor | float,
    apex_tangent: torch.Tensor,
    point_tangent: torch.Tensor,
    negative_tangent: torch.Tensor,
    k: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Positive containment plus negative rejection for one edge type.

    The positive term is the one-sided cone energy ``relu(xi - psi)``. The
    negative term must NOT be built from that energy: for a negative that has
    already fallen inside the cone the energy is clamped to zero, so
    ``relu(margin - energy)`` is a constant with a zero gradient and the item
    can never be pushed back out. Rejection therefore uses the signed angle
    difference ``relu(margin + psi - xi)``, which stays active for every
    negative inside the cone or near its boundary and vanishes only once the
    negative is comfortably outside.
    """
    apex = to_point(geometry, apex_tangent, curvature)
    point = to_point(geometry, point_tangent, curvature)
    negative = to_point(geometry, negative_tangent, curvature)
    positive = energy(geometry, apex, point, k)
    negative_gap = (
        float(margin)
        + aperture(geometry, apex, k)
        - apex_angle(geometry, apex, negative)
    )
    negative = F.relu(negative_gap)
    total = positive.mean() + negative.mean()
    return total, positive.mean(), negative.mean()


CONFORMAL_FLOOR = 1e-3


def conformal_factor(
    geometry: str, point: torch.Tensor, curvature: torch.Tensor | float
) -> torch.Tensor:
    """``lambda(x)``: how much the arm's metric stretches at ``x``.

    A cone is an angular object and the Poincare ball is conformal, so the same
    aperture selects the same points in both arms. What the geometries disagree
    about is how far a point sits from the cone boundary once distance is
    measured with the arm's own metric, and that distance is the angular gap
    multiplied by this factor. For euclid the factor is identically one.
    """
    if geometry == EUCLID:
        return torch.ones_like(point[..., 0])
    if geometry == POINCARE:
        norm_sq = point.square().sum(dim=-1)
        return 2.0 / (1.0 - float(curvature) * norm_sq).clamp_min(CONFORMAL_FLOOR)
    raise ValueError(f"Unknown geometry: {geometry}")


def metric_containment_loss(
    geometry: str,
    curvature: torch.Tensor | float,
    apex_tangent: torch.Tensor,
    point_tangent: torch.Tensor,
    negative_tangent: torch.Tensor,
    k: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Cone supervision measured in the arm's own metric.

    Each hinge is weighted by the conformal factor at the point it acts on, so
    the objective is the gap to the cone boundary in metric units rather than in
    radians. Near the ball's boundary the hyperbolic factor grows without bound,
    which is the geometric statement that a fixed angular gap buys an
    exponentially larger separation there; a Euclidean cone cannot reproduce
    that. Both weights are divided by their batch mean so the two arms keep the
    same overall loss scale and only the relative emphasis differs - for euclid
    the weights are all one and this reduces exactly to ``containment_loss``.
    """
    apex = to_point(geometry, apex_tangent, curvature)
    point = to_point(geometry, point_tangent, curvature)
    negative = to_point(geometry, negative_tangent, curvature)
    positive = energy(geometry, apex, point, k)
    negative_gap = (
        float(margin)
        + aperture(geometry, apex, k)
        - apex_angle(geometry, apex, negative)
    )
    negative = F.relu(negative_gap)
    positive_weight = conformal_factor(geometry, point, curvature)
    negative_weight = conformal_factor(geometry, negative, curvature)
    scale = 0.5 * (positive_weight.mean() + negative_weight.mean())
    scale = scale.clamp_min(CONFORMAL_FLOOR)
    positive = positive * positive_weight / scale
    negative = negative * negative_weight / scale
    total = positive.mean() + negative.mean()
    return total, positive.mean(), negative.mean(), float(scale)


def cone_separation_loss(
    geometry: str,
    curvature: torch.Tensor | float,
    apex_tangent: torch.Tensor,
    k: float,
    metric_gap: float,
) -> torch.Tensor:
    """Keep different categories' cones apart by a margin in the arm's metric.

    Containment alone is satisfied by widening cones until they cover their
    categories, and wide cones of different categories then overlap, which is
    how supervised codebooks lose the fine-category discriminability Stage3
    needs. This term asks for a gap between every pair of cones, and it asks for
    it in metric units: the required angular gap is ``metric_gap`` divided by
    the conformal factor at the pair.

    That division is the hyperbolic part. Near the ball's boundary the factor is
    large, so the same metric gap is met by a much smaller angular gap, and the
    number of cones that fit in the space grows with the volume the metric
    provides rather than with the angle alone. A Euclidean cone has no such
    factor, so it must buy separation with angle it does not have to spare.
    """
    apex = to_point(geometry, apex_tangent, curvature)
    count = apex.shape[0]
    if count < 2:
        return apex.sum() * 0.0
    # The separation of two cones is the angle between their axes, measured at
    # the origin. ``apex_angle`` cannot serve here: it is the angle at one apex
    # between the geodesic to the origin and the geodesic to the other point,
    # which for two prototypes at equal radius is already near a right angle and
    # would leave the term satisfied no matter how close the axes are. The
    # metric is conformal, so an angle at the origin is the Euclidean angle
    # between the directions, which is what this computes.
    direction = apex / apex.norm(dim=-1, keepdim=True).clamp_min(NORM_EPS)
    cosine = (direction @ direction.transpose(0, 1)).clamp(-1.0 + COS_EPS, 1.0 - COS_EPS)
    angles = torch.acos(cosine)
    psi = aperture(geometry, apex, k)
    factor = conformal_factor(geometry, apex, curvature)
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=apex.device)
    pair_factor = 0.5 * (factor.unsqueeze(1) + factor.unsqueeze(0))
    required = float(metric_gap) / pair_factor.clamp_min(CONFORMAL_FLOOR)
    deficit = (psi.unsqueeze(1) + psi.unsqueeze(0) + required - angles).clamp_min(0.0)
    return deficit[off_diagonal].mean()


class CategoryPrototypes(nn.Module):
    """Learnable tangent-space prototypes, one per coarse and fine category.

    Both arms hold the same count and dimension and draw the same directions
    from the same seed; only the map into the space differs, so a comparison
    cannot be won by having more parameters.
    """

    def __init__(
        self,
        coarse_ids: torch.Tensor,
        fine_ids: torch.Tensor,
        dim: int,
        reference_radius: float,
        coarse_radius_ratio: float,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.n_coarse = int(coarse_ids.max().item()) + 1
        self.n_fine = int(fine_ids.max().item()) + 1
        coarse = self._init_points(
            self.n_coarse,
            dim,
            reference_radius * coarse_radius_ratio,
            generator,
            device,
        )
        fine = self._init_points(
            self.n_fine, dim, reference_radius, generator, device
        )
        self.coarse = nn.Parameter(coarse)
        self.fine = nn.Parameter(fine)

    @staticmethod
    def _init_points(
        count: int,
        dim: int,
        radius: float,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        direction = torch.randn(count, dim, device=device, generator=generator)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(NORM_EPS)
        return direction * float(radius)

    def extra_repr(self) -> str:
        return f"coarse={self.n_coarse}, fine={self.n_fine}"


def radial_order_loss(
    geometry: str,
    curvature: torch.Tensor | float,
    coarse_tangent: torch.Tensor,
    fine_tangent: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Keep coarse prototypes interior to the fine prototypes of their children."""
    coarse = to_point(geometry, coarse_tangent, curvature).norm(dim=-1)
    fine = to_point(geometry, fine_tangent, curvature).norm(dim=-1)
    return F.relu(coarse.unsqueeze(1) + float(margin) - fine).mean()


def containment_rate(
    geometry: str,
    curvature: torch.Tensor | float,
    apex_tangent: torch.Tensor,
    point_tangent: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """Share of points inside the apex cone, as a boolean tensor."""
    apex = to_point(geometry, apex_tangent, curvature)
    point = to_point(geometry, point_tangent, curvature)
    return (apex_angle(geometry, apex, point) - aperture(geometry, apex, k)) <= 0.0