"""Pairwise geometry losses for the E0/H1/E1 controlled comparison.

All arms use ``weight * softplus(d(anchor, positive) - d(anchor, negative))``.
E0 measures squared Euclidean distance on raw tangent vectors. H1 and E1
first apply the same differentiable exponential map; H1 uses fixed-curvature
Poincare geodesic distance and E1 uses squared Euclidean distance. A fixed
scale of 2.0 moves real L1 vectors into the mid-ball; no trainable or
data-dependent radial anchor is added.

Numerical contract: the shared H1/E1 tangent scale is 2.0; the resulting
scaled tangent norm is capped at 2.0 before mapping, map epsilon is 1e-8,
and mapped radii are projected below ``BALL_CLAMP = 1 - 1e-5`` (the
curvature-scaled ball radius is ``BALL_CLAMP / sqrt(curvature)``).
Curvature is a caller-supplied positive scalar and is never a parameter.
Quantization and codeword assignment are not imported or modified.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BALL_CLAMP = 1.0 - 1e-5
MAP_EPS = 1e-8
RADIAL_SCALE = 2.0
NORM_CAP = 2.0
STRENGTHS = (0.5, 2.0, 8.0)


def _curvature(curvature, like: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(curvature, dtype=like.dtype, device=like.device)
    if value.numel() != 1 or not torch.isfinite(value).all() or value.item() <= 0:
        raise ValueError("curvature must be a finite positive scalar")
    return value


def map_to_ball(x: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Map tangent vectors into the open curvature-scaled Poincare ball."""
    c = _curvature(curvature, x)
    scaled = x * RADIAL_SCALE
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    cap = torch.as_tensor(NORM_CAP, dtype=x.dtype, device=x.device)
    capped_norm = torch.minimum(norm, cap)
    capped = scaled * (capped_norm / norm.clamp_min(MAP_EPS))
    root_c = torch.sqrt(c)
    scaled_norm = root_c * capped_norm
    factor = torch.tanh(scaled_norm) / scaled_norm.clamp_min(MAP_EPS)
    mapped = capped * factor
    radius = torch.linalg.vector_norm(mapped, dim=-1, keepdim=True)
    max_radius = torch.as_tensor(BALL_CLAMP, dtype=x.dtype, device=x.device) / root_c
    safe_radius = torch.minimum(radius, max_radius)
    return mapped * (safe_radius / radius.clamp_min(MAP_EPS))


def poincare_distance(x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Stable geodesic distance for points in a ball of curvature ``-c``."""
    c = _curvature(curvature, x)
    delta2 = (x - y).square().sum(dim=-1)
    nx = x.square().sum(dim=-1)
    ny = y.square().sum(dim=-1)
    denominator = ((1.0 - c * nx) * (1.0 - c * ny)).clamp_min(MAP_EPS)
    q = (2.0 * c * delta2 / denominator).clamp_min(0.0)
    acosh = torch.log1p(q + torch.sqrt(q * (q + 2.0) + MAP_EPS))
    return acosh / torch.sqrt(c)


def squared_euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance, retaining all leading broadcast dimensions."""
    return (x - y).square().sum(dim=-1)


def mapped_points(arm: str, x: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Return the coordinates on which the selected arm measures distance."""
    if arm == "E0":
        return x
    if arm in ("H1", "E1"):
        return map_to_ball(x, curvature)
    raise ValueError("unknown geometry arm: {}".format(arm))


def metric_distance(arm: str, x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Apply the selected metric to coordinates already in that arm's space."""
    if arm in ("E0", "E1"):
        return squared_euclidean_distance(x, y)
    if arm == "H1":
        return poincare_distance(x, y, curvature)
    raise ValueError("unknown geometry arm: {}".format(arm))


def distance(arm: str, x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Map each point as configured, then apply the arm's distance metric."""
    return metric_distance(
        arm,
        mapped_points(arm, x, curvature),
        mapped_points(arm, y, curvature),
        curvature,
    )


def geometry_loss(
    arm: str,
    z: torch.Tensor,
    codebook: torch.Tensor,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    *,
    curvature: float = 1.0,
    weight: float = 0.2,
) -> torch.Tensor:
    """Weighted positive-vs-negative item softplus ranking loss.

    ``anchors``, ``positives`` and ``negatives`` all index rows of ``z``. The
    codebook argument is retained for the shared call contract but is not used:
    geometry supervision is on encoder representations, not quantized vectors.
    """
    if arm not in ("E0", "H1", "E1"):
        raise ValueError("unknown geometry arm: {}".format(arm))
    anchor = z[anchors]
    positive = z[positives]
    negative = z[negatives]
    positive_distance = distance(arm, anchor, positive, curvature)
    negative_distance = distance(arm, anchor, negative, curvature)
    return float(weight) * F.softplus(positive_distance - negative_distance).mean()


def compose_loss(behavior_loss: torch.Tensor, geometry_term: torch.Tensor) -> torch.Tensor:
    """Add the geometry term, preserving behaviour loss exactly when zero."""
    if not torch.count_nonzero(geometry_term.detach()).item():
        return behavior_loss
    return behavior_loss + geometry_term
