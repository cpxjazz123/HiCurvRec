"""Fixed-strength hyperbolic mechanisms and parameter-matched Euclidean twins."""
from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn.functional as F

from hyperbolic import encode_ball, expmap0, pair_poincare_distance, poincare_distance

CURVATURES = (0.25, 1.0, 4.0)
TARGET_RADIUS_FRACTION = 0.6
MECHANISMS = (
    "geodesic", "assign", "joint", "radial", "localcurve", "perlayer",
    "hardneg", "angular", "capacity",
)
# Locked at specification time; no post-test strength tuning.
_STRENGTHS = {name: 0.10 for name in MECHANISMS}
# Exact existing TIGER model trainable shapes: encoder, decoder, and 3 codebooks.
_PARAMETER_SHAPES = (
    (512, 768), (512,), (256, 512), (256,), (128, 256), (128,),
    (32, 128), (32,), (128, 32), (128,), (256, 128), (256,),
    (512, 256), (512,), (768, 512), (768,),
    (256, 32), (256, 32), (256, 32),
)
ARM_CONFIGS = {}
for _mechanism in MECHANISMS:
    for _c in CURVATURES:
        for _prefix, _distance in (("hyp_", "poincare"), ("eucl_", "euclidean")):
            _name = f"{_prefix}{_mechanism}_c{_c:g}"
            ARM_CONFIGS[_name] = {
                "mechanism": _mechanism, "curvature": _c,
                "strength": _STRENGTHS[_mechanism], "distance": _distance,
                "parameter_shapes": _PARAMETER_SHAPES,
            }
for _c in CURVATURES:
    ARM_CONFIGS[f"radial_scale_c{_c:g}"] = {
        "mechanism": "radial_scale", "curvature": _c, "strength": 1.35,
        "distance": "poincare", "parameter_shapes": _PARAMETER_SHAPES,
    }
    ARM_CONFIGS[f"eucl_margin_c{_c:g}"] = {
        "mechanism": "hardneg", "curvature": _c, "strength": 0.10,
        "distance": "euclidean", "parameter_shapes": _PARAMETER_SHAPES,
    }


def tangent_scale_for_median_radius(latent: torch.Tensor, c: float,
                                    target_fraction: float = TARGET_RADIUS_FRACTION) -> float:
    """Pin numeric median radius uniformly to a fraction of the smallest ball radius."""
    raw_norms = torch.linalg.vector_norm(latent.detach(), dim=-1)
    median_norm = float(raw_norms.median().clamp_min(1e-12))
    batch_rms = float(latent.detach().square().sum(-1).mean().sqrt().clamp_min(1e-12))
    pinned_radius = target_fraction / math.sqrt(max(CURVATURES) * 4.0)
    curvature_fraction = math.sqrt(c) * pinned_radius
    return math.atanh(curvature_fraction) / math.sqrt(c) * batch_rms / median_norm


def shared_tangent_scale(reference: torch.Tensor, c: float,
                         target_fraction: float = TARGET_RADIUS_FRACTION) -> float:
    """One tangent scale for both sides of a distance, derived from the codebook.

    ``tangent_scale_for_median_radius`` reads the statistics of whatever tensor
    it is given, so encoding items and codewords separately placed the same
    vector in two different places and moved the mapping with every batch. The
    scale is derived from the reference alone -- and ``encode_points`` is called
    with the same reference, which also supplies the RMS ``encode_ball`` divides
    by. Both halves are needed: threading the scale alone still left the ball map
    normalising by each tensor's own RMS, so the mapping kept drifting with batch
    composition until the reference was passed through as well.
    """
    return tangent_scale_for_median_radius(reference, c, target_fraction)


def encode_points(latent: torch.Tensor, c: float,
                  tangent_scale: float | None = None,
                  reference: torch.Tensor | None = None) -> torch.Tensor:
    """Map into the ball under the scale and RMS of ``reference``.

    Every two-sided distance passes the codebook as the reference, so the
    same vector lands in the same place no matter which batch it arrived in
    and no matter which side of the distance it is on. Callers that map a
    single tensor (``pair_order``) may leave it out and keep the historical
    per-tensor behaviour.
    """
    scale = tangent_scale if tangent_scale is not None else tangent_scale_for_median_radius(latent, c)
    return encode_ball(latent, c, scale, reference)


def pair_order(latent: torch.Tensor, pairs, c: float) -> tuple[int, ...]:
    points = encode_points(latent, c)
    i = torch.as_tensor([p[0] for p in pairs], device=latent.device)
    j = torch.as_tensor([p[1] for p in pairs], device=latent.device)
    return tuple(torch.argsort(poincare_distance(points[i], points[j], c)).tolist())


def radial_scale_distances(points: torch.Tensor, pairs, scale: float, c: float = 1.0):
    """Apply the radial scale control to geodesic distance units, preserving order."""
    i = torch.as_tensor([p[0] for p in pairs], device=points.device)
    j = torch.as_tensor([p[1] for p in pairs], device=points.device)
    plain = poincare_distance(points[i], points[j], c)
    return plain * scale, plain


def _distances(x, y, c, euclidean):
    return torch.cdist(x, y, p=2) if euclidean else pair_poincare_distance(x, y, c)


def nearest_assignments(latent: torch.Tensor, codebook: torch.Tensor, c: float,
                        euclidean: bool = False) -> torch.Tensor:
    """Return non-gradient nearest-code IDs using the selected distance geometry."""
    scale = shared_tangent_scale(codebook, c)
    x = encode_points(latent, c, scale, reference=codebook)
    codes = encode_points(codebook, c, scale, reference=codebook)
    return _distances(x, codes, c, euclidean).argmin(dim=-1)


def install_nearest_assignment(layer, c: float, euclidean: bool) -> None:
    original_distances = layer._distances
    original_indices = layer._indices
    layer._hyp_assignment_active = False

    def distances(self, latent):
        if not self._hyp_assignment_active:
            return original_distances(latent)
        if euclidean:
            return torch.cdist(latent, self.get_code_embs(), p=2).square()
        book = self.get_code_embs()
        scale = shared_tangent_scale(book, c)
        x = encode_points(latent, c, scale, reference=book)
        codes = encode_points(book, c, scale, reference=book)
        return pair_poincare_distance(x, codes, c)

    # ``_indices`` is deliberately left alone. It is the only place that
    # decides Sinkhorn (training, last level) versus argmin (inference), and
    # overriding it with a plain argmin deleted the balanced-assignment
    # control, so the arm differed from the control in two variables.
    layer._distances = MethodType(distances, layer)


def mechanism_loss(mechanism: str, latent: torch.Tensor, codebook,
                   positive: torch.Tensor, c: float, strength: float,
                   degrees: torch.Tensor, euclidean: bool = False) -> torch.Tensor:
    """Compute a mechanism loss on z/codebook only; never consume a deep residual."""
    count = positive.numel()
    if latent.shape[0] == 2 * count:
        source_raw, target_raw = latent[:count], latent[count:]
    else:
        source_raw = target_raw = latent
    scale = shared_tangent_scale(target_raw, c)
    x = encode_points(source_raw, c, scale, reference=target_raw)
    target = encode_points(target_raw, c, scale, reference=target_raw)
    positive = positive.to(device=x.device, dtype=torch.long).remainder(target.shape[0])
    rows = torch.arange(count, device=x.device)
    if mechanism == "perlayer":
        codebooks = codebook if isinstance(codebook, (tuple, list)) else (codebook,) * 3
        layer_losses = []
        for level, factor in enumerate((1.0, 2.0, 4.0)):
            layer_c = c * factor
            book = codebooks[level]
            layer_scale = shared_tangent_scale(book, layer_c)
            layer_x = encode_points(source_raw, layer_c, layer_scale, reference=book)
            layer_codes = encode_points(book, layer_c, layer_scale, reference=book)
            distances = _distances(layer_x, layer_codes, layer_c, euclidean)
            assignment = distances.detach().argmin(dim=-1)
            consistency = distances[rows, assignment].mean()
            layer_losses.append(
                _info_nce(source_raw, target_raw, positive, layer_c, euclidean)
                + strength * consistency
            )
        return torch.stack(layer_losses).mean()

    code_scale = (None if isinstance(codebook, (tuple, list))
                 else shared_tangent_scale(codebook, c))
    codes = (encode_points(codebook, c, code_scale, reference=codebook)
             if code_scale is not None else None)
    if mechanism == "radial_scale":
        return _info_nce_matrix(-_distances(x, target, c, euclidean) * strength, positive)
    if mechanism == "radial":
        degree = degrees.to(device=x.device, dtype=x.dtype)
        desired = torch.log1p(degree) / torch.log1p(degree).mean().clamp_min(1e-12)
        median_radius = torch.linalg.vector_norm(x, dim=-1).median().detach()
        target_radius = median_radius * (1.0 + strength * (desired - 1.0))
        return strength * F.mse_loss(torch.linalg.vector_norm(x, dim=-1), target_radius)
    if mechanism == "localcurve":
        degree = degrees.to(device=x.device, dtype=x.dtype)
        local_c = c * (1.0 + strength * (degree / degree.mean().clamp_min(1e-12) - 1.0))
        median_norm = torch.linalg.vector_norm(source_raw.detach(), dim=-1).median().clamp_min(1e-12)
        curve_points = torch.stack([
            expmap0(
                source_raw[k] * (math.atanh(TARGET_RADIUS_FRACTION) /
                                 (math.sqrt(float(local_c[k].detach())) * median_norm)),
                float(local_c[k].detach()),
            )
            for k in range(count)
        ])
        if euclidean:
            matrix = torch.cdist(curve_points, curve_points)
        else:
            matrix = torch.stack([
                pair_poincare_distance(curve_points[k:k + 1], curve_points,
                                       float(local_c[k].detach()))[0]
                for k in range(count)
            ])
        return strength * _info_nce_matrix(matrix, positive)
    if mechanism == "hardneg":
        matrix = _distances(x, target, c, euclidean)
        pos_dist = matrix[rows, positive]
        blocked = torch.zeros_like(matrix, dtype=torch.bool)
        blocked[rows, positive] = True
        negative = matrix.masked_fill(blocked, torch.inf).min(-1).values
        return strength * F.relu(strength + pos_dist - negative).mean()
    if mechanism == "angular":
        attraction = (1.0 - (F.normalize(source_raw, dim=-1) *
                             F.normalize(target_raw, dim=-1)[positive]).sum(-1)).mean()
        radii = torch.linalg.vector_norm(x, dim=-1)
        repulsion = F.relu(0.25 - torch.cdist(radii[:, None], radii[:, None])).mean()
        return strength * (attraction + repulsion)
    if mechanism == "capacity":
        neighborhood = torch.exp(-_distances(x, x, c, euclidean).detach()).sum(-1) - 1.0
        capacity = (1.0 + neighborhood).pow(-strength)
        nearest = _distances(x, codes, c, euclidean).min(-1).values
        return strength * (capacity * nearest.square()).mean()
    if mechanism in ("assign", "joint"):
        matrix = _distances(x, codes, c, euclidean)
        assignment = matrix.detach().argmin(-1)
        consistency = matrix[torch.arange(len(x), device=x.device), assignment].mean()
        if mechanism == "assign":
            # H2 changes hard code assignment only; its gradient comes from the
            # unchanged behavior and VQ objectives, not an added assignment loss.
            return x.sum() * 0.0
        return strength * (_info_nce(source_raw, target_raw, positive, c, euclidean)
                           + consistency)
    if mechanism == "geodesic":
        return strength * _info_nce(source_raw, target_raw, positive, c, euclidean)
    raise ValueError(f"unknown hyperbolic mechanism: {mechanism}")


def _info_nce(source, target, positive, c, euclidean):
    scale = shared_tangent_scale(target, c)
    distances = _distances(encode_points(source, c, scale, reference=target),
                           encode_points(target, c, scale, reference=target),
                           c, euclidean)
    return _info_nce_matrix(distances, positive)


def _info_nce_matrix(distances, positive):
    return F.cross_entropy(-distances, positive)
