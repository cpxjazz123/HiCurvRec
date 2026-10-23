"""Fixed-strength hyperbolic mechanisms and parameter-matched Euclidean twins."""
from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn.functional as F

from hyperbolic import (busemann_boundary_matrix, busemann_boundary_poincare,
                        busemann_poincare, encode_ball, expmap0,
                        pair_poincare_distance, poincare_distance,
                        spatial_sq_distance)

CURVATURES = (0.25, 1.0, 4.0)
TARGET_RADIUS_FRACTION = 0.6
MECHANISMS = (
    "geodesic", "assign", "joint", "radial", "localcurve", "perlayer",
    "hardneg", "angular", "capacity", "horosphere", "horosphere_boundary", "product",
)
# The horosphere temperature is locked at specification time, like the strengths: it is not
# tuned against any measured metric table.
HOROSPHERE_TEMPERATURE = 0.25
# The level of a Busemann function diverges as the point approaches the base point, so an item
# whose level leaves this band is SKIPPED and counted -- never clamped (F3: a finite but
# unbounded value is not a stable one).
HOROSPHERE_LEVEL_BOUND = 20.0
# The counts the arm records per step, so the guard is evidence and not a claim.
HOROSPHERE_STATS = {"skipped_items": 0, "max_abs_level": 0.0}
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
# The product-geodesic assignment: ONE common tangent scale (measured; see
# results/hyperbolic_research_runs/subspace_111/latent_scale.json) and a RELATIVE
# curvature. F1: a curvature sweep is a rescaling, so the curvature enters here only
# as the ratio between the two factors.
PRODUCT_SCALE = 1.4937731472770945
PRODUCT_CURVATURE_RATIO = 2.0
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


# the product entries carry the second factor's curvature and the common tangent scale
for _name in list(ARM_CONFIGS):
    if ARM_CONFIGS[_name]["mechanism"] == "product":
        ARM_CONFIGS[_name]["curvature_b"] = (
            ARM_CONFIGS[_name]["curvature"] * PRODUCT_CURVATURE_RATIO)
        ARM_CONFIGS[_name]["tangent_scale"] = PRODUCT_SCALE


def tangent_scale_for_median_radius(latent: torch.Tensor, c: float,
                                    target_fraction: float = TARGET_RADIUS_FRACTION) -> float:
    """Pin numeric median radius uniformly to a fraction of the smallest ball radius."""
    raw_norms = torch.linalg.vector_norm(latent.detach(), dim=-1)
    median_norm = float(raw_norms.median().clamp_min(1e-12))
    batch_rms = float(latent.detach().square().sum(-1).mean().sqrt().clamp_min(1e-12))
    pinned_radius = target_fraction / math.sqrt(max(CURVATURES) * 4.0)
    curvature_fraction = math.sqrt(c) * pinned_radius
    return math.atanh(curvature_fraction) / math.sqrt(c) * batch_rms / median_norm


def encode_points(latent: torch.Tensor, c: float,
                  tangent_scale: float | None = None) -> torch.Tensor:
    scale = tangent_scale if tangent_scale is not None else tangent_scale_for_median_radius(latent, c)
    return encode_ball(latent, c, scale)


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


def _busemann_matrix(codes: torch.Tensor, bases: torch.Tensor, c: float) -> torch.Tensor:
    """The Busemann level of every codeword with respect to every base point.

    The result has shape (len(codes), len(bases)): the entry (j, i) is the level of the
    codeword j with respect to the horosphere family centred at the base point i.
    """
    return busemann_poincare(codes[:, None, :], bases[None, :, :], c)


def _euclidean_level_matrix(codes: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    """The matched Euclidean level: the squared distance to the base point.

    It is the same code without the horosphere: the level is unsigned, so the inside and
    the outside of the horosphere are the same level set, and the term cannot ask for the
    structure the signed Busemann level carries.
    """
    return (codes[:, None, :] - bases[None, :, :]).square().sum(-1)


def _distances(x, y, c, euclidean):
    return torch.cdist(x, y, p=2) if euclidean else pair_poincare_distance(x, y, c)


def nearest_assignments(latent: torch.Tensor, codebook: torch.Tensor, c: float,
                        euclidean: bool = False) -> torch.Tensor:
    """Return non-gradient nearest-code IDs using the selected distance geometry."""
    x, codes = encode_points(latent, c), encode_points(codebook, c)
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
        x = encode_points(latent, c)
        codes = encode_points(self.get_code_embs(), c)
        return pair_poincare_distance(x, codes, c)

    def indices(self, distance_matrix, infer_use_sk):
        if not self._hyp_assignment_active:
            return original_indices(distance_matrix, infer_use_sk)
        return distance_matrix.argmin(dim=-1)

    layer._distances = MethodType(distances, layer)
    layer._indices = MethodType(indices, layer)


def install_product_assignment(layer, c_s: float, c_b: float, scale: float) -> None:
    """The level-0 assignment minimises a sum of two factor geodesic distances.

    The codebook keeps 256 entries of 32 tangent coordinates, split into a semantic half and a
    behaviour half; each half is mapped into its own ball with the SAME tangent scale and its own
    curvature, and the assignment minimises d_{c_s}^2 + d_{c_b}^2. The layer returns the codeword's
    own tangent coordinates, so the residual chain, the decode and the commitment loss are the
    host's and the model/ directory stays byte-identical.
    """
    original_distances = layer._distances
    original_indices = layer._indices
    layer._hyp_assignment_active = False

    def distances(self, latent):
        if not self._hyp_assignment_active:
            return original_distances(latent)
        half = latent.shape[1] // 2
        x_s = expmap0(scale * latent[:, :half], c_s)
        x_b = expmap0(scale * latent[:, half:], c_b)
        codes = self.get_code_embs()
        a_s = expmap0(codes[:, :half], c_s)
        a_b = expmap0(codes[:, half:], c_b)
        return (pair_poincare_distance(x_s, a_s, c_s) ** 2
                + pair_poincare_distance(x_b, a_b, c_b) ** 2)

    def indices(self, distance_matrix, infer_use_sk):
        if not self._hyp_assignment_active:
            return original_indices(distance_matrix, infer_use_sk)
        return distance_matrix.argmin(dim=-1)

    layer._distances = MethodType(distances, layer)
    layer._indices = MethodType(indices, layer)


def mechanism_loss(mechanism: str, latent: torch.Tensor, codebook,
                   positive: torch.Tensor, c: float, strength: float,
                   degrees: torch.Tensor, euclidean: bool = False) -> torch.Tensor:
    """Compute a mechanism loss on z/codebook only; never consume a deep residual."""
    count = positive.numel()
    if latent.shape[0] == 2 * count:
        source_raw, target_raw = latent[:count], latent[count:]
    else:
        source_raw = target_raw = latent
    x, target = encode_points(source_raw, c), encode_points(target_raw, c)
    positive = positive.to(device=x.device, dtype=torch.long).remainder(target.shape[0])
    rows = torch.arange(count, device=x.device)
    if mechanism == "perlayer":
        codebooks = codebook if isinstance(codebook, (tuple, list)) else (codebook,) * 3
        layer_losses = []
        for level, factor in enumerate((1.0, 2.0, 4.0)):
            layer_c = c * factor
            layer_x = encode_points(source_raw, layer_c)
            layer_codes = encode_points(codebooks[level], layer_c)
            distances = _distances(layer_x, layer_codes, layer_c, euclidean)
            assignment = distances.detach().argmin(dim=-1)
            consistency = distances[rows, assignment].mean()
            layer_losses.append(
                _info_nce(source_raw, target_raw, positive, layer_c, euclidean)
                + strength * consistency
            )
        return torch.stack(layer_losses).mean()

    codes = encode_points(codebook, c) if not isinstance(codebook, (tuple, list)) else None
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
    if mechanism == "horosphere":
        # The horosphere constraint. The base point is the TARGET's hyperbolic position;
        # the horosphere through the SOURCE is the level set of the Busemann function of
        # that base point; the codeword the host assigns to the target has to sit ON that
        # horosphere, and the other codewords off it.
        #
        # The label is the host's OWN assignment -- the codeword the Euclidean assignment
        # gives the target -- and it is detached, so it is a hard label that does not move
        # with the term. The base point and the source level are detached as well, so the
        # gradient reaches the codebook and nothing else.
        xi = target.detach()
        codes = encode_points(codebook, c)
        if euclidean:
            source_level = spatial_sq_distance(x, xi)
            code_levels = _euclidean_level_matrix(codes, xi)
        else:
            source_level = busemann_poincare(x, xi, c)
            code_levels = _busemann_matrix(codes, xi, c)
        source_level = source_level.detach()
        label = torch.cdist(target_raw, codebook).argmin(dim=-1)
        gap = (code_levels.t() - source_level[:, None]).abs()
        return strength * F.cross_entropy(-gap / HOROSPHERE_TEMPERATURE, label)
    if mechanism == "horosphere_boundary":
        # The horosphere through the source, with the target's DIRECTION as the base point.
        #
        # The base point is ON THE BOUNDARY, which is the only place a Busemann function is
        # defined; with an interior base point the "level" is a near-radial equidistance function
        # (measured: results/hyperbolic_research_runs/issue112_busemann_audit.json). The level
        # sets here are horospheres, and the level is signed.
        target_ball = encode_points(target_raw, c)
        omega = target_ball.detach()
        omega = omega / torch.linalg.vector_norm(omega, dim=-1, keepdim=True).clamp_min(1e-12)
        omega = omega / math.sqrt(c)          # the boundary of the curvature -c ball
        source_ball = encode_points(source_raw, c)
        codes = encode_points(codebook, c)
        source_level = busemann_boundary_poincare(source_ball, omega, c)
        code_levels = busemann_boundary_matrix(codes, omega, c)
        source_level = source_level.detach()
        label = torch.cdist(target_raw, codebook).argmin(dim=-1)
        gap = (code_levels.t() - source_level[:, None]).abs()
        # the guard: an item whose levels leave the declared band is skipped and counted
        keep = (code_levels.abs() < HOROSPHERE_LEVEL_BOUND).all(dim=0) & (
            source_level.abs() < HOROSPHERE_LEVEL_BOUND)
        HOROSPHERE_STATS["skipped_items"] += int((~keep).sum())
        HOROSPHERE_STATS["max_abs_level"] = max(
            float(HOROSPHERE_STATS["max_abs_level"]),
            float(code_levels.abs().max()),
        )
        if not bool(keep.any()):
            return source_ball.sum() * 0.0
        loss = F.cross_entropy(-gap[keep] / HOROSPHERE_TEMPERATURE, label[keep])
        return strength * loss
    if mechanism == "product":
        # The product-geodesic level-0 assignment contributes no additive term: the mechanism
        # works by changing the assignment (see install_product_assignment), so its effect is
        # inside the base loss's graph.
        return source_raw.sum() * 0.0
    if mechanism == "geodesic":
        return strength * _info_nce(source_raw, target_raw, positive, c, euclidean)
    raise ValueError(f"unknown hyperbolic mechanism: {mechanism}")


def _info_nce(source, target, positive, c, euclidean):
    distances = _distances(encode_points(source, c), encode_points(target, c), c, euclidean)
    return _info_nce_matrix(distances, positive)


def _info_nce_matrix(distances, positive):
    return F.cross_entropy(-distances, positive)
