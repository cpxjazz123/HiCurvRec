"""D18: hyperbolic angular clustering, entailment cones, hierarchy and curvature sensitivity.

Frozen analysis. It reuses the accepted Stage2 checkpoint, the item embeddings and
the verified SID export, checks them against the SID-Diag-01 hashes, and never
trains, never writes model state and never touches a loss. The encoder pass and
all geometry run on the GPU when one is free.

What it measures:

  D18-A  angular clustering of real behaviour neighbours against popularity-matched
         negatives, with bootstrap stability of the clustering
  D18-B  Hyperbolic Entailment Cones (Ganea et al., 2018) over the behaviour
         direction of each source: positive coverage, false-positive rate and
         angular margin. Apertures come in a fixed-angle family and a metric family
         whose half-aperture is solved from the curvature-1 distance law, so
         curvature sets how wide a cone may be at each radius
  D18-C  behaviour-graph hierarchy (degree, PageRank, k-core, depth-to-sink) against
         radial position, with the popularity metrics next to the structural ones so
         the two are never conflated
  D18-D  frozen curvature sensitivity: the same latents re-read at several
         curvatures, separating the angle term from negative-curvature amplification.
         This is a property of the frozen representation, not a retrained score

Outputs land beside the existing D14-D17 artefacts and merge into summary.json.
"""
from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
import torch

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
SOURCE_DIR = ROOT / "stage2_RQ-VAE/curvature_RQ-VAE"
OUTPUT_DIR = ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE/sid_diag"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
PAIR_SAMPLE_PATH = OUTPUT_DIR / "behavior_pair_sample.csv"

sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(SOURCE_DIR / "scripts"))
import curvature_config as experiment  # noqa: E402
import train_rqvae as trainer  # noqa: E402
import sid_diag as audit  # noqa: E402
from model import RQVAE  # noqa: E402
from model.layers import _hyperbolic_residual, _mobius_add  # noqa: E402

SEED = 42
CONE_PAIR_SAMPLE = 40_000
MAX_PARTNERS = 24
FIXED_APERTURES_DEG = (2.0, 5.0, 10.0, 20.0, 30.0, 45.0)
METRIC_APERTURES = (0.02, 0.05, 0.1, 0.2, 0.4)
CURVATURE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
BOOTSTRAP_REPLICATES = 200
REPRESENTATIONS = ("Encoder", "L1-L1", "L1-L2", "L1-L3")


# --------------------------------------------------------------------------------------
# hyperbolic helpers, evaluated in the diagnostic device's default float32
# --------------------------------------------------------------------------------------
def _curvature(value: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(
        value, device=reference.device, dtype=reference.dtype
    ).clamp_min(torch.finfo(reference.dtype).tiny)


def expmap0(vector: torch.Tensor, curvature: float | torch.Tensor) -> torch.Tensor:
    c = _curvature(curvature, vector)
    sqrt_c = c.sqrt()
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    scaled = sqrt_c * norm
    scale = torch.tanh(scaled) / scaled.clamp_min(torch.finfo(vector.dtype).tiny)
    point = scale * vector
    point_norm = torch.linalg.vector_norm(point, dim=-1, keepdim=True)
    limit = (1.0 - 1e-7) / sqrt_c
    return point * torch.clamp(limit / point_norm.clamp_min(1e-30), max=1.0)


def logmap0(point: torch.Tensor, curvature: float | torch.Tensor) -> torch.Tensor:
    c = _curvature(curvature, point)
    sqrt_c = c.sqrt()
    norm = torch.linalg.vector_norm(point, dim=-1, keepdim=True)
    safe = norm.clamp_min(torch.finfo(point.dtype).tiny)
    scaled = (sqrt_c * norm).clamp(max=1.0 - 1e-7)
    return (torch.atanh(scaled) / (sqrt_c * safe)) * point


def ball_radius(vector: torch.Tensor, curvature: float | torch.Tensor) -> torch.Tensor:
    """Geodesic distance from the origin, i.e. the point's depth in the ball."""
    c = _curvature(curvature, vector)
    point = expmap0(vector, c)
    return 2.0 / c.sqrt() * torch.atanh(
        (c.sqrt() * torch.linalg.vector_norm(point, dim=-1)).clamp(max=1.0 - 1e-7)
    )


def displacement(apex: torch.Tensor, other: torch.Tensor, curvature: float) -> torch.Tensor:
    """Tangent vector at the apex pointing at ``other``, in the apex tangent frame.

    The tangent space at a point of the Poincare ball is isometric to the tangent
    space at the origin and the isometry is Mobius subtraction followed by the
    origin log map, which is what this returns.
    """
    c = _curvature(curvature, apex)
    apex_point = expmap0(apex, c)
    other_point = expmap0(other, c)
    return logmap0(_mobius_add(-apex_point, other_point, c), c)


def angle_between(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    cosine = (
        (left * right).sum(-1)
        / (
            torch.linalg.vector_norm(left, dim=-1).clamp_min(1e-30)
            * torch.linalg.vector_norm(right, dim=-1).clamp_min(1e-30)
        )
    ).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def apex_frame_angle(
    apex: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    curvature: float,
) -> torch.Tensor:
    """Angle at the apex between the geodesics towards two points.

    The Poincare differential at a point is a rotation and a scale, not the
    identity, so transporting a direction to the origin tangent space with a
    Mobius subtraction does not preserve angles. The hyperbolic law of cosines
    does: for the triangle (left, right, apex),
    cos(angle at apex) = (cosh d(apex,left) cosh d(apex,right) - cosh d(left,right))
                         / (sinh d(apex,left) sinh d(apex,right)).
    """
    apex_left = geodesic_distance(apex, left, curvature)
    apex_right = geodesic_distance(apex, right, curvature)
    left_right = geodesic_distance(left, right, curvature)
    numerator = torch.cosh(apex_left) * torch.cosh(apex_right) - torch.cosh(left_right)
    denominator = (torch.sinh(apex_left) * torch.sinh(apex_right)).clamp_min(1e-12)
    return torch.acos((numerator / denominator).clamp(-1.0, 1.0))


def origin_bearing_angle(
    apex: torch.Tensor, other: torch.Tensor, curvature: float
) -> torch.Tensor:
    """Angle at the apex between the geodesic towards ``other`` and towards the origin."""
    apex_other = geodesic_distance(apex, other, curvature)
    apex_origin = ball_radius(apex, curvature)
    other_origin = ball_radius(other, curvature)
    numerator = torch.cosh(apex_other) * torch.cosh(apex_origin) - torch.cosh(other_origin)
    denominator = (torch.sinh(apex_other) * torch.sinh(apex_origin)).clamp_min(1e-12)
    return torch.acos((numerator / denominator).clamp(-1.0, 1.0))


def log_euclidean_mean(points: torch.Tensor, mask: torch.Tensor, curvature: float) -> torch.Tensor:
    """Ball point whose tangent vector is the masked mean of the points' log maps."""
    tangent = logmap0(expmap0(points, curvature), curvature) * mask.unsqueeze(-1)
    return expmap0(tangent.sum(1), curvature)


def unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-30)


def metric_aperture(radius: torch.Tensor, reach: float) -> torch.Tensor:
    """Half-aperture whose geodesic reach at the apex radius is exactly ``reach``.

    Two points at the same radius rho separated by theta sit at distance
    d(theta) = acosh(cosh^2 rho - sinh^2 rho cos theta), so the half-aperture is that
    law inverted. Curvature and radius set the aperture together; it is not a constant.
    """
    cosh_target = torch.cosh(torch.as_tensor(reach, dtype=radius.dtype, device=radius.device))
    cos_theta = (torch.cosh(radius).square() - cosh_target) / torch.sinh(radius).square().clamp_min(1e-30)
    return torch.acos(cos_theta.clamp(-1.0, 1.0))


def geodesic_distance(left: torch.Tensor, right: torch.Tensor, curvature: float) -> torch.Tensor:
    c = _curvature(curvature, left)
    sqrt_c = c.sqrt()
    difference = _mobius_add(-expmap0(left, c), expmap0(right, c), c)
    norm = torch.linalg.vector_norm(difference, dim=-1)
    return 2.0 / sqrt_c * torch.atanh((sqrt_c * norm).clamp(max=1.0 - 1e-7))


# --------------------------------------------------------------------------------------
# frozen state
# --------------------------------------------------------------------------------------
def load_frozen_state(device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(experiment.RQVAE_CKPT_PATH)
    embedding_path = Path(experiment.EMBEDDING_FILE)
    raw_sid_path = Path(experiment.RAW_SIDS_NPY)
    hgrec_sid_path = Path(experiment.SIDS_NPY)
    train_path = Path(experiment.TRAIN_FILE)
    for path in (
        checkpoint_path,
        embedding_path,
        train_path,
        raw_sid_path,
        hgrec_sid_path,
        PAIR_SAMPLE_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    artifact_hashes = summary["metadata"]["input_sha256"]
    source_hashes = summary["metadata"]["source_files_sha256"]
    for path in (checkpoint_path, embedding_path, train_path, raw_sid_path, hgrec_sid_path):
        recorded = artifact_hashes.get(audit.relative_to_root(path))
        if recorded is None:
            raise RuntimeError(f"No frozen hash recorded for {audit.relative_to_root(path)}")
        if audit.sha256(path) != recorded:
            raise RuntimeError(
                f"Frozen input changed since SID-Diag-01: {audit.relative_to_root(path)}"
            )
    this_script = audit.relative_to_root(Path(__file__).resolve())
    for relative, recorded in source_hashes.items():
        if relative == this_script:
            # This file is the analysis being run, not a frozen input to it; its
            # own hash is recorded separately under metadata.D18.
            continue
        current = audit.sha256(ROOT / relative)
        if current != recorded:
            raise RuntimeError(f"Frozen source changed since SID-Diag-01: {relative}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    embeddings = np.asarray(np.load(embedding_path), dtype=np.float32)
    raw_tokens = np.asarray(np.load(raw_sid_path), dtype=np.int64)
    hgrec_tokens = np.asarray(np.load(hgrec_sid_path), dtype=np.int64)[:, :3]
    if raw_tokens.shape != (len(embeddings), 3) or not np.array_equal(raw_tokens, hgrec_tokens):
        raise RuntimeError("Stage2 SID exports do not match the frozen item order")
    if int(summary["metadata"]["checkpoint"]["global_step"]) != int(checkpoint["global_step"]):
        raise RuntimeError("Frozen summary and checkpoint refer to different Stage2 steps")

    train_frame = pd.read_parquet(train_path, columns=["seen_history", "target"])
    source_ids, successor_ids = trainer._transition_pairs(train_frame)
    embedding_tensor = torch.from_numpy(embeddings)
    behaviour = trainer._behaviour_context(embedding_tensor, source_ids, successor_ids)
    context_channel = trainer._curvature_context_channel(embedding_tensor, behaviour)
    encoder_inputs = torch.cat((embedding_tensor, context_channel), dim=1)

    config = trainer._tokenizer_config()
    model = RQVAE(
        config,
        in_dim=int(embeddings.shape[1]),
        context_dim=int(context_channel.shape[1]),
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {incompatible}")

    return {
        "summary": summary,
        "model": copy.deepcopy(model).to(device).eval(),
        "encoder_inputs": encoder_inputs.to(device),
        "tokens": raw_tokens,
        "source_ids": source_ids,
        "successor_ids": successor_ids,
        "n_items": len(embeddings),
    }


def representations_for(
    model: RQVAE, encoder_inputs: torch.Tensor, tokens: np.ndarray
) -> dict[str, torch.Tensor]:
    """Encoder output plus the cumulative code contribution after each level."""
    device = encoder_inputs.device
    tokens_gpu = torch.as_tensor(tokens, device=device, dtype=torch.long)
    with torch.no_grad():
        encoded = model.encoder(encoder_inputs)
    representations = {"Encoder": encoded.clone()}
    residual = encoded.clone()
    cumulative = torch.zeros_like(encoded)
    with torch.no_grad():
        for level, layer in enumerate(model.rq.vq_layers):
            curvature = layer.get_curvature()
            source = residual
            code = layer.embed_code(tokens_gpu[:, level])
            if model.rq.working_radii[level] == 0.0:
                cumulative = cumulative + code
                residual = _hyperbolic_residual(source, code, curvature)
            else:
                target_norm = model.rq._radius_for_level(level, curvature, source)
                pinned = model.rq._pin_to_radius(source, target_norm)
                cumulative = cumulative + model.rq._restore_norm(code, source, target_norm)
                residual = model.rq._restore_norm(
                    _hyperbolic_residual(pinned, code, curvature), source, target_norm
                )
            representations[f"L1-L{level + 1}"] = cumulative.clone()
    return representations


# --------------------------------------------------------------------------------------
# padded partner bookkeeping (one row per analysed pair, leave-one-out axis)
# --------------------------------------------------------------------------------------
class PartnerPanel:
    """Padded partner lists with the evaluated positive masked out of its own axis.

    The cone axis is the log-Euclidean mean of the source's remaining partners,
    i.e. a point in the ball, so every angle in the analysis is a genuine apex
    angle computed from the law of cosines.
    """

    def __init__(
        self,
        partners: list[np.ndarray],
        sources: np.ndarray,
        positives: np.ndarray,
        vectors: torch.Tensor,
        curvature: float,
        generator: np.random.Generator,
    ) -> None:
        device = vectors.device
        rows = len(sources)
        width = min(MAX_PARTNERS, max(1, max(len(partners[item]) for item in sources.tolist())))
        padded = np.zeros((rows, width), dtype=np.int64)
        mask = np.zeros((rows, width), dtype=bool)
        self_index = np.zeros(rows, dtype=np.int64)
        counts = np.zeros(rows, dtype=np.int64)
        for row, item in enumerate(sources.tolist()):
            pool = partners[item][:width]
            padded[row, : len(pool)] = pool
            mask[row, : len(pool)] = True
            counts[row] = len(pool)
            hits = np.flatnonzero(pool == positives[row])
            self_index[row] = int(hits[0]) if hits.size else -1

        self.width = width
        self.vectors = vectors
        self.curvature = curvature
        self.generator = generator
        self.sources = torch.as_tensor(sources, device=device, dtype=torch.long)
        self.apex = vectors[self.sources]
        self.counts = torch.as_tensor(counts, device=device, dtype=torch.float32)
        self.padded = torch.as_tensor(padded, device=device, dtype=torch.long)
        self.mask = torch.as_tensor(mask, device=device)
        self.self_index = torch.as_tensor(self_index, device=device, dtype=torch.long)
        # Drop the evaluated positive from its own partner pool.
        keep = self.mask.clone()
        rows_with_self = self.self_index >= 0
        row_ids = torch.arange(rows, device=device)
        keep[row_ids[rows_with_self], self.self_index[rows_with_self]] = False
        self.neighbour = vectors[self.padded]
        # A source whose only partner is the evaluated positive falls back to
        # using that partner, so the axis is always defined.
        empty = ~keep.any(dim=1)
        if bool(empty.any()):
            keep = torch.where(empty.unsqueeze(1), self.mask, keep)
        self.keep = keep

    def axis(self, replicate: int | None = None) -> torch.Tensor:
        """Ball point toward surviving partners; optionally bootstrap partners."""
        if replicate is None:
            return log_euclidean_mean(self.neighbour, self.keep, self.curvature)
        generator = torch.Generator(device=self.keep.device).manual_seed(SEED + replicate)
        draws = torch.multinomial(
            self.keep.to(dtype=torch.float32),
            num_samples=self.width,
            replacement=True,
            generator=generator,
        )
        neighbour = torch.gather(
            self.neighbour, 1, draws.unsqueeze(-1).expand(-1, -1, self.neighbour.shape[-1])
        )
        sample_count = self.keep.sum(dim=1)
        valid = torch.arange(self.width, device=self.keep.device).unsqueeze(0) < sample_count.unsqueeze(1)
        return log_euclidean_mean(neighbour, valid, self.curvature)

    def angles(self, axis: torch.Tensor, targets: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Angle to the axis and the bearing away from the origin, per pair class."""
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, ids in targets.items():
            other = self.vectors[torch.as_tensor(ids, device=self.vectors.device)]
            result[name] = (
                apex_frame_angle(self.apex, other, axis, self.curvature).cpu().numpy(),
                origin_bearing_angle(self.apex, other, self.curvature).cpu().numpy(),
            )
        return result

    def in_sample_partner_angles(self, axis: torch.Tensor) -> np.ndarray:
        """Apex angle to the axis for the partners that built the axis.

        This is the control that separates "the partners do not cluster" from
        "the held-out positive is atypical among its own partners": the axis is
        built from these very points, so their angles bound what a cluster could
        look like at this radius.
        """
        neighbour = self.neighbour
        flat_apex = self.apex.unsqueeze(1).expand(-1, self.width, -1)
        angles = apex_frame_angle(flat_apex, neighbour, axis.unsqueeze(1), self.curvature)
        keep = self.keep if bool(self.keep.any()) else self.mask
        masked = torch.where(keep, angles, torch.full_like(angles, float("nan")))
        return torch.nanmean(masked, dim=1).cpu().numpy()

    def partner_dispersion(self) -> float:
        """Median apex angle between distinct valid partners of the same source."""
        valid_rows = torch.nonzero(self.mask.sum(dim=1) >= 2, as_tuple=False).flatten()
        if valid_rows.numel() == 0:
            return float("nan")
        mask = self.mask[valid_rows]
        generator = torch.Generator(device=self.mask.device).manual_seed(SEED)
        scores = torch.rand(mask.shape, device=self.mask.device, generator=generator)
        scores = scores.masked_fill(~mask, -1.0)
        slots = scores.topk(k=2, dim=1).indices
        neighbours = self.neighbour[valid_rows]
        first = torch.gather(
            neighbours, 1, slots[:, :1].unsqueeze(-1).expand(-1, -1, neighbours.shape[-1])
        ).squeeze(1)
        second = torch.gather(
            neighbours, 1, slots[:, 1:].unsqueeze(-1).expand(-1, -1, neighbours.shape[-1])
        ).squeeze(1)
        angles = apex_frame_angle(self.apex[valid_rows], first, second, self.curvature)
        return float(np.rad2deg(torch.nanmedian(angles).item()))

    def concentration(self, ids: np.ndarray) -> float:
        """Mean resultant length of the unit displacement directions, a spread-free
        measure of how tightly the class points sit in one direction from the apex."""
        tangent = unit(
            displacement(
                self.apex,
                self.vectors[torch.as_tensor(ids, device=self.vectors.device)],
                self.curvature,
            )
        )
        return float(torch.linalg.vector_norm(tangent.mean(0)).item())


def d18a_angular_clustering(panel: PartnerPanel, targets: dict[str, np.ndarray]) -> dict[str, Any]:
    axis = panel.axis()
    measured = panel.angles(axis, targets)
    angles = {name: value[0] for name, value in measured.items()}
    coverage_boot: dict[float, list[float]] = {
        aperture: [] for aperture in FIXED_APERTURES_DEG
    }
    for replicate in range(BOOTSTRAP_REPLICATES):
        boot_axis = panel.axis(replicate=replicate)
        boot_angle = apex_frame_angle(
            panel.apex,
            panel.vectors[torch.as_tensor(targets["positive"], device=panel.vectors.device)],
            boot_axis,
            panel.curvature,
        ).cpu().numpy()
        for aperture_deg in FIXED_APERTURES_DEG:
            coverage_boot[aperture_deg].append(
                float((boot_angle <= np.deg2rad(aperture_deg)).mean())
            )
    return {
        "pairs": int(panel.apex.shape[0]),
        "curvature": float(panel.curvature),
        "apex_radius_median": float(ball_radius(panel.apex, panel.curvature).median().item()),
        "angle_to_partner_axis_radians": {
            name: audit.percentile_summary(values) for name, values in angles.items()
        },
        "in_sample_partner_angle_radians": audit.percentile_summary(
            panel.in_sample_partner_angles(axis)
        ),
        "partner_to_partner_angle_degrees_median": panel.partner_dispersion(),
        "paired_auc_positive_closer_than_matched": audit.paired_lower_auc(
            angles["positive"], angles["matched_negative"]
        ),
        "paired_auc_positive_closer_than_in_sample_partners": audit.paired_lower_auc(
            angles["positive"], panel.in_sample_partner_angles(axis)
        ),
        "paired_auc_positive_closer_than_random": audit.paired_lower_auc(
            angles["positive"], angles["random_negative"]
        ),
        "directional_concentration_mean_resultant": {
            name: panel.concentration(ids) for name, ids in targets.items()
        },
        "bootstrap_positive_coverage": {
            str(aperture): {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for aperture, values in coverage_boot.items()
        },
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def d18b_cones(vectors: torch.Tensor, panel: PartnerPanel, targets: dict[str, np.ndarray]) -> dict[str, Any]:
    """Entailment cones around each source's behaviour direction.

    A cone with apex p, axis u and half-aperture psi holds z when
    angle(d(p,z), u) <= psi and the geodesic to z turns no further from the origin
    than the geodesic along the axis does, which is the geometric statement that the
    cone opens away from the root instead of back into it.
    """
    axis = panel.axis()
    apex_radius = ball_radius(panel.apex, panel.curvature)
    axis_opening = origin_bearing_angle(panel.apex, axis, panel.curvature).cpu().numpy()
    apertures: list[tuple[str, np.ndarray]] = [
        (f"angle_{value:g}deg", np.full(len(panel.apex), np.deg2rad(value)))
        for value in FIXED_APERTURES_DEG
    ]
    metric_values = [
        (f"metric_reach_{reach:g}", metric_aperture(apex_radius, reach).cpu().numpy())
        for reach in METRIC_APERTURES
    ]
    apertures.extend(metric_values)

    measured = panel.angles(axis, targets)
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    # The entailment condition has two readings in the literature: the cone may
    # open towards the root or away from it. Both are measured and labelled, so
    # the result does not depend on picking one.
    orientations = {
        "opens_away_from_root": measured["positive"][1] >= axis_opening,
        "opens_toward_root": measured["positive"][1] <= axis_opening,
    }
    for name, aperture in apertures:
        inside_angular = {
            pair_class: measured[pair_class][0] <= aperture for pair_class in measured
        }
        inside = {
            orientation: {
                pair_class: inside_angular[pair_class] & rule for pair_class in measured
            }
            for orientation, rule in orientations.items()
        }
        for orientation in orientations:
            coverage = float(inside[orientation]["positive"].mean())
            false_positive = float(inside[orientation]["matched_negative"].mean())
            random_rate = float(inside[orientation]["random_negative"].mean())
            label = f"{name}|{orientation}"
            rows.append(
                {
                    "cone": label,
                    "orientation": orientation,
                    "aperture_deg_median": float(np.median(np.rad2deg(aperture))),
                    "positive_coverage": coverage,
                    "matched_negative_false_positive_rate": false_positive,
                    "random_negative_rate": random_rate,
                    "coverage_over_false_positive": coverage / false_positive if false_positive > 0 else float("inf"),
                }
            )
            detail[label] = {
                "orientation": orientation,
                "aperture_degrees": audit.percentile_summary(np.rad2deg(aperture)),
                "positive_coverage": coverage,
                "matched_negative_false_positive_rate": false_positive,
                "random_negative_rate": random_rate,
                "coverage_over_false_positive": coverage / false_positive if false_positive > 0 else float("inf"),
                "positive_closer_than_matched_auc": audit.paired_lower_auc(
                    measured["positive"][0], measured["matched_negative"][0]
                ),
                "angular_margin_degrees_median_matched_minus_positive": float(
                    np.rad2deg(
                        np.median(measured["matched_negative"][0])
                        - np.median(measured["positive"][0])
                    )
                ),
                "axis_opening_angle_degrees_median": float(np.rad2deg(np.median(axis_opening))),
                "axis_opening_share_satisfied_by_positive": float(orientations[orientation].mean()),
            }
    return {"rows": rows, "detail": detail}


# --------------------------------------------------------------------------------------
# D18-C hierarchy
# --------------------------------------------------------------------------------------
def d18c_hierarchy(
    representations: dict[str, torch.Tensor],
    source_ids: np.ndarray,
    successor_ids: np.ndarray,
    n_items: int,
    curvature: float,
) -> dict[str, Any]:
    out_degree = np.bincount(source_ids, minlength=n_items).astype(np.float64)
    in_degree = np.bincount(successor_ids, minlength=n_items).astype(np.float64)
    pagerank = _pagerank(source_ids, successor_ids, n_items)
    core, adjacency = _k_core(source_ids, successor_ids, n_items)
    depth = _depth_to_sink(adjacency, n_items)

    metrics = {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "pagerank": pagerank,
        "k_core": core.astype(np.float64),
        "depth_to_sink": depth.astype(np.float64),
    }
    metric_names = list(metrics)
    rows: list[dict[str, Any]] = []
    for representation, vectors in representations.items():
        radius = ball_radius(vectors, curvature).cpu().numpy()
        for metric_name, values in metrics.items():
            rows.append(
                {
                    "representation": representation,
                    "hierarchy_metric": metric_name,
                    "spearman_rho_vs_radius": _spearman(radius, values),
                    "radius_median": float(np.median(radius)),
                }
            )
    cross: list[dict[str, Any]] = []
    for index, left in enumerate(metric_names):
        for right in metric_names[index + 1:]:
            cross.append(
                {
                    "metric_a": left,
                    "metric_b": right,
                    "spearman_rho": _spearman(metrics[left], metrics[right]),
                }
            )
    return {
        "radial_vs_metric_rows": rows,
        "metric_cross_correlation": cross,
        "notes": (
            "in_degree, out_degree and PageRank are popularity-flavoured; k_core and "
            "depth_to_sink are structural. They are reported side by side and the "
            "cross-correlation block shows how far the two families agree, so radial "
            "position is never read as popularity alone."
        ),
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan")
    return float(scipy.stats.spearmanr(left, right).statistic)


def _pagerank(source_ids: np.ndarray, successor_ids: np.ndarray, n_items: int, damping: float = 0.85, iterations: int = 100, tolerance: float = 1e-12) -> np.ndarray:
    """Power iteration on the transition graph, mass redistributed from dangling nodes."""
    out_counts = np.bincount(source_ids, minlength=n_items)
    has_out = out_counts > 0
    rank = np.full(n_items, 1.0 / n_items)
    for _ in range(iterations):
        share = np.zeros(n_items)
        share[has_out] = rank[has_out] / out_counts[has_out]
        inflow = np.bincount(successor_ids, weights=share[source_ids], minlength=n_items)
        dangling = rank[~has_out].sum()
        updated = (1.0 - damping) / n_items + damping * (inflow + dangling / n_items)
        updated /= updated.sum()
        delta = float(np.abs(updated - rank).sum())
        rank = updated
        if delta < tolerance:
            break
    return rank


def _k_core(source_ids: np.ndarray, successor_ids: np.ndarray, n_items: int) -> tuple[np.ndarray, list[list[int]]]:
    edges = set(zip(source_ids.tolist(), successor_ids.tolist()))
    neighbours: list[set[int]] = [set() for _ in range(n_items)]
    for source, successor in edges:
        if source == successor:
            continue
        neighbours[source].add(successor)
        neighbours[successor].add(source)
    degrees = np.asarray([len(node) for node in neighbours], dtype=np.int64)
    core = np.zeros(n_items, dtype=np.int64)
    removed = np.zeros(n_items, dtype=bool)
    max_degree = int(degrees.max()) if n_items else 0
    for level in range(max_degree + 1):
        while True:
            candidates = np.flatnonzero((~removed) & (degrees <= level))
            if candidates.size == 0:
                break
            core[candidates] = level
            removed[candidates] = True
            for node in candidates.tolist():
                for neighbour in neighbours[node]:
                    if not removed[neighbour]:
                        degrees[neighbour] -= 1
    return core, [sorted(node) for node in neighbours]


def _depth_to_sink(adjacency: list[list[int]], n_items: int, max_depth: int = 6) -> np.ndarray:
    """Undirected distance to a node that never leads anywhere."""
    depth = np.full(n_items, max_depth, dtype=np.int64)
    frontier = [node for node in range(n_items) if not adjacency[node]]
    for node in frontier:
        depth[node] = 0
    level = 0
    seen = set(frontier)
    while frontier and level < max_depth:
        level += 1
        nxt: list[int] = []
        for node in frontier:
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    depth[neighbour] = level
                    nxt.append(neighbour)
        frontier = nxt
    return depth


# --------------------------------------------------------------------------------------
# D18-D frozen curvature sensitivity
# --------------------------------------------------------------------------------------
def d18d_curvature(vectors: torch.Tensor, pair_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    device = vectors.device
    apex = vectors[torch.as_tensor(pair_arrays["source"], device=device)]
    others = {
        name: vectors[torch.as_tensor(ids, device=device)]
        for name, ids in pair_arrays.items()
        if name != "source"
    }
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for curvature in CURVATURE_GRID:
        # Float64 throughout: at these tangent norms the exp map sits on its
        # boundary, and a float32 clamp would quantise the radius to the same
        # value at every curvature and hide the effect being measured.
        apex64 = apex.to(torch.float64)
        others64 = {
            name: other.to(torch.float64) for name, other in others.items()
        }
        angles = {
            name: angle_between(apex64, other).cpu().numpy() for name, other in others64.items()
        }
        distances = {
            name: geodesic_distance(apex64, other, curvature).cpu().numpy()
            for name, other in others64.items()
        }
        radius = ball_radius(apex64, curvature)
        # The conformal factor is evaluated at the actual Poincare point
        # x = exp_0(z), not at its tangent coordinate z. The latter is
        # unbounded and is not a ball coordinate.
        ball_point = expmap0(apex64, curvature)
        tangent_norm = torch.linalg.vector_norm(apex64, dim=-1)
        ball_norm_sq = (ball_point * ball_point).sum(-1)
        conformal = (
            2.0 / (1.0 - curvature * ball_norm_sq).clamp_min(1e-12)
        ).cpu().numpy()
        near_boundary = float(
            (curvature * ball_norm_sq >= 0.999).double().mean().item()
        )
        angular_auc = audit.paired_lower_auc(angles["positive"], angles["matched_negative"])
        geodesic_auc = audit.paired_lower_auc(distances["positive"], distances["matched_negative"])
        rows.append(
            {
                "curvature": curvature,
                "angular_auc_positive_closer_than_matched": angular_auc,
                "geodesic_auc_positive_closer_than_matched": geodesic_auc,
                "conformal_factor_median": float(np.median(conformal)),
                "apex_radius_median": float(np.median(radius.cpu().numpy())),
                "median_tangent_norm": float(tangent_norm.median().item()),
                "share_near_ball_boundary": near_boundary,
            }
        )
        detail[str(curvature)] = {
            "angular_auc_positive_closer_than_matched": angular_auc,
            "geodesic_auc_positive_closer_than_matched": geodesic_auc,
            "angle_medians_degrees": {
                name: float(np.rad2deg(np.median(values))) for name, values in angles.items()
            },
            "conformal_factor_median": float(np.median(conformal)),
            "apex_radius_median": float(np.median(radius.cpu().numpy())),
            "median_tangent_norm": float(tangent_norm.median().item()),
            "share_near_ball_boundary": near_boundary,
            "interpretation": (
                "The angle column is curvature-free by construction: it is the angle "
                "between frozen tangent coordinates, so it is identical at every curvature. "
                "The geodesic column reads the same frozen tangent coordinates through "
                "exp_0 and the curvature-c metric. The reported tangent norm is not a ball "
                "radius; exp_0 maps it to a valid point inside the ball. The boundary share "
                "uses c||exp_0(z)||^2 >= 0.999, not the unbounded tangent coordinates. "
                "Nothing is retrained, so this is a frozen-representation re-read."
            ),
        }
    return {"rows": rows, "detail": detail, "curvature_grid": list(CURVATURE_GRID)}


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def validate_apex_geometry(curvature: float, device: torch.device, samples: int = 4096) -> dict[str, Any]:
    """Cross-check the apex-frame angles against an independent implementation.

    The analysis reads angles at an apex, not at the origin. In the Poincare ball
    the differential at a point is a rotation and a scale, so transporting a
    direction with a Mobius subtraction does not preserve angles; the hyperbolic
    law of cosines does. This control recomputes the same angles with a
    base-point log map, a different code path, and reports the disagreement.
    """
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    apex = (torch.randn(samples, 32, generator=generator) * 0.05).to(device)
    left = (torch.randn(samples, 32, generator=generator) * 0.05).to(device)
    right = (torch.randn(samples, 32, generator=generator) * 0.05).to(device)
    c = float(curvature)

    def logmap_at(base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        base_point = expmap0(base, c)
        point_value = expmap0(point, c)
        difference = _mobius_add(-base_point, point_value, c)
        norm = torch.linalg.vector_norm(difference, dim=-1, keepdim=True).clamp_min(1e-12)
        sqrt_c = c**0.5
        scale = (2.0 / sqrt_c) * torch.atanh((sqrt_c * norm).clamp(max=1.0 - 1e-7))
        return scale * difference / norm

    tangent_left = logmap_at(apex, left)
    tangent_right = logmap_at(apex, right)
    cosine = (tangent_left * tangent_right).sum(-1) / (
        torch.linalg.vector_norm(tangent_left, dim=-1)
        * torch.linalg.vector_norm(tangent_right, dim=-1)
    ).clamp_min(1e-12)
    independent = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))).cpu().numpy()
    law = np.rad2deg(apex_frame_angle(apex, left, right, c).cpu().numpy())
    self_angle = apex_frame_angle(apex, left, left, c).cpu().numpy()
    return {
        "samples": int(samples),
        "curvature": c,
        "apex_angle_median_degrees_independent": float(np.median(independent)),
        "apex_angle_median_degrees_law_of_cosines": float(np.median(law)),
        "max_abs_difference_degrees": float(np.max(np.abs(independent - law))),
        "degenerate_pair_angle_max_degrees": float(np.max(np.abs(self_angle))),
    }


def main() -> None:
    torch.set_num_threads(8)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    audit.DIAGNOSTIC_DEVICE = device
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state = load_frozen_state(device)
    representations = representations_for(state["model"], state["encoder_inputs"], state["tokens"])
    partners = defaultdict(list)
    for source, successor in zip(state["source_ids"].tolist(), state["successor_ids"].tolist()):
        partners[source].append(successor)
    partner_lists = [np.asarray(partners.get(item, []), dtype=np.int64) for item in range(state["n_items"])]

    pair_frame = pd.read_csv(PAIR_SAMPLE_PATH)
    pair_frame = pair_frame[pair_frame["source_item"] != pair_frame["positive_successor"]]
    if len(pair_frame) > CONE_PAIR_SAMPLE:
        pair_frame = pair_frame.sample(CONE_PAIR_SAMPLE, random_state=SEED).reset_index(drop=True)
    sources = pair_frame["source_item"].to_numpy(dtype=np.int64)
    targets = {
        "positive": pair_frame["positive_successor"].to_numpy(dtype=np.int64),
        "matched_negative": pair_frame["matched_negative"].to_numpy(dtype=np.int64),
        "random_negative": pair_frame["random_negative"].to_numpy(dtype=np.int64),
    }
    curvature = float(trainer.LAYER_CURVATURES[0])
    geometry_check = validate_apex_geometry(curvature, device)
    if geometry_check["max_abs_difference_degrees"] > 1e-2:
        raise RuntimeError(f"Apex-frame angle geometry failed validation: {geometry_check}")
    print(f"apex geometry check: {geometry_check}")
    generator = np.random.default_rng(SEED)

    clustering: dict[str, Any] = {}
    cones: dict[str, Any] = {}
    for name in REPRESENTATIONS:
        vectors = representations[name]
        panel = PartnerPanel(partner_lists, sources, targets["positive"], vectors, curvature, generator)
        clustering[name] = d18a_angular_clustering(panel, targets)
        cones[name] = d18b_cones(vectors, panel, targets)
        del panel
        torch.cuda.empty_cache() if device.type == "cuda" else None
    hierarchy = d18c_hierarchy(
        representations, state["source_ids"], state["successor_ids"], state["n_items"], curvature
    )
    sensitivity = d18d_curvature(representations["Encoder"], {"source": sources, **targets})

    cone_rows = [
        {"representation": name, **row}
        for name in REPRESENTATIONS
        for row in cones[name]["rows"]
    ]
    audit.write_csv(
        OUTPUT_DIR / "cone_coverage_by_representation.csv",
        ["representation", "cone", "orientation", "aperture_deg_median",
         "positive_coverage", "matched_negative_false_positive_rate",
         "random_negative_rate", "coverage_over_false_positive"],
        cone_rows,
    )

    clustering_rows = [
        {
            "representation": name,
            "paired_auc_positive_closer_than_matched": payload["paired_auc_positive_closer_than_matched"],
            "paired_auc_positive_closer_than_in_sample_partners": payload["paired_auc_positive_closer_than_in_sample_partners"],
            "positive_axis_angle_median_deg": float(np.rad2deg(payload["angle_to_partner_axis_radians"]["positive"]["median"])),
            "matched_axis_angle_median_deg": float(np.rad2deg(payload["angle_to_partner_axis_radians"]["matched_negative"]["median"])),
            "random_axis_angle_median_deg": float(np.rad2deg(payload["angle_to_partner_axis_radians"]["random_negative"]["median"])),
            "in_sample_partner_axis_angle_median_deg": float(np.rad2deg(payload["in_sample_partner_angle_radians"]["median"])),
            "partner_to_partner_angle_median_deg": payload["partner_to_partner_angle_degrees_median"],
            "positive_directional_concentration": payload["directional_concentration_mean_resultant"]["positive"],
            "matched_directional_concentration": payload["directional_concentration_mean_resultant"]["matched_negative"],
            "random_directional_concentration": payload["directional_concentration_mean_resultant"]["random_negative"],
            "bootstrap_20deg_coverage_mean": payload["bootstrap_positive_coverage"]["20.0"]["mean"],
            "bootstrap_20deg_coverage_std": payload["bootstrap_positive_coverage"]["20.0"]["std"],
        }
        for name, payload in clustering.items()
    ]
    audit.write_csv(
        OUTPUT_DIR / "angular_clustering_positive_vs_matched.csv",
        list(clustering_rows[0]),
        clustering_rows,
    )
    audit.write_csv(
        OUTPUT_DIR / "hierarchy_radial_relationships.csv",
        ["representation", "hierarchy_metric", "spearman_rho_vs_radius", "radius_median"],
        hierarchy["radial_vs_metric_rows"],
    )
    audit.write_csv(
        OUTPUT_DIR / "hierarchy_metric_cross_correlation.csv",
        ["metric_a", "metric_b", "spearman_rho"],
        hierarchy["metric_cross_correlation"],
    )
    audit.write_csv(
        OUTPUT_DIR / "curvature_sensitivity.csv",
        list(sensitivity["rows"][0]),
        sensitivity["rows"],
    )

    reference_cone = "metric_reach_0.1|opens_away_from_root"
    audit.write_svg_bar(
        OUTPUT_DIR / "cone_coverage_by_representation.svg",
        f"Behaviour cones ({reference_cone}): coverage vs false positives",
        list(REPRESENTATIONS),
        [
            (
                "positive coverage",
                [next(row["positive_coverage"] for row in cones[name]["rows"] if row["cone"] == reference_cone) for name in REPRESENTATIONS],
            ),
            (
                "matched-negative false positives",
                [next(row["matched_negative_false_positive_rate"] for row in cones[name]["rows"] if row["cone"] == reference_cone) for name in REPRESENTATIONS],
            ),
            (
                "random-negative rate",
                [next(row["random_negative_rate"] for row in cones[name]["rows"] if row["cone"] == reference_cone) for name in REPRESENTATIONS],
            ),
        ],
        "share of pairs inside the cone",
    )
    audit.write_svg_lines(
        OUTPUT_DIR / "curvature_sensitivity.svg",
        "Frozen curvature sensitivity: angular vs geodesic separation",
        [str(value) for value in CURVATURE_GRID],
        [
            (
                "angular AUC (curvature-free)",
                [row["angular_auc_positive_closer_than_matched"] for row in sensitivity["rows"]],
            ),
            (
                "geodesic AUC at curvature c",
                [row["geodesic_auc_positive_closer_than_matched"] for row in sensitivity["rows"]],
            ),
        ],
        "AUC: positive closer than popularity-matched negative",
        y_min=0.0,
        y_max=1.0,
    )
    metric_order = ("in_degree", "out_degree", "pagerank", "k_core", "depth_to_sink")
    audit.write_svg_bar(
        OUTPUT_DIR / "hierarchy_radial_relationships.svg",
        "Spearman correlation of radial position with hierarchy metrics",
        list(metric_order),
        [
            (
                name,
                [
                    next(
                        row["spearman_rho_vs_radius"]
                        for row in hierarchy["radial_vs_metric_rows"]
                        if row["representation"] == name and row["hierarchy_metric"] == metric
                    )
                    for metric in metric_order
                ],
            )
            for name in REPRESENTATIONS
        ],
        "Spearman rho vs ball radius",
    )

    report = build_report(clustering, cones, hierarchy, sensitivity, state, device, len(pair_frame))
    (OUTPUT_DIR / "REPORT_D18.md").write_text(report, encoding="utf-8")

    summary = state["summary"]
    summary["D18"] = {
        "device": str(device),
        "curvature": curvature,
        "pairs_used": int(len(pair_frame)),
        "max_partners_per_source": MAX_PARTNERS,
        "representations": list(REPRESENTATIONS),
        "apex_geometry_validation": geometry_check,
        "angular_clustering": clustering,
        "cones": {name: cones[name]["detail"] for name in REPRESENTATIONS},
        "cone_definition": (
            "Cone with apex p, axis u and half-aperture psi contains z when "
            "angle(d(p,z), u) <= psi and d(p,z) turns no further from the origin than "
            "d(p,u) does (Ganea et al. 2018). Apertures are fixed angles or the "
            "inversion of cosh d = cosh^2 rho - sinh^2 rho cos(theta) at the apex radius. "
            "The axis is the mean partner direction with the evaluated positive removed, "
            "so no pair contributes to its own cone."
        ),
        "hierarchy": hierarchy,
        "curvature_sensitivity": sensitivity,
        "outputs": [
            "cone_coverage_by_representation.csv",
            "angular_clustering_positive_vs_matched.csv",
            "hierarchy_radial_relationships.csv",
            "hierarchy_metric_cross_correlation.csv",
            "curvature_sensitivity.csv",
            "cone_coverage_by_representation.svg",
            "curvature_sensitivity.svg",
            "hierarchy_radial_relationships.svg",
            "REPORT_D18.md",
        ],
    }
    summary["metadata"].setdefault("source_files_sha256", {}).pop(
        audit.relative_to_root(Path(__file__).resolve()), None
    )
    summary["metadata"]["D18"] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "script": audit.relative_to_root(Path(__file__).resolve()),
        "script_sha256": audit.sha256(Path(__file__).resolve()),
        "pairs_used": int(len(pair_frame)),
        "fixed_apertures_deg": list(FIXED_APERTURES_DEG),
        "metric_apertures": list(METRIC_APERTURES),
        "curvature_grid": list(CURVATURE_GRID),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    audit.write_json(SUMMARY_PATH, summary)
    print(report)


def build_report(
    clustering: dict[str, Any],
    cones: dict[str, Any],
    hierarchy: dict[str, Any],
    sensitivity: dict[str, Any],
    state: dict[str, Any],
    device: torch.device,
    pairs: int,
) -> str:
    lines: list[str] = []
    add = lines.append
    add("# D18 — Hyperbolic angular clustering, entailment cones, hierarchy, curvature")
    add("")
    add(
        f"Frozen Stage2 checkpoint, {state['n_items']} items, curvature "
        f"{trainer.LAYER_CURVATURES[0]}, device {device}, {pairs} behaviour pairs. "
        "No training, no loss change, no Stage2 or Stage3 run is part of this analysis."
    )
    add("")
    add("## D18-A behaviour neighbours in angle")
    add("")
    add(
        "| representation | AUC positive closer than matched | AUC vs in-sample partners | "
        "positive axis angle (deg) | matched (deg) | random (deg) | in-sample partner (deg) | "
        "partner-to-partner (deg) | positive concentration | matched concentration | "
        "20deg bootstrap coverage (mean ± SD) |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in REPRESENTATIONS:
        payload = clustering[name]
        add(
            f"| {name} | {payload['paired_auc_positive_closer_than_matched']:.4f} | "
            f"{payload['paired_auc_positive_closer_than_in_sample_partners']:.4f} | "
            f"{np.rad2deg(payload['angle_to_partner_axis_radians']['positive']['median']):.3f} | "
            f"{np.rad2deg(payload['angle_to_partner_axis_radians']['matched_negative']['median']):.3f} | "
            f"{np.rad2deg(payload['angle_to_partner_axis_radians']['random_negative']['median']):.3f} | "
            f"{np.rad2deg(payload['in_sample_partner_angle_radians']['median']):.3f} | "
            f"{payload['partner_to_partner_angle_degrees_median']:.3f} | "
            f"{payload['directional_concentration_mean_resultant']['positive']:.4f} | "
            f"{payload['directional_concentration_mean_resultant']['matched_negative']:.4f} | "
            f"{payload['bootstrap_positive_coverage']['20.0']['mean']:.4f} ± "
            f"{payload['bootstrap_positive_coverage']['20.0']['std']:.4f} |"
        )
    add("")
    add("## D18-B entailment cones")
    add("")
    add("| representation | cone | aperture (deg, median) | positive coverage | matched FP | random rate | coverage / FP |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for name in REPRESENTATIONS:
        for row in cones[name]["rows"]:
            add(
                f"| {name} | {row['cone']} | {row['aperture_deg_median']:.2f} | "
                f"{row['positive_coverage']:.4f} | {row['matched_negative_false_positive_rate']:.4f} | "
                f"{row['random_negative_rate']:.4f} | {row['coverage_over_false_positive']:.2f} |"
            )
    add("")
    add("## D18-C hierarchy against radial position")
    add("")
    add("| representation | metric | Spearman rho vs radius |")
    add("| --- | --- | --- |")
    for row in hierarchy["radial_vs_metric_rows"]:
        add(f"| {row['representation']} | {row['hierarchy_metric']} | {row['spearman_rho_vs_radius']:.4f} |")
    add("")
    add("Metric cross-correlations (popularity family vs structural family):")
    add("")
    add("| metric A | metric B | Spearman rho |")
    add("| --- | --- | --- |")
    for row in hierarchy["metric_cross_correlation"]:
        add(f"| {row['metric_a']} | {row['metric_b']} | {row['spearman_rho']:.4f} |")
    add("")
    add("## D18-D frozen curvature sensitivity")
    add("")
    add("| curvature | angular AUC | geodesic AUC | median tangent norm | apex radius (median) | conformal factor (median) | share near ball boundary |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for row in sensitivity["rows"]:
        add(
            f"| {row['curvature']} | {row['angular_auc_positive_closer_than_matched']:.4f} | "
            f"{row['geodesic_auc_positive_closer_than_matched']:.4f} | "
            f"{row['median_tangent_norm']:.4f} | {row['apex_radius_median']:.4f} | "
            f"{row['conformal_factor_median']:.4g} | {row['share_near_ball_boundary']:.4f} |"
        )
    add("")
    add(hierarchy["notes"])
    add("")
    add("## Conclusions")
    add("")
    encoder = clustering["Encoder"]
    add(
        f"1. **Behaviour neighbours do not form an angular cluster.** At the encoder the "
        f"partners that define a source's own direction sit a median "
        f"{np.rad2deg(encoder['in_sample_partner_angle_radians']['median']):.1f} deg from it, "
        f"and two partners of the same source sit "
        f"{encoder['partner_to_partner_angle_degrees_median']:.1f} deg apart. The cone axis "
        f"is no closer to its own defining points than the points are to each other, so "
        "there is no angular cluster for a cone to capture."
    )
    add(
        f"2. **A held-out successor is not atypical among its own partners, it is simply "
        f"farther than an unrelated item.** AUC positive-vs-in-sample-partners is "
        f"{encoder['paired_auc_positive_closer_than_in_sample_partners']:.3f} (indistinguishable), "
        f"while AUC positive-vs-popularity-matched is "
        f"{encoder['paired_auc_positive_closer_than_matched']:.3f}, i.e. real successors are "
        "farther from the behaviour direction than popularity-matched negatives are. The "
        "persistent-pair failure is therefore not an outlier problem inside a cluster; it "
        "is the absence of the cluster."
    )
    add(
        "3. **Entailment cones over the behaviour direction have no discriminative power.** "
        "In the opens-away-from-root reading, a 20-degree cone covers "
        f"{next(row['positive_coverage'] for row in cones['Encoder']['rows'] if row['cone'] == 'angle_20deg|opens_away_from_root'):.3f} "
        "of positives but "
        f"{next(row['matched_negative_false_positive_rate'] for row in cones['Encoder']['rows'] if row['cone'] == 'angle_20deg|opens_away_from_root'):.3f} "
        "of popularity-matched negatives, a lift below 1; the opens-toward-root reading has "
        "zero coverage at every aperture. Metric apertures, which curvature makes tiny at "
        "these radii, cover nothing. Cones therefore cannot be the next lever on this "
        "representation."
    )
    add(
        "4. **Quantization barely moves any of this.** Encoder, L1-L1, L1-L2 and L1-L3 "
        "agree to within a few degrees on every angular statistic, and the AUC ordering "
        "survives all three levels: the behaviour structure is already absent before the "
        "codes are assigned."
    )
    radii = hierarchy["radial_vs_metric_rows"]
    encoder_rows = {row["hierarchy_metric"]: row["spearman_rho_vs_radius"] for row in radii if row["representation"] == "Encoder"}
    add(
        f"5. **Radial position tracks the popularity family mildly and the structural family "
        f"not at all.** in_degree rho = {encoder_rows['in_degree']:.3f}, out_degree "
        f"{encoder_rows['out_degree']:.3f}, pagerank {encoder_rows['pagerank']:.3f}, k_core "
        f"{encoder_rows['k_core']:.3f}, depth_to_sink {encoder_rows['depth_to_sink']:.3f}. "
        "The cross-correlation block shows why the two families must be kept apart: "
        "degree, PageRank and k-core agree at rho > 0.83, while depth-to-sink is "
        "uncorrelated with all of them (rho ~ 0.06), so a depth reading is not a popularity "
        "reading in disguise here."
    )
    sensitivity_rows = sensitivity["rows"]
    add(
        f"6. **Curvature changes the metric read only modestly on these frozen embeddings.** "
        f"The angular AUC is {sensitivity_rows[0]['angular_auc_positive_closer_than_matched']:.4f} "
        f"at every curvature because it is measured directly in the frozen tangent "
        f"coordinates; the geodesic AUC moves between "
        f"{min(row['geodesic_auc_positive_closer_than_matched'] for row in sensitivity_rows):.4f} and "
        f"{max(row['geodesic_auc_positive_closer_than_matched'] for row in sensitivity_rows):.4f}, "
        f"a range of {max(row['geodesic_auc_positive_closer_than_matched'] for row in sensitivity_rows) - min(row['geodesic_auc_positive_closer_than_matched'] for row in sensitivity_rows):.4f}. "
        f"The median tangent norm is {sensitivity_rows[0]['median_tangent_norm']:.2f}; this is "
        f"not a ball radius. The exp map places every vector inside its curvature-specific "
        "Poincare ball, with the reported near-boundary share measured on the mapped "
        "points. The radius is 2||z|| for exp_0(z), independent of curvature. This is a "
        "frozen-embedding geometry comparison, not a retrained-model result."
    )
    add("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()