"""Frozen-representation Euclidean and Poincare geometry diagnostics."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from config import ARMS, BOOTSTRAP_REPLICATES, LEVELS, OUT_DIR, PRIMARY_PAIRS, SECONDARY_PAIRS, SEED

CURVATURES = (0.25, 0.5, 1.0, 2.0, 4.0)
TARGET_RADIUS_FRACTION = 0.7
BOUNDARY_FRACTION = 0.99


def auc_from_scores(positive: np.ndarray, negative: np.ndarray) -> float:
    """Matched AUC: mean(pos < its matched negatives), with half credit for ties."""
    p = np.asarray(positive, dtype=np.float64).reshape(-1)
    n = np.asarray(negative, dtype=np.float64)
    if n.ndim == 1:
        if len(n) != len(p):
            raise ValueError("One-dimensional negative scores must align one-to-one with positives")
        n = n[:, None]
    if not len(p) or not n.size or n.shape[0] != len(p):
        raise ValueError("AUC requires nonempty matched positive/negative distances")
    return float(np.mean((p[:, None] < n) + 0.5 * (p[:, None] == n)))


def score_variants(positive: np.ndarray, negative: np.ndarray, rms: float, global_scale: float) -> dict:
    """Produce monotone control scores from raw Euclidean comparison distances."""
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    variants = {
        "raw": (positive, negative),
        "rms_normalized": (positive / rms, negative / rms),
        "matched_euclidean": (positive * global_scale, negative * global_scale),
        "log1p_euclidean": (np.log1p(positive), np.log1p(negative)),
    }
    return {name: {"positive": p, "negative": n, "auc": auc_from_scores(p, n)}
            for name, (p, n) in variants.items()}


def ordering_fractions(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Fractions of pairwise orderings concordant, discordant, or tied."""
    x, y = np.asarray(reference, dtype=np.float64).reshape(-1), np.asarray(candidate, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Orderings must contain the same comparisons")
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    upper = np.triu_indices(len(x), k=1)
    a, b = dx[upper], dy[upper]
    tie = (a == 0) | (b == 0)
    comparable = ~tie
    concordant = np.count_nonzero((a[comparable] * b[comparable]) > 0)
    discordant = np.count_nonzero((a[comparable] * b[comparable]) < 0)
    total = len(a)
    return {"concordant_fraction": float(concordant / total) if total else 0.0,
            "discordant_fraction": float(discordant / total) if total else 0.0,
            "tie_fraction": float(np.count_nonzero(tie) / total) if total else 0.0}
def matched_ordering_fractions(reference_pos, reference_neg, candidate_pos, candidate_neg) -> dict:
    """Compare positive-vs-negative orderings for matched anchor/negative pairs."""
    a = np.asarray(reference_pos)[:, None] - np.asarray(reference_neg).reshape(len(reference_pos), -1)
    b = np.asarray(candidate_pos)[:, None] - np.asarray(candidate_neg).reshape(len(candidate_pos), -1)
    tie = (a == 0) | (b == 0)
    total = a.size
    return {"concordant_fraction": float(np.count_nonzero((a * b > 0) & ~tie) / total),
            "discordant_fraction": float(np.count_nonzero((a * b < 0) & ~tie) / total),
            "tie_fraction": float(np.count_nonzero(tie) / total)}


def _pointwise_poincare(a: np.ndarray, b: np.ndarray, c: float) -> np.ndarray:
    import torch
    hb = _hyperbolic_module()
    with torch.no_grad():
        return hb.poincare_distance(torch.as_tensor(a, dtype=torch.float64),
                                    torch.as_tensor(b, dtype=torch.float64), float(c)).cpu().numpy()


def _hyperbolic_module():
    path = Path(__file__).resolve().parents[2] / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/hyperbolic.py"
    spec = importlib.util.spec_from_file_location("crowding_hyperbolic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ball_map(points: np.ndarray, c: float, tangent_scale: float, already_mapped: bool = False):
    """Map points via the existing expmap; report any raw output outside the ball."""
    import torch
    x = torch.as_tensor(np.asarray(points), dtype=torch.float64)
    if already_mapped:
        mapped = x
        scaled_radius = np.sqrt(c) * torch.linalg.vector_norm(mapped, dim=-1)
    else:
        hb = _hyperbolic_module()
        mapped = hb.encode_ball(x, float(c), float(tangent_scale))
        scaled_radius = np.sqrt(c) * torch.linalg.vector_norm(mapped, dim=-1)
    outside = scaled_radius >= 1.0
    return mapped.cpu().numpy(), {
        "would_leave_ball_count": int(outside.sum().item()),
        "would_leave_ball_fraction": float(outside.double().mean().item()) if outside.numel() else 0.0,
        "max_sqrt_c_norm": float(scaled_radius.max().item()) if scaled_radius.numel() else 0.0,
    }


def pairwise_poincare(anchors: np.ndarray, others: np.ndarray, c: float) -> np.ndarray:
    import torch
    hb = _hyperbolic_module()
    a = torch.as_tensor(np.asarray(anchors), dtype=torch.float64)
    b = torch.as_tensor(np.asarray(others), dtype=torch.float64)
    distances = hb.pair_poincare_distance(a, b, float(c))
    if a.shape == b.shape and torch.equal(a, b):
        distances.fill_diagonal_(0.0)
    return distances.cpu().numpy()


def _load_pairs(path: Path) -> dict:
    data = np.load(path)
    return {key: data[key].astype(np.int64) for key in ("anchors", "positives", "negatives")}


def _pair_distances(points: np.ndarray, pairs: dict, metric: str, c: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    anchors, positives, negatives = pairs["anchors"], pairs["positives"], pairs["negatives"]
    flat_anchor = np.repeat(anchors, negatives.shape[1])
    flat_negative = negatives.reshape(-1)
    if metric == "euclidean":
        pos = np.linalg.norm(points[anchors] - points[positives], axis=-1)
        neg = np.linalg.norm(points[flat_anchor] - points[flat_negative], axis=-1)
    else:
        pos = _pointwise_poincare(points[anchors], points[positives], c)
        chunks = []
        for start in range(0, len(flat_anchor), 4096):
            chunks.append(_pointwise_poincare(points[flat_anchor[start:start + 4096]],
                                              points[flat_negative[start:start + 4096]], c))
        neg = np.concatenate(chunks)
    return pos, neg


def _comparison_metrics(pos: np.ndarray, neg: np.ndarray) -> dict:
    # Negatives are matched to anchors; intrusion means any is closer than its positive.
    neg_by_anchor = neg.reshape(len(pos), -1)
    auc = auc_from_scores(pos, neg_by_anchor)
    return {"auc": auc, "ordering_error_rate": float(1.0 - auc),
            "hard_negative_intrusion_rate": float(np.mean(np.any(neg_by_anchor <= pos[:, None], axis=1))),
            "positive_pairs": int(len(pos)), "negative_pairs": int(len(neg))}


def _bootstrap_auc_interval(pos: np.ndarray, neg: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n, k = len(pos), len(neg) // len(pos)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    neg_matrix = neg.reshape(n, k)
    for b in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, n, size=n)
        values[b] = auc_from_scores(pos[idx], neg_matrix[idx])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _load_codebooks(label: str):
    import torch
    root = Path(__file__).resolve().parents[2]
    model_root = root / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
    sys.path.insert(0, str(model_root))
    from model import RQVAE
    arm = ARMS[label]
    # Use the same model configuration and initializer as residual extraction.
    spec = importlib.util.spec_from_file_location("crowding_trainer", model_root / "train_rqvae.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    model = RQVAE(trainer._tokenizer_config(), in_dim=768)
    trainer.initialize_tiger_weights(model)
    checkpoint = torch.load(arm["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return [layer.get_code_embs().detach().cpu().numpy().astype(np.float64) for layer in model.rq.vq_layers]


def _radial_stats(mapped: np.ndarray, c: float) -> dict:
    r = np.sqrt(c) * np.linalg.norm(mapped, axis=-1)
    q = np.percentile(r, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    spread = float(np.std(r))
    return {"c": c, "sqrt_c_norm_quantiles": {f"p{p:02d}": float(v) for p, v in zip((0, 1, 5, 25, 50, 75, 95, 99, 100), q)},
            "near_boundary_fraction_ge_0.99": float(np.mean(r >= BOUNDARY_FRACTION)),
            "radius_std": spread, "radius_iqr": float(np.percentile(r, 75) - np.percentile(r, 25)),
            "radial_information": "degenerate_constant_radius" if spread < 1e-8 else "variable_radius",
            "saturated": bool(np.any(r >= BOUNDARY_FRACTION))}


def _reassignment_fraction(items: np.ndarray, codebook: np.ndarray, c: float, tangent_scale: float) -> float:
    import torch
    combined = np.concatenate((items, codebook), axis=0)
    combined_rms = float(np.sqrt(np.mean(np.sum(combined ** 2, axis=-1))))
    item_rms = float(np.sqrt(np.mean(np.sum(items ** 2, axis=-1))))
    combined_ball, _ = ball_map(combined, c, tangent_scale * combined_rms / max(item_rms, 1e-12))
    item_ball = combined_ball[:len(items)]
    code_ball = combined_ball[len(items):]
    euclidean_idx = []
    for start in range(0, len(items), 512):
        delta = items[start:start + 512, None, :] - codebook[None, :, :]
        euclidean_idx.append(np.argmin(np.sum(delta * delta, axis=-1), axis=1))
    euclidean_idx = np.concatenate(euclidean_idx)
    hb = _hyperbolic_module()
    x = torch.as_tensor(item_ball, dtype=torch.float64)
    cb = torch.as_tensor(code_ball, dtype=torch.float64)
    chosen = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            dist = hb.poincare_distance(x[start:start + 256, None, :], cb[None, :, :], float(c))
            chosen.append(torch.argmin(dist, dim=-1).cpu().numpy())
    return float(np.mean(euclidean_idx != np.concatenate(chosen)))


def run():
    OUT_DIR.joinpath("geometry").mkdir(parents=True, exist_ok=True)
    output_dir = OUT_DIR / "geometry"
    pairs_by_set = {"primary_valid": _load_pairs(PRIMARY_PAIRS), "secondary_train": _load_pairs(SECONDARY_PAIRS)}
    all_rows, radial_rows, reassignment_rows, verdicts, sensitivity_rows = [], [], [], [], []
    for label in ARMS:
        residual_file = OUT_DIR / f"residuals_{label}.npz"
        if not residual_file.exists():
            raise FileNotFoundError(f"Slice A residual input not yet available: {residual_file}")
        z = np.load(residual_file)
        residuals = np.asarray(z["residuals"], dtype=np.float64)
        codebooks = _load_codebooks(label)
        for level, level_name in enumerate(LEVELS):
            points = residuals[:, level]
            coord_rms = float(np.sqrt(np.mean(points ** 2)))
            point_rms = float(np.sqrt(np.mean(np.sum(points ** 2, axis=-1))))
            median_norm = float(np.median(np.linalg.norm(points, axis=-1)))
            if median_norm <= 0 or point_rms <= 0:
                raise ValueError(f"Degenerate level scale: {label} {level_name}")
            raw_pos, raw_neg = _pair_distances(points, pairs_by_set["primary_valid"], "euclidean")
            raw_concat_median = float(np.median(np.concatenate((raw_pos, raw_neg))))
            # RMS coordinate normalization; global scale uses the exact median of all matched comparison distances.
            rms_pos, rms_neg = raw_pos / coord_rms, raw_neg / coord_rms
            matched_scale = raw_concat_median / float(np.median(np.concatenate((rms_pos, rms_neg))))
            if matched_scale <= 0:
                raise ValueError("Median distance scale must be positive")
            scales = {"raw": (raw_pos, raw_neg),
                      "rms_normalized": (rms_pos, rms_neg),
                      "matched_euclidean": (raw_pos * matched_scale, raw_neg * matched_scale),
                      "log1p_euclidean": (np.log1p(raw_pos), np.log1p(raw_neg))}
            neg_matrix = raw_neg.reshape(len(raw_pos), -1)
            invariant_auc = auc_from_scores(raw_pos, neg_matrix)
            for monotone_name in ("rms_normalized", "matched_euclidean", "log1p_euclidean"):
                monotone_pos, monotone_neg = scales[monotone_name]
                assert auc_from_scores(monotone_pos, monotone_neg.reshape(len(monotone_pos), -1)) == invariant_auc, (
                    f"Monotone AUC invariant failed for {label}/{level_name}/{monotone_name}"
                )
            raw_reference = np.concatenate((raw_pos, raw_neg))
            for name, (pos, neg) in scales.items():
                result = _comparison_metrics(pos, neg)
                result.update({"model": label, "pair_set": "primary_valid", "level": level_name,
                               "variant": name, "spearman_vs_raw_euclidean": float(spearmanr(np.concatenate((pos, neg)), raw_reference).statistic),
                               "discordant_fraction_vs_raw": matched_ordering_fractions(raw_pos, raw_neg, pos, neg)["discordant_fraction"]})
                if name == "raw":
                    result["bootstrap_auc_95_ci_low"], result["bootstrap_auc_95_ci_high"] = _bootstrap_auc_interval(pos, neg, SEED + level)
                else:
                    result["bootstrap_auc_95_ci_low"] = result["bootstrap_auc_95_ci_high"] = None
                result.pop("_raw_auc", None)
                all_rows.append(result)
            for c in CURVATURES:
                normalizer = float(np.arctanh(TARGET_RADIUS_FRACTION) / np.sqrt(c) / median_norm)
                tangent_scale = normalizer * point_rms
                ball, map_report = ball_map(points, c, tangent_scale)
                radial = _radial_stats(ball, c)
                radial.update({"model": label, "level": level_name, "normalizer": normalizer,
                               "tangent_scale": tangent_scale, **map_report})
                radial_rows.append(radial)
                hyp_pos, hyp_neg = _pair_distances(ball, pairs_by_set["primary_valid"], "poincare", c)
                hyp_scores = np.concatenate((hyp_pos, hyp_neg))
                metrics = _comparison_metrics(hyp_pos, hyp_neg)
                fraction = matched_ordering_fractions(raw_pos, raw_neg, hyp_pos, hyp_neg)
                metrics.update({"model": label, "pair_set": "primary_valid", "level": level_name,
                                "variant": f"poincare_c{c:g}",
                                "spearman_vs_raw_euclidean": float(spearmanr(hyp_scores, raw_reference).statistic),
                                "discordant_fraction_vs_raw": fraction["discordant_fraction"],
                                "concordant_fraction_vs_raw": fraction["concordant_fraction"],
                                "tie_fraction_vs_raw": fraction["tie_fraction"],
                                "bootstrap_auc_95_ci_low": None, "bootstrap_auc_95_ci_high": None,
                                "map_would_leave_ball_count": map_report["would_leave_ball_count"]})
                all_rows.append(metrics)
                changed = _reassignment_fraction(points, codebooks[level], c, tangent_scale)
                reassignment_rows.append({"model": label, "level": level_name, "c": c,
                                          "code_reassignment_fraction": changed,
                                          "n_items": int(len(points))})
                raw_row = next(row for row in all_rows if row["model"] == label and row["level"] == level_name and row["variant"] == "raw")
                ci_contains = raw_row["bootstrap_auc_95_ci_low"] <= metrics["auc"] <= raw_row["bootstrap_auc_95_ci_high"]
                negligible = fraction["discordant_fraction"] < 0.01
                saturated = radial["saturated"]
                if ci_contains and negligible:
                    verdict = "NEGATIVE: hyperbolic distance acts as a rescaling here and is not doing additional work"
                elif not ci_contains and not negligible:
                    verdict = ("Poincare DOES MORE THAN RESCALING raw Euclidean: AUC is outside the raw Euclidean 95% bootstrap interval "
                               "and matched positive-negative orderings are discordant; curvature values are ranking-equivalent under "
                               "this radius-matched map. " + ("RADIAL SATURATION FLAG: do not interpret this as a curvature advantage."
                               if saturated else "Radial coordinate remains variable without saturation."))
                elif ci_contains and not negligible:
                    verdict = ("Poincare changes matched positive-negative ordering, but its AUC remains inside the raw Euclidean bootstrap interval; "
                               "it is not a mere rescaling, yet this comparison does not establish an AUC shift. "
                               + ("RADIAL SATURATION FLAG: do not interpret this as a curvature advantage."
                                  if saturated else "Radial coordinate remains variable without saturation."))
                elif not ci_contains and negligible:
                    verdict = "AUC is outside the raw bootstrap interval, but matched order discordance is negligible; negative-result rule is inconclusive."
                verdicts.append({"model": label, "level": level_name, "c": c,
                                 "raw_auc": raw_row["auc"], "poincare_auc": metrics["auc"],
                                 "raw_auc_bootstrap_95_ci": [raw_row["bootstrap_auc_95_ci_low"], raw_row["bootstrap_auc_95_ci_high"]],
                                 "raw_auc_ci_contains_poincare_auc": bool(ci_contains),
                                 "discordant_fraction": fraction["discordant_fraction"],
                                 "discordant_fraction_negligible_lt_0.01": bool(negligible),
                                 "radial_saturation_flag": bool(saturated),
                                 "verdict": verdict})
            base_normalizer = float(np.arctanh(TARGET_RADIUS_FRACTION) / median_norm)
            matched_tangent_scale = base_normalizer * point_rms
            sensitivity_scales = (
                ("half_target_radius_scale", 0.5 * matched_tangent_scale),
                ("target_radius_scale", matched_tangent_scale),
                ("double_target_radius_scale", 2.0 * matched_tangent_scale),
                ("normalizer_passed_directly", base_normalizer),
            )
            for scale_name, tested_scale in sensitivity_scales:
                sensitivity_ball, sensitivity_map = ball_map(points, 1.0, tested_scale)
                sensitivity_pos, sensitivity_neg = _pair_distances(
                    sensitivity_ball, pairs_by_set["primary_valid"], "poincare", 1.0
                )
                sensitivity_metrics = _comparison_metrics(sensitivity_pos, sensitivity_neg)
                sensitivity_rows.append({
                    "model": label, "level": level_name, "curvature": 1.0,
                    "scale_name": scale_name, "tangent_scale_passed": tested_scale,
                    "effective_raw_vector_multiplier": tested_scale / point_rms,
                    "point_rms_used_by_encode_ball": point_rms,
                    "median_raw_norm": median_norm,
                    "auc": sensitivity_metrics["auc"],
                    "ordering_error_rate": sensitivity_metrics["ordering_error_rate"],
                    "discordant_fraction_vs_euclidean": matched_ordering_fractions(
                        raw_pos, raw_neg, sensitivity_pos, sensitivity_neg
                    )["discordant_fraction"],
                    "max_sqrt_c_norm": sensitivity_map["max_sqrt_c_norm"],
                    "near_boundary_fraction_ge_0.99": float(
                        np.mean(np.linalg.norm(sensitivity_ball, axis=-1) >= BOUNDARY_FRACTION)
                    ),
                })
        # Secondary train-derived comparisons receive the same full distance-variant scoring.
        for level, level_name in enumerate(LEVELS):
            points = residuals[:, level]
            pairs = pairs_by_set["secondary_train"]
            raw_pos, raw_neg = _pair_distances(points, pairs, "euclidean")
            raw_median = float(np.median(np.concatenate((raw_pos, raw_neg))))
            coord_rms = float(np.sqrt(np.mean(points ** 2)))
            rms_pos, rms_neg = raw_pos / coord_rms, raw_neg / coord_rms
            matched_scale = raw_median / float(np.median(np.concatenate((rms_pos, rms_neg))))
            euclidean_variants = {
                "raw": (raw_pos, raw_neg),
                "rms_normalized": (rms_pos, rms_neg),
                "matched_euclidean": (raw_pos * matched_scale, raw_neg * matched_scale),
                "log1p_euclidean": (np.log1p(raw_pos), np.log1p(raw_neg)),
            }
            neg_matrix = raw_neg.reshape(len(raw_pos), -1)
            invariant_auc = auc_from_scores(raw_pos, neg_matrix)
            for name, (pos, neg) in euclidean_variants.items():
                if name != "raw":
                    assert auc_from_scores(pos, neg.reshape(len(pos), -1)) == invariant_auc, (
                        f"Monotone AUC invariant failed for {label}/{level_name}/{name}/secondary_train"
                    )
                result = _comparison_metrics(pos, neg)
                result.update({"model": label, "pair_set": "secondary_train", "level": level_name,
                               "variant": name,
                               "spearman_vs_raw_euclidean": float(spearmanr(np.concatenate((pos, neg)),
                                                                              np.concatenate((raw_pos, raw_neg))).statistic),
                               "discordant_fraction_vs_raw": matched_ordering_fractions(raw_pos, raw_neg, pos, neg)["discordant_fraction"],
                               "bootstrap_auc_95_ci_low": None, "bootstrap_auc_95_ci_high": None})
                result.pop("_raw_auc", None)
                all_rows.append(result)
            median_norm = float(np.median(np.linalg.norm(points, axis=-1)))
            point_rms = float(np.sqrt(np.mean(np.sum(points ** 2, axis=-1))))
            for c in CURVATURES:
                tangent_scale = float(np.arctanh(TARGET_RADIUS_FRACTION) / np.sqrt(c) / median_norm * point_rms)
                ball, _ = ball_map(points, c, tangent_scale)
                pos, neg = _pair_distances(ball, pairs, "poincare", c)
                fraction = matched_ordering_fractions(raw_pos, raw_neg, pos, neg)
                result = _comparison_metrics(pos, neg)
                result.update({"model": label, "pair_set": "secondary_train", "level": level_name,
                               "variant": f"poincare_c{c:g}",
                               "spearman_vs_raw_euclidean": float(spearmanr(np.concatenate((pos, neg)),
                                                                              np.concatenate((raw_pos, raw_neg))).statistic),
                               "discordant_fraction_vs_raw": fraction["discordant_fraction"],
                               "concordant_fraction_vs_raw": fraction["concordant_fraction"],
                               "tie_fraction_vs_raw": fraction["tie_fraction"],
                               "bootstrap_auc_95_ci_low": None, "bootstrap_auc_95_ci_high": None})
                result.pop("_raw_auc", None)
                all_rows.append(result)
    _write_csv(output_dir / "auc_metrics.csv", all_rows)
    _write_csv(output_dir / "radial_distribution.csv", radial_rows)
    _write_csv(output_dir / "code_reassignment.csv", reassignment_rows)
    _write_csv(output_dir / "tangent_scale_sensitivity.csv", sensitivity_rows)
    report = {
        "falsification_framing": "Raw, RMS-normalized, globally median-distance-matched Euclidean, and log1p(Euclidean) preserve within-level ordering; their AUC must be bit-identical. Only Poincare may change ordering.",
        "verdict_rule": "Negative only when Poincare AUC is inside the raw-Euclidean anchor-bootstrap 95% interval AND matched positive-negative discordance is below 0.01.",
        "curvature_interpretation": "The specified radius-matched map makes tested curvature rankings equivalent; curvature changes the distance scale, not rankings, in these frozen representations.",
        "encode_ball_scale_semantics": "encode_ball divides latent by point_rms = sqrt(mean(sum(latent**2))); the target-radius tangent_scale passed is normalizer * point_rms, so the effective raw-vector multiplier is normalizer. The normalizer-only sensitivity row deliberately reproduces the smaller scale obtained by passing normalizer directly.",
        "tangent_scale_sensitivity": sensitivity_rows,
        "verdicts": verdicts,
    }
    (output_dir / "verdicts.json").write_text(json.dumps(report, indent=2) + "\n")
    _figures(output_dir, all_rows, radial_rows)
    print("FALSIFICATION: monotone/affine Euclidean transforms must preserve AUC exactly; runtime assertions passed for both pair sets.")
    print("model,pair_set,level,variant,auc,ordering_error_rate,hard_negative_intrusion_rate,spearman_vs_raw_euclidean,discordant_fraction_vs_raw")
    for row in all_rows:
        print(f"{row['model']},{row['pair_set']},{row['level']},{row['variant']},{row['auc']:.8f},{row['ordering_error_rate']:.8f},{row['hard_negative_intrusion_rate']:.8f},{row['spearman_vs_raw_euclidean']:.8f},{row['discordant_fraction_vs_raw']:.8f}")
    print("RADIAL DISTRIBUTION")
    for row in radial_rows:
        print(f"{row['model']},{row['level']},c={row['c']},tangent_scale={row['tangent_scale']:.10g},median={row['sqrt_c_norm_quantiles']['p50']:.8f},p99={row['sqrt_c_norm_quantiles']['p99']:.8f},near_boundary={row['near_boundary_fraction_ge_0.99']:.8f},radius_std={row['radius_std']:.8f},flag={row['radial_information']},saturated={row['saturated']}")
    print("TANGENT SCALE SENSITIVITY (c=1, matched primary comparisons)")
    for row in sensitivity_rows:
        print(f"{row['model']},{row['level']},{row['scale_name']},passed={row['tangent_scale_passed']:.10g},effective_multiplier={row['effective_raw_vector_multiplier']:.10g},auc={row['auc']:.8f},discordance={row['discordant_fraction_vs_euclidean']:.8f},max_sqrt_c_norm={row['max_sqrt_c_norm']:.8f},near_boundary={row['near_boundary_fraction_ge_0.99']:.8f}")
    print("CODE REASSIGNMENT")
    for row in reassignment_rows:
        print(f"{row['model']},{row['level']},c={row['c']},fraction={row['code_reassignment_fraction']:.8f}")
    print("NAMED VERDICTS")
    for row in verdicts:
        print(f"{row['model']},{row['level']},c={row['c']}: {row['verdict']}")
    print(f"Outputs: {output_dir}")


def _write_csv(path: Path, rows: list[dict]):
    import csv
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _figures(output_dir: Path, metrics: list[dict], radial: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for label in ARMS:
        rows = [r for r in metrics if r["model"] == label and r["pair_set"] == "primary_valid"]
        fig, ax = plt.subplots(figsize=(11, 5))
        variants = [r for r in rows if r["variant"] == "raw" or r["variant"].startswith("poincare_")]
        for variant in sorted({r["variant"] for r in variants}):
            subset = [r for r in variants if r["variant"] == variant]
            ax.plot([LEVELS.index(r["level"]) + 1 for r in subset], [r["auc"] for r in subset], marker="o", label=variant)
        ax.set(xticks=(1, 2, 3), xticklabels=LEVELS, ylabel="positive-vs-negative AUC", title=f"Frozen geometry AUC: {label}")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"auc_{label}.png", dpi=160)
        plt.close(fig)
        rows = [r for r in radial if r["model"] == label]
        fig, ax = plt.subplots(figsize=(8, 5))
        for level in LEVELS:
            subset = [r for r in rows if r["level"] == level]
            ax.plot([r["c"] for r in subset], [r["radius_std"] for r in subset], marker="o", label=level)
        ax.set(xscale="log", xlabel="curvature c", ylabel="std(sqrt(c) ||z||)", title=f"Radial spread: {label}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"radial_{label}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    run()
