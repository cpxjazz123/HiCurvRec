from __future__ import annotations

import numpy as np


def _distribution(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            **{f"p{p:02d}": float(np.percentile(x, p)) for p in (0, 5, 25, 50, 75, 95, 99, 100)}}


def map_to_ball(x, radial_scale=2.0, clamp=1.0 - 1e-5):
    x = np.asarray(x, dtype=np.float64)
    scaled = x * radial_scale
    norm = np.linalg.norm(scaled, axis=-1, keepdims=True)
    capped = np.minimum(norm, 2.0)
    mapped_radius = np.tanh(capped)
    mapped = scaled * (mapped_radius / np.maximum(norm, 1e-12))
    radius = np.linalg.norm(mapped, axis=-1, keepdims=True)
    factor = np.minimum(1.0, clamp / np.maximum(radius, 1e-12))
    return mapped * factor, {"pre_clamp_fraction": float(np.mean(radius.reshape(-1) >= clamp)),
                              "norm_cap_fraction": float(np.mean(norm.reshape(-1) > 2.0))}


def radius_statistics(points, reference_points=None):
    p = np.asarray(points, dtype=np.float64)
    radii = np.linalg.norm(p, axis=-1)
    return {"radius": {**_distribution(radii), "below_0.1_fraction": float(np.mean(radii < .1)),
                        "above_0.9_fraction": float(np.mean(radii > .9))},
            "conformal_factor": _distribution(2.0 / np.maximum(1.0 - radii ** 2, 1e-12)),
            "nonfinite_points": int(np.size(radii) - np.isfinite(radii).sum())}


def pairwise_angle_statistics(points, max_points=2048, seed=42):
    x = np.asarray(points, dtype=np.float64)
    if len(x) > max_points:
        take = np.random.default_rng(seed).choice(len(x), max_points, replace=False)
        x = x[take]
    norms = np.linalg.norm(x, axis=1)
    keep = norms > 1e-12
    x, norms = x[keep], norms[keep]
    if len(x) < 2:
        return {"pair_count": 0, "degrees": _distribution([])}
    unit = x / norms[:, None]
    i, j = np.triu_indices(len(unit), 1)
    angle = np.degrees(np.arccos(np.clip(np.sum(unit[i] * unit[j], axis=1), -1, 1)))
    return {"pair_count": int(len(angle)), "degrees": _distribution(angle)}


def angle_radius_dependence(points, max_points=2048, seed=42):
    x = np.asarray(points, dtype=np.float64)
    if len(x) > max_points:
        x = x[np.random.default_rng(seed).choice(len(x), max_points, replace=False)]
    r = np.linalg.norm(x, axis=1)
    unit = x / np.maximum(r[:, None], 1e-12)
    i, j = np.triu_indices(len(x), 1)
    angle = np.arccos(np.clip(np.sum(unit[i] * unit[j], axis=1), -1, 1))
    mean_radius = (r[i] + r[j]) / 2
    if len(angle) < 2 or np.std(angle) == 0 or np.std(mean_radius) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(angle, mean_radius)[0, 1])
    return {"pearson_angle_vs_mean_pair_radius": corr, "pair_count": int(len(angle))}


def ranking_difference(euclidean_distances, geodesic_distances):
    e, g = np.asarray(euclidean_distances), np.asarray(geodesic_distances)
    ae = float(np.mean(e[:, 1] > e[:, 0]) + .5 * np.mean(e[:, 1] == e[:, 0]))
    ag = float(np.mean(g[:, 1] > g[:, 0]) + .5 * np.mean(g[:, 1] == g[:, 0]))
    return {"mapped_euclidean_auc": ae, "geodesic_auc": ag, "auc_difference": ag - ae,
            "ranking_disagreement_fraction": float(np.mean(np.sign(e[:, 1] - e[:, 0]) !=
                                                              np.sign(g[:, 1] - g[:, 0])))}


def shared_local_scales(euclidean_vectors, hyperbolic_vectors, anchors, reference_ids):
    """Same anchors/reference IDs/median rule; distances remain each metric's native units."""
    anchors, refs = np.asarray(anchors, dtype=np.int64), np.unique(reference_ids)
    output = {}
    for name, vectors in (("euclidean", euclidean_vectors), ("hyperbolic", hyperbolic_vectors)):
        d = np.linalg.norm(vectors[anchors, None] - vectors[refs][None], axis=-1)
        d[anchors[:, None] == refs[None]] = np.nan
        with np.errstate(all="ignore"):
            output[name] = np.nanmedian(d, axis=1)
    return output


def pipeline_spaces(vectors, cumulative_reconstruction):
    return {"continuous_encoder": vectors["encoder"],
            "per_level_residual": {k: vectors[k] for k in ("L1", "L2", "L3")},
            "cumulative_reconstruction": np.asarray(cumulative_reconstruction).copy()}


def codebook_statistics(codes, size=256):
    codes = np.asarray(codes, dtype=np.int64)
    out = {}
    for level in range(codes.shape[1]):
        counts = np.bincount(codes[:, level], minlength=size)
        p = counts[counts > 0] / max(counts.sum(), 1)
        sorted_counts = np.sort(counts.astype(float))
        gini = (2 * np.dot(np.arange(1, size + 1), sorted_counts) / (size * sorted_counts.sum()) -
                (size + 1) / size) if sorted_counts.sum() else 0.
        out[f"L{level+1}"] = {"occupied": int(np.count_nonzero(counts)), "capacity": int(size),
                               "entropy_nats": float(-np.sum(p * np.log(p))), "gini": float(gini)}
    unique = len(np.unique(codes, axis=0))
    out["sid"] = {"unique": int(unique), "items": int(len(codes)),
                  "collision_rate": float(1 - unique / len(codes))}
    return out
