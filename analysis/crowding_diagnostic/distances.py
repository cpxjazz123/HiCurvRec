
import numpy as np


def load_pairs(path):
    with np.load(path) as data:
        return {key: data[key] for key in ("anchors", "positives", "negatives")}


def euclidean(vectors):
    x = np.asarray(vectors, dtype=np.float64)
    sq = np.sum(x * x, axis=1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0))


def _pair_distances(vectors, pairs):
    a, p, n = pairs["anchors"], pairs["positives"], pairs["negatives"]
    pos = np.linalg.norm(vectors[a] - vectors[p], axis=1)
    neg = np.linalg.norm(vectors[a, None, :] - vectors[n], axis=2)
    return pos, neg


def _paired_indicators(pos, neg):
    pos = np.asarray(pos).ravel()
    neg = np.asarray(neg)
    if neg.ndim == 2 and neg.shape[0] == len(pos):
        return neg > pos[:, None], neg == pos[:, None], neg < pos[:, None]
    neg = neg.ravel()
    greater = neg[None, :] > pos[:, None]
    ties = neg[None, :] == pos[:, None]
    less = neg[None, :] < pos[:, None]
    return greater, ties, less


def auc_neg_farther_than_pos(pos, neg):
    greater, ties, _ = _paired_indicators(pos, neg)
    return float(np.mean(greater) + .5 * np.mean(ties))


def ordering_error_rate(pos, neg):
    _, ties, less = _paired_indicators(pos, neg)
    return float(np.mean(less) + .5 * np.mean(ties))


def _distribution(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std()), "min": float(values.min()), "max": float(values.max()),
            **{f"p{p:02d}": float(np.percentile(values, p)) for p in (1, 5, 25, 75, 95, 99)}}


def _overlap(pos, neg):
    lo, hi = min(pos.min(), neg.min()), max(pos.max(), neg.max())
    if lo == hi:
        return 1.0, 1.0
    bins = np.linspace(lo, hi, 101)
    hp, _ = np.histogram(pos, bins=bins, density=True)
    hn, _ = np.histogram(neg, bins=bins, density=True)
    width = bins[1] - bins[0]
    histogram = float(np.minimum(hp, hn).sum() * width)
    # Overlapping coefficient = integral min(f_pos, f_neg), estimated on the same grid.
    grid = np.linspace(lo, hi, 1001)
    p_density = np.histogram(pos, bins=grid, density=True)[0]
    n_density = np.histogram(neg, bins=grid, density=True)[0]
    coefficient = float(np.minimum(p_density, n_density).sum() * (grid[1] - grid[0]))
    return histogram, coefficient


def _anchor_auc_ci(pos, neg, anchors, seed, replicates):
    unique = np.unique(anchors)
    if len(unique) < 2:
        return None
    rng = np.random.default_rng(seed)
    rows = {anchor: np.flatnonzero(anchors == anchor) for anchor in unique}
    values = []
    for _ in range(replicates):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate([rows[a] for a in chosen])
        values.append(auc_neg_farther_than_pos(pos[indexes], neg[indexes]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def analyze_distances(residuals, pairs, variants, seed=42, replicates=200):
    results, rows = [], []
    anchors = pairs["anchors"]
    auc_ci_cache = {}
    for model_label, residual in residuals.items():
        for level in range(3):
            vectors = np.asarray(residual[:, level, :], dtype=np.float64)
            rms = float(np.sqrt(np.mean(vectors * vectors)))
            for variant in variants:
                scaled = vectors if variant == "euclidean" else vectors / rms
                pos, neg_matrix = _pair_distances(scaled, pairs)
                neg = neg_matrix.ravel()
                diff = neg_matrix - pos[:, None]
                histogram, overlap = _overlap(pos, neg)
                auc = auc_neg_farther_than_pos(pos, neg_matrix)
                key = (model_label, level)
                if variant == "euclidean" or key not in auc_ci_cache:
                    auc_ci_cache[key] = _anchor_auc_ci(pos, neg_matrix, anchors, seed + level, replicates)
                ci = auc_ci_cache[key]
                rec = {"model": model_label, "level": f"L{level+1}", "variant": variant,
                       "n_pairs": int(len(pos)), "n_negative_comparisons": int(len(neg)),
                       "positive": _distribution(pos), "negative": _distribution(neg),
                       "difference_negative_minus_positive": _distribution(diff),
                       "fraction_negative_closer": float(np.mean(diff < 0)),
                       "ordering_error_rate": ordering_error_rate(pos, neg_matrix),
                       "histogram_bin_overlap_coefficient": histogram,
                       "overlapping_coefficient": overlap,
                       "auc_neg_farther_than_pos": auc,
                       "auc_neg_farther_than_pos_anchor_bootstrap_95ci": ci,
                       "rms_scale": rms}
                results.append(rec)
                for label, distribution in (("positive", rec["positive"]), ("negative", rec["negative"]),
                                             ("difference", rec["difference_negative_minus_positive"])):
                    rows.append({"model": model_label, "level": rec["level"], "variant": variant,
                                 "distribution": label, **distribution})
                rows.append({"model": model_label, "level": rec["level"], "variant": variant,
                             "distribution": "comparison", "fraction_negative_closer": rec["fraction_negative_closer"],
                             "ordering_error_rate": rec["ordering_error_rate"],
                             "histogram_bin_overlap_coefficient": histogram, "overlapping_coefficient": overlap,
                             "auc_neg_farther_than_pos": auc,
                             "auc_neg_farther_than_pos_ci_low": None if ci is None else ci[0],
                             "auc_neg_farther_than_pos_ci_high": None if ci is None else ci[1]})
    return results, rows
