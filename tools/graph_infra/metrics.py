"""Production evaluation metrics for issues #243/#244.

The #243 evaluation used a hand-rolled rank correlation without tie handling and a Jaccard
proxy (`getnnz` of a summed adjacency), neither of which is the quantity the issue asks
for.  These are the production versions; the tests compare them against hand-computed
values and against ``reference.py`` (which derives them independently).
"""
from __future__ import annotations

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based average ranks; ties share the mean of the positions they occupy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        end = position
        while end + 1 < values.size and values[order[end + 1]] == values[order[position]]:
            end += 1
        ranks[order[position:end + 1]] = 0.5 * (position + end) + 1.0
        position = end + 1
    return ranks


def spearman(x, y) -> float | None:
    """Spearman rank correlation with average ranks; a constant input returns ``None`` (NA)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or x.size != y.size:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    rx = average_ranks(x)
    ry = average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denominator == 0.0:
        return None
    return float((rx * ry).sum() / denominator)


def jaccard(left, right) -> float:
    """|A ∩ B| / |A ∪ B|; empty union returns 0."""
    left = set(int(x) for x in left)
    right = set(int(x) for x in right)
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def neighbour_jaccard(out_csr, in_csr, i: int, j: int, directed: bool = False) -> float:
    """Jaccard of the undirected (or in/out-averaged) neighbourhoods of two items."""
    def undirected(node):
        start, end = out_csr.indptr[node], out_csr.indptr[node + 1]
        out_side = set(int(x) for x in out_csr.indices[start:end])
        start, end = in_csr.indptr[node], in_csr.indptr[node + 1]
        in_side = set(int(x) for x in in_csr.indices[start:end])
        return out_side | in_side

    def out_side(node):
        start, end = out_csr.indptr[node], out_csr.indptr[node + 1]
        return set(int(x) for x in out_csr.indices[start:end])

    def in_side(node):
        start, end = in_csr.indptr[node], in_csr.indptr[node + 1]
        return set(int(x) for x in in_csr.indices[start:end])

    if directed:
        return 0.5 * (jaccard(out_side(i), out_side(j)) + jaccard(in_side(i), in_side(j)))
    return jaccard(undirected(i), undirected(j))


def prefix(sequence, length: int) -> tuple:
    return tuple(int(x) for x in sequence[:length])


def prefix_share(sid_left, sid_right, length: int) -> int:
    """Number of leading positions two SIDs agree on, capped at ``length``."""
    count = 0
    for a, b in zip(sid_left, sid_right):
        if count >= length or int(a) != int(b):
            break
        count += 1
    return count


def pearson(x, y) -> float | None:
    """Plain Pearson; ``None`` for a constant input or mismatched lengths."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or x.size != y.size:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    dx = x - x.mean()
    dy = y - y.mean()
    denominator = float(np.linalg.norm(dx) * np.linalg.norm(dy))
    if denominator == 0.0:
        return None
    return float((dx * dy).sum() / denominator)


def distortion(distance, graph_distance, unit: str = "graph hops") -> dict:
    """``|(d/g)^2 - 1|`` over finite pairs; the unit is reported, never assumed."""
    d = np.asarray(distance, dtype=np.float64)
    g = np.asarray(graph_distance, dtype=np.float64)
    mask = np.isfinite(d) & np.isfinite(g) & (g > 0)
    if not mask.any():
        return {"n": 0, "mean": None, "median": None, "unit": unit}
    values = np.abs((d[mask] / g[mask]) ** 2 - 1.0)
    return {"n": int(mask.sum()), "mean": float(values.mean()),
            "median": float(np.median(values)), "unit": unit,
            "unreachable": int((~np.isfinite(g)).sum())}


def prefix_at_k(pairs, k: int) -> float:
    """Fraction of pairs whose first ``k`` SID positions are identical, position by position."""
    pairs = list(pairs)
    if not pairs:
        return float("nan")
    hits = 0
    for left, right in pairs:
        if all(int(a) == int(b) for a, b in zip(left[:k], right[:k])) and len(left) >= k == len(right[:k]):
            hits += 1
    return hits / len(pairs)


def collision_rate(sids) -> float:
    """1 - distinct / total over the first three codes (the SID collision of the Stage2 gate)."""
    rows = [tuple(int(x) for x in row[:3]) for row in sids]
    if not rows:
        return float("nan")
    return 1.0 - len(set(rows)) / len(rows)


def duplicate_item_fraction(sids) -> float:
    """Share of items whose full SID tuple is shared with at least one other item."""
    rows = [tuple(int(x) for x in row) for row in sids]
    if not rows:
        return float("nan")
    counts = {}
    for row in rows:
        counts[row] = counts.get(row, 0) + 1
    duplicated = sum(1 for count in counts.values() if count > 1 for _ in range(count))
    return duplicated / len(rows)


def codebook_stats(tokens) -> list[dict]:
    """Per-level usage, entropy and maximum bucket share for a token matrix (n, levels)."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("tokens must be (items, levels)")
    report = []
    for level in range(tokens.shape[1]):
        column = tokens[:, level]
        size = int(column.max()) + 1 if column.size else 0
        counts = np.bincount(column, minlength=size).astype(np.float64)
        probabilities = counts / counts.sum()
        positive = probabilities[probabilities > 0]
        report.append({
            "level": level,
            "used": int((counts > 0).sum()),
            "entropy": float(-(positive * np.log(positive)).sum()),
            "max_share": float(probabilities.max()) if probabilities.size else 0.0,
        })
    return report


def reconstruction_error(target, prediction) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError("shape mismatch")
    difference = prediction - target
    return {"mse": float((difference ** 2).mean()),
            "rmse": float(np.sqrt((difference ** 2).mean())),
            "mae": float(np.abs(difference).mean())}


def compare_sid_tables(left, right) -> dict:
    """Item-by-item comparison of two ``{item_id: [codes]}`` tables; missing keys are listed."""
    left_keys, right_keys = set(left), set(right)
    common = left_keys & right_keys
    differing = [key for key in common if list(left[key]) != list(right[key])]
    return {
        "items_left": len(left_keys),
        "items_right": len(right_keys),
        "only_left": len(left_keys - right_keys),
        "only_right": len(right_keys - left_keys),
        "differing": len(differing),
        "differing_examples": sorted(differing)[:5],
    }


def subgraph_summary(subgraph_values, threshold: float) -> dict:
    """Aggregate per-center, per-subgraph scores without conflating the two denominators.

    ``subgraph_values`` is a sequence of arrays, one per *center item*, holding that center's
    per-subgraph scores.  Center-level and subgraph-level statistics are reported separately,
    and the "all subgraphs pass" fraction is never mixed with "some subgraph passes".
    """
    arrays = [np.asarray(values, dtype=np.float64) for values in subgraph_values]
    per_subgraph = np.concatenate(arrays) if arrays else np.zeros(0)
    center_mean = np.array([values.mean() for values in arrays if values.size], dtype=np.float64)
    center_worst = np.array([values.min() for values in arrays if values.size], dtype=np.float64)
    all_pass = np.array([bool((values >= threshold).all()) for values in arrays if values.size], dtype=bool)
    any_pass = np.array([bool((values >= threshold).any()) for values in arrays if values.size], dtype=bool)
    return {
        "threshold": float(threshold),
        "centers": int(center_mean.size),
        "subgraphs": int(per_subgraph.size),
        "subgraph_pass_fraction": float((per_subgraph >= threshold).mean()) if per_subgraph.size else None,
        "center_mean": float(center_mean.mean()) if center_mean.size else None,
        "center_worst": float(center_worst.mean()) if center_worst.size else None,
        "center_p90": float(np.percentile(center_mean, 90)) if center_mean.size else None,
        "center_all_pass_fraction": float(all_pass.mean()) if all_pass.size else None,
        "center_any_pass_fraction": float(any_pass.mean()) if any_pass.size else None,
        "policy": "subgraph_pass_fraction and center_all_pass_fraction are different denominators "
                  "and must be reported side by side, never as one number",
    }


def stratified(values, strata) -> dict:
    """Group a metric by a categorical stratum; empty strata are absent, not zero."""
    values = np.asarray(values, dtype=np.float64)
    strata = np.asarray(strata)
    if values.shape != strata.shape:
        raise ValueError("values and strata differ in shape")
    report = {}
    for name in np.unique(strata):
        selected = values[strata == name]
        report[str(name)] = {"n": int(selected.size),
                             "mean": float(selected.mean()) if selected.size else None,
                             "p10": float(np.percentile(selected, 10)) if selected.size else None,
                             "p90": float(np.percentile(selected, 90)) if selected.size else None}
    return report
