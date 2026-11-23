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
        left = list(left)
        right = list(right)
        if len(left) < k or len(right) < k:
            continue
        if all(int(a) == int(b) for a, b in zip(left[:k], right[:k])):
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


def codebook_stats(tokens, unit: str = "nats") -> list[dict]:
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
        base = np.e if unit == "nats" else 2.0
        report.append({
            "level": level,
            "used": int((counts > 0).sum()),
            "entropy": float(-(positive * np.log(positive) / np.log(base)).sum()),
            "entropy_unit": unit,
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


def subgraph_summary(subgraph_values, threshold: float, direction: str = "minimize") -> dict:
    """Aggregate per-centre, per-subgraph scores without conflating the denominators.

    ``direction="minimize"`` is for distortions (lower is better): pass = value <= threshold,
    worst = max, per-centre p90 = quantile(values, 0.9).  ``"maximize"`` is the mirror for
    similarities.  The three pass fractions use different denominators and are always reported
    together; centres without a valid subgraph are counted, never silently dropped.
    """
    if direction not in ("minimize", "maximize"):
        raise ValueError(f"direction must be 'minimize' or 'maximize', got {direction!r}")
    arrays = [np.asarray(values, dtype=np.float64).ravel() for values in subgraph_values]
    valid = [values for values in arrays if values.size]
    per_subgraph = np.concatenate(valid) if valid else np.zeros(0)
    if direction == "minimize":
        passed = per_subgraph <= threshold
        worst = np.array([values.max() for values in valid], dtype=np.float64)
        per_center = np.array([np.percentile(values, 90) for values in valid], dtype=np.float64)
        all_pass = np.array([bool((values <= threshold).all()) for values in valid], dtype=bool)
        any_pass = np.array([bool((values <= threshold).any()) for values in valid], dtype=bool)
        quantile = 0.9
    else:
        passed = per_subgraph >= threshold
        worst = np.array([values.min() for values in valid], dtype=np.float64)
        per_center = np.array([np.percentile(values, 10) for values in valid], dtype=np.float64)
        all_pass = np.array([bool((values >= threshold).all()) for values in valid], dtype=bool)
        any_pass = np.array([bool((values >= threshold).any()) for values in valid], dtype=bool)
        quantile = 0.1
    centers = len(valid)
    return {
        "threshold": float(threshold),
        "direction": direction,
        "centers": centers,
        "centers_total": len(arrays),
        "centers_without_valid_subgraph": len(arrays) - centers,
        "subgraphs": int(per_subgraph.size),
        "subgraph_pass_fraction": float(passed.mean()) if per_subgraph.size else None,
        "center_all_pass_fraction": float(all_pass.mean()) if centers else None,
        "center_any_pass_fraction": float(any_pass.mean()) if centers else None,
        "worst": float(worst.mean()) if centers else None,
        "p90": float(per_center.mean()) if centers else None,
        "center_quantile": float(quantile),
        "center_mean": float(np.mean([values.mean() for values in valid])) if centers else None,
        "denominator_policy": "subgraph pass uses all valid subgraphs; the two centre fractions use "
                              "centres with at least one valid subgraph; centres without one are "
                              "reported in centers_without_valid_subgraph and never counted as pass",
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


def sid_structure(codes) -> dict:
    """Collision and duplicate rates over the prefix-3 codes and over the full codes."""
    rows = [tuple(int(x) for x in code) for code in codes]
    total = len(rows)
    if total == 0:
        return {"total": 0}
    prefix = [row[:3] for row in rows]

    def pair(keys):
        counts = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        repeated = sum(count for count in counts.values() if count > 1)
        return 1.0 - len(counts) / total, repeated / total

    prefix_collision, prefix_duplicate = pair(prefix)
    full_collision, full_duplicate = pair(rows)
    return {
        "total": total,
        "distinct_prefix3": len(set(prefix)),
        "distinct_full": len(set(rows)),
        "collision_prefix3": prefix_collision,
        "duplicate_prefix3": prefix_duplicate,
        "collision_full": full_collision,
        "duplicate_full": full_duplicate,
        "policy": "prefix-3 and full-length (including the disambiguation token) are reported "
                  "separately; they answer different questions",
    }


def sid_audit(table, item_ids, width: int, max_token: int) -> dict:
    """Validate a SID table before any metric is computed from it."""
    normalized = {int(k): [int(x) for x in values] for k, values in table.items()}
    keys = sorted(normalized)
    expected = [int(x) for x in item_ids]
    bad_length, out_of_range, duplicates = [], [], []
    seen = {}
    for key in keys:
        row = normalized[key]
        if len(row) != width:
            bad_length.append(key)
            continue
        if any(value < 0 or value > max_token for value in row):
            out_of_range.append(key)
        tuple_row = tuple(row)
        if tuple_row in seen:
            duplicates.append(key)
        else:
            seen[tuple_row] = key
    missing = sorted(set(expected) - set(keys))
    extra = sorted(set(keys) - set(expected))
    return {
        "rows": len(keys),
        "expected": len(expected),
        "missing_items": missing[:20],
        "missing_count": len(missing),
        "extra_items": extra[:20],
        "extra_count": len(extra),
        "bad_length": bad_length[:20],
        "bad_length_count": len(bad_length),
        "out_of_range": out_of_range[:20],
        "out_of_range_count": len(out_of_range),
        "duplicate_rows": duplicates[:20],
        "duplicate_count": len(duplicates),
        "width": int(width),
        "max_token": int(max_token),
        "ok": not missing and not extra and not bad_length and not out_of_range and not duplicates,
    }
