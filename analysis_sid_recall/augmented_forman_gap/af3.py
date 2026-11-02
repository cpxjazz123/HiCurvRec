"""Issue #136: AF3 curvature and the original-graph within/cross edge gap.

Frozen convention (definitions.json): on a binary, unweighted, simple graph H,
for every undirected edge e=(u,v)
    F_H(e) = 4 - deg_H(u) - deg_H(v)
    T_H(e) = |N_H(u) & N_H(v)|
    AF3_H(e) = F_H(e) + 3*T_H(e)
Degrees and triangles always belong to the graph the edge is measured in.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def degree_of(matrix: sp.csr_matrix) -> np.ndarray:
    return np.asarray(matrix.sum(axis=1)).ravel()


def edge_arrays(matrix: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Every undirected edge exactly once, in CSR order (row-major)."""
    coo = matrix.tocoo()
    keep = coo.row < coo.col
    return coo.row[keep], coo.col[keep]


def triangles_per_edge(matrix: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T_H(e) for every undirected edge; edges without triangles are 0, not missing."""
    rows, cols = edge_arrays(matrix)
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    triangles = np.zeros(rows.size, dtype=np.int64)
    size = int(matrix.shape[0])
    if rows.size and size:
        integer = matrix.astype(np.int32)
        product = (integer @ integer).multiply(integer).tocoo()
        keep = product.row < product.col
        product_rows = product.row[keep].astype(np.int64)
        product_cols = product.col[keep].astype(np.int64)
        counts = np.rint(product.data[keep]).astype(np.int64)
        key = product_rows * size + product_cols
        target = rows.astype(np.int64) * size + cols.astype(np.int64)
        positions = np.searchsorted(target, key)
        if positions.size and (positions.max() >= target.size
                               or not np.array_equal(target[positions], key)):
            raise RuntimeError("the triangle pass returned an edge outside the graph")
        triangles[positions] = counts
    return rows, cols, triangles


def edge_curvatures(matrix: sp.csr_matrix) -> dict:
    """F, T and AF3 on every edge of one graph, aligned by edge order."""
    rows, cols = edge_arrays(matrix)
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    tri_rows, tri_cols, triangles = triangles_per_edge(matrix)
    if not (np.array_equal(rows, tri_rows) and np.array_equal(cols, tri_cols)):
        raise RuntimeError("edge order mismatch between adjacency and triangle pass")
    degree = degree_of(matrix)
    ordinary = 4.0 - degree[rows] - degree[cols]
    return {"rows": rows, "cols": cols, "triangles": triangles,
            "forman": ordinary, "af3": ordinary + 3.0 * triangles.astype(float),
            "degree": degree}


def _summarise(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None,
                "negative_fraction": None, "edges": 0}
    p10, p90 = np.quantile(values, [0.1, 0.9])
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(p10), "p90": float(p90),
            "negative_fraction": float((values < 0).mean()), "edges": int(values.size)}


def group_af3(graph: sp.csr_matrix, labels: np.ndarray) -> dict:
    """AF3 on every occupied prefix's induced subgraph, plus the census."""
    occupied = labels[labels >= 0]
    if occupied.size == 0:
        return {"groups": 0, "status": "UNDEFINED_NO_LABELS"}
    groups = np.unique(occupied)
    members = {int(value): np.nonzero(labels == value)[0] for value in groups}
    per_group, af3_values, f_values, weights = [], [], [], []
    singleton = no_edge = 0
    for value in groups:
        nodes = members[int(value)]
        if nodes.size < 2:
            singleton += 1
            continue
        inner = graph[nodes][:, nodes]
        if inner.nnz == 0:
            no_edge += 1
            continue
        curvature = edge_curvatures(inner)
        af3_stats = _summarise(curvature["af3"])
        forman_stats = _summarise(curvature["forman"])
        per_group.append({"group": int(value), "items": int(nodes.size),
                          "edges": af3_stats["edges"],
                          "af3_mean": af3_stats["mean"], "af3_median": af3_stats["median"],
                          "af3_p10": af3_stats["p10"], "af3_p90": af3_stats["p90"],
                          "af3_negative_fraction": af3_stats["negative_fraction"],
                          "forman_mean": forman_stats["mean"],
                          "triangles_mean": float(curvature["triangles"].mean())
                          if curvature["triangles"].size else None})
        af3_values.append(af3_stats["mean"])
        f_values.append(forman_stats["mean"])
        weights.append(af3_stats["edges"])
    af3_values = np.asarray(af3_values, dtype=float)
    f_values = np.asarray(f_values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return {"groups": int(groups.size), "eligible_groups": int(af3_values.size),
            "singleton_groups": singleton, "no_edge_groups": no_edge,
            "edges": int(weights.sum()),
            "af3_mean": float(af3_values.mean()) if af3_values.size else None,
            "af3_median": float(np.median(af3_values)) if af3_values.size else None,
            "af3_p10": float(np.quantile(af3_values, 0.1)) if af3_values.size else None,
            "af3_p90": float(np.quantile(af3_values, 0.9)) if af3_values.size else None,
            "af3_negative_fraction": float((af3_values < 0).mean())
            if af3_values.size else None,
            "af3_edge_weighted": float(np.average(af3_values, weights=weights))
            if af3_values.size and weights.sum() > 0 else None,
            "forman_mean": float(f_values.mean()) if f_values.size else None,
            "group_coverage": float(af3_values.size / max(groups.size, 1)),
            "per_group": per_group,
            "status": "DEFINED" if af3_values.size else "UNDEFINED_NO_EDGES"}


def gap_metrics(rows: np.ndarray, cols: np.ndarray, curvature: np.ndarray,
                labels: np.ndarray) -> dict:
    """Classify the SAME original edges as within/cross and difference the means."""
    left, right = labels[rows], labels[cols]
    unlabelled = (left < 0) | (right < 0)
    eligible = ~unlabelled
    within = eligible & (left == right)
    cross = eligible & (left != right)
    result = {"edges_total": int(rows.size), "edges_unlabelled": int(unlabelled.sum()),
              "edges_eligible": int(eligible.sum()), "edges_within": int(within.sum()),
              "edges_cross": int(cross.sum())}
    if within.sum() == 0:
        result.update({"status": "UNDEFINED_INSUFFICIENT_WITHIN_EDGES",
                       "gap": None, "within_mean": None, "cross_mean": None})
        return result
    if cross.sum() == 0:
        result.update({"status": "UNDEFINED_INSUFFICIENT_CROSS_EDGES",
                       "gap": None, "within_mean": None, "cross_mean": None})
        return result
    inside = _summarise(curvature[within])
    outside = _summarise(curvature[cross])
    result.update({"status": "DEFINED", "within_mean": inside["mean"],
                   "cross_mean": outside["mean"], "gap": inside["mean"] - outside["mean"],
                   "within_median": inside["median"], "cross_median": outside["median"],
                   "within_negative_fraction": inside["negative_fraction"],
                   "cross_negative_fraction": outside["negative_fraction"],
                   "within_triangles_mean": None})
    return result


def shuffle_labels(labels: np.ndarray, degree: np.ndarray, design: str,
                   rng: np.random.Generator) -> np.ndarray:
    """A label permutation that keeps every label (and its size) on labelled items."""
    out = np.asarray(labels, dtype=np.int64).copy()
    masked = np.nonzero(out >= 0)[0]
    if design == "global":
        out[masked] = rng.permutation(out[masked])
    elif design == "degree_stratified":
        buckets = np.floor(np.log2(np.maximum(degree[masked], 1.0))).astype(int)
        for bucket in np.unique(buckets):
            positions = masked[buckets == bucket]
            out[positions] = rng.permutation(out[positions])
    else:
        raise ValueError(f"unknown null design {design!r}")
    return out


def gap_null(rows: np.ndarray, cols: np.ndarray, curvature: np.ndarray,
             labels: np.ndarray, degree: np.ndarray, design: str,
             replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(replicates):
        record = gap_metrics(rows, cols, curvature,
                             shuffle_labels(labels, degree, design, rng))
        if record["status"] == "DEFINED":
            gaps.append(record["gap"])
    values = np.asarray(gaps, dtype=float)
    if values.size < max(2, replicates // 2):
        return {"n": int(values.size), "status": "INSUFFICIENT_DEFINED_NULLS"}
    return {"n": int(values.size), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)), "status": "DEFINED"}
