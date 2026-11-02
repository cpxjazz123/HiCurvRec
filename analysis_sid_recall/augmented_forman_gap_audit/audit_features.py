"""Corrected AF3 feature semantics for Issue #137.

B3: #136 published ``(per-group AF3 mean < 0).mean()`` under the name
"negative-edge fraction". The two quantities are different and are now named
and implemented separately:

* ``group_mean_negative_fraction``      -- groups whose MEAN AF3 is negative;
* ``edge_negative_fraction``            -- negative AF3 EDGES over all eligible
                                           induced edges (full edge census);
* ``mean_group_negative_edge_fraction`` -- unweighted mean of the per-group
                                           edge-negative fractions.

Ordinary-Forman analogues use the same denominators. ``gap_block`` adds the
within/cross triangle means so the identity
``gap_af3 - gap_forman = 3*(T_within - T_cross)`` can be checked on every row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

GAP_DIR = Path(__file__).resolve().parents[1] / "augmented_forman_gap"
if str(GAP_DIR) not in sys.path:
    sys.path.insert(0, str(GAP_DIR))

import af3  # noqa: E402


def _summarise(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None,
                "negative_fraction": None, "edges": 0}
    p10, p90 = np.quantile(values, [0.1, 0.9])
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(p10), "p90": float(p90),
            "negative_fraction": float((values < 0).mean()), "edges": int(values.size)}


def intra_features(graph: sp.csr_matrix, labels: np.ndarray) -> dict:
    occupied = labels[labels >= 0]
    groups = np.unique(occupied)
    per_group = []
    singleton = no_edge = 0
    total_edges = total_negative = total_triangles = 0.0
    af3_group_means, forman_group_means = [], []
    af3_group_edge_fractions, forman_group_edge_fractions = [], []
    af3_edge_values, forman_edge_values = [], []
    for value in groups:
        nodes = np.nonzero(labels == value)[0]
        if nodes.size < 2:
            singleton += 1
            continue
        inner = graph[nodes][:, nodes]
        if inner.nnz == 0:
            no_edge += 1
            continue
        curvature = af3.edge_curvatures(inner)
        edges = curvature["af3"].size
        af3_negative = int((curvature["af3"] < 0).sum())
        forman_negative = int((curvature["forman"] < 0).sum())
        af3_mean = float(curvature["af3"].mean())
        forman_mean = float(curvature["forman"].mean())
        triangles_mean = float(curvature["triangles"].mean()) if edges else None
        per_group.append({"group": int(value), "items": int(nodes.size), "edges": edges,
                          "af3_mean": af3_mean, "forman_mean": forman_mean,
                          "af3_negative_edges": af3_negative,
                          "forman_negative_edges": forman_negative,
                          "triangles_mean": triangles_mean})
        af3_group_means.append(af3_mean)
        forman_group_means.append(forman_mean)
        af3_group_edge_fractions.append(af3_negative / edges)
        forman_group_edge_fractions.append(forman_negative / edges)
        total_edges += edges
        total_negative += af3_negative
        total_triangles += float(curvature["triangles"].sum())
        af3_edge_values.append(curvature["af3"])
        forman_edge_values.append(curvature["forman"])
    af3_group_means = np.asarray(af3_group_means, dtype=float)
    forman_group_means = np.asarray(forman_group_means, dtype=float)
    af3_edge_values = np.concatenate(af3_edge_values) if af3_edge_values else np.zeros(0)
    forman_edge_values = (np.concatenate(forman_edge_values) if forman_edge_values
                          else np.zeros(0))
    expected_edges = 0
    if total_edges:
        coo = graph.tocoo()
        keep = coo.row < coo.col
        left, right = labels[coo.row[keep]], labels[coo.col[keep]]
        expected_edges = int(((left >= 0) & (left == right)).sum())
    return {
        "groups": int(groups.size),
        "eligible_groups": int(af3_group_means.size),
        "singleton_groups": singleton,
        "no_edge_groups": no_edge,
        "edges": int(total_edges),
        "edge_counts_consistent": bool(int(total_edges) == expected_edges),
        "group_mean_negative_fraction": (float((af3_group_means < 0).mean())
                                         if af3_group_means.size else None),
        "edge_negative_fraction": (total_negative / total_edges
                                   if total_edges else None),
        "mean_group_negative_edge_fraction": (float(np.mean(af3_group_edge_fractions))
                                              if af3_group_edge_fractions else None),
        "forman_group_mean_negative_fraction": (
            float((forman_group_means < 0).mean()) if forman_group_means.size else None),
        "forman_edge_negative_fraction": (
            float((forman_edge_values < 0).mean()) if forman_edge_values.size else None),
        "forman_mean_group_negative_edge_fraction": (
            float(np.mean(forman_group_edge_fractions))
            if forman_group_edge_fractions else None),
        "af3_mean": float(af3_group_means.mean()) if af3_group_means.size else None,
        "af3_edge_weighted": (float(af3_edge_values.mean())
                              if af3_edge_values.size else None),
        "forman_mean": float(forman_group_means.mean()) if forman_group_means.size else None,
        "triangles_mean": (float(total_triangles / total_edges)
                           if total_edges else None),
        "af3_minus_forman": (float(af3_edge_values.mean() - forman_edge_values.mean())
                             if total_edges else None),
        "af3_median": (float(np.median(af3_edge_values)) if af3_edge_values.size else None),
        "af3_p10": (float(np.quantile(af3_edge_values, 0.1))
                    if af3_edge_values.size else None),
        "af3_p90": (float(np.quantile(af3_edge_values, 0.9))
                    if af3_edge_values.size else None),
        "forman_edge_weighted": (float(forman_edge_values.mean())
                                 if forman_edge_values.size else None),
        "per_group": per_group,
        "status": "DEFINED" if af3_group_means.size else "UNDEFINED_NO_EDGES"}


def gap_block(curvature: dict, labels: np.ndarray) -> dict:
    """Within/cross edge gap on the fixed original-edge curvature cache."""
    rows, cols = curvature["rows"], curvature["cols"]
    left, right = labels[rows], labels[cols]
    unlabelled = (left < 0) | (right < 0)
    eligible = ~unlabelled
    within = eligible & (left == right)
    cross = eligible & (left != right)
    block = {"edges_total": int(rows.size), "edges_unlabelled": int(unlabelled.sum()),
             "edges_eligible": int(eligible.sum()), "edges_within": int(within.sum()),
             "edges_cross": int(cross.sum())}
    if within.sum() == 0:
        block.update({"status": "UNDEFINED_INSUFFICIENT_WITHIN_EDGES",
                      "gap_af3": None, "gap_forman": None})
        return block
    if cross.sum() == 0:
        block.update({"status": "UNDEFINED_INSUFFICIENT_CROSS_EDGES",
                      "gap_af3": None, "gap_forman": None})
        return block
    af3_values, forman_values, triangles = (curvature["af3"], curvature["forman"],
                                            curvature["triangles"])
    inside = _summarise(af3_values[within])
    outside = _summarise(af3_values[cross])
    inside_f = _summarise(forman_values[within])
    outside_f = _summarise(forman_values[cross])
    block.update({
        "status": "DEFINED",
        "gap_af3": inside["mean"] - outside["mean"],
        "gap_forman": inside_f["mean"] - outside_f["mean"],
        "within_mean": inside["mean"], "cross_mean": outside["mean"],
        "within_median": inside["median"], "cross_median": outside["median"],
        "within_negative_fraction": inside["negative_fraction"],
        "cross_negative_fraction": outside["negative_fraction"],
        "forman_within_mean": inside_f["mean"], "forman_cross_mean": outside_f["mean"],
        "triangles_within_mean": float(triangles[within].mean()),
        "triangles_cross_mean": float(triangles[cross].mean()),
        "triangle_increment": float(3.0 * (triangles[within].mean()
                                         - triangles[cross].mean())),
        "edge_retention": block["edges_within"] / block["edges_eligible"]})
    if abs(block["gap_af3"] - block["gap_forman"] - block["triangle_increment"]) > 1e-9:
        block["status"] = "IDENTITY_VIOLATION"
    return block


def quotient_features(matrix: sp.csr_matrix) -> dict:
    if matrix.nnz == 0:
        return {"status": "UNDEFINED_EMPTY_QUOTIENT"}
    curvature = af3.edge_curvatures(matrix)
    af3_values, forman_values = curvature["af3"], curvature["forman"]
    degree = curvature["degree"]
    nodes = int(degree.size)
    edges = int(af3_values.size)
    wedges = float((degree.astype(np.float64) * (degree - 1.0)).sum() / 2.0)
    triangles = float(curvature["triangles"].sum() / 3.0)
    block = _summarise(af3_values)
    forman = _summarise(forman_values)
    return {"status": "DEFINED", "nodes": nodes, "edges": edges,
            "density": edges / (nodes * (nodes - 1) / 2.0) if nodes > 1 else None,
            "transitivity": (3.0 * triangles / wedges) if wedges > 0 else None,
            "af3_mean": block["mean"], "af3_median": block["median"],
            "af3_p10": block["p10"], "af3_p90": block["p90"],
            "af3_negative_edge_fraction": block["negative_fraction"],
            "forman_mean": forman["mean"], "forman_negative_edge_fraction":
            forman["negative_fraction"],
            "af3_minus_forman": block["mean"] - forman["mean"],
            "triangles_mean": float(curvature["triangles"].mean())}
