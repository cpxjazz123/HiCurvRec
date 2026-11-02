"""Issue #135: aggregation/conditioning/normalisation variants of the SAME geometry.

All functions take a binary, symmetric, self-loop-free quotient graph (CSR) and
return plain scalars with an explicit coverage/status. Nothing here reads the
Recall labels; the candidate registry is frozen in ``candidates.py``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from candidates import NULL_FEATURES, NULL_REPLICATES


def degree_of(matrix: sp.csr_matrix) -> np.ndarray:
    return np.asarray(matrix.sum(axis=1)).ravel()


def _triangles_at_nodes(matrix: sp.csr_matrix, degree: np.ndarray) -> np.ndarray:
    """t_v = edges among N(v); the wedge sum over the edges at v is 2 t_v."""
    squared = (matrix @ matrix).multiply(matrix)
    wedge = np.asarray(squared.sum(axis=1)).ravel()
    values = wedge / 2.0
    rounded = np.rint(values)
    return np.where(np.abs(values - rounded) < 1e-6, rounded, values)


def _quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None, "iqr": None}
    p10, p90 = np.quantile(values, [0.1, 0.9])
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(p10), "p90": float(p90), "iqr": float(p90 - p10)}


def local_block(matrix: sp.csr_matrix, degree: np.ndarray) -> dict:
    triangles = _triangles_at_nodes(matrix, degree)
    nodes = int(matrix.shape[0])
    clustering = np.full(nodes, np.nan)
    ego_density = np.full(nodes, np.nan)
    several = degree >= 2
    clustering[several] = (2.0 * triangles[several]
                           / (degree[several] * (degree[several] - 1.0)))
    one_plus = degree >= 1
    ego_density[one_plus] = (2.0 * (degree[one_plus] + triangles[one_plus])
                             / (degree[one_plus] * (degree[one_plus] + 1.0)))
    stats = _quantiles(clustering[np.isfinite(clustering)])
    ego = _quantiles(ego_density[np.isfinite(ego_density)])
    return {"local_transitivity_mean": stats["mean"],
            "local_transitivity_median": stats["median"],
            "local_transitivity_p10": stats["p10"],
            "local_transitivity_p90": stats["p90"],
            "local_transitivity_iqr": stats["iqr"],
            "local_transitivity_coverage": float(several.mean()) if nodes else 0.0,
            "triangles_per_node_mean": float(triangles.mean()) if nodes else None,
            "triangle_incidence_sum": float(3.0 * triangles.sum()),
            "ego_density_mean": ego["mean"], "ego_density_median": ego["median"],
            "ego_density_coverage": float(one_plus.mean()) if nodes else 0.0,
            "ego_cycle_surplus_mean": float(triangles.mean()) if nodes else None,
            "ego_cycle_surplus_median": float(np.median(triangles)) if nodes else None}


def cycle_block(matrix: sp.csr_matrix, degree: np.ndarray) -> dict:
    nodes = int(matrix.shape[0])
    edges = int(matrix.nnz // 2)
    if nodes == 0:
        return {"cycle_rank_per_node": None, "cycle_rank_per_edge": None,
                "cycle_surplus_per_component_mean": None,
                "cycle_surplus_per_component_median": None,
                "cycle_surplus_per_component_coverage": 0.0, "components": 0}
    count, labels = connected_components(matrix, directed=False)
    surplus = []
    for component in range(count):
        members = np.nonzero(labels == component)[0]
        if members.size < 2:
            continue
        inner = float(degree[members].sum()) / 2.0
        surplus.append(inner - members.size + 1.0)
    surplus = np.asarray(surplus)
    return {"cycle_rank_per_node": (edges - nodes + int(count)) / nodes,
            "cycle_rank_per_edge": (edges - nodes + int(count)) / edges if edges else None,
            "cycle_surplus_per_component_mean":
                float(surplus.mean()) if surplus.size else None,
            "cycle_surplus_per_component_median":
                float(np.median(surplus)) if surplus.size else None,
            "cycle_surplus_per_component_coverage":
                float(surplus.size / max(count, 1)),
            "components": int(count)}


def degree_block(degree: np.ndarray) -> dict:
    if degree.size == 0:
        return {"degree_mean": None, "degree_p10": None, "degree_p90": None,
                "degree_median": None, "degree_iqr": None, "degree_cv": None,
                "degree_p90_over_mean": None, "degree_median_over_mean": None}
    mean = float(degree.mean())
    p10, p90 = np.quantile(degree, [0.1, 0.9])
    median = float(np.median(degree))
    deviations = float(degree.std())
    return {"degree_mean": mean, "degree_p10": float(p10), "degree_p90": float(p90),
            "degree_median": median, "degree_iqr": float(p90 - p10),
            "degree_cv": deviations / mean if mean > 0 else None,
            "degree_p90_over_mean": float(p90) / mean if mean > 0 else None,
            "degree_median_over_mean": median / mean if mean > 0 else None}


def forman_block(matrix: sp.csr_matrix, degree: np.ndarray) -> dict:
    coo = matrix.tocoo()
    keep = coo.row < coo.col
    left, right = coo.row[keep], coo.col[keep]
    if left.size == 0:
        return {"forman_mean": None, "forman_median": None, "forman_p10": None,
                "forman_p90": None, "forman_negative_fraction": None,
                "forman_node_incident_mean": None,
                "forman_node_incident_median": None, "forman_edges": 0}
    curvature = 4.0 - degree[left] - degree[right]
    sums = np.bincount(left, weights=curvature, minlength=matrix.shape[0])
    sums += np.bincount(right, weights=curvature, minlength=matrix.shape[0])
    counts = np.bincount(left, minlength=matrix.shape[0])
    counts += np.bincount(right, minlength=matrix.shape[0])
    incident = sums[counts > 0] / counts[counts > 0]
    stats = _quantiles(curvature)
    return {"forman_mean": stats["mean"], "forman_median": stats["median"],
            "forman_p10": stats["p10"], "forman_p90": stats["p90"],
            "forman_negative_fraction": float((curvature < 0).mean()),
            "forman_node_incident_mean": float(incident.mean()) if incident.size else None,
            "forman_node_incident_median": float(np.median(incident)) if incident.size else None,
            "forman_edges": int(left.size)}


def base_block(matrix: sp.csr_matrix, degree: np.ndarray) -> dict:
    nodes = int(matrix.shape[0])
    edges = int(matrix.nnz // 2)
    if nodes == 0 or edges == 0:
        return {"density": None, "transitivity": None, "triangles": None,
                "wedges": 0, "nodes": nodes, "edges": edges}
    wedges = float((degree * (degree - 1.0)).sum() / 2.0)
    triangles = float(_triangles_at_nodes(matrix, degree).sum() / 3.0)
    return {"density": edges / (nodes * (nodes - 1) / 2.0),
            "transitivity": (3.0 * triangles / wedges) if wedges > 0 else None,
            "triangles": triangles, "wedges": wedges, "nodes": nodes, "edges": edges}


def quotient_blocks(matrix: sp.csr_matrix) -> dict:
    degree = degree_of(matrix)
    record = {"base": base_block(matrix, degree)}
    record.update(cycle_block(matrix, degree))
    record.update(degree_block(degree))
    record.update(forman_block(matrix, degree))
    record.update(local_block(matrix, degree))
    return record


def induced_subgraph(matrix: sp.csr_matrix, keep: np.ndarray) -> sp.csr_matrix:
    nodes = np.nonzero(keep)[0]
    return matrix[nodes][:, nodes]


def sibling_blocks(matrix: sp.csr_matrix, parent_of_node: np.ndarray,
                   child_items: np.ndarray) -> dict:
    """Sibling quotient inside every parent; no artificial parent for L1."""
    parent_of_node = np.asarray(parent_of_node, dtype=np.int64)
    child_items = np.asarray(child_items, dtype=float)
    coo = matrix.tocoo()
    keep = coo.row < coo.col
    rows, cols = coo.row[keep], coo.col[keep]
    same = parent_of_node[rows] == parent_of_node[cols]
    rows, cols = rows[same], cols[same]
    quotient_edges = int(keep.sum())
    sibling_edges = int(rows.size)
    per_parent = []
    parent_of_edge = parent_of_node[rows] if rows.size else np.zeros(0, dtype=np.int64)
    order = np.argsort(parent_of_edge, kind="stable")
    rows, cols = rows[order], cols[order]
    parent_of_edge = parent_of_edge[order]
    for parent in np.unique(parent_of_node):
        children = np.nonzero(parent_of_node == parent)[0]
        start = int(np.searchsorted(parent_of_edge, parent, side="left"))
        stop = int(np.searchsorted(parent_of_edge, parent, side="right"))
        left = np.searchsorted(children, rows[start:stop])
        right = np.searchsorted(children, cols[start:stop])
        if left.size and (left.max() >= children.size or right.max() >= children.size):
            raise RuntimeError("sibling edge endpoint outside its parent")
        inner = sp.coo_matrix((np.ones(left.size), (left, right)),
                              shape=(children.size, children.size)).tocsr()
        inner = inner.maximum(inner.T)
        inner.setdiag(0)
        inner.eliminate_zeros()
        inner_degree = degree_of(inner)
        metrics = {"parent": int(parent), "children": int(children.size),
                   "child_items": float(child_items[children].sum())}
        metrics.update(base_block(inner, inner_degree))
        metrics.update(cycle_block(inner, inner_degree))
        metrics.update(degree_block(inner_degree))
        metrics.update(forman_block(inner, inner_degree))
        per_parent.append(metrics)

    def collect(key: str, predicate) -> np.ndarray:
        values = [entry[key] for entry in per_parent
                  if predicate(entry) and entry.get(key) is not None
                  and np.isfinite(entry[key])]
        return np.asarray(values, dtype=float)

    aggregates = {}
    for key in ("density", "transitivity", "cycle_rank_per_node", "degree_median",
                "degree_mean", "degree_p90", "forman_mean", "forman_negative_fraction"):
        if key == "density":
            eligibility = lambda entry: entry["children"] >= 2      # noqa: E731
        elif key == "transitivity":
            eligibility = lambda entry: entry["wedges"] > 0         # noqa: E731
        elif key in ("forman_mean", "forman_negative_fraction"):
            eligibility = lambda entry: entry["edges"] > 0          # noqa: E731
        else:
            eligibility = lambda entry: entry["children"] >= 1      # noqa: E731
        values = collect(key, eligibility)
        children = np.asarray([entry["children"] for entry in per_parent
                               if eligibility(entry) and entry.get(key) is not None
                               and np.isfinite(entry[key])], dtype=float)
        items = np.asarray([entry["child_items"] for entry in per_parent
                            if eligibility(entry) and entry.get(key) is not None
                            and np.isfinite(entry[key])], dtype=float)
        stats = _quantiles(values)
        aggregates[f"sibling_{key}_mean"] = stats["mean"]
        aggregates[f"sibling_{key}_median_parent"] = stats["median"]
        aggregates[f"sibling_{key}_p10"] = stats["p10"]
        aggregates[f"sibling_{key}_p90"] = stats["p90"]
        aggregates[f"sibling_{key}_child_weighted"] = (
            float(np.average(values, weights=children)) if values.size else None)
        aggregates[f"sibling_{key}_item_weighted"] = (
            float(np.average(values, weights=items)) if values.size else None)
        aggregates[f"sibling_{key}_parents"] = int(values.size)
    aggregates["sibling_edge_share"] = (sibling_edges / quotient_edges
                                       if quotient_edges else None)
    aggregates["sibling_edges"] = sibling_edges
    aggregates["quotient_edges"] = quotient_edges
    aggregates["cross_parent_edges"] = quotient_edges - sibling_edges
    aggregates["parents_total"] = len(per_parent)
    aggregates["parents_multi_child"] = int(sum(1 for entry in per_parent
                                                if entry["children"] >= 2))
    aggregates["parents_with_sibling_edge"] = int(sum(1 for entry in per_parent
                                                      if entry["edges"] > 0))
    aggregates["children_total"] = int(sum(entry["children"] for entry in per_parent))
    aggregates["child_item_coverage"] = (
        float(sum(entry["child_items"] for entry in per_parent) / max(child_items.sum(), 1)))
    return {"per_parent": per_parent, "aggregates": aggregates}


def null_labels(labels: np.ndarray, behaviour_degree: np.ndarray, design: str,
                rng: np.random.Generator,
                parent_of_node: np.ndarray | None = None) -> np.ndarray:
    """One label permutation; the caller rebuilds the quotient from it."""
    labels = np.asarray(labels, dtype=np.int64)
    out = labels.copy()
    masked = np.nonzero(labels >= 0)[0]
    if design == "global":
        out[masked] = rng.permutation(labels[masked])
    elif design == "degree_stratified":
        buckets = np.floor(np.log2(np.maximum(behaviour_degree[masked], 1.0))).astype(int)
        for bucket in np.unique(buckets):
            positions = masked[buckets == bucket]
            out[positions] = rng.permutation(labels[positions])
    elif design == "parent_conditioned":
        if parent_of_node is None:
            raise ValueError("the parent-conditioned null needs the parent partition")
        parents = parent_of_node[masked]
        for parent in np.unique(parents):
            positions = masked[parents == parent]
            out[positions] = rng.permutation(labels[positions])
    else:
        raise ValueError(f"unknown null design {design!r}")
    return out


def null_blocks(labels: np.ndarray, behaviour_degree: np.ndarray, design: str,
                build, parent_of_node: np.ndarray | None = None,
                replicates: int = NULL_REPLICATES, seed: int = 0) -> dict:
    """``build(labels)`` returns the scalar feature dict for one label vector."""
    rng = np.random.default_rng(seed)
    values = {}
    for _ in range(replicates):
        record = build(null_labels(labels, behaviour_degree, design, rng,
                                   parent_of_node))
        for key, value in record.items():
            if value is None or not np.isfinite(value):
                continue
            values.setdefault(key, []).append(float(value))
    out = {}
    for key, series in values.items():
        array = np.asarray(series, dtype=float)
        if array.size != replicates:
            out[key] = {"n": int(array.size), "status": "INCOMPLETE_NULL"}
            continue
        mean, sd = float(array.mean()), float(array.std(ddof=1))
        out[key] = {"n": int(array.size), "mean": mean, "sd": sd,
                    "mc_se": sd / np.sqrt(array.size)}
    return out


def null_relative(real: dict, nulls: dict, keys) -> dict:
    record = {}
    for key in keys:
        value = real.get(key)
        block = nulls.get(key)
        if value is None or not np.isfinite(value) or block is None or "mean" not in block:
            record[f"{key}"] = None
            continue
        mean, sd = block["mean"], block["sd"]
        record[f"{key}"] = {
            "delta": float(value) - mean,
            "z": (float(value) - mean) / sd if sd > 0 else None,
            "ratio": float(value) / mean if abs(mean) > 1e-12 else None,
            "null_mean": mean, "null_sd": sd, "null_n": block["n"],
            "null_mc_se": block.get("mc_se")}
    return record


NULL_KEYS = NULL_FEATURES
