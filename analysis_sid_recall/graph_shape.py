"""Is the train behaviour graph tree-like, community-like, or both at once?

The graph is the TRAIN behaviour graph: an undirected edge for every consecutive
item pair inside a user's history, deduplicated.  Every edge is kept -- sampling
would drop edges and triangles and change the structural readings, and the graph
(it is a quarter of a million edges) fits in memory as it is.  No SID is
involved, so this describes the data every Stage 2 arm is asked to quantize.

Readings
--------
clustering   local clustering coefficient per node (triangles / wedges)
edges        share of edges that sit in no triangle, in >=3, in >=5
cores        k-core sizes by iterative peeling
concentration share of all triangles held by the top 1% of nodes

    python3 analysis_sid_recall/graph_shape.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kurtosis

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
OUT = Path(__file__).resolve().parent / "graph_shape.json"


def edges() -> np.ndarray:
    frame = pd.read_parquet(TRAIN)
    chunks = []
    for history in frame["history"]:
        if history is None or len(history) < 2:
            continue
        items = np.asarray([int(x) for x in history])
        chunks.append(np.stack([items[:-1], items[1:]], axis=1))
    pairs = np.unique(np.sort(np.concatenate(chunks), axis=1), axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]


def shape(pairs: np.ndarray) -> dict:
    nodes, inverse = np.unique(pairs, return_inverse=True)
    index = inverse.reshape(-1, 2)
    n = len(nodes)
    A = sparse.coo_matrix((np.ones(len(index)), (index[:, 0], index[:, 1])),
                          shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    S = A @ A
    edge_tri = np.asarray(S[index[:, 0], index[:, 1]]).ravel()
    node_tri = np.asarray(S.multiply(A).sum(1)).ravel()   # (A^3)_vv = 2 triangles at v
    with np.errstate(invalid="ignore", divide="ignore"):
        clustering = np.where(deg >= 2, node_tri / (deg * (deg - 1.0)), np.nan)
    values = clustering[~np.isnan(clustering)]
    core = deg.copy().astype(int)
    alive = np.ones(n, bool)
    level = 0
    while alive.any():
        level += 1
        while True:
            alive_idx = np.flatnonzero(alive)
            d = np.asarray(A[alive][:, alive].sum(1)).ravel()
            drop = alive_idx[d < level]
            if not len(drop):
                break
            alive[drop] = False
        core[alive] = level
    components, labels = sparse.csgraph.connected_components(A, directed=False)
    share = node_tri / node_tri.sum()
    top = np.argsort(-node_tri)[: max(1, int(0.01 * n))]
    hubs, leaves = deg >= 50, deg <= 3
    return {
        "nodes": int(n), "edges": int(len(index)),
        "mean_degree": float(deg.mean()), "median_degree": float(np.median(deg)),
        "clustering_mean": float(values.mean()), "clustering_median": float(np.median(values)),
        "clustering_percentiles": {("p%d" % q): float(np.percentile(values, q))
                                   for q in (10, 25, 50, 75, 90, 99)},
        "clustering_kurtosis": float(kurtosis(values)),
        "share_clustering_zero": float((values == 0).mean()),
        "share_clustering_below_0_1": float(((values > 0) & (values < 0.1)).mean()),
        "share_clustering_0_1_to_0_3": float(((values >= 0.1) & (values < 0.3)).mean()),
        "share_clustering_at_least_0_3": float((values >= 0.3).mean()),
        "share_edges_no_triangle": float((edge_tri == 0).mean()),
        "share_edges_ge3_triangles": float((edge_tri >= 3).mean()),
        "share_edges_ge5_triangles": float((edge_tri >= 5).mean()),
        "max_kcore": int(core.max()),
        "share_core_ge5": float((core >= 5).mean()),
        "share_core_ge10": float((core >= 10).mean()),
        "share_triangles_top1pct_nodes": float(share[top].sum()),
        "clustering_hubs_deg_ge50": float(np.nanmean(clustering[hubs])) if hubs.any() else None,
        "clustering_leaves_deg_le3": float(np.nanmean(clustering[leaves])) if leaves.any() else None,
        "components": int(components),
        "share_nodes_in_largest_component": float(np.bincount(labels).max() / n),
    }


def main() -> None:
    readings = shape(edges())
    OUT.write_text(json.dumps(readings, indent=2) + "\n")
    for key in ("nodes", "edges", "clustering_mean", "clustering_median",
                "share_clustering_zero", "share_clustering_at_least_0_3",
                "share_edges_no_triangle", "share_edges_ge5_triangles",
                "max_kcore", "share_triangles_top1pct_nodes",
                "clustering_hubs_deg_ge50", "clustering_leaves_deg_le3",
                "share_nodes_in_largest_component"):
        print("%-32s %s" % (key, readings[key]))


if __name__ == "__main__":
    main()
