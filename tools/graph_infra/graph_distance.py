"""Graph distances on the visible directed graph (issue #245 §4, C12).

Two objects, deliberately separate names, never interchangeable:

``bfs_distances``       shortest paths on the **undirected projection of the whole graph**;
                        the target ``g`` of every distortion metric.
``subgraph_distances``  shortest paths **restricted to one induced subgraph's node set**;
                        used when a local subgraph is evaluated on its own.

``reference_bfs`` is a plain Python breadth-first search used only by the tests: it is the
definition, not a faster copy of the same code.

Unreachable pairs are ``inf`` and must stay masked; nothing here imputes a distance.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph


def undirected_projection(csr: sp.csr_matrix) -> sp.csr_matrix:
    """Simple undirected projection: an edge exists if either direction does."""
    projection = (csr + csr.T).tocsr()
    projection.data[:] = 1
    return projection


def bfs_distances(undirected: sp.csr_matrix, roots, depth: int | None = None) -> np.ndarray:
    """Unweighted distances from each root; shape (len(roots), n), ``inf`` where unreachable."""
    roots = np.atleast_1d(np.asarray(roots, dtype=np.int64))
    distance = csgraph.shortest_path(undirected, method="D", unweighted=True, indices=roots)
    distance = np.atleast_2d(distance).astype(np.float64)
    if depth is not None:
        distance[distance > depth] = np.inf
    return distance


def subgraph_distances(nodes, out_csr: sp.csr_matrix, depth: int | None = None) -> np.ndarray:
    """Distances restricted to the induced subgraph on ``nodes`` (undirected inside)."""
    nodes = [int(n) for n in nodes]
    index = {node: k for k, node in enumerate(nodes)}
    rows, cols, data = [], [], []
    for node in nodes:
        start, end = out_csr.indptr[node], out_csr.indptr[node + 1]
        for target in out_csr.indices[start:end]:
            target = int(target)
            if target in index:
                rows.append(index[node]); cols.append(index[target]); data.append(1.0)
                rows.append(index[target]); cols.append(index[node]); data.append(1.0)
    if not rows:
        return np.full((len(nodes), len(nodes)), np.inf)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    distance = csgraph.shortest_path(matrix, method="D", unweighted=True).astype(np.float64)
    if depth is not None:
        distance[distance > depth] = np.inf
    return distance


def reference_bfs(edges, n_nodes: int, root: int, depth: int | None = None) -> dict:
    """Plain BFS from the edge list; the definition the fast path is checked against."""
    adjacency = [[] for _ in range(n_nodes)]
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        if int(right) != int(left):
            adjacency[int(right)].append(int(left))
    seen = {int(root): 0}
    frontier = [int(root)]
    while frontier:
        nxt = []
        for node in frontier:
            if depth is not None and seen[node] >= depth:
                continue
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen[neighbour] = seen[node] + 1
                    nxt.append(neighbour)
        frontier = nxt
    return seen


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean over the valid entries only; ``None`` when nothing is valid."""
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    return float(values[mask].mean())


def ppr(csr: sp.csr_matrix, source: int, restart: float = 0.15, tolerance: float = 1e-8,
        max_iterations: int = 200, top_k: int | None = 50) -> dict:
    """Personalised PageRank with a fixed restart, tolerance and truncation.

    PPR is a propagation score, not a distance: it is returned with its own name and unit so no
    caller can report it as a geometric distance.  Convergence is declared on the L1 change.
    """
    if not 0.0 < restart < 1.0:
        raise ValueError("restart must be in (0, 1)")
    n = csr.shape[0]
    row_sums = np.asarray(csr.sum(axis=1)).ravel()
    inverse = np.where(row_sums > 0, 1.0 / np.maximum(row_sums, 1e-30), 0.0)
    transition = sp.diags(inverse) @ csr
    teleport = np.zeros(n)
    teleport[int(source)] = 1.0
    dangling = row_sums == 0
    vector = teleport.copy()
    iterations = 0
    delta = float("inf")
    for iterations in range(1, max_iterations + 1):
        updated = (1.0 - restart) * (transition.T @ vector) + restart * teleport
        # a dangling row would silently destroy probability mass; it is returned to the source
        dangling_mass = float(vector[dangling].sum()) if dangling.any() else 0.0
        if dangling_mass > 0.0:
            updated = updated + (1.0 - restart) * dangling_mass * teleport
        delta = float(np.abs(updated - vector).sum())
        vector = updated
        if delta < tolerance:
            break
    order = np.argsort(-vector)
    if top_k is not None:
        order = order[: int(top_k)]
    scores = vector[order]
    keep = scores > 0
    order, scores = order[keep], scores[keep]
    return {"nodes": order.astype(np.int64), "scores": scores.astype(np.float64),
            "iterations": int(iterations), "l1_delta": delta,
            "converged": bool(delta < tolerance), "restart": float(restart),
            "tolerance": float(tolerance), "unit": "propagation score (not a distance)",
            "source": int(source)}


def pair_metrics(model_distance, graph_hop, upper_mask=None) -> dict:
    """Distance diagnostics on exactly the pairs the caller marked valid.

    Full-graph and subgraph distances must be passed under separate names by the caller; this
    helper only computes on the arrays it is given and reports the mask it used.
    """
    d = np.asarray(model_distance, dtype=np.float64)
    g = np.asarray(graph_hop, dtype=np.float64)
    if d.shape != g.shape:
        raise ValueError("model distance and graph hop arrays differ in shape")
    mask = np.isfinite(d) & np.isfinite(g)
    if upper_mask is not None:
        mask &= np.asarray(upper_mask, dtype=bool)
    if not mask.any():
        return {"n": 0, "unit": "graph hops"}
    difference = d[mask] - g[mask]
    return {
        "n": int(mask.sum()),
        "unit": "graph hops",
        "mean_model": float(d[mask].mean()),
        "mean_graph": float(g[mask].mean()),
        "bias": float(difference.mean()),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt((difference ** 2).mean())),
        "spearman": None,
        "note": "monotonicity is reported by metrics.spearman; masked pairs are excluded, not zero-filled",
        "masked_out": int((~mask).sum()),
    }
