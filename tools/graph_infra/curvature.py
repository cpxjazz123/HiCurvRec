"""Exact Ollivier–Ricci curvature on a graph, for a fixed reference set (issue #246 C23).

    W1(μ_x, μ_y) = min over couplings π of E_{(u,v)~π}[d(u,v)]
    κ(x, y)      = 1 − W1(μ_x, μ_y) / d(x, y)

with uniform neighbour measures.  The transport problem is solved exactly by a small LP; the
reference implementation in the tests enumerates integer plans instead (independent route).
"""
from __future__ import annotations

import numpy as np
import scipy.optimize as optimize
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


def neighbour_sets(undirected: sp.csr_matrix, node: int) -> np.ndarray:
    start, end = undirected.indptr[int(node)], undirected.indptr[int(node) + 1]
    return undirected.indices[start:end].astype(np.int64)


def local_distances(undirected: sp.csr_matrix, sources, limit: int = 4) -> dict:
    """Hop distances from each source, restricted to ``limit`` hops (small neighbourhoods)."""
    result = {}
    for source in np.asarray(sources).tolist():
        hop = {int(source): 0}
        frontier = [int(source)]
        for depth in range(1, limit + 1):
            nxt = []
            for node in frontier:
                for neighbour in neighbour_sets(undirected, node).tolist():
                    if neighbour not in hop:
                        hop[neighbour] = depth
                        nxt.append(neighbour)
            frontier = nxt
            if not frontier:
                break
        result[int(source)] = hop
    return result


def transport_w1(left: np.ndarray, right: np.ndarray, distance: np.ndarray) -> float:
    """Exact W1 between two uniform measures; rows and columns sum to 1/n and 1/m."""
    n, m = len(left), len(right)
    cost = np.asarray(distance, dtype=np.float64).ravel()
    rows, cols, values = [], [], []
    for i in range(n):
        for j in range(m):
            rows.append(i)
            cols.append(j)
            values.append(1.0)
    a_eq = np.zeros((n + m, n * m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            a_eq[i, i * m + j] = 1.0
            a_eq[n + j, i * m + j] = 1.0
    b_eq = np.concatenate([np.full(n, 1.0 / n), np.full(m, 1.0 / m)])
    result = optimize.linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"transport LP failed: {result.message}")
    return float(result.fun)


def orc_edge(undirected: sp.csr_matrix, left: int, right: int, max_neighbours: int = 64,
             seed: int = 0) -> float | None:
    """κ for one undirected edge; ``None`` when an endpoint has no neighbour (dangling).

    The neighbour sets are capped at ``max_neighbours`` per side (a deterministic seeded subset),
    because a hub with 1,295 neighbours would make the transport problem a million-variable LP.
    The cap is a documented approximation and is reported with the curvature summary.
    """
    first = neighbour_sets(undirected, left)
    second = neighbour_sets(undirected, right)
    if len(first) > max_neighbours or len(second) > max_neighbours:
        rng = np.random.default_rng(seed + int(left) * 1000003 + int(right))
        if len(first) > max_neighbours:
            first = np.sort(rng.choice(first, size=max_neighbours, replace=False))
        if len(second) > max_neighbours:
            second = np.sort(rng.choice(second, size=max_neighbours, replace=False))
    if first.size == 0 or second.size == 0:
        return None
    # the cost matrix needs the hop distance between the two neighbour sets; scipy's C
    # implementation is used because a python walk would sweep the whole component per pair
    rows_second = dijkstra(undirected, directed=False, unweighted=True, indices=second)
    matrix = np.full((first.size, second.size), np.inf)
    for i, a in enumerate(first.tolist()):
        matrix[i, :] = rows_second[:, a]
    if not np.all(np.isfinite(matrix)):
        return None
    distance = 1.0 if undirected[left, right] > 0 else float("nan")
    w1 = transport_w1(first, second, matrix)
    return 1.0 - w1 / distance


def node_curvature(undirected: sp.csr_matrix, nodes, max_edges_per_node: int, seed: int) -> dict:
    """Mean κ over a frozen sample of incident edges for each requested node."""
    rng = np.random.default_rng(seed)
    results = {}
    sampled_edges = []
    for node in np.asarray(nodes).tolist():
        neighbours = neighbour_sets(undirected, int(node))
        if neighbours.size == 0:
            results[int(node)] = {"mean": None, "edges": 0, "reason": "isolated"}
            continue
        take = min(max_edges_per_node, neighbours.size)
        chosen = rng.choice(neighbours, size=take, replace=False) if take < neighbours.size else neighbours
        values = []
        for other in np.sort(chosen).tolist():
            value = orc_edge(undirected, int(node), int(other), seed=seed + int(node))
            sampled_edges.append((int(node), int(other)))
            if value is not None:
                values.append(value)
        results[int(node)] = {"mean": float(np.mean(values)) if values else None,
                              "edges": len(values),
                              "reason": None if values else "no finite transport"}
    return {"nodes": results, "sampled_edges": len(sampled_edges)}


def summarise(values) -> dict:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "positive": 0, "negative": 0, "near_zero": 0}
    array = np.asarray(values)
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "positive": int((array > 0.05).sum()),
        "negative": int((array < -0.05).sum()),
        "near_zero": int((np.abs(array) <= 0.05).sum()),
        "min": float(array.min()),
        "max": float(array.max()),
        "unit": "dimensionless (1 - W1/hop)",
    }
