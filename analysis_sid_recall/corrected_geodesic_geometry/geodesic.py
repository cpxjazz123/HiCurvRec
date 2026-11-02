"""True-geodesic Ollivier-Ricci curvature and exact boundary counting (Issue #134).

Defects fixed relative to #133:

* **D1** — the transport ground cost was the 0/1/2-clipped metric instead of the
  true shortest-path distance in the full quotient graph. For support vertices of
  an adjacent pair the true distance is at most 3, so the exact cost matrix is
  built with adjacency (1 hop), a full-graph 2-hop product (2 hops, paths may
  pass outside the support) and 3 for everything else. The Wasserstein distance is
  then computed exactly for costs in {0, 1, 2, 3} by three max flows:

      W1 = 3*T - F0 - F1 - F2

  with ``F_k`` the maximum mass matchable using arcs of cost <= k. ``w1_hop12``
  survives only as the explicitly *clipped-metric diagnostic*.

* **D2** — ``cut = sum_{i in S} deg_B(i) - 2*E_B(S)`` already counts every
  undirected crossing edge exactly once, so the reported boundary is ``cut``, not
  ``cut / 2``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

MAX_DISTANCE = 3
SCALE = 10 ** 9


def adjacency(binary: sp.csr_matrix) -> sp.csr_matrix:
    return binary


def support_distances(graph: sp.csr_matrix, support: np.ndarray) -> np.ndarray:
    """True shortest-path distances among ``support`` nodes, capped at 3.

    Any support pair is at distance at most 3 because both endpoints are within
    one hop of the adjacent pair's ends; the 2-hop values use a full-graph
    product, so paths through nodes outside the support are included.
    """
    support = np.asarray(support, dtype=np.int64)
    size = support.size
    block = graph[support][:, support].tocsr()
    two_hop = (graph[support, :] @ graph[:, support]).tocsr()
    distance = np.full((size, size), 3, dtype=np.int64)
    np.fill_diagonal(distance, 0)
    rows, cols = block.nonzero()
    distance[rows, cols] = 1
    rows, cols = two_hop.nonzero()
    current = distance[rows, cols]
    distance[rows, cols] = np.where(current > 2, 2, current)
    np.fill_diagonal(distance, 0)
    return distance


def lazy_measure(graph: sp.csr_matrix, node: int, position: dict) -> np.ndarray:
    size = len(position)
    weight = np.zeros(size)
    index = position.get(int(node))
    if index is not None:
        weight[index] += 0.5
    neighbours = graph.indices[graph.indptr[node]:graph.indptr[node + 1]]
    degree = max(float(neighbours.size), 1.0)
    for neighbour in neighbours.tolist():
        target = position.get(int(neighbour))
        if target is not None:
            weight[target] += 0.5 / degree
    total = weight.sum()
    if total <= 0:
        raise ValueError("a node without a lazy measure")
    return weight / total


def _max_flow(left: np.ndarray, right: np.ndarray, arcs: list[tuple[int, int]]) -> int:
    """Maximum mass matchable on ``arcs`` with the marginals, in scaled integers."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    size = left.size
    source, sink = 0, 2 * size + 1
    rows, cols, capacity = [], [], []
    for index in range(size):
        if left[index] > 0:
            rows.append(source); cols.append(1 + index); capacity.append(int(left[index]))
        if right[index] > 0:
            rows.append(size + 1 + index); cols.append(sink); capacity.append(int(right[index]))
    for a, b in arcs:
        if left[a] > 0 and right[b] > 0:
            rows.append(1 + a); cols.append(size + 1 + b)
            capacity.append(int(min(left[a], right[b])))
    if not rows:
        return 0
    graph = csr_matrix((np.asarray(capacity, dtype=np.int64), (rows, cols)),
                       shape=(sink + 1, sink + 1))
    return int(maximum_flow(graph, source, sink).flow_value)


def w1_exact(left: np.ndarray, right: np.ndarray, distance: np.ndarray) -> float:
    """Exact 1-Wasserstein for integer costs in {0, 1, 2, 3} (sparse LP).

    The 0/1/2-clipped max-flow identity does not generalise to three cost classes
    (a single transport must optimise all thresholds at once), so the exact value
    is solved as a sparse transportation LP. The constraint matrix has two
    nonzeros per variable, so ``highs`` handles supports of a few hundred nodes.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, vstack

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    size = left.size
    if size == 0:
        return 0.0
    cost = distance.astype(float).reshape(-1)
    rows = np.repeat(np.arange(size), size)          # row constraints
    cols = np.tile(np.arange(size), size)            # column constraints
    n = size * size
    indices = np.arange(n)
    row_matrix = csr_matrix((np.ones(n), (rows, indices)), shape=(size, n))
    col_matrix = csr_matrix((np.ones(n), (cols, indices)), shape=(size, n))
    matrix = vstack([row_matrix, col_matrix]).tocsr()
    rhs = np.concatenate([left, right])
    solution = linprog(cost, A_eq=matrix, b_eq=rhs, bounds=(0, None), method="highs")
    if not solution.success:
        raise ValueError("the transport LP did not solve: " + str(solution.message))
    return float(cost @ solution.x)


def orc_edge(graph: sp.csr_matrix, u: int, v: int) -> dict:
    """True-geodesic ORC of one quotient edge (costs up to 3, exact W1)."""
    support = np.unique(np.concatenate((
        np.asarray([u]), graph.indices[graph.indptr[u]:graph.indptr[u + 1]],
        np.asarray([v]), graph.indices[graph.indptr[v]:graph.indptr[v + 1]])))
    position = {int(node): index for index, node in enumerate(support.tolist())}
    distance = support_distances(graph, support)
    left = lazy_measure(graph, u, position)
    right = lazy_measure(graph, v, position)
    value = 1.0 - w1_exact(left, right, distance)
    return {"curvature": float(value), "support": int(support.size),
            "max_distance": int(distance.max()),
            "distance_three_pairs": int((distance == 3).sum())}


def orc_edge_clipped(graph: sp.csr_matrix, u: int, v: int) -> float:
    """The #133 estimator: 0/1/2-clipped cost. Diagnostic only, never ORC."""
    support = np.unique(np.concatenate((
        np.asarray([u]), graph.indices[graph.indptr[u]:graph.indptr[u + 1]],
        np.asarray([v]), graph.indices[graph.indptr[v]:graph.indptr[v + 1]])))
    position = {int(node): index for index, node in enumerate(support.tolist())}
    inside = graph[support][:, support].tocsr()
    left = lazy_measure(graph, u, position)
    right = lazy_measure(graph, v, position)
    size = support.size
    cost = np.full((size, size), 2)
    np.fill_diagonal(cost, 0)
    rows, cols = inside.nonzero()
    cost[rows, cols] = 1
    np.fill_diagonal(cost, 0)
    return float(1.0 - _w1_hop12(left, right, inside))


def _w1_hop12(left: np.ndarray, right: np.ndarray, inside: sp.csr_matrix) -> float:
    """The clipped 0/1/2 identity, kept for the diagnostic comparison."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    size = left.size
    left_scaled = np.rint(left * SCALE).astype(np.int64)
    right_scaled = np.rint(right * SCALE).astype(np.int64)
    total = int(min(left_scaled.sum(), right_scaled.sum()))
    if total <= 0:
        return 0.0
    zero_cost = np.minimum(left_scaled, right_scaled)
    m0 = int(zero_cost.sum())
    left_residual = left_scaled - zero_cost
    right_residual = right_scaled - zero_cost
    source, sink = 0, 2 * size + 1
    rows, cols, capacity = [], [], []
    for index in range(size):
        if left_residual[index] > 0:
            rows.append(source); cols.append(1 + index); capacity.append(int(left_residual[index]))
        if right_residual[index] > 0:
            rows.append(size + 1 + index); cols.append(sink); capacity.append(int(right_residual[index]))
        if left_residual[index] <= 0:
            continue
        for neighbour in inside.indices[inside.indptr[index]:inside.indptr[index + 1]]:
            rows.append(1 + index); cols.append(size + 1 + int(neighbour))
            capacity.append(total)
    graph = csr_matrix((np.asarray(capacity, dtype=np.int64), (rows, cols)),
                       shape=(sink + 1, sink + 1))
    one_cost = int(maximum_flow(graph, source, sink).flow_value)
    return float(max(0.0, (2 * total - 2 * m0 - one_cost)) / SCALE)


# --------------------------------------------------------------- boundaries
def group_cut(binary: sp.csr_matrix, members: np.ndarray,
              degree: np.ndarray) -> tuple[float, int]:
    """``(cut, crossing_edges)`` with the crossing edges enumerated for the check."""
    block = binary[members][:, members].tocsr()
    internal = int(block.nnz // 2)
    cut = float(degree[members].sum() - 2 * internal)
    inside = np.zeros(binary.shape[0], dtype=bool)
    inside[members] = True
    coo = binary.tocoo()
    crossing = int((inside[coo.row] & ~inside[coo.col]).sum())
    return cut, crossing


def cross_prefix_edges(binary: sp.csr_matrix, labels: np.ndarray) -> int:
    """The number of original item edges whose endpoints are in different prefixes."""
    coo = binary.tocoo()
    keep = (labels[coo.row] >= 0) & (labels[coo.col] >= 0)
    return int((keep & (labels[coo.row] != labels[coo.col])).sum() // 2)
