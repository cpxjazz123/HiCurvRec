"""The undirected behaviour quotient graph and its Ollivier-Ricci curvature.

This is the authority for the curvature measurement (Issue #128). Both code
paths that measure it -- the analysis (``analysis_sid_recall/curvature_analysis.py``)
and the candidate trees (``model/curvature_allocation.py``) -- use these
semantics:

* the behaviour graph is UNDIRECTED: an item pair that co-occurs in a training
  sequence is one edge, and the edge set carries no direction from the item
  ids. A directed next-item transition graph is a different signal and does not
  use this metric;
* the prefix quotient graph is UNDIRECTED for the same reason: a group pair that
  some item pair connects is one edge;
* the ground metric of the transport problem is the EXACT unweighted shortest
  path in that quotient, restricted to the support of the two measures. A
  bounded approximation is not used: it biases the curvature by an amount that
  depends on the graph's diameter, and the null graph has a different diameter;
* a disconnected support is a FAILURE, not a zero: the curvature of a pair that
  cannot be reached is undefined and is reported as such.
"""
from __future__ import annotations

import collections
import math

import numpy as np
import scipy.sparse as sp

#: Below this curvature the hyperboloid branch is replaced by the Euclidean
#: limit, which is the same rule the ball uses for its own boundaries.
LORENTZ_EPS = 1e-6


def symmetrise(matrix: sp.spmatrix) -> sp.csr_matrix:
    """The undirected graph of a matrix that may hold one orientation per edge.

    ``A.maximum(A.T)`` keeps the larger weight of the two orientations, so an
    edge that is already symmetric is not counted twice.
    """
    out = matrix.tocsr()
    out = out.maximum(out.transpose().tocsr()).tocsr()
    out.setdiag(0)
    out.eliminate_zeros()
    return out


def build_behaviour_graph(frame) -> tuple:
    """The training split's UNDIRECTED co-occurrence graph.

    Two items are adjacent when they follow each other in some user's training
    sequence; the edge is one edge, in both directions, with the weight of the
    number of times the pair was seen in either order. No validation or test
    interaction is read.
    """
    weights: dict[tuple[int, int], float] = {}
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for first, second in zip(values[:-1], values[1:]):
            if first == second:
                continue
            key = (first, second) if first < second else (second, first)
            weights[key] = weights.get(key, 0.0) + 0.5
    nodes = sorted({item for edge in weights for item in edge})
    index = {item: position for position, item in enumerate(nodes)}
    rows, cols, data = [], [], []
    for (first, second), weight in weights.items():
        rows.append(index[first])
        cols.append(index[second])
        data.append(weight)
    graph = sp.coo_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    return nodes, index, symmetrise(graph)


def quotient_graph(graph: sp.csr_matrix, assignment: np.ndarray,
                   n_groups: int) -> sp.csr_matrix:
    """The behaviour graph contracted into the prefix groups, UNDIRECTED.

    The adjacency of the quotient is the OR of its members' adjacencies: a pair
    of groups is adjacent when some pair of their items is.
    """
    coo = graph.tocoo()
    first = assignment[coo.row]
    second = assignment[coo.col]
    keep = (first != second) & (first >= 0) & (second >= 0)
    first, second = first[keep], second[keep]
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    key = low.astype(np.int64) * np.int64(max(n_groups, 1)) + high.astype(np.int64)
    unique = np.unique(key)
    rows = (unique // max(n_groups, 1)).astype(np.int32)
    cols = (unique % max(n_groups, 1)).astype(np.int32)
    data = np.ones(rows.shape[0], dtype=np.float32)
    quotient = sp.coo_matrix((data, (rows, cols)), shape=(n_groups, n_groups))
    return symmetrise(quotient)


def exact_hop_costs(graph: sp.csr_matrix, source: int, support: np.ndarray) -> np.ndarray:
    """The EXACT unweighted shortest-path distances from ``source`` to the support.

    The distance is computed on the full undirected quotient, so it does not
    depend on the support's size or on the order of the ids. A support node that
    cannot be reached gets infinity, and the caller fails closed on it.
    """
    from scipy.sparse.csgraph import shortest_path

    unweighted = graph.tocsr()
    distances = shortest_path(unweighted, method="D", unweighted=True,
                              indices=[int(source)], directed=False)
    return distances[0][support]


def lazy_measure(graph: sp.csr_matrix, node: int, support: np.ndarray) -> np.ndarray:
    """The lazy random walk's measure at ``node``, restricted to the support."""
    degree = np.asarray(graph.sum(axis=1)).ravel()
    if degree[node] <= 0:
        raise ValueError("a node without neighbours has no lazy measure")
    neighbours = graph.indices[graph.indptr[node]:graph.indptr[node + 1]]
    weight = np.zeros(support.shape[0])
    position = {int(value): index for index, value in enumerate(support.tolist())}
    index = position.get(int(node))
    if index is not None:
        weight[index] = 0.5
    for neighbour in neighbours.tolist():
        index = position.get(int(neighbour))
        if index is not None:
            weight[index] += 0.5 / degree[node]
    total = weight.sum()
    if total <= 0:
        raise ValueError("the support does not contain the node's neighbourhood")
    return weight / total


def transport_cost(graph: sp.csr_matrix, a: int, b: int) -> float:
    """The exact 1-Wasserstein distance between the two lazy measures.

    The cost matrix is the exact hop distance between the support's cells, so it
    is symmetric, has a zero diagonal and is finite whenever the support is
    connected.
    """
    from scipy.sparse.csgraph import shortest_path

    support = np.unique(np.concatenate((
        np.asarray([a]),
        graph.indices[graph.indptr[a]:graph.indptr[a + 1]],
        np.asarray([b]),
        graph.indices[graph.indptr[b]:graph.indptr[b + 1]],
    )))
    left = lazy_measure(graph, a, support)
    right = lazy_measure(graph, b, support)
    # the EXACT distances, from every support node at once: the shortest paths
    # are computed on the full undirected quotient, not on a bounded
    # neighbourhood, so the metric does not depend on the support's size
    hops = shortest_path(graph.tocsr(), method="D", unweighted=True,
                         directed=False, indices=support.tolist())
    if not np.isfinite(hops).all():
        raise ValueError("the support is not connected: the curvature is undefined")
    # the rows are the support's own distances to every node; the support's
    # pairwise distances are the columns of the support
    cost = np.asarray(hops[:, support], dtype=np.float64)
    cost = 0.5 * (cost + cost.T)
    np.fill_diagonal(cost, 0.0)
    if (cost < 0).any():
        raise ValueError("the transport cost is negative")
    return transport(left, right, cost)


def transport(left: np.ndarray, right: np.ndarray, cost: np.ndarray) -> float:
    """The exact 1-Wasserstein distance between two measures on a cost matrix.

    The transport problem is solved as a linear program, so the value does not
    depend on an entropy regularisation (which would bias the curvature by an
    amount that depends on the graph's diameter) and no extra package is
    needed in the Stage 2 environment.
    """
    from scipy.optimize import linprog

    size = left.shape[0]
    # the variables are ordered (i, j) with the flat index i * size + j: the row
    # constraint of i covers its own block, the column constraint of j covers the
    # entries (i, j) for every i
    matrix = np.zeros((2 * size, size * size))
    matrix[:size] = np.kron(np.eye(size), np.ones((1, size)))
    matrix[size:] = np.kron(np.ones((size, 1)), np.eye(size)).T
    bounds = (0.0, None)
    solution = linprog(cost.reshape(-1), A_eq=matrix,
                       b_eq=np.concatenate([left, right]),
                       bounds=bounds, method="highs")
    if not solution.success:
        raise ValueError("the transport problem did not solve: " + str(solution.message))
    return float(cost.reshape(-1) @ solution.x)


def edge_curvature(graph: sp.csr_matrix, a: int, b: int) -> float:
    """The Ollivier curvature of ONE edge of an undirected quotient."""
    hop = float(exact_hop_costs(graph, a, np.asarray([b]))[0])
    if not math.isfinite(hop) or hop <= 0:
        raise ValueError("the edge is not connected: the curvature is undefined")
    return 1.0 - transport_cost(graph, a, b) / hop


def quotient_curvature(graph: sp.csr_matrix, rng: np.random.Generator,
                       edges: int = 32) -> float:
    """The mean Ollivier curvature of the lazy walk on an UNDIRECTED quotient.

    For an edge ``(a, b)`` the curvature is ``1 - W1(m_a, m_b) / d(a, b)``
    with the exact hop distance ``d`` and the lazy walk measures ``m``.
    """
    graph = graph.tocsr()
    coo = graph.tocoo()
    pairs = list(zip(coo.row.tolist(), coo.col.tolist()))
    if not pairs:
        raise ValueError("the quotient has no usable edges")
    if len(pairs) > edges:
        weight = np.asarray(coo.data, dtype=np.float64)
        probability = weight / max(weight.sum(), 1e-12)
        chosen = rng.choice(len(pairs), size=min(edges, len(pairs)),
                            replace=False, p=probability)
        pairs = [pairs[int(index)] for index in chosen]
    values = []
    for a, b in pairs:
        distance = exact_hop_costs(graph, a, np.asarray([b]))
        if not np.isfinite(distance).all():
            raise ValueError("the sampled edge is not connected")
        hop = float(distance[0])
        if hop <= 0:
            continue
        values.append(1.0 - transport_cost(graph, a, b) / hop)
    if not values:
        raise ValueError("the quotient has no usable edges")
    return float(np.mean(values))


def standardised_curvature(graph: sp.csr_matrix, assignment: np.ndarray,
                            n_groups: int, rng: np.random.Generator,
                            nulls: int = 3, edges: int = 32,
                            min_groups: int = 8) -> dict:
    """The curvature of one level and its size-preserving null.

    The null relabels the ITEMS into the groups with the group sizes preserved,
    so it keeps the size of every group and removes only the behaviour.
    """
    real = quotient_curvature(quotient_graph(graph, assignment, n_groups), rng, edges)
    sizes = np.bincount(assignment[assignment >= 0], minlength=n_groups)
    nulls_values = []
    order = np.arange(assignment.shape[0])
    for _ in range(nulls):
        rng.shuffle(order)
        shuffled = np.empty_like(assignment)
        cursor = 0
        for group in range(n_groups):
            size = int(sizes[group])
            shuffled[order[cursor:cursor + size]] = group
            cursor += size
        try:
            nulls_values.append(quotient_curvature(
                quotient_graph(graph, shuffled, n_groups), rng, edges))
        except ValueError:
            continue
    if not nulls_values or int((sizes > 0).sum()) < min_groups:
        return {"kappa": real, "kappa_hat": None, "null_mean": None, "null_std": None,
                "occupied_prefixes": int((sizes > 0).sum()),
                "status": "NOT_MEASURED",
                "reason": "the null is degenerate: too few groups or no spread"}
    mean = float(np.mean(nulls_values))
    std = float(np.std(nulls_values))
    if std < 1e-6:
        return {"kappa": real, "kappa_hat": None, "null_mean": mean, "null_std": std,
                "occupied_prefixes": int((sizes > 0).sum()),
                "status": "NOT_MEASURED",
                "reason": "the null has no spread: the deviation is undefined"}
    return {"kappa": real, "kappa_hat": (real - mean) / std, "null_mean": mean,
            "null_std": std, "occupied_prefixes": int((sizes > 0).sum()),
            "status": "MEASURED", "reason": None}
