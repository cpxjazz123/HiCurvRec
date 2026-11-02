"""The behaviour-graph curvature that sets each level's geometry.

The mechanism of the selected proposal: the TRAINING-ONLY behaviour graph is
contracted into the prefix groups of every level, the Ollivier curvature of the
lazy random walk is measured on that quotient, the value is standardised against
a null that keeps the group sizes and shuffles the members, and the standardised
profile sets each level's curvature and code radius.

Everything here is measured OUTSIDE the gradient path: the curvature is a
schedule, not a loss, so it carries no gradient and is refreshed every 200
epochs.

The switches are the proposal's: ``BEHAVIOUR_OFF`` keeps the geometry but reads
no behaviour (the curvature stays at the unit value), ``GEOMETRY_OFF`` keeps the
behaviour-set radius but flattens the curvature, and ``BOTH_ON`` is the full
loop.
"""
from __future__ import annotations

import numpy as np

from . import quotient_curvature as estimator

#: The three arms of the controlled comparison. Only ``BOTH_ON`` is trained.
BEHAVIOUR_OFF = False
GEOMETRY_OFF = False

#: The unit curvature the level starts from, and the clip on the standardised
#: deviation: one standard deviation maps to a factor e.
C0 = 1.0
DELTA_CLIP = 1.0

#: How the level geometry follows the standardised curvature.
DELTA_TO_CURVATURE = -1.0      # c_l = c0 * exp(-delta_l)
DELTA_TO_RADIUS = 0.5          # rho_l = rho0 * exp(delta_l / 2)

#: The measurement's own knobs: a compute budget, not tuned parameters.
NULL_RELABELLINGS = 3
SAMPLED_EDGES = 32
#: The transport problem is solved on the support of the two measures, and a
#: dense quotient's neighbourhood can hold hundreds of cells: the support is
#: capped, or the refresh would cost minutes instead of seconds.
MAX_SUPPORT = 48
MIN_OCCUPIED_PREFIXES = 8
MIN_NULL_SIGMA = 1e-6
EMA = 0.5


def behaviour_graph(train_frame) -> tuple:
    """The training split's co-occurrence graph, as a sparse weight matrix.

    A row's history plus its target is one training sequence, and consecutive
    items in it are adjacent. No validation or test interaction is read.
    """
    return estimator.build_behaviour_graph(train_frame)


def _legacy_behaviour_graph(train_frame):
    import scipy.sparse as sp

    weights: dict[tuple[int, int], float] = {}
    for history, target in zip(train_frame["history"].tolist(), train_frame["target"].tolist()):
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
    graph = sp.coo_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()
    return nodes, index, estimator.symmetrise(graph)


def quotient_graph(graph, assignment: np.ndarray, n_groups: int):
    """The behaviour graph contracted into the level's prefix groups, UNDIRECTED."""
    return estimator.quotient_graph(graph, assignment, n_groups)


def _legacy_quotient_graph(graph, assignment: np.ndarray, n_groups: int):
    import scipy.sparse as sp

    coo = graph.tocoo()
    first = assignment[coo.row]
    second = assignment[coo.col]
    keep = first != second
    first, second = first[keep], second[keep]
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    key = low.astype(np.int64) * np.int64(max(n_groups, 1)) + high.astype(np.int64)
    unique = np.unique(key)
    rows = (unique // max(n_groups, 1)).astype(np.int32)
    cols = (unique % max(n_groups, 1)).astype(np.int32)
    data = np.ones(rows.shape[0], dtype=np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(n_groups, n_groups)).tocsr()


def _hop_costs(adjacency, source: int, support: np.ndarray, depth: int = 3) -> np.ndarray:
    """The hop distances from ``source`` to the support, bounded in depth."""
    import collections

    neighbours = [set(adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]].tolist())
                  for node in range(adjacency.shape[0])]
    distance = {source: 0}
    frontier = collections.deque([source])
    level = 0
    while frontier and level < depth:
        level += 1
        for _ in range(len(frontier)):
            current = frontier.popleft()
            for neighbour in neighbours[current]:
                if neighbour not in distance:
                    distance[neighbour] = level
                    frontier.append(neighbour)
    out = np.full(support.shape[0], float(level + 1))
    for index, node in enumerate(support.tolist()):
        if node in distance:
            out[index] = float(distance[node])
    return out


def _exact_w1(a: np.ndarray, b: np.ndarray, cost: np.ndarray) -> float:
    """The exact 1-Wasserstein distance between two measures on a cost matrix.

    The transport problem is solved exactly (a linear program), because the
    measurement has to be the SAME estimator as the one whose standardised
    value was found to track the recall: an entropic approximation biases the
    value by an amount that depends on the graph's diameter, and the null has a
    different diameter.
    """
    from scipy.optimize import linprog

    size = a.shape[0]
    left = a / max(a.sum(), 1e-12)
    right = b / max(b.sum(), 1e-12)
    # the transport plan on the support, row sums left, column sums right
    cost_matrix = np.zeros((size, size))
    index = 0
    for i in range(size):
        for j in range(size):
            cost_matrix[i, j] = cost[i, j]
            index += 1
    rows = []
    columns = []
    for i in range(size):
        for j in range(size):
            rows.append(i)
            columns.append(size + j)
    objective = cost_matrix.reshape(-1)
    equality_rows = list(range(size)) + list(range(size, 2 * size))
    equality_columns = list(range(size * size)) + list(range(size * size))
    matrix = np.zeros((2 * size, size * size))
    for i in range(size):
        matrix[i, i * size:(i + 1) * size] = 1.0
    for j in range(size):
        matrix[size + j, j::size] = 1.0
    bounds = [(0.0, None)] * (size * size)
    solution = linprog(objective, A_eq=matrix, b_eq=np.concatenate([left, right]),
                       bounds=bounds, method="highs")
    if not solution.success:
        return float("nan")
    return float(objective @ solution.x)


def quotient_curvature(adjacency, rng: np.random.Generator) -> float:
    """The mean Ollivier curvature of the lazy walk on the UNDIRECTED quotient.

    The estimator is the framework's: the ground metric is the EXACT unweighted
    shortest path in the quotient, and the transport problem is solved exactly.
    """
    return estimator.quotient_curvature(adjacency, rng, edges=SAMPLED_EDGES)


def standardised_curvature(adjacency, assignment: np.ndarray, n_groups: int,
                            rng: np.random.Generator) -> dict:
    """The standardised curvature of one level and its null.

    The null relabels the ITEMS into the prefixes with the group sizes
    preserved, so it keeps the size of every group and removes only the
    behaviour. The same estimator is applied to both.
    """
    real = quotient_curvature(quotient_graph(adjacency, assignment, n_groups), rng)
    sizes = np.bincount(assignment, minlength=n_groups)
    nulls = []
    order = np.arange(assignment.shape[0])
    for _ in range(NULL_RELABELLINGS):
        rng.shuffle(order)
        shuffled = np.empty_like(assignment)
        cursor = 0
        for group in range(n_groups):
            size = int(sizes[group])
            shuffled[order[cursor:cursor + size]] = group
            cursor += size
        value = quotient_curvature(quotient_graph(adjacency, shuffled, n_groups), rng)
        if value == value:
            nulls.append(value)
    if not nulls or len(sizes) < MIN_OCCUPIED_PREFIXES:
        return {"kappa": real, "kappa_hat": 0.0, "null_mean": None, "null_std": None,
                "occupied_prefixes": int(len(sizes)), "fallback": "too few prefixes"}
    mean = float(np.mean(nulls))
    std = float(np.std(nulls))
    if std < MIN_NULL_SIGMA:
        return {"kappa": real, "kappa_hat": 0.0, "null_mean": mean, "null_std": std,
                "occupied_prefixes": int(len(sizes)), "fallback": "null spread too small"}
    return {"kappa": real, "kappa_hat": (real - mean) / std, "null_mean": mean,
            "null_std": std, "occupied_prefixes": int(len(sizes)), "fallback": None}


def update_delta(delta: np.ndarray, kappa_hat: np.ndarray) -> np.ndarray:
    """The mean-zero, clipped, EMA-smoothed deviation of the curvature profile.

    The mean-zero constraint keeps the total curvature budget: a level cannot
    buy curvature by taking it from another one.
    """
    deviation = kappa_hat - float(np.mean(kappa_hat))
    largest = float(np.abs(deviation).max()) if deviation.size else 0.0
    if largest > DELTA_CLIP:
        # a uniform scaling keeps the mean-zero constraint EXACTLY, where a clip
        # would break it
        deviation = deviation * (DELTA_CLIP / largest)
    return EMA * delta + EMA * deviation


def level_geometry(delta: np.ndarray, rho0: float) -> tuple:
    """The per-level curvature and code radius the deviation asks for."""
    curvature = C0 * np.exp(DELTA_TO_CURVATURE * delta)
    radius = rho0 * np.exp(DELTA_TO_RADIUS * delta)
    return curvature, radius
