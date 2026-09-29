from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit
from scipy.sparse import coo_matrix

import curvature_config as experiment


FEATURE_NAMES = ("negative_orc_need", "two_hop_expansion", "transition_entropy")
_WORK_OUT_NEIGHBORS = None
_WORK_OUT_PROBABILITIES = None
_WORK_ADJACENCY_BITS = None
_WORK_TWO_HOP_BITS = None


def _init_transport_worker():
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except ImportError:
        pass


def _exact_transport_cost_matrix(left, right):
    right_words = right >> 6
    right_bits = np.left_shift(np.uint64(1), (right & 63).astype(np.uint64))
    adjacency = _WORK_ADJACENCY_BITS[left[:, None], right_words[None, :]]
    two_hop = _WORK_TWO_HOP_BITS[left[:, None], right_words[None, :]]
    direct = (adjacency & right_bits[None, :]) != 0
    within_two = (two_hop & right_bits[None, :]) != 0
    costs = np.full((len(left), len(right)), 3.0, dtype=np.float64)
    costs[within_two] = 2.0
    costs[direct] = 1.0
    costs[left[:, None] == right[None, :]] = 0.0
    return costs


def _edge_negative_curvature_contribution(edge):
    source, target = edge
    left = _WORK_OUT_NEIGHBORS[source]
    left_probabilities = _WORK_OUT_PROBABILITIES[source]
    right = _WORK_OUT_NEIGHBORS[target]
    right_probabilities = _WORK_OUT_PROBABILITIES[target]
    if len(right) == 0:
        right = np.asarray([target], dtype=np.int32)
        right_probabilities = np.asarray([1.0], dtype=np.float64)
    if len(left) == 0:
        raise RuntimeError(f"Observed transition source {source} has no outgoing measure")

    cost = _exact_transport_cost_matrix(left, right)
    rows, columns = cost.shape
    variables = rows * columns
    variable_ids = np.arange(variables, dtype=np.int32)
    source_rows = np.repeat(np.arange(rows, dtype=np.int32), columns)
    target_columns = np.tile(np.arange(columns, dtype=np.int32), rows)
    keep_target_rows = target_columns < columns - 1
    constraint_rows = np.concatenate(
        (source_rows, rows + target_columns[keep_target_rows])
    )
    constraint_columns = np.concatenate(
        (variable_ids, variable_ids[keep_target_rows])
    )
    constraint_values = np.ones(len(constraint_rows), dtype=np.float64)
    constraints = coo_matrix(
        (constraint_values, (constraint_rows, constraint_columns)),
        shape=(rows + columns - 1, variables),
    ).tocsr()
    demand = np.concatenate((left_probabilities, right_probabilities[:-1]))
    solution = linprog(
        cost.reshape(-1),
        A_eq=constraints,
        b_eq=demand,
        bounds=(0.0, None),
        method="highs-ds",
    )
    if not solution.success or not np.isfinite(solution.fun):
        raise RuntimeError(
            f"Exact W1 failed on transition {source}->{target}: {solution.message}"
        )
    # Each observed edge has unit length in the undirected support metric.
    negative_part = max(0.0, float(solution.fun) - 1.0)
    target_index = int(np.searchsorted(left, target))
    if target_index >= len(left) or int(left[target_index]) != target:
        raise RuntimeError(f"Observed transition {source}->{target} is absent from P_source")
    edge_probability = float(left_probabilities[target_index])
    return source, edge_probability * negative_part


def _make_outgoing_tables(edge_sources, edge_targets, edge_counts, item_count):
    out_neighbors = [[] for _ in range(item_count)]
    out_counts = [[] for _ in range(item_count)]
    for source, target, count in zip(edge_sources, edge_targets, edge_counts):
        out_neighbors[int(source)].append(int(target))
        out_counts[int(source)].append(float(count))
    probabilities = []
    for item_id in range(item_count):
        neighbors = np.asarray(out_neighbors[item_id], dtype=np.int32)
        counts = np.asarray(out_counts[item_id], dtype=np.float64)
        if len(neighbors):
            order = np.argsort(neighbors, kind="stable")
            neighbors = neighbors[order]
            counts = counts[order]
            counts /= counts.sum()
        out_neighbors[item_id] = neighbors
        probabilities.append(counts)
    return out_neighbors, probabilities


def _undirected_graph_tables(edge_sources, edge_targets, item_count):
    neighbors = [set() for _ in range(item_count)]
    for source, target in zip(edge_sources, edge_targets):
        source, target = int(source), int(target)
        if source == target:
            raise ValueError("Self-loop transitions have undefined ORC denominator")
        neighbors[source].add(target)
        neighbors[target].add(source)
    return [np.asarray(sorted(row), dtype=np.int32) for row in neighbors]


def _distance_bitsets(undirected_neighbors, item_count):
    word_count = (item_count + 63) // 64
    adjacency_bits = np.zeros((item_count, word_count), dtype=np.uint64)
    for node, neighbors in enumerate(undirected_neighbors):
        for neighbor in neighbors:
            adjacency_bits[node, int(neighbor) >> 6] |= np.uint64(1) << np.uint64(
                int(neighbor) & 63
            )
    two_hop_bits = np.zeros_like(adjacency_bits)
    for node, neighbors in enumerate(undirected_neighbors):
        if len(neighbors):
            two_hop_bits[node] = np.bitwise_or.reduce(adjacency_bits[neighbors], axis=0)
    return adjacency_bits, two_hop_bits


def _transition_records(frame: pd.DataFrame, item_count: int):
    sources = []
    targets = []
    for history, target in frame[["history", "target"]].itertuples(
        index=False, name=None
    ):
        if history is None or not isinstance(history, (list, tuple, np.ndarray)) or len(history) == 0:
            continue
        source = int(history[-1])
        target = int(target)
        if min(source, target) < 0 or max(source, target) >= item_count:
            raise ValueError("Training transition item IDs do not match embedding rows")
        sources.append(source)
        targets.append(target)
    if not sources:
        raise ValueError("Training split contains no valid consecutive item transitions")
    keys = np.asarray(sources, dtype=np.int64) * item_count + np.asarray(
        targets, dtype=np.int64
    )
    unique_keys, counts = np.unique(keys, return_counts=True)
    edge_sources = (unique_keys // item_count).astype(np.int32)
    edge_targets = (unique_keys % item_count).astype(np.int32)
    return edge_sources, edge_targets, counts.astype(np.int64)


def build_item_curvatures(frame: pd.DataFrame, item_count: int, workers=None):
    global _WORK_OUT_NEIGHBORS, _WORK_OUT_PROBABILITIES
    global _WORK_ADJACENCY_BITS, _WORK_TWO_HOP_BITS
    edge_sources, edge_targets, edge_counts = _transition_records(frame, item_count)
    out_neighbors, out_probabilities = _make_outgoing_tables(
        edge_sources, edge_targets, edge_counts, item_count
    )
    undirected_neighbors = _undirected_graph_tables(
        edge_sources, edge_targets, item_count
    )
    adjacency_bits, two_hop_bits = _distance_bitsets(undirected_neighbors, item_count)

    records = (
        (int(source), int(target))
        for source, target in zip(edge_sources, edge_targets)
    )
    negative_orc_need = np.zeros(item_count, dtype=np.float64)
    worker_count = int(workers if workers is not None else experiment.CURVATURE_WORKERS)
    if worker_count <= 0:
        raise ValueError("Curvature worker count must be positive")

    print(
        f"[BehaviorRicci] exact W1 for {len(edge_sources)} weighted train edges; "
        f"nodes={item_count} workers={worker_count}",
        flush=True,
    )
    _WORK_OUT_NEIGHBORS = out_neighbors
    _WORK_OUT_PROBABILITIES = out_probabilities
    _WORK_ADJACENCY_BITS = adjacency_bits
    _WORK_TWO_HOP_BITS = two_hop_bits
    context = mp.get_context("fork")
    with context.Pool(
        processes=worker_count,
        initializer=_init_transport_worker,
    ) as pool:
        for done, (source, contribution) in enumerate(
            pool.imap(_edge_negative_curvature_contribution, records, chunksize=64),
            start=1,
        ):
            negative_orc_need[source] += contribution
            if done % 20_000 == 0 or done == len(edge_sources):
                print(f"[BehaviorRicci] exact W1 edges {done}/{len(edge_sources)}", flush=True)
    _WORK_OUT_NEIGHBORS = None
    _WORK_OUT_PROBABILITIES = None
    _WORK_ADJACENCY_BITS = None
    _WORK_TWO_HOP_BITS = None

    two_hop_expansion = np.zeros(item_count, dtype=np.float64)
    transition_entropy = np.zeros(item_count, dtype=np.float64)
    for item_id, (neighbors, probabilities) in enumerate(
        zip(out_neighbors, out_probabilities)
    ):
        if len(neighbors):
            two_hop_shell = set()
            one_hop = set(int(item) for item in neighbors)
            for neighbor in neighbors:
                two_hop_shell.update(int(item) for item in out_neighbors[int(neighbor)])
            two_hop_shell.discard(item_id)
            two_hop_shell.difference_update(one_hop)
            two_hop_expansion[item_id] = np.log(
                (len(two_hop_shell) + 1.0) / (len(one_hop) + 1.0)
            )
            positive = probabilities[probabilities > 0.0]
            transition_entropy[item_id] = -float(np.dot(positive, np.log(positive)))

    signals = np.column_stack(
        (negative_orc_need, two_hop_expansion, transition_entropy)
    ).astype(np.float32)
    means = signals.mean(axis=0, dtype=np.float64)
    scales = signals.std(axis=0, dtype=np.float64)
    scales[scales < 1e-8] = 1.0
    normalized = (signals.astype(np.float64) - means) / scales
    weights = np.asarray(
        (
            experiment.RICCI_WEIGHT_R,
            experiment.RICCI_WEIGHT_G,
            experiment.RICCI_WEIGHT_H,
        ),
        dtype=np.float64,
    )
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Fixed Ricci coefficients must be nonnegative and sum to one")
    q = normalized @ weights
    curvatures = experiment.CURVATURE_MIN + (
        experiment.CURVATURE_MAX - experiment.CURVATURE_MIN
    ) * expit(q)
    curvatures = curvatures.astype(np.float32)
    if (
        not np.isfinite(signals).all()
        or not np.isfinite(curvatures).all()
        or curvatures.min() < experiment.CURVATURE_MIN
        or curvatures.max() > experiment.CURVATURE_MAX
    ):
        raise RuntimeError("Behavior-Ricci features produced invalid item curvatures")
    return curvatures, signals
