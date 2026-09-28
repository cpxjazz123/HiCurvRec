"""Deterministic sampled multi-scale Ollivier-Ricci curvature on train behavior."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment


MAPPING_ID = "train_graph_lazy_walk_ollivier_ricci_base_relative"
BASE_CURVATURES = (1.3660953164241916, 0.7347829661951981, 0.6439958072706683)
RICCI_EDGE_SAMPLE_COUNT = 4096
RICCI_WALKS_PER_EDGE = 32
RICCI_EDGE_SAMPLE_SEED = 42
LAZY_PROBABILITY = 0.5
GEODESIC_METRIC_CAP = 3
LAYER_HOPS = (3, 2, 1)
C_MIN = 0.05
C_MAX = 1.50


def _undirected_train_graph(
    next_items: Sequence[Sequence[int]],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    n_items = len(next_items)
    if n_items == 0:
        raise ValueError("train-transition graph has no items")

    undirected_edges: set[tuple[int, int]] = set()
    for source, targets in enumerate(next_items):
        for target in targets:
            if isinstance(target, (bool, np.bool_)) or not isinstance(
                target, (int, np.integer)
            ):
                raise ValueError(f"train graph target for source {source} is not an integer")
            target_id = int(target)
            if target_id < 0 or target_id >= n_items:
                raise ValueError(
                    f"train graph target {target_id} outside item range [0, {n_items})"
                )
            if target_id != source:
                undirected_edges.add((min(source, target_id), max(source, target_id)))
    if not undirected_edges:
        raise ValueError("train-transition graph has no non-self undirected edges")

    edge_array = np.asarray(sorted(undirected_edges), dtype=np.int32)
    rows = np.concatenate((edge_array[:, 0], edge_array[:, 1]))
    columns = np.concatenate((edge_array[:, 1], edge_array[:, 0]))
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, columns)),
        shape=(n_items, n_items),
        dtype=np.int32,
    )
    adjacency.data[:] = 1
    adjacency.sort_indices()
    return adjacency, edge_array


def _bounded_geodesic_costs(
    left_samples: np.ndarray,
    right_samples: np.ndarray,
    adjacency: sparse.csr_matrix,
    radius_two: sparse.csr_matrix,
) -> np.ndarray:
    """Exact shortest-path cost capped at three, a bounded graph metric."""
    count = len(left_samples)
    costs = np.empty((count, count), dtype=np.float64)
    for row_index, source in enumerate(left_samples):
        row_start, row_end = adjacency.indptr[source : source + 2]
        neighbors = adjacency.indices[row_start:row_end]
        adjacent_positions = np.searchsorted(neighbors, right_samples)
        adjacent = np.zeros(count, dtype=bool)
        if len(neighbors):
            valid = adjacent_positions < len(neighbors)
            adjacent[valid] = (
                neighbors[adjacent_positions[valid]] == right_samples[valid]
            )

        row_start, row_end = radius_two.indptr[source : source + 2]
        within_two = radius_two.indices[row_start:row_end]
        two_positions = np.searchsorted(within_two, right_samples)
        distance_two = np.zeros(count, dtype=bool)
        if len(within_two):
            valid = two_positions < len(within_two)
            distance_two[valid] = (
                within_two[two_positions[valid]] == right_samples[valid]
            )

        costs[row_index] = np.where(
            right_samples == source,
            0.0,
            np.where(
                adjacent,
                1.0,
                np.where(distance_two, 2.0, float(GEODESIC_METRIC_CAP)),
            ),
        )
    return costs


def _sample_lazy_walks(
    sampled_edges: np.ndarray,
    adjacency: sparse.csr_matrix,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    walks = np.repeat(
        sampled_edges[:, :, None], RICCI_WALKS_PER_EDGE, axis=2
    ).astype(np.int32, copy=True)
    endpoints_by_hop = []
    flat_walks = walks.reshape(-1)
    for _ in range(max(LAYER_HOPS)):
        move_positions = np.flatnonzero(rng.random(flat_walks.size) >= LAZY_PROBABILITY)
        for position in move_positions:
            node = int(flat_walks[position])
            start, end = adjacency.indptr[node : node + 2]
            if start == end:
                raise RuntimeError("lazy random walk reached an isolated item")
            neighbor_index = start + int(rng.integers(end - start))
            flat_walks[position] = adjacency.indices[neighbor_index]
        endpoints_by_hop.append(walks.copy())
    return endpoints_by_hop


def _layer_curvatures(ricci_by_hop: list[float]) -> dict:
    if len(ricci_by_hop) != len(BASE_CURVATURES):
        raise ValueError("Ricci signal must have one value for each of three scales")
    ricci_by_layer = list(reversed(ricci_by_hop))
    multipliers = [1.0 - value for value in ricci_by_layer]
    if not all(math.isfinite(value) and value > 0.0 for value in multipliers):
        raise ValueError("Ricci correction multipliers must be finite and positive")
    mean_multiplier = math.fsum(multipliers) / len(multipliers)
    curvatures = [
        base * multiplier / mean_multiplier
        for base, multiplier in zip(BASE_CURVATURES, multipliers)
    ]
    if not all(math.isfinite(value) and C_MIN <= value <= C_MAX for value in curvatures):
        raise ValueError(
            "Ricci-adjusted curvature falls outside the inherited model domain; "
            "refusing to clip or min-max rescale"
        )
    return {
        "ricci_by_layer": ricci_by_layer,
        "g_by_layer": multipliers,
        "mean_g": mean_multiplier,
        "base_curvatures": list(BASE_CURVATURES),
        "c_by_layer": curvatures,
        "sectional_curvature_l": [-value for value in curvatures],
    }


def compute_multiscale_ricci_curvature(
    next_items: Sequence[Sequence[int]],
) -> dict:
    """Estimate h-step lazy-walk Ollivier-Ricci signals for h=1,2,3.

    Curvature is averaged over a fixed-seed uniform sample of undirected train
    edges. Each endpoint law is the empirical measure of equally weighted lazy
    walks. W1 uses the graph shortest-path metric capped at three; the cap keeps
    the metric bounded while preserving exact distances through the studied
    three-hop local geometry. The edge base distance is one.
    """
    adjacency, edge_array = _undirected_train_graph(next_items)
    sample_count = min(RICCI_EDGE_SAMPLE_COUNT, len(edge_array))
    rng = np.random.default_rng(RICCI_EDGE_SAMPLE_SEED)
    sample_indices = rng.choice(len(edge_array), size=sample_count, replace=False)
    sampled_edges = edge_array[sample_indices]

    exact_two_hop = (adjacency @ adjacency).tocsr()
    exact_two_hop.data[:] = 1
    exact_two_hop.eliminate_zeros()
    radius_two = (adjacency + exact_two_hop).tocsr()
    radius_two.data[:] = 1
    radius_two.setdiag(0)
    radius_two.eliminate_zeros()
    radius_two.sort_indices()

    endpoints_by_hop = _sample_lazy_walks(sampled_edges, adjacency, rng)
    ricci_by_hop = []
    for endpoints in endpoints_by_hop:
        total_curvature = 0.0
        for edge_index in range(sample_count):
            transport_costs = _bounded_geodesic_costs(
                endpoints[edge_index, 0],
                endpoints[edge_index, 1],
                adjacency,
                radius_two,
            )
            rows, columns = linear_sum_assignment(transport_costs)
            wasserstein = float(transport_costs[rows, columns].mean())
            total_curvature += 1.0 - wasserstein
        ricci_by_hop.append(total_curvature / sample_count)

    derived = _layer_curvatures(ricci_by_hop)
    return {
        "mapping": MAPPING_ID,
        "n_items": adjacency.shape[0],
        "unique_undirected_train_edges": len(edge_array),
        "sampled_edges": sample_count,
        "walks_per_endpoint": RICCI_WALKS_PER_EDGE,
        "lazy_probability": LAZY_PROBABILITY,
        "geodesic_metric_cap": GEODESIC_METRIC_CAP,
        "edge_sample_seed": RICCI_EDGE_SAMPLE_SEED,
        "layer_hops": list(LAYER_HOPS),
        "ricci_by_hop": ricci_by_hop,
        **derived,
    }
