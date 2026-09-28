"""Map train-transition graph reachability to fixed layer curvatures."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import sparse

C_MIN = 0.05
C_MAX = 1.50
MAPPING_ID = "train_graph_log1p_hop_fanout_minmax"
LAYER_HOPS = (3, 2, 1)


def compute_closed_form_curvature(branching: object) -> dict[str, list[float]]:
    """Min-max map log1p of ordered coarse-to-fine graph fan-out into c bounds."""
    if not isinstance(branching, (list, tuple, np.ndarray)) or len(branching) != 3:
        raise ValueError("branching must be an ordered length-three vector")
    values = []
    for index, value in enumerate(branching):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"branching[{index}] must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"branching[{index}] must be finite and nonnegative")
        values.append(number)
    if any(values[index] < values[index + 1] for index in range(2)):
        raise ValueError("branching must be ordered coarse-to-fine, nonincreasing")

    log_values = [math.log1p(value) for value in values]
    low, high = min(log_values), max(log_values)
    if high <= low:
        raise ValueError("graph branching scales have no range for min-max mapping")
    normalized = [(value - low) / (high - low) for value in log_values]
    curvatures = [C_MIN + weight * (C_MAX - C_MIN) for weight in normalized]
    if not all(math.isfinite(value) and C_MIN <= value <= C_MAX for value in curvatures):
        raise ValueError("computed fixed curvatures are non-finite or out of bounds")
    return {
        "branching_by_layer": values,
        "log1p_branching": log_values,
        "normalized_log1p_branching": normalized,
        "closed_form_c_l": curvatures,
        "sectional_curvature_l": [-value for value in curvatures],
    }


def compute_behavior_graph_curvature(
    next_items: Sequence[Sequence[int]], block_size: int = 256
) -> dict:
    """Derive three hop-radius fan-outs from Stage2's train-only edge lists.

    Layer 0 uses cumulative 3-hop reachability, layer 1 uses 2-hop, and layer 2
    uses 1-hop. The source item and duplicate/self edges do not count as branches.
    """
    n_items = len(next_items)
    if n_items == 0:
        raise ValueError("train-transition graph has no items")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    rows: list[int] = []
    columns: list[int] = []
    for source, targets in enumerate(next_items):
        unique_targets = set()
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
                unique_targets.add(target_id)
        for target_id in unique_targets:
            rows.append(source)
            columns.append(target_id)
    if not rows:
        raise ValueError("train-transition graph has no non-self directed edges")

    adjacency = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, columns)),
        shape=(n_items, n_items),
        dtype=np.int32,
    )
    adjacency.data[:] = 1
    exact_two_hop = (adjacency @ adjacency).tocsr()
    exact_two_hop.data[:] = 1
    exact_two_hop.eliminate_zeros()

    radius_two = (adjacency + exact_two_hop).tocsr()
    radius_two.data[:] = 1
    radius_two.setdiag(0)
    radius_two.eliminate_zeros()

    fanout_one = np.diff(adjacency.indptr).astype(np.int64)
    fanout_two = np.diff(radius_two.indptr).astype(np.int64)
    active_sources = fanout_one > 0
    if not active_sources.any():
        raise ValueError("train-transition graph has no active source items")

    fanout_three = np.zeros(n_items, dtype=np.int64)
    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        exact_three_hop = (exact_two_hop[start:end] @ adjacency).tocsr()
        radius_three = (radius_two[start:end] + exact_three_hop).tocsr()
        radius_three.data[:] = 1
        radius_three.eliminate_zeros()
        radius_three.sort_indices()
        counts = np.diff(radius_three.indptr).astype(np.int64)
        for local_row, source in enumerate(range(start, end)):
            row_indices = radius_three.indices[
                radius_three.indptr[local_row] : radius_three.indptr[local_row + 1]
            ]
            position = np.searchsorted(row_indices, source)
            if position < len(row_indices) and row_indices[position] == source:
                counts[local_row] -= 1
        fanout_three[start:end] = counts

    branching = [
        float(fanout_three[active_sources].mean()),
        float(fanout_two[active_sources].mean()),
        float(fanout_one[active_sources].mean()),
    ]
    derived = compute_closed_form_curvature(branching)
    return {
        "mapping": MAPPING_ID,
        "layer_hops": list(LAYER_HOPS),
        "n_items": n_items,
        "active_sources": int(active_sources.sum()),
        "unique_nonself_train_edges": int(adjacency.nnz),
        **derived,
    }
