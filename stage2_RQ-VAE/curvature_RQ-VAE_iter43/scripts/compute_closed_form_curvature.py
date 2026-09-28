"""Map train-transition hop-shell growth to fixed layer curvatures."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import sparse

BASE_MEAN_CURVATURE = 0.915
MAPPING_ID = "train_graph_hop_shell_squared_log_mean_budget"
LAYER_HOPS = (3, 2, 1)



def compute_closed_form_curvature(growth_by_hop: object) -> dict:
    """Allocate Iter32's mean curvature by squared log hop-shell growth."""
    if not isinstance(growth_by_hop, (list, tuple, np.ndarray)) or len(growth_by_hop) != 3:
        raise ValueError("growth_by_hop must be an ordered length-three vector")
    hop_values = []
    for index, value in enumerate(growth_by_hop):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"growth_by_hop[{index}] must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"growth_by_hop[{index}] must be finite and nonnegative")
        hop_values.append(number)

    layer_values = list(reversed(hop_values))
    log1p_by_layer = [math.log1p(value) for value in layer_values]
    squared_log_by_layer = [value * value for value in log1p_by_layer]
    mean_squared_log = math.fsum(squared_log_by_layer) / len(squared_log_by_layer)
    if not math.isfinite(mean_squared_log) or mean_squared_log <= 0.0:
        raise ValueError("hop-shell growth has no positive squared-log scale")
    curvatures = [
        BASE_MEAN_CURVATURE * value / mean_squared_log
        for value in squared_log_by_layer
    ]
    if not all(math.isfinite(value) and value >= 0.0 for value in curvatures):
        raise ValueError("computed fixed curvatures are non-finite or negative")
    if not math.isclose(
        math.fsum(curvatures) / len(curvatures),
        BASE_MEAN_CURVATURE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("curvature allocation did not preserve the mean budget")
    return {
        "growth_by_hop": hop_values,
        "growth_by_layer": layer_values,
        "log1p_growth_by_layer": log1p_by_layer,
        "squared_log_growth_by_layer": squared_log_by_layer,
        "curvature_allocation_by_layer": [
            value / mean_squared_log for value in squared_log_by_layer
        ],
        "base_mean_curvature": BASE_MEAN_CURVATURE,
        "closed_form_c_l": curvatures,
        "sectional_curvature_l": [-value for value in curvatures],
    }



def compute_behavior_graph_curvature(
    next_items: Sequence[Sequence[int]], block_size: int = 256
) -> dict:
    """Map train-only shortest-hop shell growth to fixed coarse-to-fine curvatures.

    S1 is averaged over active source items. q2 and q3 are aggregate ratios of
    newly reached source-destination pairs, giving branching-weighted shell
    expansion rather than a mean of per-source ratios.
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

    shell_one = np.diff(adjacency.indptr).astype(np.int64)
    radius_two_counts = np.diff(radius_two.indptr).astype(np.int64)
    shell_two = radius_two_counts - shell_one
    active_sources = shell_one > 0
    if not active_sources.any():
        raise ValueError("train-transition graph has no active source items")

    shell_three = np.zeros(n_items, dtype=np.int64)
    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        exact_three_hop = (exact_two_hop[start:end] @ adjacency).tocsr()
        radius_three = (radius_two[start:end] + exact_three_hop).tocsr()
        radius_three.data[:] = 1
        radius_three.eliminate_zeros()
        radius_three.sort_indices()
        radius_three_counts = np.diff(radius_three.indptr).astype(np.int64)
        for local_row, source in enumerate(range(start, end)):
            row_indices = radius_three.indices[
                radius_three.indptr[local_row] : radius_three.indptr[local_row + 1]
            ]
            position = np.searchsorted(row_indices, source)
            if position < len(row_indices) and row_indices[position] == source:
                radius_three_counts[local_row] -= 1
        shell_three[start:end] = radius_three_counts - np.diff(
            radius_two.indptr[start : end + 1]
        )
    if np.any(shell_two < 0) or np.any(shell_three < 0):
        raise RuntimeError("shortest-hop shell counts are not monotonically nested")

    shell_totals = [
        int(shell_one[active_sources].sum()),
        int(shell_two[active_sources].sum()),
        int(shell_three[active_sources].sum()),
    ]
    if shell_totals[0] == 0 or shell_totals[1] == 0:
        raise ValueError("train graph has no second- or third-hop growth")
    growth_by_hop = [
        float(shell_one[active_sources].mean()),
        shell_totals[1] / shell_totals[0],
        shell_totals[2] / shell_totals[1],
    ]
    derived = compute_closed_form_curvature(growth_by_hop)
    return {
        "mapping": MAPPING_ID,
        "layer_hops": list(LAYER_HOPS),
        "n_items": n_items,
        "active_sources": int(active_sources.sum()),
        "unique_nonself_train_edges": int(adjacency.nnz),
        "shell_totals_by_hop": shell_totals,
        **derived,
    }

