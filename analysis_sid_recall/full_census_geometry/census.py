"""Full-census local geometry of the SID prefix groups (Issue #132).

No group-count, group-size or item-coverage cap is applied to the structural
census: every occupied prefix group of every admitted SID is enumerated at L1,
L2 and L3, and the exact structural metrics are computed from its induced
subgraph of the audited #128 TRAIN-only graph.

Per group: size, internal edges, connected components and their sizes, largest
component's item fraction, triangles, transitivity (UNDEFINED when no connected
triple exists), cycle rank, external boundary edges, volume and conductance.

Expensive metrics:

* **four-point Gromov deviation** — enumerated EXACTLY over all 4-tuples of every
  connected component with at most ``FOUR_POINT_MAX`` nodes; larger components
  are reported ``BLOCKED_NOT_FULL_CENSUS`` with their counts. It is never
  estimated on a cherry-picked tuple sample.
* **path detour** — computed EXACTLY for every connected pair of every group up
  to ``DETOUR_MAX`` items, using exact BFS distances inside the group and in the
  full graph. Larger groups are ``BLOCKED_NOT_FULL_CENSUS`` with their pair
  counts: exact all-pairs over every item and every SID needs one full-graph BFS
  per item (about 2.7 million BFS for this table) and is reported as a blocker,
  never replaced by a sample. Pairs without a within-group path contribute to
  the ``unavailable`` count, not to the distance ratio.

Every metric carries its own measured group and item coverage, so a metric that
is only defined on part of the census is never reported as an all-group value.
"""
from __future__ import annotations

import itertools
from collections import deque

import numpy as np
import scipy.sparse as sp

FOUR_POINT_MAX = 24
DETOUR_MAX = 16
TRIANGLE_MAX = 2000


# ---------------------------------------------------------------- distances
def bfs_rows_fast(adjacency: sp.csr_matrix, sources: np.ndarray,
                  targets: np.ndarray) -> np.ndarray:
    """Level-synchronous BFS with numpy frontiers; exact unweighted distances."""
    indptr = adjacency.indptr
    indices = adjacency.indices
    targets = np.asarray(targets, dtype=np.int64)
    lookup = np.full(adjacency.shape[0], -1, dtype=np.int64)
    lookup[targets] = np.arange(targets.size)
    out = np.full((sources.size, targets.size), -1, dtype=np.int32)
    for row, source in enumerate(np.asarray(sources, dtype=np.int64).tolist()):
        seen = np.zeros(adjacency.shape[0], dtype=bool)
        frontier = np.asarray([source], dtype=np.int64)
        seen[source] = True
        depth = 0
        discovered = 0
        while frontier.size and discovered < targets.size:
            hits = lookup[frontier]
            found = hits >= 0
            if found.any():
                out[row, hits[found]] = depth
                discovered += int(found.sum())
            # expand
            starts = indptr[frontier]
            ends = indptr[frontier + 1]
            if not np.any(ends > starts):
                break
            pieces = [indices[start:end] for start, end in zip(starts.tolist(), ends.tolist())]
            neighbours = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)
            neighbours = neighbours[~seen[neighbours]]
            if neighbours.size == 0:
                break
            frontier = np.unique(neighbours)
            seen[frontier] = True
            depth += 1
    return out


def bfs_distances(adjacency: sp.csr_matrix, sources: np.ndarray,
                  targets: np.ndarray) -> np.ndarray:
    """Exact unweighted distances from ``sources`` to ``targets`` (BFS rows)."""
    indptr, indices = adjacency.indptr, adjacency.indices
    targets = np.asarray(targets, dtype=np.int64)
    position = {int(node): index for index, node in enumerate(targets.tolist())}
    out = np.full((sources.size, targets.size), -1, dtype=np.int32)
    for row, source in enumerate(np.asarray(sources, dtype=np.int64).tolist()):
        distance = {source: 0}
        frontier = deque([source])
        found = 1 if source in position else 0
        while frontier and found < targets.size:
            node = frontier.popleft()
            step = distance[node] + 1
            for neighbour in indices[indptr[node]:indptr[node + 1]].tolist():
                if neighbour in distance:
                    continue
                distance[neighbour] = step
                frontier.append(neighbour)
                if neighbour in position:
                    out[row, position[neighbour]] = step
                    found += 1
        if source in position:
            out[row, position[source]] = 0
    return out


def gromov_four_point(distances: np.ndarray) -> dict:
    """Exact four-point deviation over ALL 4-tuples of a component.

    ``delta = max pairing sum - second max pairing sum``; a tree gives 0.
    """
    size = distances.shape[0]
    if size < 4:
        return {"status": "UNDEFINED_BY_DEFINITION", "quadruples": 0, "delta_max": None}
    total = 0
    maximum = 0
    for a, b, c, d in itertools.combinations(range(size), 4):
        sums = sorted((distances[a, b] + distances[c, d],
                       distances[a, c] + distances[b, d],
                       distances[a, d] + distances[b, c]), reverse=True)
        maximum = max(maximum, sums[0] - sums[1])
        total += 1
    return {"status": "EXACT", "quadruples": total, "delta_max": int(maximum)}


# ------------------------------------------------------------------ census
def groups(labels: np.ndarray) -> list[np.ndarray]:
    usable = np.nonzero(labels >= 0)[0]
    if usable.size == 0:
        return []
    order = np.argsort(labels[usable], kind="mergesort")
    nodes = usable[order]
    values = labels[nodes]
    boundaries = np.flatnonzero(np.diff(values)) + 1
    return [chunk for chunk in np.split(nodes, boundaries) if chunk.size]


def group_metrics(adjacency: sp.csr_matrix, degree: np.ndarray,
                  members: np.ndarray, *, four_point_enabled: bool = True) -> dict:
    """Exact structural metrics of one induced subgraph (any size)."""
    size = int(members.size)
    induced = adjacency[members][:, members].tocsr()
    internal = int(induced.nnz // 2)
    from scipy.sparse.csgraph import connected_components
    count, labels = connected_components(induced, directed=False)
    sizes = np.bincount(labels) if count else np.zeros(0, dtype=int)
    largest = int(sizes.max()) if sizes.size else 0
    volume = float(degree[members].sum())
    boundary = float(max(volume - 2.0 * internal, 0.0))
    total_volume = float(degree.sum())
    denominator = min(volume, max(total_volume - volume, 1e-12))
    conductance = boundary / denominator if (volume > 0 and denominator > 0) else None
    if size <= TRIANGLE_MAX:
        # exact triangle count: trace(A^3)/6 on the binary adjacency
        binary = (induced.toarray() > 0).astype(np.int8)
        triangles = int(round(np.trace(binary @ binary @ binary) / 6.0))
        degrees = binary.sum(axis=1).astype(np.int64)
        triples = int((degrees * (degrees - 1)).sum() // 2)
        transitivity = (3.0 * triangles / triples) if triples > 0 else None
        triangle_status = "EXACT"
    else:
        # the cubic dense product is not computable at this size; the metric is
        # declared blocked for the group instead of being approximated
        triangles = None
        transitivity = None
        triangle_status = "BLOCKED_NOT_FULL_CENSUS"
    # exact four-point, only where the component is small enough to enumerate
    four = ({"status": "BLOCKED_NOT_FULL_CENSUS", "quadruples": 0, "delta_max": None}
            if four_point_enabled
            else {"status": "NOT_COMPUTED", "quadruples": 0, "delta_max": None})
    flagged = False
    per_component = []
    for index in range(count) if four_point_enabled else ():
        component = np.nonzero(labels == index)[0]
        if component.size >= 4:
            if component.size <= FOUR_POINT_MAX:
                distances = bfs_distances(induced, component, component)
                exact = gromov_four_point(distances)
                per_component.append({"size": int(component.size), **exact})
            else:
                per_component.append({"size": int(component.size),
                                      "status": "BLOCKED_NOT_FULL_CENSUS",
                                      "quadruples": 0, "delta_max": None})
    if per_component:
        exact = [entry for entry in per_component if entry["status"] == "EXACT"]
        blocked = [entry for entry in per_component if entry["status"] != "EXACT"]
        if exact and not blocked:
            four = {"status": "EXACT",
                    "quadruples": int(sum(entry["quadruples"] for entry in exact)),
                    "delta_max": int(max(entry["delta_max"] for entry in exact))}
        elif exact and blocked:
            four = {"status": "PARTIAL_BLOCKED",
                    "quadruples": int(sum(entry["quadruples"] for entry in exact)),
                    "delta_max": int(max(entry["delta_max"] for entry in exact)),
                    "blocked_components": len(blocked),
                    "blocked_quadruples_lower_bound": None}
        else:
            four = {"status": "BLOCKED_NOT_FULL_CENSUS", "quadruples": 0,
                    "delta_max": None, "blocked_components": len(blocked)}
    return {"size": size, "internal_edges": internal, "components": int(count),
            "component_sizes": sorted(int(value) for value in sizes.tolist()),
            "largest_component": largest,
            "largest_component_fraction": largest / size,
            "triangles": triangles, "triangle_status": triangle_status,
            "transitivity": transitivity,
            "cycle_rank": internal - size + int(count),
            "cycle_rank_normalised": (internal - size + int(count)) / size,
            "volume": volume, "boundary_edges": boundary,
            "conductance": conductance, "four_point": four}


def detour(adjacency: sp.csr_matrix, members: np.ndarray) -> dict:
    """Exact within-group vs full-graph distance ratio over all connected pairs."""
    size = int(members.size)
    if size < 2:
        return {"status": "UNDEFINED_BY_DEFINITION", "pairs": 0, "unavailable": 0,
                "mean_ratio": None, "coverage": 0.0}
    local = np.arange(size)
    inside = bfs_rows_fast(adjacency[members][:, members].tocsr(), local, local)
    full = bfs_rows_fast(adjacency, members, members)
    pairs = size * (size - 1) // 2
    upper = np.triu_indices(size, k=1)
    d_inside = inside[upper]
    d_full = full[upper]
    usable = (d_inside >= 0) & (d_full > 0)
    unavailable = int(pairs - usable.sum())
    ratios = d_inside[usable] / d_full[usable]
    return {"status": "EXACT" if usable.any() else "UNDEFINED_NO_CONNECTED_PAIR",
            "pairs": int(pairs), "unavailable": unavailable,
            "mean_ratio": float(ratios.mean()) if ratios.size else None,
            "coverage": float(usable.sum() / pairs) if pairs else 0.0}


def _singleton_metrics(adjacency, degree, member: int, total_volume: float) -> dict:
    """Analytic metrics of a one-item group (no subgraph allocation)."""
    volume = float(degree[member])
    boundary = volume
    denominator = min(volume, max(total_volume - volume, 1e-12))
    return {"size": 1, "internal_edges": 0, "components": 1, "component_sizes": [1],
            "largest_component": 1, "largest_component_fraction": 1.0,
            "triangles": 0, "transitivity": None, "cycle_rank": 0,
            "cycle_rank_normalised": 0.0, "volume": volume, "boundary_edges": boundary,
            "conductance": (boundary / denominator) if denominator > 0 else None,
            "four_point": {"status": "UNDEFINED_BY_DEFINITION", "quadruples": 0,
                           "delta_max": None},
            "detour": {"status": "UNDEFINED_BY_DEFINITION", "pairs": 0,
                       "unavailable": 0, "mean_ratio": None, "coverage": 0.0}}


def census(adjacency: sp.csr_matrix, degree: np.ndarray, labels: np.ndarray,
           *, detour_enabled: bool = False, four_point_enabled: bool = True) -> dict:
    """The full census of one (SID, level): every group, no cap."""
    groups_list = groups(labels)
    per_group = []
    total_volume = float(degree.sum())
    for members in groups_list:
        if members.size == 1:
            per_group.append(_singleton_metrics(adjacency, degree, int(members[0]),
                                                total_volume))
            continue
        entry = group_metrics(adjacency, degree, members,
                              four_point_enabled=four_point_enabled)
        if detour_enabled and 2 <= entry["size"] <= DETOUR_MAX:
            try:
                entry["detour"] = detour(adjacency, members)
            except MemoryError:                      # pragma: no cover - guard
                entry["detour"] = {"status": "BLOCKED_NOT_FULL_CENSUS",
                                   "pairs": entry["size"] * (entry["size"] - 1) // 2,
                                   "unavailable": None, "mean_ratio": None,
                                   "coverage": 0.0}
        elif detour_enabled and entry["size"] > DETOUR_MAX:
            # exact enumeration over every pair of every group is the cost of a
            # full-graph BFS per item; above this size it is declared blocked
            entry["detour"] = {"status": "BLOCKED_NOT_FULL_CENSUS",
                               "pairs": entry["size"] * (entry["size"] - 1) // 2,
                               "unavailable": None, "mean_ratio": None, "coverage": 0.0}
        elif detour_enabled:
            entry["detour"] = {"status": "UNDEFINED_BY_DEFINITION", "pairs": 0,
                               "unavailable": 0, "mean_ratio": None, "coverage": 0.0}
        else:
            entry["detour"] = {"status": "BLOCKED_NOT_FULL_CENSUS",
                               "pairs": entry["size"] * (entry["size"] - 1) // 2,
                               "unavailable": None, "mean_ratio": None, "coverage": 0.0}
        per_group.append(entry)
    return {"groups": per_group, "support": support(per_group)}


def support(per_group: list[dict]) -> dict:
    sizes = np.asarray([entry["size"] for entry in per_group]) if per_group else np.zeros(0)
    if sizes.size == 0:
        return {"occupied": 0, "singleton": 0, "multi": 0, "ge4": 0, "items": 0,
                "item_coverage_multi": 0.0, "item_coverage_ge4": 0.0,
                "size_histogram": {}, "eligible": {}, "measured": {}}
    multi = sizes >= 2
    ge4 = sizes >= 4
    eligible = {
        "transitivity": int(((sizes >= 3)).sum()),          # needs a connected triple
        "four_point": int(((sizes >= 4)).sum()),             # needs 4 connected nodes
        "detour": int(multi.sum()),
    }
    measured = {
        "transitivity": int(sum(1 for entry in per_group if entry["transitivity"] is not None)),
        "four_point": int(sum(1 for entry in per_group
                              if entry["four_point"]["status"] == "EXACT")),
        "detour": int(sum(1 for entry in per_group
                          if entry["detour"]["status"] == "EXACT")),
    }
    return {"occupied": int(sizes.size), "singleton": int((sizes == 1).sum()),
            "multi": int(multi.sum()), "ge4": int(ge4.sum()),
            "items": int(sizes.sum()),
            "item_coverage_multi": float(sizes[multi].sum() / sizes.sum()),
            "item_coverage_ge4": float(sizes[ge4].sum() / sizes.sum()),
            "size_histogram": {int(k): int(v) for k, v in
                               zip(*np.unique(sizes, return_counts=True))},
            "size_median": float(np.median(sizes)), "size_max": int(sizes.max()),
            "eligible": eligible, "measured": measured,
            "measured_item_coverage": {
                "transitivity": float(sum(entry["size"] for entry in per_group
                                          if entry["transitivity"] is not None) / sizes.sum()),
                "four_point": float(sum(entry["size"] for entry in per_group
                                        if entry["four_point"]["status"] == "EXACT") / sizes.sum()),
                "detour": float(sum(entry["size"] for entry in per_group
                                    if entry["detour"]["status"] == "EXACT") / sizes.sum())}}


def aggregates(per_group: list[dict], support_block: dict) -> dict:
    """Group- and item-weighted summaries for each metric, with coverage."""
    out = {}
    sizes = np.asarray([entry["size"] for entry in per_group], dtype=float)
    for key in ("transitivity", "cycle_rank_normalised", "conductance",
                "largest_component_fraction", "internal_edges", "boundary_edges",
                "components", "triangles"):
        values = np.asarray([entry[key] if entry[key] is not None else np.nan
                             for entry in per_group], dtype=float)
        finite = np.isfinite(values)
        block = {"groups": int(finite.sum()),
                 "group_coverage": float(finite.mean()) if values.size else 0.0,
                 "group_mean": float(values[finite].mean()) if finite.any() else None,
                 "item_weighted": (float(np.average(values[finite], weights=sizes[finite]))
                                   if finite.any() else None),
                 "median": float(np.median(values[finite])) if finite.any() else None,
                 "p90": float(np.quantile(values[finite], 0.9)) if finite.any() else None}
        out[key] = block
    four = [entry["four_point"] for entry in per_group]
    exact = [entry for entry in four if entry["status"] == "EXACT"]
    blocked = [entry for entry in four if entry["status"].startswith("BLOCKED")
               or entry["status"] == "PARTIAL_BLOCKED"]
    out["four_point"] = {
        "groups_exact": len(exact), "groups_blocked": len(blocked),
        "quadruples": int(sum(entry["quadruples"] for entry in exact)),
        "mean_delta_max": float(np.mean([entry["delta_max"] for entry in exact]))
        if exact else None,
        "status": "EXACT" if exact and not blocked else
                  ("BLOCKED_NOT_FULL_CENSUS" if not exact else "PARTIAL_BLOCKED")}
    detours = [entry["detour"] for entry in per_group]
    usable = [entry for entry in detours if entry["status"] == "EXACT"]
    out["detour"] = {
        "groups_exact": len(usable), "groups_undefined": len(detours) - len(usable),
        "pairs": int(sum(entry["pairs"] for entry in usable)),
        "pairs_unavailable": int(sum(entry["unavailable"] for entry in usable)),
        "mean_ratio": float(np.mean([entry["mean_ratio"] for entry in usable]))
        if usable else None}
    return out
