"""Singleton-aware local geometry of the SID prefix groups (Issue #131).

Definitions (frozen before any Recall analysis):

* ``G(l, p) = {i : prefix_l(SID(i)) = p}``; the *induced* subgraph is
  ``H[G]`` of the audited #128 TRAIN-only undirected item graph ``H``.
* Occupied / singleton / multi-item / adequately-supported groups are reported
  separately. Items of singleton groups stay in ``H`` for every boundary and
  control computation; their size-1 induced graphs are excluded only from the
  measures that need at least 2 or 4 nodes.
* Cheap metrics are computed for every group with ``2 <= |G| <= 64``
  (declared eligibility); larger groups are reported as excluded, never dropped
  silently.
* The four-point deviation is sampled on connected components with >= 4 nodes
  and normalised by the component's diameter; a tree gives exactly 0.
* The path-detour ratio is computed on a declared subsample of connected groups
  (``detour_groups`` per level, ``detour_pairs`` pairs each): the within-group
  distance over the distance in ``H``; pairs whose within-group path does not
  exist are counted, not dropped.

Nulls: a fixed-size global label shuffle; a parent-conditioned shuffle that
permutes children only inside their own parent (preserving the child-size
multiset); and an activity-stratified shuffle. All use the same estimator and
the same eligible-group rule as the real condition.
"""
from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np
import scipy.sparse as sp


# ------------------------------------------------------------------ groups
def groups_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    """The item positions of every occupied label, in ascending label order."""
    usable = np.nonzero(labels >= 0)[0]
    if usable.size == 0:
        return []
    order = np.argsort(labels[usable], kind="mergesort")
    nodes = usable[order]
    values = labels[nodes]
    boundaries = np.flatnonzero(np.diff(values)) + 1
    return [chunk for chunk in np.split(nodes, boundaries) if chunk.size]


def support_audit(groups: list[np.ndarray]) -> dict:
    sizes = np.asarray([group.size for group in groups])
    if sizes.size == 0:
        return {"occupied": 0, "singleton": 0, "multi": 0, "ge4": 0,
                "size_histogram": {}, "item_coverage_multi": 0.0,
                "item_coverage_ge4": 0.0}
    multi = sizes >= 2
    ge4 = sizes >= 4
    return {"occupied": int(sizes.size), "singleton": int((sizes == 1).sum()),
            "multi": int(multi.sum()), "ge4": int(ge4.sum()),
            "size_histogram": {int(k): int(v) for k, v in
                               zip(*np.unique(sizes, return_counts=True))},
            "size_median": float(np.median(sizes)),
            "size_p90": float(np.quantile(sizes, 0.9)),
            "size_max": int(sizes.max()),
            "item_coverage_multi": float(sizes[multi].sum() / sizes.sum()),
            "item_coverage_ge4": float(sizes[ge4].sum() / sizes.sum()),
            "item_total": int(sizes.sum())}


# --------------------------------------------------------------- local metrics
def _bfs(adjacency: sp.csr_matrix, source: int) -> dict[int, int]:
    distance = {source: 0}
    frontier = [source]
    while frontier:
        nxt = []
        for node in frontier:
            for neighbour in adjacency.indices[adjacency.indptr[node]:
                                               adjacency.indptr[node + 1]].tolist():
                if neighbour not in distance:
                    distance[neighbour] = distance[node] + 1
                    nxt.append(neighbour)
        frontier = nxt
    return distance


def _graph_distance(graph: sp.csr_matrix, source: int) -> np.ndarray:
    from scipy.sparse.csgraph import shortest_path
    return shortest_path(graph, method="D", unweighted=True, directed=False,
                         indices=[int(source)])[0]


def local_metrics(adjacency: sp.csr_matrix, members: np.ndarray, degree: np.ndarray,
                  *, quadruples: int = 8, seed: int = 0) -> dict:
    """Geometry of the induced subgraph on ``members`` (size in [2, 64])."""
    size = int(members.size)
    induced = adjacency[members][:, members].tocsr()
    internal = int(induced.nnz // 2)
    from scipy.sparse.csgraph import connected_components

    count, labels = connected_components(induced, directed=False)
    component_sizes = np.bincount(labels)
    largest = int(component_sizes.max()) if count else 0
    volume = float(degree[members].sum())
    boundary = float(max(volume - 2.0 * internal, 0.0))
    total_volume = float(degree.sum())
    conductance = boundary / min(volume, max(total_volume - volume, 1e-12)) \
        if volume > 0 else None
    dense = (induced.toarray() > 0).astype(float)     # structure, not interaction weight
    triangles = float(np.trace(dense @ dense @ dense) / 6.0)
    triples = float((dense.sum(axis=1) * (dense.sum(axis=1) - 1)).sum() / 2.0)
    transitivity = (3.0 * triangles / triples) if triples > 0 else None
    surplus = internal - size + int(count)
    four = {"deviation": None, "n": 0, "normalised": None}
    components = [np.nonzero(labels == index)[0] for index in range(count)]
    rng = np.random.default_rng(seed)
    big = [component for component in components if component.size >= 4]
    if big:
        component = big[int(rng.integers(len(big)))]
        sub = induced[component][:, component].tocsr()
        diameter = None
        deviations = []
        for _ in range(quadruples):
            pick = rng.choice(component.size, size=4, replace=False)
            distances = {}
            for position in pick.tolist():
                distances[position] = _bfs(sub, position)
            def distance(a, b):
                return distances[int(a)].get(int(b), None)
            pairs = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]
            sums = []
            usable = True
            for a, b, c, d in pairs:
                first = distance(pick[a], pick[b])
                second = distance(pick[c], pick[d])
                if first is None or second is None:
                    usable = False
                    break
                sums.append(first + second)
            if not usable:
                continue
            sums.sort(reverse=True)
            deviations.append(sums[0] - sums[1])
            if diameter is None:
                diameter = max(max(row.values()) for row in distances.values())
        if deviations and diameter:
            four = {"deviation": float(np.mean(deviations)), "n": len(deviations),
                    "normalised": float(np.mean(deviations) / max(diameter, 1))}
    return {"size": size, "internal_edges": internal, "components": int(count),
            "largest_component": largest,
            "largest_component_fraction": largest / size,
            "boundary_edges": boundary, "volume": volume,
            "conductance": conductance, "triangles": triangles,
            "transitivity": transitivity, "cycle_surplus": surplus,
            "cycle_surplus_normalised": surplus / max(size, 1),
            "four_point": four}


def path_detour(adjacency: sp.csr_matrix, members: np.ndarray, *, pairs: int = 2,
                seed: int = 0) -> dict:
    """Within-group distance over the distance in H, on sampled connected pairs."""
    size = int(members.size)
    if size < 4:
        return {"detour": None, "n": 0, "unavailable": 0}
    induced = adjacency[members][:, members].tocsr()
    from scipy.sparse.csgraph import connected_components

    count, labels = connected_components(induced, directed=False)
    if count == 0:
        return {"detour": None, "n": 0, "unavailable": 0}
    rng = np.random.default_rng(seed)
    ratios, unavailable = [], 0
    eligible = [index for index in range(count) if (labels == index).sum() >= 2]
    if not eligible:
        return {"detour": None, "n": 0, "unavailable": pairs}
    for _ in range(pairs):
        component = int(eligible[int(rng.integers(len(eligible)))])
        members_of = np.nonzero(labels == component)[0]
        a, b = rng.choice(members_of, size=2, replace=False)
        inside = _bfs(induced, int(a)).get(int(b))
        full = _graph_distance(adjacency, int(members[int(a)]))[int(members[int(b)])]
        if inside is None or not np.isfinite(full) or full <= 0:
            unavailable += 1
            continue
        ratios.append(inside / float(full))
    return {"detour": float(np.mean(ratios)) if ratios else None,
            "n": len(ratios), "unavailable": unavailable}


def aggregate_level(per_group: list[dict], sizes: list[int]) -> dict:
    """Group-weighted and item-weighted summaries with their coverage."""
    if not per_group:
        return {"status": "UNDEFINED_NO_ELIGIBLE_GROUP", "groups": 0, "items": 0}
    keys = ("four_point", "transitivity", "cycle_surplus_normalised", "conductance",
            "largest_component_fraction", "components", "boundary_edges")
    out = {"groups": len(per_group), "items": int(sum(sizes)), "status": "MEASURED",
           "group_weighted": {}, "item_weighted": {}, "quantiles": {}}
    weights = np.asarray(sizes, dtype=float)
    for key in keys:
        values = np.asarray([entry[key] for entry in per_group], dtype=object)
        numeric = np.asarray([value if isinstance(value, (int, float)) else np.nan
                              for value in values], dtype=float)
        block = {}
        finite = np.isfinite(numeric)
        block["groups_defined"] = int(finite.sum())
        block["coverage_group"] = float(finite.mean())
        if finite.any():
            block["group_mean"] = float(numeric[finite].mean())
            block["item_weighted"] = (float(np.average(numeric[finite],
                                                       weights=weights[finite]))
                                      if weights[finite].sum() > 0 else None)
            block["median"] = float(np.median(numeric[finite]))
            block["p90"] = float(np.quantile(numeric[finite], 0.9))
        out["group_weighted"][key] = block
    four = [entry["four_point"] for entry in per_group if entry["four_point"]["n"] > 0]
    if four:
        out["four_point"] = {
            "groups": len(four),
            "mean_deviation": float(np.mean([entry["deviation"] for entry in four])),
            "mean_normalised": float(np.mean([entry["normalised"] for entry in four
                                              if entry["normalised"] is not None])),
            "samples": int(sum(entry["n"] for entry in four))}
    else:
        out["four_point"] = {"groups": 0, "status": "UNDEFINED_NO_4_COMPONENT"}
    return out


# --------------------------------------------------------------------- nulls
def global_shuffle(labels: np.ndarray, rng) -> np.ndarray:
    shuffled = labels.copy()
    usable = np.nonzero(labels >= 0)[0]
    shuffled[usable] = labels[usable][rng.permutation(usable.size)]
    return shuffled


def parent_conditioned_shuffle(labels: np.ndarray, parents: np.ndarray, rng) -> np.ndarray:
    """Permute children only inside their parent, preserving the child sizes."""
    groups: dict[int, list[int]] = defaultdict(list)
    for position, parent in enumerate(parents.tolist()):
        if parent >= 0 and labels[position] >= 0:
            groups[int(parent)].append(position)
    shuffled = labels.copy()
    for members in groups.values():
        members = np.asarray(members)
        shuffled[members] = labels[members][rng.permutation(members.size)]
    return shuffled


def activity_stratified_shuffle(labels: np.ndarray, strata: np.ndarray, rng) -> np.ndarray:
    shuffled = labels.copy()
    for stratum in np.unique(strata):
        positions = np.nonzero((strata == stratum) & (labels >= 0))[0]
        if positions.size:
            shuffled[positions] = labels[positions][rng.permutation(positions.size)]
    return shuffled
