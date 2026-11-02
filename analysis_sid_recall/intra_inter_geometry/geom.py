"""Corrected intra-group and new inter-group geometry (Issue #133).

All structural metrics use the **binary** symmetric item graph B (no diagonal),
built from the audited #128 TRAIN-only weighted graph; the weighted data are kept
separately and are never mixed into a binary cut/volume.

Fixes over the #132 implementation:

1. **Singleton contamination** — every occupied prefix group is in the census, but
   the intra-group averages use only groups with >= 2 items. Singletons keep
   their items and edges in the source graph and in the inter-group quotient.
2. **Triangle overflow** — exact sparse ``(A @ A).multiply(A).sum() // 6`` with an
   int32 accumulator; no dense int8 cube, no size cap unless the square's
   nonzero count would exceed ``TRIANGLE_NNZ_MAX`` (declared, with a status).
3. **Conductance units** — binary cut ``sum(deg_B(S)) - 2*internal_B(S)`` over binary
   volumes ``min(vol_B(S), vol_B(V\\S))``; a zero denominator is
   ``UNDEFINED_ZERO_VOLUME`` and the value is asserted to lie in [0, 1].
4. **Inter-group quotient** — one node per occupied prefix (singletons included),
   an undirected edge iff at least one source edge crosses the prefixes, no
   self-loops. Metrics: connectivity, LCC share, density, degree quantiles,
   triangles, transitivity, cycle rank per node, bridge fraction, Gromov status.
5. **Curvature** — exact Forman (``4 - deg(u) - deg(v)``) on every quotient edge;
   Ollivier-Ricci on a declared sampled set with the exact 0/1/2-hop transport
   computed by the validated max-flow identity, labelled ``SAMPLED_ESTIMATE``
   with its edge coverage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "full_census_geometry"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interaction_strength"))

TRIANGLE_NNZ_MAX = 2 * 10 ** 8
GROMOV_MAX = 24
ORC_EDGES = 512


def binary_graph(weighted: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetric binary adjacency without the diagonal, from the weighted graph."""
    binary = (weighted != 0).astype(np.int8).tocsr()
    binary = binary.maximum(binary.T)
    binary.setdiag(0)
    binary.eliminate_zeros()
    degrees = np.asarray(binary.sum(axis=1)).ravel()
    assert int(degrees.sum()) == 2 * int(binary.nnz // 2), "degree sum must be 2E"
    return binary


def groups_of(labels: np.ndarray) -> list[np.ndarray]:
    usable = np.nonzero(labels >= 0)[0]
    if usable.size == 0:
        return []
    order = np.argsort(labels[usable], kind="mergesort")
    nodes = usable[order]
    values = labels[nodes]
    boundaries = np.flatnonzero(np.diff(values)) + 1
    return [chunk for chunk in np.split(nodes, boundaries) if chunk.size]


def triangles_sparse(block: sp.csr_matrix) -> tuple[int, str]:
    """Exact triangle count; returns (count, status)."""
    if block.nnz == 0:
        return 0, "EXACT"
    square = block.astype(np.int32) @ block.astype(np.int32)
    if square.nnz > TRIANGLE_NNZ_MAX:
        return -1, "BLOCKED_NOT_FULL_CENSUS"
    count = int(square.multiply(block.astype(np.int32)).sum() // 6)
    return count, "EXACT"


def intra_metrics(binary: sp.csr_matrix, degree_b: np.ndarray, members: np.ndarray,
                  total_volume: float) -> dict:
    """Binary structural metrics of one induced subgraph (any size >= 2)."""
    size = int(members.size)
    block = binary[members][:, members].tocsr()
    internal = int(block.nnz // 2)
    from scipy.sparse.csgraph import connected_components
    components, labels = connected_components(block, directed=False)
    sizes = np.bincount(labels) if components else np.zeros(0, dtype=int)
    largest = int(sizes.max()) if sizes.size else 0
    volume = float(degree_b[members].sum())
    cut = float(max(volume - 2.0 * internal, 0.0))
    complement = float(total_volume - volume)
    denominator = min(volume, complement)
    if denominator <= 0:
        conductance, conductance_status = None, "UNDEFINED_ZERO_VOLUME"
    else:
        conductance = cut / denominator
        conductance_status = "MEASURED"
        assert -1e-9 <= conductance <= 1.0 + 1e-9, "binary conductance outside [0, 1]"
        conductance = float(min(max(conductance, 0.0), 1.0))
    triangles, triangle_status = triangles_sparse(block)
    if triangle_status != "EXACT":
        transitivity = None
    else:
        row_degree = np.asarray(block.sum(axis=1)).ravel().astype(np.int64)
        triples = int((row_degree * (row_degree - 1)).sum() // 2)
        transitivity = (3.0 * triangles / triples) if triples > 0 else None
    return {"size": size, "internal_edges": internal, "components": int(components),
            "component_sizes": sorted(int(value) for value in sizes.tolist()),
            "largest_component": largest,
            "largest_component_fraction": largest / size,
            "triangles": triangles if triangle_status == "EXACT" else None,
            "triangle_status": triangle_status,
            "transitivity": transitivity,
            "cycle_rank": internal - size + int(components),
            "cycle_rank_normalised": (internal - size + int(components)) / size,
            "volume_binary": volume, "cut_binary": cut,
            "conductance": conductance, "conductance_status": conductance_status,
            "boundary_edges": cut / 2.0}


def singleton_metrics(binary: sp.csr_matrix, degree_b: np.ndarray, member: int,
                      total_volume: float) -> dict:
    volume = float(degree_b[member])
    cut = volume
    denominator = min(volume, float(total_volume - volume))
    conductance = (cut / denominator) if denominator > 0 else None
    return {"size": 1, "internal_edges": 0, "components": 1, "component_sizes": [1],
            "largest_component": 1, "largest_component_fraction": 1.0,
            "triangles": 0, "triangle_status": "EXACT", "transitivity": None,
            "cycle_rank": 0, "cycle_rank_normalised": 0.0,
            "volume_binary": volume, "cut_binary": cut,
            "conductance": conductance,
            "conductance_status": "MEASURED" if denominator > 0 else "UNDEFINED_ZERO_VOLUME",
            "boundary_edges": cut / 2.0}


# ------------------------------------------------------------------ quotient
def quotient(binary: sp.csr_matrix, labels: np.ndarray) -> sp.csr_matrix:
    """One node per occupied prefix; an edge iff a source edge crosses prefixes."""
    usable = np.nonzero(labels >= 0)[0]
    if usable.size == 0:
        return sp.csr_matrix((0, 0))
    remap = _remap(labels, usable)
    coo = binary.tocoo()
    left = remap[coo.row]
    right = remap[coo.col]
    keep = (left >= 0) & (right >= 0) & (left != right)
    if not keep.any():
        return sp.csr_matrix((int(remap[usable].max()) + 1,) * 2)
    size = int(remap.max()) + 1
    matrix = sp.coo_matrix((np.ones(int(keep.sum())), (left[keep], right[keep])),
                           shape=(size, size)).tocsr()
    matrix.data[:] = 1.0
    matrix = matrix.maximum(matrix.T)
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


def _remap(labels: np.ndarray, usable: np.ndarray) -> np.ndarray:
    values = labels[usable]
    unique, inverse = np.unique(values, return_inverse=True)
    out = np.full(labels.shape[0], -1, dtype=np.int64)
    out[usable] = inverse
    return out


def quotient_gromov(matrix: sp.csr_matrix) -> dict:
    """Exact four-point deviation on components up to GROMOV_MAX nodes, else BLOCKED."""
    from scipy.sparse.csgraph import connected_components
    import itertools
    from census import bfs_distances                      # the validated BFS

    count, labels = connected_components(matrix, directed=False)
    values, blocked, quadruples = [], 0, 0
    for index in range(count):
        component = np.nonzero(labels == index)[0]
        if component.size < 4:
            continue
        if component.size > GROMOV_MAX:
            blocked += 1
            continue
        distances = bfs_distances(matrix[component][:, component].tocsr(),
                                  np.arange(component.size), np.arange(component.size))
        local_max = 0
        for a, b, c, d in itertools.combinations(range(component.size), 4):
            sums = sorted((distances[a, b] + distances[c, d],
                           distances[a, c] + distances[b, d],
                           distances[a, d] + distances[b, c]), reverse=True)
            local_max = max(local_max, sums[0] - sums[1])
            quadruples += 1
        values.append(local_max)
    if not values:
        return {"status": "BLOCKED_NOT_FULL_CENSUS" if blocked else
                          "UNDEFINED_BY_DEFINITION",
                "components_exact": 0, "components_blocked": blocked,
                "quadruples": quadruples, "delta_max": None}
    status = "EXACT" if blocked == 0 else "PARTIAL_BLOCKED"
    return {"status": status, "components_exact": len(values),
            "components_blocked": blocked, "quadruples": quadruples,
            "delta_max": int(max(values))}


def forman_edges(matrix: sp.csr_matrix) -> np.ndarray:
    """Combinatorial Forman curvature 4 - deg(u) - deg(v) on every quotient edge."""
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    coo = matrix.tocoo()
    keep = coo.row < coo.col
    return (4.0 - degree[coo.row[keep]] - degree[coo.col[keep]]).astype(np.float64)


def orc_sampled(matrix: sp.csr_matrix, *, edges: int = ORC_EDGES,
                seed: int = 20261102) -> dict:
    """Ollivier-Ricci on a declared sampled edge set, exact 0/1/2-hop transport."""
    from strength import w1_hop12

    coo = matrix.tocoo()
    keep = coo.row < coo.col
    pairs = list(zip(coo.row[keep].tolist(), coo.col[keep].tolist()))
    total = len(pairs)
    if total == 0:
        return {"status": "UNDEFINED_NO_EDGES", "edges_total": 0, "edges_measured": 0,
                "mean": None, "median": None, "negative_fraction": None,
                "edge_coverage": 0.0}
    rng = np.random.default_rng(seed)
    chosen = pairs if total <= edges else [pairs[int(i)] for i in
                                           rng.choice(total, size=edges, replace=False)]
    measures = []
    for a, b in chosen:
        support = np.unique(np.concatenate((
            np.asarray([a]), matrix.indices[matrix.indptr[a]:matrix.indptr[a + 1]],
            np.asarray([b]), matrix.indices[matrix.indptr[b]:matrix.indptr[b + 1]])))
        position = {int(node): index for index, node in enumerate(support.tolist())}
        inside = matrix[support][:, support].tocsr()
        left = _lazy(matrix, a, position)
        right = _lazy(matrix, b, position)
        measures.append(1.0 - w1_hop12(left, right, inside))
    values = np.asarray(measures)
    status = "EXACT" if len(chosen) == total else "SAMPLED_ESTIMATE"
    return {"status": status, "edges_total": total, "edges_measured": len(chosen),
            "edge_coverage": len(chosen) / total,
            "mean": float(values.mean()), "median": float(np.median(values)),
            "sd": float(values.std()),
            "negative_fraction": float((values < 0).mean())}


def _lazy(matrix: sp.csr_matrix, node: int, position: dict) -> np.ndarray:
    size = len(position)
    weight = np.zeros(size)
    index = position.get(int(node))
    if index is not None:
        weight[index] += 0.5
    neighbours = matrix.indices[matrix.indptr[node]:matrix.indptr[node + 1]]
    degree = max(float(matrix.indptr[node + 1] - matrix.indptr[node]), 1.0)
    for neighbour in neighbours.tolist():
        target = position.get(int(neighbour))
        if target is not None:
            weight[target] += 0.5 / degree
    total = weight.sum()
    return weight / total if total > 0 else weight


def quotient_metrics(matrix: sp.csr_matrix) -> dict:
    from scipy.sparse.csgraph import connected_components
    nodes = int(matrix.shape[0])
    edges = int(matrix.nnz // 2)
    if nodes == 0 or edges == 0:
        return {"nodes": nodes, "edges": edges, "status": "UNDEFINED_EMPTY_QUOTIENT"}
    count, labels = connected_components(matrix, directed=False)
    sizes = np.bincount(labels)
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    triangles, triangle_status = triangles_sparse(matrix)
    row_degree = degree.astype(np.int64)
    triples = int((row_degree * (row_degree - 1)).sum() // 2)
    transitivity = (3.0 * triangles / triples) if (triples > 0
                                                   and triangle_status == "EXACT") else None
    forman = forman_edges(matrix) if edges else np.zeros(0)
    return {"nodes": nodes, "edges": edges, "components": int(count),
            "largest_component_fraction": float(sizes.max() / nodes),
            "density": edges / (nodes * (nodes - 1) / 2),
            "degree_median": float(np.median(degree)), "degree_p90": float(np.quantile(degree, 0.9)),
            "degree_max": int(degree.max()),
            "triangles": triangles if triangle_status == "EXACT" else None,
            "triangle_status": triangle_status, "transitivity": transitivity,
            "cycle_rank_per_node": (edges - nodes + int(count)) / nodes,
            "forman_mean": float(forman.mean()) if forman.size else None,
            "forman_median": float(np.median(forman)) if forman.size else None,
            "forman_negative_fraction": float((forman < 0).mean()) if forman.size else None,
            "forman_edges": int(forman.size)}
