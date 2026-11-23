"""Full-resolution sectional curvature (issue #248).

Everything here is derived from the training-visible directed graph only. The statistic is the
GraphMoRE Appendix A.1 sectional curvature of the geodesic triangle (centre m, neighbours b and c),
with the **full** reachable reference set:

    kappa(m;b,c) = mean_{a in R(m)} [ g(a,m)^2 + g(b,c)^2/4 - ( g(a,b)^2 + g(a,c)^2 )/2 ]

``g`` is the unweighted shortest-path distance on the undirected projection of the visible graph.
The mean over the full reference set has the exact closed form

    kappa(m;b,c) = ( S2(m) + N * g(b,c)^2/4 - ( S2(b) + S2(c) )/2 ) / N

with ``S2(v) = sum_{a in R(v)} g(a,v)^2`` and ``N = |R(m)|``; this is an algebraic rearrangement,
not an approximation. See results/research/issue248/CONTRACT.md.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path, connected_components

from tools.graph_infra.sectional import projection, ZERO_TOLERANCE

#: Canonical 3-node directed code bits, in the order used by tools/graph_infra/graphlet.py.
ORDERED_PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


def component_labels(undirected: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Component id and size of every node in the undirected projection."""
    count, labels = connected_components(undirected, directed=False)
    sizes = np.bincount(labels, minlength=count).astype(np.int64)
    return labels.astype(np.int64), sizes


def distance_intermediates(undirected: sp.csr_matrix, chunk: int = 2000) -> dict:
    """Exact per-node distance histogram and its moments, computed by BFS from every node.

    The histogram is the lossless intermediate: every moment used downstream (S1, S2, the
    eccentricity, the component size) is recomputed from it, so no distance matrix is stored.
    """
    n = undirected.shape[0]
    counts = []
    offsets = np.zeros(n + 1, dtype=np.int64)
    s1 = np.zeros(n, dtype=np.float64)
    s2 = np.zeros(n, dtype=np.float64)
    eccentricity = np.zeros(n, dtype=np.int64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        distances = shortest_path(undirected, directed=False, unweighted=True,
                                  indices=np.arange(start, stop))
        for row in range(stop - start):
            values = distances[row]
            finite = values[np.isfinite(values)]
            histogram = np.bincount(finite.astype(np.int64), minlength=1)
            counts.append(histogram.astype(np.int64))
            levels = np.arange(len(histogram), dtype=np.float64)
            s1[start + row] = float((levels * histogram).sum())
            s2[start + row] = float((levels ** 2 * histogram).sum())
            eccentricity[start + row] = len(histogram) - 1
    flat = np.concatenate(counts) if counts else np.zeros(0, dtype=np.int64)
    offsets[1:] = np.cumsum([len(c) for c in counts])
    labels, sizes = component_labels(undirected)
    # `component_size` is indexed by node; `component_sizes_by_id` is indexed by component id.
    return {'histogram_offsets': offsets, 'histogram_counts': flat, 's1': s1, 's2': s2,
            'eccentricity': eccentricity, 'component': labels,
            'component_size': sizes[labels], 'component_sizes_by_id': sizes}


def moments_from_histogram(intermediates: dict) -> dict:
    """Recompute S1/S2/eccentricity from the stored histogram only."""
    offsets = intermediates['histogram_offsets']
    counts = intermediates['histogram_counts']
    n = len(offsets) - 1
    s1 = np.zeros(n); s2 = np.zeros(n); ecc = np.zeros(n, dtype=np.int64)
    for node in range(n):
        histogram = counts[offsets[node]:offsets[node + 1]]
        levels = np.arange(len(histogram), dtype=np.float64)
        s1[node] = float((levels * histogram).sum())
        s2[node] = float((levels ** 2 * histogram).sum())
        ecc[node] = len(histogram) - 1
    return {'s1': s1, 's2': s2, 'eccentricity': ecc}


def curvature_closed_form(s2_m, s2_b, s2_c, references, bc_dist_sq):
    """Exact value of Eq. A.1 over the full reachable reference set."""
    return (s2_m + references * bc_dist_sq / 4.0 - (s2_b + s2_c) / 2.0) / references


def curvature_excluding_triple(s2_m, s2_b, s2_c, references, bc_dist_sq):
    """The same statistic with the triple's own three vertices removed from the reference set."""
    count = references - 3
    if count <= 0:
        return np.nan
    return ((s2_m - 2.0) + count * bc_dist_sq / 4.0
            - ((s2_b - 1.0 - bc_dist_sq) + (s2_c - 1.0 - bc_dist_sq)) / 2.0) / count


def _directed_edge_keys(directed: sp.csr_matrix) -> np.ndarray:
    coo = directed.tocoo()
    keys = coo.row.astype(np.int64) * directed.shape[0] + coo.col.astype(np.int64)
    return np.unique(keys)


def _edge_lookup(keys: np.ndarray, n: int, sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Boolean membership of the directed edges (sources, targets) in a sorted key array."""
    probe = sources.astype(np.int64) * n + targets.astype(np.int64)
    position = np.searchsorted(keys, probe)
    position = np.clip(position, 0, len(keys) - 1)
    return keys[position] == probe


def triple_chunks(undirected: sp.csr_matrix, directed: sp.csr_matrix, intermediates: dict,
                   centres: np.ndarray, projection_keys: np.ndarray, directed_keys: np.ndarray) -> list[dict]:
    """All unordered neighbour pairs of the given centres, with their curvature and structure class.

    A triple is (m, b, c) with b < c; the neighbour set is the undirected projection's, so the
    pair is always at distance 1 or 2 in the projection. The structure class is the canonical
    directed 3-node graphlet class of the induced subgraph on {m, b, c} in the **visible directed**
    graph, so direction is preserved in the classification.
    """
    from tools.graph_infra.graphlet import canonical_lookup
    lookup = canonical_lookup()
    n = undirected.shape[0]
    s2 = intermediates['s2']
    sizes = intermediates['component_size']
    out = []
    for centre in np.asarray(centres, dtype=np.int64):
        neighbours = undirected.indices[undirected.indptr[centre]:undirected.indptr[centre + 1]]
        degree = len(neighbours)
        if degree < 2:
            continue
        first, second = np.triu_indices(degree, 1)
        b = neighbours[first].astype(np.int64)
        c = neighbours[second].astype(np.int64)
        adjacent = (_edge_lookup(projection_keys, n, b, c)
                    | _edge_lookup(projection_keys, n, c, b))
        bc_dist_sq = np.where(adjacent, 1, 4).astype(np.int8)
        references = np.full(len(b), sizes[centre], dtype=np.int64)
        curvature = curvature_closed_form(s2[centre], s2[b], s2[c], references, bc_dist_sq)
        m = np.full(len(b), centre, dtype=np.int64)
        code = np.zeros(len(b), dtype=np.int64)
        for left, right in ORDERED_PAIRS:
            source = (m, b, c)[left]
            target = (m, b, c)[right]
            bit = _edge_lookup(directed_keys, n, source, target).astype(np.int64)
            code = (code << 1) | bit
        out.append({'centre': m, 'b': b, 'c': c, 'bc_dist_sq': bc_dist_sq,
                    'curvature': curvature, 'effective_references': references,
                    'valid': np.ones(len(b), dtype=bool),
                    'invalid_reason': np.zeros(len(b), dtype=np.int8),
                    'structure_class': lookup[code].astype(np.int8)})
    return out


def item_aggregates(centres: np.ndarray, chunk: dict, zero_tolerance: float = ZERO_TOLERANCE) -> dict:
    """Per-item distribution summary of one centre's triples."""
    values = chunk['curvature']
    valid = chunk['valid'] & np.isfinite(values)
    finite = values[valid]
    total = len(values)
    count = int(valid.sum())
    result = {'total': total, 'valid': count, 'invalid': total - count,
              'positive': int((finite > zero_tolerance).sum()),
              'zero': int((np.abs(finite) <= zero_tolerance).sum()),
              'negative': int((finite < -zero_tolerance).sum()),
              'mean': np.nan, 'variance': np.nan, 'minimum': np.nan, 'maximum': np.nan,
              'quantiles': np.full(5, np.nan)}
    if count:
        result['mean'] = float(finite.mean())
        result['variance'] = float(finite.var())
        result['minimum'] = float(finite.min())
        result['maximum'] = float(finite.max())
        result['quantiles'] = np.quantile(finite, [0, .25, .5, .75, 1])
    return result


def zero_ratio(count: int, values: np.ndarray, zero_tolerance: float = ZERO_TOLERANCE) -> float:
    if not len(values):
        return np.nan
    return float(np.mean(np.abs(values) <= zero_tolerance))
