"""Typed structural motifs, computed from the graph and never from a sampling (issue #245 §3).

The contract the issue insists on: a *type* is a structural predicate on the real adjacency,
so two different samplings of the same ego subgraph must produce the same labels.  Nothing in
this module draws random nodes; sampling only decides which nodes are *reported*.  A type is
therefore never "a second drawing of the same ego".
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

MOTIFS = ("chain", "fork", "merge", "triangle", "reciprocal")

FAMILY_NAMES = None  # filled after FAMILY_BY_CLASS is built


def _sorted_adjacency(csr: sp.csr_matrix, node: int) -> set[int]:
    start, end = csr.indptr[node], csr.indptr[node + 1]
    return set(int(x) for x in csr.indices[start:end]) - {int(node)}


def _has_edge(undirected: sp.csr_matrix, left: int, right: int) -> bool:
    start, end = undirected.indptr[left], undirected.indptr[left + 1]
    return bool(np.any(undirected.indices[start:end] == right))


def root_motifs(root: int, out_csr: sp.csr_matrix, in_csr: sp.csr_matrix,
                undirected: sp.csr_matrix) -> dict:
    """All five predicates for one root, on the real graph.

    chain       a directed path root -> a -> b with b != root and no edge root--b (induced 3-path)
    fork        at least two successors and no edge between any pair of them
    merge       at least two predecessors and no edge between any pair of them
    triangle    an undirected triangle containing the root
    reciprocal  at least one neighbour with edges in both directions
    """
    root = int(root)
    successors = _sorted_adjacency(out_csr, root)
    predecessors = _sorted_adjacency(in_csr, root)

    chain = False
    for first in successors:
        for second in _sorted_adjacency(out_csr, first):
            if second != root and not _has_edge(undirected, root, second):
                chain = True
                break
        if chain:
            break

    def _has_independent_pair(nodes: set[int]) -> bool:
        """One pair of neighbours with no edge between them is enough (existential, not universal)."""
        listed = sorted(nodes)
        for index, left in enumerate(listed):
            for right in listed[index + 1:]:
                if not _has_edge(undirected, left, right):
                    return True
        return False

    fork = len(successors) >= 2 and _has_independent_pair(successors)
    merge = len(predecessors) >= 2 and _has_independent_pair(predecessors)
    reciprocal = len(successors & predecessors) > 0

    triangle = False
    neighbours = successors | predecessors
    for left in neighbours:
        for right in _sorted_adjacency(undirected, left) & neighbours:
            if right != root and _has_edge(undirected, right, root):
                triangle = True
                break
        if triangle:
            break

    return {"chain": chain, "fork": fork, "merge": merge,
            "triangle": triangle, "reciprocal": reciprocal,
            "degree_in": len(predecessors), "degree_out": len(successors)}


def label_nodes(out_csr: sp.csr_matrix, in_csr: sp.csr_matrix,
                undirected: sp.csr_matrix, items) -> dict:
    """Per-node labels, degrees and per-type coverage/overlap statistics."""
    rows = {}
    for item in items:
        rows[int(item)] = root_motifs(int(item), out_csr, in_csr, undirected)
    degrees = np.array([rows[key]["degree_out"] + rows[key]["degree_in"] for key in rows], dtype=np.int64)

    per_type = {}
    for name in MOTIFS:
        mask = np.array([rows[key][name] for key in rows], dtype=bool)
        type_degrees = degrees[mask]
        per_type[name] = {
            "items": int(mask.sum()),
            "coverage": float(mask.mean()) if mask.size else 0.0,
            "mean_degree": float(type_degrees.mean()) if type_degrees.size else None,
            "median_degree": float(np.median(type_degrees)) if type_degrees.size else None,
        }

    overlap = {}
    for index, left in enumerate(MOTIFS):
        for right in MOTIFS[index + 1:]:
            both = sum(1 for key in rows if rows[key][left] and rows[key][right])
            overlap[f"{left}+{right}"] = int(both)

    empty = sum(1 for key in rows if not any(rows[key][name] for name in MOTIFS))
    return {
        "rows": rows,
        "per_type": per_type,
        "overlap": overlap,
        "no_motif": int(empty),
        "n_items": int(len(rows)),
        "mutual_exclusion_note": "types overlap by construction; the overlap counts are reported "
                                 "so a later arm cannot claim independently sampled structures",
    }


def degree_stratified(labels: dict, buckets: int = 5) -> dict:
    """Carrier rate per degree bucket plus the carriers' degree shift.

    This is a descriptive table, not a null model: no degree-preserving rewiring was run, so
    the numbers say how much a type tracks degree, and must not be read as "structure beyond
    degree".  A G(n, m) null would confound degree with everything else and is deliberately
    not reported as if it were degree-matched.
    """
    rows = labels["rows"]
    keys = np.array(sorted(rows), dtype=np.int64)
    degrees = np.array([rows[int(k)]["degree_in"] + rows[int(k)]["degree_out"] for k in keys],
                       dtype=np.int64)
    edges = np.quantile(degrees, np.linspace(0, 1, buckets + 1))
    buckets_idx = np.clip(np.searchsorted(edges[1:-1], degrees, side="right"), 0, buckets - 1)
    report = {}
    for name in MOTIFS:
        carriers = np.array([rows[int(k)][name] for k in keys], dtype=bool)
        per_bucket = []
        for bucket in range(buckets):
            selected = buckets_idx == bucket
            per_bucket.append({
                "bucket": bucket,
                "degree_min": int(degrees[selected].min()) if selected.any() else None,
                "degree_max": int(degrees[selected].max()) if selected.any() else None,
                "n": int(selected.sum()),
                "rate": float(carriers[selected].mean()) if selected.any() else None,
            })
        report[name] = {
            "carriers": int(carriers.sum()),
            "rate": float(carriers.mean()) if carriers.size else None,
            "mean_degree_carriers": float(degrees[carriers].mean()) if carriers.any() else None,
            "mean_degree_non_carriers": float(degrees[~carriers].mean()) if (~carriers).any() else None,
            "by_degree_bucket": per_bucket,
        }
    return {"buckets": buckets, "report": report,
            "null_model": "none: no degree-preserving rewiring was run in this pass"}

# ---------------------------------------------------------------- exact 3-node graphlets (v2)
import itertools as _itertools

ORDERED_PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
_PERMUTATIONS = tuple(_itertools.permutations(range(3)))


def matrix_from_code(code: int) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int8)
    for bit, (i, j) in enumerate(ORDERED_PAIRS):
        matrix[i, j] = (int(code) >> (len(ORDERED_PAIRS) - 1 - bit)) & 1
    return matrix


def code_from_matrix(matrix: np.ndarray) -> int:
    code = 0
    for i, j in ORDERED_PAIRS:
        code = (code << 1) | int(matrix[i, j])
    return int(code)


def canonical_with_perm(code: int) -> tuple[int, tuple]:
    """Minimum code over the six relabelings and the permutation that attains it.

    Convention: `relabel(A, perm)[i, j] = A[perm[i], perm[j]]`, i.e. new index i is old index
    perm[i]; the old position `p` becomes new position `perm.index(p)`.
    """
    matrix = matrix_from_code(code)
    best_code, best_perm = None, None
    for perm in _PERMUTATIONS:
        candidate = matrix[np.ix_(list(perm), list(perm))]
        value = code_from_matrix(candidate)
        if best_code is None or value < best_code:
            best_code, best_perm = value, perm
    return int(best_code), tuple(best_perm)


def canonical_code(code: int) -> int:
    return canonical_with_perm(code)[0]


def orbits(code: int) -> list[list[int]]:
    """Automorphism orbits of the canonical pattern, as sorted position lists."""
    canonical = matrix_from_code(canonical_code(code))
    seen: set = set()
    result = []
    for start in range(3):
        if start in seen:
            continue
        orbit = set()
        for perm in _PERMUTATIONS:
            if np.array_equal(canonical[np.ix_(list(perm), list(perm))], canonical):
                orbit.add(perm[start])
        orbit = sorted(orbit)
        seen.update(orbit)
        result.append(orbit)
    return sorted(result)


def orbit_of_position(code: int, position: int) -> int:
    canonical, perm = canonical_with_perm(code)
    new_position = perm.index(int(position))
    for index, orbit in enumerate(orbits(canonical)):
        if new_position in orbit:
            return index
    raise AssertionError("every position belongs to an orbit")


def _base_family(code: int) -> str:
    """Readable base name; uniqueness is enforced by `FAMILY_BY_CLASS`."""
    canonical = canonical_code(code)
    matrix = matrix_from_code(canonical)
    pairs = [(i, j) for i, j in ORDERED_PAIRS if matrix[i, j]]
    count = len(pairs)
    reciprocal_pairs = [tuple(sorted((i, j))) for i, j in _itertools.combinations(range(3), 2)
                        if matrix[i, j] and matrix[j, i]]
    if count == 0:
        return "empty"
    if count == 1:
        return "single_edge"
    if count == 2 and not reciprocal_pairs:
        if matrix.sum(axis=1).max() == 2:
            return "out_star"
        if matrix.sum(axis=0).max() == 2:
            return "in_star"
        return "chain"
    if count == 2 and reciprocal_pairs:
        return "reciprocal_pair"
    if count == 3 and not reciprocal_pairs:
        cycle = ((matrix[0, 1] and matrix[1, 2] and matrix[2, 0])
                 or (matrix[0, 2] and matrix[2, 1] and matrix[1, 0]))
        return "fb_triangle" if cycle else "ff_triangle"
    if count == 6:
        return "complete_bidirectional"
    # reciprocal-bearing classes: name by the third node's in/out degree so every class differs
    thirds = [node for node in range(3)
              if not any(node in pair for pair in reciprocal_pairs)]
    if not thirds:
        return f"reciprocal_all_{count}edges"
    third = thirds[0]
    third_in = int(matrix[:, third].sum())
    third_out = int(matrix[third, :].sum())
    if len(reciprocal_pairs) == 2:
        base = "double_reciprocal"
    elif len(reciprocal_pairs) == 1 and count == 5:
        base = "reciprocal_chain"
    elif len(reciprocal_pairs) == 1 and count == 4:
        base = "reciprocal_plus"
    else:
        base = "reciprocal_with"
    return f"{base}_third_in{third_in}_out{third_out}"


def _build_family_table() -> dict:
    from collections import Counter
    classes = sorted({canonical_code(code) for code in range(64)})
    base = {code: _base_family(code) for code in classes}
    counts = Counter(base.values())
    table = {}
    for code, name in base.items():
        table[code] = name if counts[name] == 1 else f"{name}_{code}"
    if len(set(table.values())) != len(table):
        raise AssertionError("family names must be one-to-one")
    return table


FAMILY_BY_CLASS = _build_family_table()


def family_of(code: int) -> str:
    """Deterministic, one-to-one family name for each of the 16 unlabeled classes."""
    return FAMILY_BY_CLASS[canonical_code(code)]


def _csr_set(csr: np.ndarray, node: int) -> set:
    start, end = csr.indptr[int(node)], csr.indptr[int(node) + 1]
    return set(int(x) for x in csr.indices[start:end]) - {int(node)}


def fork_relaxed(root: int, out_csr, in_csr) -> bool:
    """∃ two out-neighbours with no edge between them, either direction (issue #246 C17)."""
    successors = sorted(_csr_set(out_csr, root))
    for index, left in enumerate(successors):
        for right in successors[index + 1:]:
            if right not in _csr_set(out_csr, left) and left not in _csr_set(out_csr, right):
                return True
    return False


def merge_relaxed(root: int, out_csr, in_csr) -> bool:
    return fork_relaxed(root, in_csr, out_csr)


def motif_instances(root: int, out_csr, in_csr, graph_version: str = "v2",
                    max_pairs: int | None = None) -> list:
    """Every induced 3-node instance containing the centre, with its exact class and orbit.

    ``max_pairs`` caps the enumeration for very high-degree centres; the record then says
    ``truncated`` so a sampled instance is never presented as a complete neighbourhood.
    """
    root = int(root)
    neighbours = sorted(_csr_set(out_csr, root) | _csr_set(in_csr, root))
    adjacency = {node: _csr_set(out_csr, node) for node in [root] + neighbours}
    instances = []
    truncated = False
    for index, (first, second) in enumerate(_itertools.combinations(neighbours, 2)):
        if max_pairs is not None and index >= max_pairs:
            truncated = True
            break
        triple = (root, first, second)
        matrix = np.zeros((3, 3), dtype=np.int8)
        local = {root: 0, first: 1, second: 2}
        edges = []
        for source in triple:
            for target in adjacency[source]:
                if target in local and target != source:
                    matrix[local[source], local[target]] = 1
                    edges.append((source, target))
        code = code_from_matrix(matrix)
        graphlet_id = canonical_code(code)
        centre_in = sum(1 for source, target in edges if target == root)
        centre_out = sum(1 for source, target in edges if source == root)
        instances.append({
            "node_ids": triple,
            "center": root,
            "code": code,
            "graphlet_id": graphlet_id,
            "family": family_of(code),
            "orbit": orbit_of_position(code, 0),
            "center_role": (centre_in, centre_out),
            "edges": edges,
            "graph_version": graph_version,
            "kind": "strict_instance",
            "truncated": truncated,
        })
    return instances


# ---------------------------------------------------------------- vectorised 3-node counts
def _edge_keys(csr: sp.csr_matrix) -> np.ndarray:
    coo = csr.tocoo()
    return np.sort(coo.row.astype(np.int64) * csr.shape[0] + coo.col.astype(np.int64))


def _contains(sorted_keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    if sorted_keys.size == 0 or values.size == 0:
        return np.zeros(values.shape, dtype=bool)
    positions = np.searchsorted(sorted_keys, values)
    positions = np.clip(positions, 0, sorted_keys.size - 1)
    return sorted_keys[positions] == values


def canonical_lookup() -> np.ndarray:
    global _CANONICAL_LOOKUP
    try:
        return _CANONICAL_LOOKUP
    except NameError:
        pass
    _CANONICAL_LOOKUP = np.array([canonical_code(code) for code in range(64)], dtype=np.int64)
    return _CANONICAL_LOOKUP


def family_index() -> dict:
    global _FAMILY_INDEX
    try:
        return _FAMILY_INDEX
    except NameError:
        pass
    classes = sorted(FAMILY_BY_CLASS)
    _FAMILY_INDEX = {code: index for index, code in enumerate(classes)}
    _FAMILY_NAMES = [FAMILY_BY_CLASS[code] for code in classes]
    globals()["_FAMILY_NAMES"] = _FAMILY_NAMES
    return _FAMILY_INDEX


def family_names() -> list:
    family_index()
    return list(globals()["_FAMILY_NAMES"])


def family_counts(out_csr: sp.csr_matrix, in_csr: sp.csr_matrix, chunk: int = 500_000) -> dict:
    """Exact per-centre counts of every 3-node family (vectorised, no sampling)."""
    n = out_csr.shape[0]
    keys = _edge_keys(out_csr)
    lookup = canonical_lookup()
    index = family_index()
    names = family_names()
    slot_of_code = np.full(64, -1, dtype=np.int64)
    for code, slot in index.items():
        slot_of_code[code] = slot
    counts = np.zeros((n, len(names)), dtype=np.int64)
    instances = np.zeros(n, dtype=np.int64)
    for centre in range(n):
        out_row = out_csr.indices[out_csr.indptr[centre]:out_csr.indptr[centre + 1]].astype(np.int64)
        in_row = in_csr.indices[in_csr.indptr[centre]:in_csr.indptr[centre + 1]].astype(np.int64)
        neighbours = np.union1d(out_row, in_row)
        neighbours = neighbours[neighbours != centre]
        if neighbours.size < 2:
            continue
        first, second = np.triu_indices(neighbours.size, 1)
        total = first.size
        instances[centre] = total
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            a = neighbours[first[start:stop]]
            b = neighbours[second[start:stop]]
            bit0 = np.isin(a, out_row)
            bit1 = np.isin(b, out_row)
            bit2 = np.isin(a, in_row)
            bit4 = np.isin(b, in_row)
            bit3 = _contains(keys, a * n + b)
            bit5 = _contains(keys, b * n + a)
            code = ((bit0.astype(np.int64) << 5) | (bit1.astype(np.int64) << 4)
                    | (bit2.astype(np.int64) << 3) | (bit3.astype(np.int64) << 2)
                    | (bit4.astype(np.int64) << 1) | bit5.astype(np.int64))
            slots = slot_of_code[lookup[code]]
            counts[centre] += np.bincount(slots, minlength=len(names))[: len(names)]
    return {"counts": counts.astype(np.int64), "instances": instances, "names": names,
            "vertices": 3, "sampling": "none (exact over all neighbour pairs)"}


def sampled_instances(out_csr: sp.csr_matrix, in_csr: sp.csr_matrix, centres,
                      max_per_centre: int, seed: int, graph_version: str = "v2") -> list:
    """At most ``max_per_centre`` instances per centre, evenly spaced over its pair list."""
    rng = np.random.default_rng(seed)
    records = []
    for centre in centres:
        produced = motif_instances(int(centre), out_csr, in_csr, graph_version,
                                   max_pairs=max(4 * max_per_centre, 200))
        if len(produced) > max_per_centre:
            pick = np.sort(rng.choice(len(produced), size=max_per_centre, replace=False))
            produced = [produced[index] for index in pick.tolist()]
        records.extend(produced)
    return records


def canonical_code_generic(code: int, vertices: int) -> int:
    """Canonical code for n-node directed patterns (n=4 uses 12 bits, 24 permutations)."""
    if vertices == 3:
        return canonical_code(code)
    pairs = [(i, j) for i in range(vertices) for j in range(vertices) if i != j]
    matrix = np.zeros((vertices, vertices), dtype=np.int8)
    for bit, (i, j) in enumerate(pairs):
        matrix[i, j] = (int(code) >> (len(pairs) - 1 - bit)) & 1
    best = None
    for perm in _itertools.permutations(range(vertices)):
        candidate = code_from_matrix_generic(matrix[np.ix_(list(perm), list(perm))], pairs)
        if best is None or candidate < best:
            best = candidate
    return int(best)


def code_from_matrix_generic(matrix: np.ndarray, pairs) -> int:
    code = 0
    for i, j in pairs:
        code = (code << 1) | int(matrix[i, j])
    return int(code)


def four_node_instances(centre: int, out_csr, in_csr, max_instances: int, seed: int,
                        graph_version: str = "v2") -> list:
    """Sampled 4-node induced patterns containing the centre; sampling is declared, not full."""
    rng = np.random.default_rng(seed)
    root = int(centre)
    neighbours = sorted(_csr_set(out_csr, root) | _csr_set(in_csr, root))[:12]
    if len(neighbours) < 3:
        return []
    adjacency = {node: _csr_set(out_csr, node) for node in [root] + neighbours}
    triples = list(_itertools.combinations(neighbours, 3))
    if len(triples) > max_instances:
        pick = np.sort(rng.choice(len(triples), size=max_instances, replace=False))
        triples = [triples[index] for index in pick.tolist()]
    pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
    records = []
    for triple in triples:
        nodes = (root,) + triple
        local = {node: index for index, node in enumerate(nodes)}
        matrix = np.zeros((4, 4), dtype=np.int8)
        edges = []
        for source in nodes:
            for target in adjacency[source]:
                if target in local and target != source:
                    matrix[local[source], local[target]] = 1
                    edges.append((source, target))
        code = code_from_matrix_generic(matrix, pairs)
        records.append({
            "node_ids": nodes,
            "center": root,
            "code": code,
            "graphlet_id": canonical_code_generic(code, 4),
            "family": "sampled_4node",
            "center_role": (sum(1 for s, t in edges if t == root),
                            sum(1 for s, t in edges if s == root)),
            "edges": edges,
            "graph_version": graph_version,
            "kind": "strict_instance",
            "vertices": 4,
        })
    return records
