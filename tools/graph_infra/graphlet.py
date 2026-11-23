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

    def _independent(nodes: set[int]) -> bool:
        listed = sorted(nodes)
        for index, left in enumerate(listed):
            for right in listed[index + 1:]:
                if _has_edge(undirected, left, right):
                    return False
        return True

    fork = len(successors) >= 2 and _independent(successors)
    merge = len(predecessors) >= 2 and _independent(predecessors)
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
