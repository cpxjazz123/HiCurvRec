"""Per-pair features on the visible graph (issue #246 C20-C22).

Every field names its unit; unreachable distances are -1 with a mask column, never 0.  PPR is a
probability mass and is stored under its own name.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from tools.graph_infra import pairs as pair_ops
from tools.graph_infra import ppr as ppr_ops
from tools.graph_infra import topology


def _row(csr: sp.csr_matrix, node: int) -> np.ndarray:
    start, end = csr.indptr[int(node)], csr.indptr[int(node) + 1]
    return csr.indices[start:end].astype(np.int64)


def _row_weights(csr: sp.csr_matrix, node: int):
    start, end = csr.indptr[int(node)], csr.indptr[int(node) + 1]
    return csr.indices[start:end].astype(np.int64), csr.data[start:end].astype(np.float64)


def _group_by_source(pairs):
    index = {}
    for position, (source, target) in enumerate(np.asarray(pairs, dtype=np.int64).tolist()):
        index.setdefault(int(source), []).append((position, int(target)))
    return index


def graph_hops(csr: sp.csr_matrix, pairs, *, directed: bool, chunk: int = 200) -> np.ndarray:
    """Unweighted hop count per pair; -1 when unreachable.

    The distance rows come from scipy's C implementation: a python breadth-first walk would
    have to sweep the whole connected component for every unreachable pair, which is orders of
    magnitude slower on this graph.
    """
    pairs = np.asarray(pairs, dtype=np.int64)
    index = _group_by_source(pairs)
    sources = sorted(index)
    out = np.full(pairs.shape[0], -1.0, dtype=np.float64)
    for start in range(0, len(sources), chunk):
        block = np.array(sources[start:start + chunk], dtype=np.int64)
        distance = dijkstra(csr, directed=directed, unweighted=True, indices=block)
        for row, source in enumerate(block.tolist()):
            for position, target in index[source]:
                value = distance[row, target]
                out[position] = value if np.isfinite(value) else -1.0
    return out


def subgraph_hops(forward: sp.csr_matrix, reverse: sp.csr_matrix, pairs, radius: int = 1) -> np.ndarray:
    """Shortest directed path inside the fixed ego subgraph; -1 when not connected inside."""
    pairs = np.asarray(pairs, dtype=np.int64)
    out = np.full(pairs.shape[0], -1.0, dtype=np.float64)
    cache: dict = {}
    for position, (source, target) in enumerate(pairs.tolist()):
        block = cache.get(int(source))
        if block is None:
            block = topology.ego_full(int(source), forward, reverse, radius=radius)
            cache[int(source)] = block
        local = {int(node): index for index, node in enumerate(block["nodes"].tolist())}
        if int(target) not in local:
            continue
        adjacency = {}
        for left, right in block["edges"].tolist():
            adjacency.setdefault(left, []).append(right)
        start, goal = local[int(source)], local[int(target)]
        hop = {start: 0}
        frontier = [start]
        while frontier and goal not in hop:
            nxt = []
            for node in frontier:
                for neighbour in adjacency.get(node, ()):
                    if neighbour not in hop:
                        hop[neighbour] = hop[node] + 1
                        nxt.append(neighbour)
            frontier = nxt
        if goal in hop:
            out[position] = float(hop[goal])
    return out


def two_hop_counts(forward: sp.csr_matrix, reverse: sp.csr_matrix, pairs) -> np.ndarray:
    out = np.zeros(len(pairs), dtype=np.int64)
    for position, (source, target) in enumerate(np.asarray(pairs, dtype=np.int64).tolist()):
        out[position] = np.intersect1d(_row(forward, source), _row(reverse, target)).size
    return out


def jaccard_fields(forward: sp.csr_matrix, reverse: sp.csr_matrix, pairs) -> dict:
    out_out = np.full(len(pairs), np.nan)
    out_in = np.full(len(pairs), np.nan)
    for position, (source, target) in enumerate(np.asarray(pairs, dtype=np.int64).tolist()):
        a, b = _row(forward, source), _row(forward, target)
        value = pair_ops.jaccard(a.tolist(), b.tolist())
        if value is not None:
            out_out[position] = value
        a, b = _row(reverse, source), _row(reverse, target)
        value = pair_ops.jaccard(a.tolist(), b.tolist())
        if value is not None:
            out_in[position] = value
    return {"jaccard_out": out_out, "jaccard_in": out_in}


def bhattacharyya_out(forward: sp.csr_matrix, pairs) -> np.ndarray:
    out = np.full(len(pairs), np.nan)
    for position, (source, target) in enumerate(np.asarray(pairs, dtype=np.int64).tolist()):
        left, right = _row_weights(forward, source), _row_weights(forward, target)
        left_total, right_total = left[1].sum(), right[1].sum()
        if left_total <= 0 or right_total <= 0:
            continue
        distribution_left = {int(i): float(w) / left_total for i, w in zip(left[0], left[1])}
        distribution_right = {int(i): float(w) / right_total for i, w in zip(right[0], right[1])}
        value = pair_ops.bhattacharyya(distribution_left, distribution_right)
        if value is not None:
            out[position] = value
    return out


def witness_fields(forward: sp.csr_matrix, reverse: sp.csr_matrix, pairs, group: str) -> dict:
    witness = np.full((len(pairs), 2), -1, dtype=np.int64)
    kinds = np.zeros(len(pairs), dtype=np.int64)
    kind_of = {
        "direct": 0, "shared_successor": 1, "shared_predecessor": 2, "two_hop_chain": 3,
        "degree_matched": 4,
    }
    for position, (source, target) in enumerate(np.asarray(pairs, dtype=np.int64).tolist()):
        evidence = pair_ops.label_evidence(forward, reverse, (source, target), group)
        for slot, node in enumerate(evidence["witness"][:2]):
            witness[position, slot] = int(node)
        kinds[position] = kind_of.get(evidence["kind"], 9)
    return {"witness": witness, "label_kind": kinds,
            "kind_names": np.array(["direct", "shared_successor", "shared_predecessor",
                                    "two_hop_chain", "degree_matched", "other"])}


def compute(pairs, forward, reverse, undirected, *, with_ppr: bool = True, solver=None,
            ppr_values=None, ppr_meta=None) -> dict:
    """All per-pair fields, computed on the visible graph only."""
    pairs = np.asarray(pairs, dtype=np.int64)
    direct_weight = np.array([int(forward[s, t]) for s, t in pairs.tolist()], dtype=np.int64)
    hop_full = graph_hops(undirected, pairs, directed=False)
    hop_directed = graph_hops(forward, pairs, directed=True)
    hop_subgraph = subgraph_hops(forward, reverse, pairs, radius=1)
    fields = {
        "direct_weight_visible": direct_weight,
        "hop_full_visible": hop_full,
        "reachable_undirected": (hop_full >= 0).astype(np.int8),
        "hop_directed_src_dst": hop_directed,
        "reachable_directed": (hop_directed >= 0).astype(np.int8),
        "hop_subgraph": hop_subgraph,
        "reachable_in_subgraph": (hop_subgraph >= 0).astype(np.int8),
        "two_hop_count": two_hop_counts(forward, reverse, pairs),
    }
    fields.update(jaccard_fields(forward, reverse, pairs))
    fields["jaccard_out_missing"] = np.isnan(fields["jaccard_out"]).astype(np.int8)
    fields["jaccard_in_missing"] = np.isnan(fields["jaccard_in"]).astype(np.int8)
    fields["bhattacharyya_out"] = bhattacharyya_out(forward, pairs)
    fields["bhattacharyya_out_missing"] = np.isnan(fields["bhattacharyya_out"]).astype(np.int8)
    if with_ppr:
        if ppr_values is None:
            solver = solver or ppr_ops.Solver(forward, alpha=0.15)
            solved = solver.solve_pairs(pairs)
            fields["ppr"] = solved["values"]
            meta = {"residual_max": solved["residual_max"], "alpha": solved["alpha"],
                    "coverage": solved["coverage"], "unit": solved["unit"],
                    "method": solved.get("method", "sparse LU (exact)"),
                    "iterations_max": solved.get("iterations_max")}
        else:
            # the whole set of banks was solved in one pass; reuse the values
            fields["ppr"] = np.asarray(ppr_values, dtype=np.float64)
            meta = dict(ppr_meta or {})
        fields["ppr_meta"] = meta
    return fields
