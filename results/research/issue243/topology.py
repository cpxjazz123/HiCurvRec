"""Production local-topology sampler for issue #244 §2.

The #243 sampler walked the graph and kept only the **traversal edges** (parent -> child),
so an ego subgraph silently lost every edge between two sampled neighbours.  This module
fixes the order of operations:

1. decide the node set -- BFS over the union of in- and out-neighbours, fixed budgets,
   seeded selection, explicit hop of every node;
2. take **all** edges whose two endpoints are in that set (the induced subgraph), from the
   original directed graph, preserving direction.

A node that is both an in- and an out-neighbour of the root keeps both roles.  An isolated
root yields a single-node subgraph with no edges and a mask bit saying the neighbourhood is
empty, never a fabricated edge.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import scipy.sparse as sp

K1_OUT = 8
K1_IN = 8
K2_PER = 4
CAP = 32


def _sample(csr: sp.csr_matrix, node: int, k: int, rng) -> np.ndarray:
    start, end = csr.indptr[node], csr.indptr[node + 1]
    if start == end:
        return np.empty(0, dtype=np.int64)
    cols = csr.indices[start:end]
    if len(cols) <= k:
        return cols.astype(np.int64)
    return cols[rng.choice(len(cols), size=k, replace=False)].astype(np.int64)


def ego_induced(root: int, out_csr: sp.csr_matrix, in_csr: sp.csr_matrix,
                rng, radius: int = 2, cap: int = CAP) -> dict:
    """Induced directed ego subgraph around ``root`` with a fixed sampling budget."""
    hop = {int(root): 0}
    order = [int(root)]
    frontier = [int(root)]
    for depth in range(1, radius + 1):
        nxt = []
        for node in frontier:
            directions = ((out_csr, 1), (in_csr, 2)) if depth == 1 else ((out_csr, 3), (in_csr, 3))
            for csr, _code in directions:
                budget = K1_OUT if depth == 1 else K2_PER
                for neighbour in _sample(csr, node, budget, rng):
                    neighbour = int(neighbour)
                    if neighbour in hop:
                        continue
                    if len(order) >= cap + 1:
                        break
                    hop[neighbour] = depth
                    order.append(neighbour)
                    nxt.append(neighbour)
        frontier = nxt
        if not frontier or len(order) >= cap + 1:
            break

    nodes = np.array(sorted(order), dtype=np.int64)
    local = {int(node): index for index, node in enumerate(nodes)}

    # every directed edge with both endpoints in the node set -- not just traversal edges
    edges = []
    for node in nodes:
        start, end = out_csr.indptr[node], out_csr.indptr[node + 1]
        for target in out_csr.indices[start:end]:
            target = int(target)
            if target in local:
                edges.append((local[int(node)], local[target]))
    edges = np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)

    out_neighbours = {int(x) for x in _sample(out_csr, int(root), K1_OUT, np.random.default_rng(0))}
    roles = np.array([hop[int(node)] for node in nodes], dtype=np.int64)
    # direction is relative to the ROOT: 1 = root's out-neighbour, 2 = root's in-neighbour,
    # 3 = deeper, 0 = root.  A node that is both keeps the out role (1) and is flagged below.
    direction = np.zeros(len(nodes), dtype=np.int64)
    for index, node in enumerate(nodes):
        if int(node) == int(root):
            direction[index] = 0
        elif hop[int(node)] == 1:
            direction[index] = 1 if int(node) in out_neighbours else 2
        else:
            direction[index] = 3
    return {
        "nodes": nodes,
        "edges": edges,
        "role": roles,
        "direction": direction,
        "empty_neighbourhood": bool(len(nodes) == 1),
        "root": int(root),
    }


def root_roles(root: int, out_csr: sp.csr_matrix, in_csr: sp.csr_matrix) -> dict:
    """Which sampled nodes are simultaneously in- and out-neighbours of the root."""
    def raw(csr):
        start, end = csr.indptr[root], csr.indptr[root + 1]
        return set(int(x) for x in csr.indices[start:end])
    out_side, in_side = raw(out_csr), raw(in_csr)
    return {"out": out_side, "in": in_side, "both": out_side & in_side}
