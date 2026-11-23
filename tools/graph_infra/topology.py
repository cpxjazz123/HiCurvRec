"""Local topology: induced ego subgraphs, real root roles, true hops, batching (issue #246 §3).

Fixes against v1:

* the root's roles come from the **real CSR** (`root_roles`), never from a second random sample;
* `hop` is the shortest hop in the source graph, recomputed for every included node, and is kept
  apart from `traversal_depth` (the BFS depth at which the sampling found it);
* `root_local` is explicit, so the centre may be any id;
* `cap` counts **neighbours**, the root is extra (`cap_includes_root = False`);
* `batch_egos` pads edges and features as well as nodes, with masks.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

K1_OUT = 8
K1_IN = 8
K2_PER = 4
CAP = 32

FEATURE_NAMES = ("is_root", "hop", "degree_out", "degree_in",
                 "has_in_edge", "has_out_edge", "is_mutual")

ROLE_ROOT = 0
ROLE_OUT = 1
ROLE_IN = 2
ROLE_MUTUAL = 3
ROLE_DEEPER = 4


def _neighbours(csr: sp.csr_matrix, node: int) -> np.ndarray:
    start, end = csr.indptr[int(node)], csr.indptr[int(node) + 1]
    return csr.indices[start:end].astype(np.int64)


def _sample(csr: sp.csr_matrix, node: int, k: int, rng) -> np.ndarray:
    cols = _neighbours(csr, node)
    if len(cols) == 0 or k <= 0:
        return np.zeros(0, dtype=np.int64)
    if len(cols) <= k:
        return cols.copy()
    return cols[rng.choice(len(cols), size=k, replace=False)].astype(np.int64)


def root_roles(root: int, out_csr: sp.csr_matrix, in_csr: sp.csr_matrix) -> dict:
    """Real CSR membership of the root's neighbours; nothing here is sampled."""
    out_side = set(int(x) for x in _neighbours(out_csr, root)) - {int(root)}
    in_side = set(int(x) for x in _neighbours(in_csr, root)) - {int(root)}
    return {"out": out_side, "in": in_side, "both": out_side & in_side}


def reference_hops(root: int, out_csr: sp.csr_matrix, in_csr: sp.csr_matrix,
                   radius: int = 2) -> dict:
    """Shortest hop over the union of in/out edges, by plain BFS (the definition)."""
    hop = {int(root): 0}
    frontier = [int(root)]
    for depth in range(1, radius + 1):
        nxt = []
        for node in frontier:
            for csr in (out_csr, in_csr):
                for neighbour in _neighbours(csr, node).tolist():
                    if neighbour not in hop:
                        hop[neighbour] = depth
                        nxt.append(neighbour)
        frontier = nxt
        if not frontier:
            break
    return hop


def _induced_edges(nodes: np.ndarray, out_csr: sp.csr_matrix) -> np.ndarray:
    local = {int(node): index for index, node in enumerate(nodes.tolist())}
    edges = []
    for node in nodes.tolist():
        for target in _neighbours(out_csr, node).tolist():
            index = local.get(target)
            if index is not None and target != node:
                edges.append((local[node], index))
    return np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)


def ego_features(nodes, root: int, out_csr: sp.csr_matrix, in_csr: sp.csr_matrix,
                 hop, role: dict) -> dict:
    """Model-independent node features in the frozen order `FEATURE_NAMES`."""
    nodes = np.asarray(nodes, dtype=np.int64)
    hop = np.asarray(hop, dtype=np.int64)
    if hop.shape != nodes.shape:
        raise ValueError("hop labels must match the node array")
    out_degree = np.array([len(_neighbours(out_csr, n)) for n in nodes.tolist()], dtype=np.int64)
    in_degree = np.array([len(_neighbours(in_csr, n)) for n in nodes.tolist()], dtype=np.int64)
    is_root = nodes == int(root)
    has_out = np.array([int(n) in role["out"] for n in nodes.tolist()], dtype=np.int64)
    has_in = np.array([int(n) in role["in"] for n in nodes.tolist()], dtype=np.int64)
    mutual = np.array([int(n) in role["both"] for n in nodes.tolist()], dtype=np.int64)
    features = np.stack([is_root.astype(np.int64), hop, out_degree, in_degree,
                         has_in, has_out, mutual], axis=1)
    return {
        "feature_names": FEATURE_NAMES,
        "features": features,
        "root_local_index": int(np.flatnonzero(is_root)[0]),
        "node_mask": np.ones(nodes.shape, dtype=bool),
        "empty": bool(nodes.shape[0] == 1),
        "nodes": nodes,
        "hop": hop,
        "role_out": has_out.astype(bool),
        "role_in": has_in.astype(bool),
    }


def _pack(root: int, nodes: np.ndarray, edges: np.ndarray, true_hop: np.ndarray,
          traversal_depth: np.ndarray, radius: int, cap: int,
          out_csr: sp.csr_matrix, in_csr: sp.csr_matrix) -> dict:
    roles = root_roles(root, out_csr, in_csr)
    direction = np.full(len(nodes), ROLE_DEEPER, dtype=np.int64)
    for index, node in enumerate(nodes.tolist()):
        if node == int(root):
            direction[index] = ROLE_ROOT
        elif node in roles["both"]:
            direction[index] = ROLE_MUTUAL
        elif node in roles["out"]:
            direction[index] = ROLE_OUT
        elif node in roles["in"]:
            direction[index] = ROLE_IN
    features = ego_features(nodes, root, out_csr, in_csr, true_hop, roles)
    return {
        "nodes": nodes,
        "edges": edges,
        "role": true_hop,                     # v1 name kept: the true hop
        "hop": true_hop,
        "traversal_depth": traversal_depth,
        "direction": direction,
        "role_out": features["role_out"],
        "role_in": features["role_in"],
        "features": features["features"],
        "feature_names": FEATURE_NAMES,
        "root_local": int(np.flatnonzero(nodes == int(root))[0]),
        "empty_neighbourhood": bool(len(nodes) == 1),
        "root": int(root),
        "radius": int(radius),
        "cap": int(cap),
        "cap_includes_root": False,
    }


def ego_induced(root: int, out_csr: sp.csr_matrix, in_csr: sp.csr_matrix,
                rng, radius: int = 2, cap: int = CAP) -> dict:
    """Seeded, budgeted induced ego subgraph with true hops and real roles."""
    root = int(root)
    depth = {root: 0}
    order = [root]
    frontier = [root]
    for current in range(1, radius + 1):
        nxt = []
        for node in frontier:
            directions = ((out_csr, K1_OUT), (in_csr, K1_IN)) if current == 1 else ((out_csr, K2_PER), (in_csr, K2_PER))
            for csr, budget in directions:
                for neighbour in _sample(csr, node, budget, rng).tolist():
                    if neighbour in depth or neighbour == root:
                        continue
                    if len(order) >= cap + 1:
                        break
                    depth[neighbour] = current
                    order.append(neighbour)
                    nxt.append(neighbour)
        frontier = nxt
        if not frontier or len(order) >= cap + 1:
            break

    nodes = np.array(sorted(order), dtype=np.int64)
    first_hop = root_roles(root, out_csr, in_csr)
    first_hop = first_hop["out"] | first_hop["in"]
    if radius <= 2:
        # BFS cannot exceed the radius, so a non-direct node is exactly two hops away; a node
        # discovered at depth 2 that is a direct neighbour of the root is corrected to 1
        true_hop = np.array([0 if int(node) == root else 1 if int(node) in first_hop else 2
                             for node in nodes.tolist()], dtype=np.int64)
    else:
        true = reference_hops(root, out_csr, in_csr, radius=radius)
        true_hop = np.array([true.get(int(node), depth[int(node)]) for node in nodes.tolist()],
                            dtype=np.int64)
    traversal = np.array([depth[int(node)] for node in nodes.tolist()], dtype=np.int64)
    return _pack(root, nodes, _induced_edges(nodes, out_csr), true_hop, traversal,
                 radius, cap, out_csr, in_csr)


def ego_full(root: int, out_csr: sp.csr_matrix, in_csr: sp.csr_matrix, radius: int = 1) -> dict:
    """Complete (deterministic, unsampled) ego subgraph; radius 1 is exact and cheap."""
    root = int(root)
    if radius != 1:
        hops = reference_hops(root, out_csr, in_csr, radius=radius)
        nodes = np.array(sorted(hops), dtype=np.int64)
        true_hop = np.array([hops[int(node)] for node in nodes.tolist()], dtype=np.int64)
    else:
        nodes = np.array(sorted({root} | root_roles(root, out_csr, in_csr)["out"]
                                | root_roles(root, out_csr, in_csr)["in"]), dtype=np.int64)
        true_hop = np.array([0 if int(n) == root else 1 for n in nodes.tolist()], dtype=np.int64)
    block = _pack(root, nodes, _induced_edges(nodes, out_csr), true_hop,
                  true_hop.copy(), radius, int(len(nodes) - 1), out_csr, in_csr)
    block["cap"] = None
    block["cap_includes_root"] = False
    block["complete"] = True
    return block


def batch_egos(blocks, indices) -> dict:
    """Pad nodes, edges and features into a batch, with masks; never attend padded entries."""
    picked = [blocks[int(i)] for i in indices]
    if not picked:
        raise ValueError("empty batch")
    widths = [len(block["nodes"]) for block in picked]
    edge_counts = [len(block.get("edges", np.zeros((0, 2), dtype=np.int64))) for block in picked]
    width = max(widths)
    edge_width = max(1, max(edge_counts))
    nodes = np.zeros((len(picked), width), dtype=np.int64)
    node_mask = np.zeros((len(picked), width), dtype=bool)
    edges = np.zeros((len(picked), edge_width, 2), dtype=np.int64)
    edge_mask = np.zeros((len(picked), edge_width), dtype=bool)
    root_row = np.zeros(len(picked), dtype=np.int64)
    features = np.zeros((len(picked), width, len(FEATURE_NAMES)), dtype=np.int64)
    features_complete = True
    for row, block in enumerate(picked):
        count = len(block["nodes"])
        nodes[row, :count] = block["nodes"]
        node_mask[row, :count] = True
        root_row[row] = block["root_local"] if "root_local" in block else block["root_local_index"]
        block_edges = np.asarray(block.get("edges", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
        if len(block_edges):
            edges[row, :len(block_edges)] = block_edges
            edge_mask[row, :len(block_edges)] = True
        if "features" in block:
            features[row, :count] = np.asarray(block["features"], dtype=np.int64)
        else:
            features_complete = False
            hop = np.asarray(block.get("hop", np.zeros(count, dtype=np.int64)), dtype=np.int64)
            features[row, :count, 0] = (
                block["nodes"] == block["nodes"][int(root_row[row])]).astype(np.int64)
            features[row, :count, 1] = hop
    return {
        "nodes": nodes,
        "node_mask": node_mask,
        "edges": edges,
        "edge_mask": edge_mask,
        "features": features,
        "feature_names": FEATURE_NAMES,
        "features_complete": features_complete,
        "root_row": root_row,
        "batch_size": len(picked),
        "width": width,
        "max_edges": edge_width,
        "padding_policy": "right padding; masked nodes and edges are never attendable",
    }
