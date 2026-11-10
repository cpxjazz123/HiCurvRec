"""Build the frozen structure prior of issue #150, once, before training.

The builder reads the TRAIN split only, rebuilds the behaviour graph the frozen
AF3 cache was measured on and verifies the rebuild edge for edge, then writes

    results/research/issue150/structure_prior.npz     tau, layers, rho, edges, triangles
    results/research/issue150/structure_prior.json    its SHA-256 and its counts

Run it with the Stage 2 interpreter (see CLAUDE.md §2 for the path). Nothing is
written outside the repository (§86).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import struct_prior  # noqa: E402

TRAIN_RELATIVE = ("results", "stage0_build_parquet", "train.parquet")
EMBEDDING_RELATIVE = ("stage1_GeneEmbedding", "output", "sentence_t5.npy")
AF3_CACHE_RELATIVE = ("results", "research", "issue121", "community_curvature",
                      "train_edge_af3.npz")
WEIGHT_STEP = 0.5
LOUVAIN_RESOLUTION = 1.0
SEED = 42


def _repository_root() -> Path:
    for parent in TREE.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(TREE))


def _train_graph(root: Path) -> dict:
    """The TRAIN binary graph, verified against the frozen AF3 cache."""
    import pandas as pd
    import scipy.sparse as sp

    frame = pd.read_parquet(root.joinpath(*TRAIN_RELATIVE), columns=["history", "target"])
    weights: dict[tuple[int, int], float] = {}
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for first, second in zip(values[:-1], values[1:]):
            if first == second:
                continue
            key = (first, second) if first < second else (second, first)
            weights[key] = weights.get(key, 0.0) + WEIGHT_STEP
    nodes = np.asarray(sorted({item for edge in weights for item in edge}), dtype=np.int64)
    index = {int(item): position for position, item in enumerate(nodes)}
    rows = np.fromiter((index[first] for first, _ in weights), dtype=np.int64, count=len(weights))
    cols = np.fromiter((index[second] for _, second in weights), dtype=np.int64, count=len(weights))
    data = np.fromiter(weights.values(), dtype=np.float64, count=len(weights))
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()
    binary = (matrix != 0).astype(np.int8).tocsr()
    binary = binary.maximum(binary.T)
    binary.setdiag(0)
    binary.eliminate_zeros()
    full_rows, full_cols = binary.nonzero()
    keep = full_rows < full_cols
    edge_rows, edge_cols = full_rows[keep], full_cols[keep]
    order = np.lexsort((edge_cols, edge_rows))
    edge_rows, edge_cols = edge_rows[order], edge_cols[order]
    degree = np.asarray(binary.sum(axis=1)).ravel().astype(np.int64)
    size = int(binary.shape[0])
    integer = binary.astype(np.int32)
    product = (integer @ integer).multiply(integer).tocoo()
    inside = product.row < product.col
    key = product.row[inside].astype(np.int64) * size + product.col[inside].astype(np.int64)
    counts = np.rint(product.data[inside]).astype(np.int64)
    per_edge = np.zeros(edge_rows.size, dtype=np.int64)
    per_edge[np.searchsorted(edge_rows.astype(np.int64) * size + edge_cols, key)] = counts
    af3 = 4.0 - degree[edge_rows] - degree[edge_cols] + 3.0 * per_edge
    cache = root.joinpath(*AF3_CACHE_RELATIVE)
    status = "NOT_CHECKED"
    if cache.is_file():
        frozen = np.load(cache)
        if (int(frozen["nodes"]) != nodes.size or int(frozen["edges"]) != edge_rows.size
                or not np.array_equal(frozen["rows"].astype(np.int64), edge_rows)
                or not np.array_equal(frozen["cols"].astype(np.int64), edge_cols)
                or not np.array_equal(frozen["af3"].astype(np.float64), af3)):
            raise ValueError("the TRAIN graph rebuild does not reproduce the frozen AF3 cache")
        status = "MATCHED"
    return {"nodes": nodes, "binary": binary, "rows": edge_rows, "cols": edge_cols,
            "degree": degree, "triangles_per_edge": per_edge, "af3": af3,
            "af3_cache": str(cache), "af3_status": status}


def _communities(binary) -> np.ndarray:
    import networkx as nx
    graph = nx.from_scipy_sparse_array(binary.tocsr())
    groups = nx.community.louvain_communities(graph, weight=None,
                                              resolution=LOUVAIN_RESOLUTION, seed=SEED)
    labels = np.full(int(binary.shape[0]), -1, dtype=np.int64)
    for label, members in enumerate(groups):
        labels[np.fromiter(sorted(members), dtype=np.int64, count=len(members))] = label
    if (labels < 0).any():
        raise RuntimeError("the community pass left a node unlabelled")
    return labels


def _all_triangles(adjacency, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Every triangle of the graph once, as sorted triples of graph positions."""
    neighbours = []
    for node in range(int(adjacency.shape[0])):
        start, stop = int(adjacency.indptr[node]), int(adjacency.indptr[node + 1])
        neighbours.append(np.asarray(adjacency.indices[start:stop], dtype=np.int64))
    found: set[tuple[int, int, int]] = set()
    for first, second in zip(rows.tolist(), cols.tolist()):
        common = np.intersect1d(neighbours[first], neighbours[second], assume_unique=True)
        for third in common.tolist():
            found.add(tuple(sorted((first, second, int(third)))))
    return np.asarray(sorted(found), dtype=np.int64).reshape(-1, 3)


def build(root: Path) -> dict:
    graph = _train_graph(root)
    edge_triangle = graph["triangles_per_edge"] > 0
    # tau_i: share of incident edges that lie in at least one triangle
    incident = np.zeros(graph["nodes"].size, dtype=np.float64)
    positive = np.zeros(graph["nodes"].size, dtype=np.float64)
    np.add.at(incident, graph["rows"], 1.0)
    np.add.at(incident, graph["cols"], 1.0)
    np.add.at(positive, graph["rows"], edge_triangle.astype(np.float64))
    np.add.at(positive, graph["cols"], edge_triangle.astype(np.float64))
    tau = np.divide(positive, incident, out=np.zeros_like(positive), where=incident > 0)
    layer, rho = struct_prior.stratify(tau)
    communities = _communities(graph["binary"])
    triangles = _all_triangles(graph["binary"], graph["rows"], graph["cols"])

    item_ids = graph["nodes"]
    n_items = int(np.load(root.joinpath(*EMBEDDING_RELATIVE), mmap_mode="r").shape[0])
    if item_ids.size and int(item_ids.max()) >= n_items:
        raise ValueError("the TRAIN graph names an item outside the embedding rows")
    edges = np.stack([item_ids[graph["rows"]], item_ids[graph["cols"]]], axis=1)
    triangle_items = item_ids[triangles] if triangles.size else np.zeros((0, 3), dtype=np.int64)
    layer_items = np.full(n_items, -1, dtype=np.int64)
    layer_items[item_ids] = layer
    edge_layer = np.where(layer[graph["rows"]] == layer[graph["cols"]],
                          layer[graph["rows"]], -1)
    metadata = {
        "issue": 150,
        "mechanism": "structure-stratified target marginal of the balanced assignment",
        "n_items": n_items,
        "n_nodes": int(item_ids.size),
        "n_edges": int(edges.shape[0]),
        "n_triangles": int(triangle_items.shape[0]),
        "groups": int(rho.size),
        "rho": [float(value) for value in rho],
        "tau_cuts": [float(value) for value in struct_prior.TAU_CUTS],
        "tau_zero_layer": bool(struct_prior.TAU_ZERO_LAYER),
        "layer_sizes": [int(value) for value in np.bincount(layer)],
        "tau_mean": float(tau.mean()),
        "louvain_resolution": LOUVAIN_RESOLUTION,
        "seed": SEED,
        "af3_cache": graph["af3_cache"],
        "af3_cache_status": graph["af3_status"],
        "items_without_structure": int((layer_items < 0).sum()),
        "train_parquet_sha256": struct_prior.sha256_of(root.joinpath(*TRAIN_RELATIVE)),
        "embedding_sha256": struct_prior.sha256_of(root.joinpath(*EMBEDDING_RELATIVE)),
        "input_rule": ("TRAIN split only; the validation and test splits and any future "
                       "behaviour are never read"),
    }
    return {"tau": tau.astype(np.float64), "layer": layer, "rho": rho,
            "item_ids": item_ids, "community": communities, "edges": edges.astype(np.int32),
            "triangles": triangle_items.astype(np.int32),
            "edge_layer": edge_layer.astype(np.int64), "metadata": metadata}


def main() -> int:
    root = _repository_root()
    payload = build(root)
    path = struct_prior.table_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, tau=payload["tau"], layer=payload["layer"], rho=payload["rho"],
        item_ids=payload["item_ids"], community=payload["community"],
        edges=payload["edges"], triangles=payload["triangles"],
        edge_layer=payload["edge_layer"],
        metadata=np.asarray(json.dumps(payload["metadata"], sort_keys=True)),
    )
    manifest = {"table": str(path), "table_sha256": struct_prior.sha256_of(path),
                "metadata": payload["metadata"]}
    struct_prior.manifest_path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: manifest["metadata"][key] for key in
                      ("n_items", "n_nodes", "n_edges", "n_triangles", "groups", "rho",
                       "items_without_structure", "af3_cache_status")}, indent=2))
    print("table_sha256", manifest["table_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
