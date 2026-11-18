"""Freeze the QDA evaluation sample before any run is inspected.

The experiment's first hard requirement: the TRAIN behaviour edges, their matched
non-edges and every graphlet/community label are fixed on disk with their hashes
*before* a single metric is computed, and the sample is split by an edge hash into
a selection half (used to pick analysis methods) and a final half (used once).

Nothing here reads a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
ROLES = REPO / "results" / "research" / "issue215" / "orca_roles_frozen.csv"
OUT = HERE / "sample"
SEED = 42
N_ITEMS = 24587


def sha256_of(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def behaviour_edges() -> np.ndarray:
    frame = pd.read_parquet(TRAIN)
    pairs = set()
    for history, target in zip(frame["history"].to_numpy(), frame["target"].to_numpy()):
        history = list(history)
        for left, right in zip(history, history[1:]):
            pairs.add((left, right) if left < right else (right, left))
        if history:
            tail = history[-1]
            pairs.add((tail, target) if tail < target else (target, tail))
    return np.array(sorted(pairs), dtype=np.int64)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    edges = behaviour_edges()
    adjacency = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(N_ITEMS, N_ITEMS)
    ).tocsr()
    adjacency = (adjacency + adjacency.T).tocsr()
    triangles = np.rint(np.asarray(
        (adjacency @ adjacency)[edges[:, 0], edges[:, 1]]).ravel()).astype(np.int32)
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.int64)
    neighbours = [set(adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]].tolist())
                  for i in range(N_ITEMS)]

    graph = nx.from_scipy_sparse_array(adjacency)
    communities = nx.community.louvain_communities(graph, seed=SEED, resolution=1.0)
    community = np.full(N_ITEMS, -1, dtype=np.int32)
    for index, block in enumerate(communities):
        community[np.fromiter(block, dtype=np.int64)] = index
    inside = (community[edges[:, 0]] == community[edges[:, 1]]).astype(np.int8)

    # K1,3 present on the edge: one endpoint has at least three other neighbours.
    star = ((degree[edges[:, 0]] - 1 >= 3) | (degree[edges[:, 1]] - 1 >= 3)).astype(np.int8)

    # One degree-quintile matched non-edge per edge, drawn on the target endpoint.
    cuts = np.quantile(degree, [0.2, 0.4, 0.6, 0.8])
    degree_bin = np.searchsorted(cuts, degree, side="right")
    by_bin = [np.flatnonzero(degree_bin == b) for b in range(5)]
    rng = np.random.default_rng(SEED)
    negatives = np.empty_like(edges)
    for index, (left, right) in enumerate(edges):
        pool = by_bin[degree_bin[right]]
        candidate = int(pool[rng.integers(pool.size)])
        for _ in range(20):
            if candidate == left or candidate in neighbours[right]:
                candidate = int(pool[rng.integers(pool.size)])
                continue
            break
        negatives[index] = (left, candidate)

    split = np.array([
        int(hashlib.sha256(f"{int(a)}:{int(b)}".encode()).hexdigest()[:8], 16) % 2
        for a, b in edges], dtype=np.int8)

    roles = np.full(N_ITEMS, -1, dtype=np.int8)
    if ROLES.is_file():
        table = pd.read_csv(ROLES)
        roles[table["item_id"].to_numpy(dtype=np.int64)] = table["role"].to_numpy(dtype=np.int8)

    payload = {
        "edges": edges.astype(np.int32),
        "negatives": negatives.astype(np.int32),
        "triangles": triangles,
        "star": star,
        "inside": inside,
        "community": community,
        "degree": degree.astype(np.int32),
        "role": roles,
        "split": split,
    }
    np.savez_compressed(OUT / "sample.npz", **payload)
    meta = {
        "items": N_ITEMS,
        "edges": int(len(edges)),
        "negatives": int(len(negatives)),
        "communities": len(communities),
        "inside_share": float(inside.mean()),
        "star_share": float(star.mean()),
        "triangles_mean": float(triangles.mean()),
        "split_selection": int((split == 0).sum()),
        "split_final": int((split == 1).sum()),
        "roles_labelled": int((roles >= 0).sum()),
        "sha256": {name: sha256_of(array) for name, array in payload.items()},
        "sources": {
            "train_parquet": str(TRAIN.relative_to(REPO)),
            "roles": str(ROLES.relative_to(REPO)) if ROLES.is_file() else None,
        },
        "seed": SEED,
        "rule": ("positives: TRAIN consecutive history pairs + history-tail to target, undirected; "
                 "negatives: one per edge on the same degree quintile of the target endpoint, "
                 "not adjacent; split: sha256(u:v) % 2"),
    }
    (OUT / "sample_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("edges", "communities", "inside_share",
                                           "star_share", "triangles_mean",
                                           "split_selection", "split_final")}, indent=1))


if __name__ == "__main__":
    main()
