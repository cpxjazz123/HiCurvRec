"""Graph, edge splits, ego neighbourhoods and negatives for the GraphMoRE arm.

Everything here obeys the leakage rule in issue #237 C13: the propagation graph, the
ego neighbourhoods, the shortest paths used by the distortion loss and the negative
samples are all built from the **training** edge split only.  Validation and test edges
are held out before any of that is computed.

The edge split is over the undirected edges of the frozen behaviour graph
(``results/research/behavior_graph/graph.npz``), whose CSR ``data`` holds AF3 values and
therefore cannot be used as adjacency weights (C3).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp

def _repo_root() -> Path:
    """Walk up until the repository marker directories are present."""
    for candidate in Path(__file__).resolve().parents:
        marker = candidate / "stage2_RQ-VAE" / "baseline"
        if marker.is_dir() and (candidate / "CLAUDE.md").is_file():
            return candidate
    raise RuntimeError("repository root not found above " + __file__)


REPO = _repo_root()
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
METADATA = REPO / "results" / "research" / "behavior_graph" / "metadata.json"
EMBEDDINGS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
SPLIT_SEED = 42
VAL_FRACTION = 0.08
TEST_FRACTION = 0.08
EGO_CAP = {2: 64, 3: 128}


def load_graph() -> tuple[sp.csr_matrix, dict]:
    """Unit adjacency from the CSR pattern; ``data`` is AF3, not adjacency."""
    raw = np.load(GRAPH, allow_pickle=True)
    n = len(raw["indptr"]) - 1
    adjacency = sp.csr_matrix(
        (np.ones(len(raw["indices"]), dtype=np.int8), raw["indices"], raw["indptr"]),
        shape=(n, n))
    return adjacency, json.loads(METADATA.read_text(encoding="utf-8"))


def undirected_edges(adjacency: sp.csr_matrix) -> np.ndarray:
    """Unique ``(i, j)`` pairs with ``i < j``."""
    upper = sp.triu(adjacency, k=1).tocoo()
    return np.stack([upper.row, upper.col], axis=1).astype(np.int64)


def split_edges(edges: np.ndarray, seed: int = SPLIT_SEED) -> dict:
    """Deterministic train/val/test split over undirected edges."""
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(edges.shape[0])
    n_val = int(round(edges.shape[0] * VAL_FRACTION))
    n_test = int(round(edges.shape[0] * TEST_FRACTION))
    val = shuffled[:n_val]
    test = shuffled[n_val:n_val + n_test]
    train = shuffled[n_val + n_test:]
    return {"train": edges[train], "val": edges[val], "test": edges[test],
            "sizes": {"train": int(train.size), "val": int(n_val), "test": int(n_test)}}


def adjacency_from_edges(n: int, edges: np.ndarray) -> sp.csr_matrix:
    """Symmetric unit adjacency restricted to ``edges``."""
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(rows.size, dtype=np.int8)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def sample_negatives(n: int, count: int, forbidden: set, seed: int) -> np.ndarray:
    """Random ``(i, j)`` pairs absent from ``forbidden`` (the whole observed graph).

    They are *unobserved* pairs, not verified non-interactions -- issue #237 C12.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        batch = max(int(count * 1.5) - len(out), 64)
        i = rng.integers(0, n, size=batch)
        j = rng.integers(0, n, size=batch)
        keep = i != j
        for a, b in zip(i[keep], j[keep]):
            key = (int(min(a, b)), int(max(a, b)))
            if key in forbidden:
                continue
            out.append(key)
            if len(out) == count:
                break
    return np.array(out, dtype=np.int64)


def edge_key_set(edges: np.ndarray) -> set:
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    return {(int(a), int(b)) for a, b in zip(lo, hi)}


def ego_indicator(adjacency: sp.csr_matrix, hops: int, cap: int) -> sp.csr_matrix:
    """Row-normalised ego-neighbourhood indicator (self included), capped by BFS order.

    Running a GCN on the full graph and mean-pooling with this matrix equals running the
    GCN on each ego subgraph and pooling it -- the receptive field of an L-layer GCN is
    contained in the ego set whenever the cap is at least L.  The cap exists because the
    3-hop neighbourhood of a degree-2220 hub would otherwise be the whole graph.
    """
    n = adjacency.shape[0]
    indptr, indices = adjacency.indptr, adjacency.indices
    rows, cols, counts = [], [], np.zeros(n, dtype=np.int64)
    for node in range(n):
        seen = {node: 0}
        frontier = [node]
        for step in range(1, hops + 1):
            nxt = []
            for current in frontier:
                for neighbour in indices[indptr[current]:indptr[current + 1]]:
                    key = int(neighbour)
                    if key not in seen:
                        seen[key] = step
                        nxt.append(key)
                        if len(seen) >= cap:
                            break
                if len(seen) >= cap:
                    break
            if len(seen) >= cap or not nxt:
                break
            frontier = nxt
        members = sorted(seen, key=lambda item: (seen[item], item))
        counts[node] = len(members)
        rows.append(np.full(len(members), node, dtype=np.int64))
        cols.append(np.array(members, dtype=np.int64))
    row_index = np.concatenate(rows)
    col_index = np.concatenate(cols)
    values = 1.0 / counts[row_index]
    matrix = sp.csr_matrix((values, (row_index, col_index)), shape=(n, n))
    return matrix


def graph_distances(edges: np.ndarray, pairs: np.ndarray, n: int) -> np.ndarray:
    """Shortest-path length on the training edge set for each pair."""
    adjacency = adjacency_from_edges(n, edges)
    sources = np.unique(pairs[:, 0])
    distance = {}
    for source in sources:
        distance[int(source)] = sp.csgraph.dijkstra(adjacency, indices=int(source),
                                                   unweighted=True)
    out = np.empty(pairs.shape[0], dtype=np.float64)
    for position, (a, b) in enumerate(pairs):
        value = distance[int(a)][int(b)]
        out[position] = value if np.isfinite(value) else float("inf")
    return out


def load_embeddings() -> np.ndarray:
    return np.load(EMBEDDINGS).astype(np.float32)


def build(seed: int = SPLIT_SEED) -> dict:
    """Assemble every array the training script needs, with the splits frozen."""
    adjacency, metadata = load_graph()
    n = adjacency.shape[0]
    edges = undirected_edges(adjacency)
    split = split_edges(edges, seed)
    train_edges = split["train"]
    train_adjacency = adjacency_from_edges(n, train_edges)
    observed = edge_key_set(edges)
    negatives = {
        "train": sample_negatives(n, train_edges.shape[0], observed, seed + 1),
        "val": sample_negatives(n, split["val"].shape[0], observed, seed + 2),
        "test": sample_negatives(n, split["test"].shape[0], observed, seed + 3),
    }
    # distortion pairs: a fixed sample of training edges, with their graph distance
    rng = np.random.default_rng(seed + 4)
    sample = rng.choice(train_edges.shape[0], size=min(20000, train_edges.shape[0]),
                        replace=False)
    distortion_pairs = train_edges[sample]
    distortion_target = graph_distances(train_edges, distortion_pairs, n)
    finite = np.isfinite(distortion_target) & (distortion_target > 0)
    return {
        "n": n,
        "metadata": metadata,
        "graph_sha256": metadata["outputs"]["graph.npz"]["sha256"],
        "edges_all": edges,
        "split": {key: split[key] for key in ("train", "val", "test")},
        "split_sizes": split["sizes"],
        "train_adjacency": train_adjacency,
        "negatives": negatives,
        "distortion_pairs": distortion_pairs,
        "distortion_target": distortion_target,
        "distortion_mask": finite,
        "ego": {hops: ego_indicator(train_adjacency, hops, EGO_CAP[hops])
                for hops in (2, 3)},
        "embeddings": load_embeddings(),
    }


if __name__ == "__main__":
    bundle = build()
    summary = {
        "n": bundle["n"],
        "split_sizes": bundle["split_sizes"],
        "graph_sha256": bundle["graph_sha256"],
        "negatives": {key: int(value.shape[0]) for key, value in bundle["negatives"].items()},
        "ego_nnz": {hops: int(matrix.nnz) for hops, matrix in bundle["ego"].items()},
        "distortion_pairs": int(bundle["distortion_pairs"].shape[0]),
        "distortion_finite": int(bundle["distortion_mask"].sum()),
        "embeddings": list(bundle["embeddings"].shape),
    }
    print(json.dumps(summary, indent=1))
