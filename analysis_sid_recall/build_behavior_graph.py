"""Build the behaviour graph in the project's canonical layout.

Layout (results/research/behavior_graph/):

    edges.parquet      source, target, weight      raw relations, human diffable
    graph.npz          nodes, rows, cols, af3, ... computation form (COO + CSR + AF3 + triangles)
    node_features.npy  item embeddings             RQ-VAE input, aligned by node id
    metadata.json      rule, sources, counts, sha256 of every output

The edge set is the adjacent-transition graph of the training sequences:

    E = { {i_t, i_{t+1}} | (i_1..i_T) in S_train }

A user's sequence is that user's longest history plus its target; only
consecutive items are joined (A-B-C-D gives A-B, B-C, C-D), edges are undirected
and deduplicated, and the directed transition counts are kept as the edge weight.

The AF3 target is the frozen degree-and-triangle quantity the AF3 trees read:
``kappa_ij = 4 - d_i - d_j + 3 * T_ij`` with ``d`` the degree and ``T_ij`` the
number of triangles through the edge, both on this graph.  `nodes`, `rows`,
`cols`, `af3`, `median`, `p90` are exactly the keys
``stage2_RQ-VAE/*/model/af3_structure.py::load_graph`` validates, so the file is
consumable without conversion.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        analysis_sid_recall/build_behavior_graph.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
ITEMS = REPO / "results" / "stage0_build_parquet" / "items.parquet"
FEATURES = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT = REPO / "results" / "research" / "behavior_graph"
SCHEMA = "genrec.behavior_graph.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sequences() -> list:
    """One ordered item sequence per user: the longest history plus its target."""
    frame = pd.read_parquet(TRAIN)
    longest = {}
    for user, history in zip(frame["user"].to_numpy(), frame["history"]):
        key = int(user)
        if key not in longest or len(history) > len(longest[key]):
            longest[key] = list(history)
    by_user = {}
    for user, history, target in zip(frame["user"].to_numpy(), frame["history"], frame["target"]):
        key = int(user)
        if len(history) == len(longest[key]):
            by_user[key] = [int(item) for item in history] + [int(target)]
    return [by_user[user] for user in sorted(by_user)]


def edge_triangles(rows: np.ndarray, cols: np.ndarray, n_nodes: int) -> np.ndarray:
    """Triangles through each edge, from sorted neighbour intersections."""
    neighbours: list = [None] * n_nodes
    buckets = defaultdict(list)
    for left, right in zip(rows, cols):
        buckets[int(left)].append(int(right))
        buckets[int(right)].append(int(left))
    for node, values in buckets.items():
        neighbours[node] = np.array(sorted(set(values)), dtype=np.int64)
    counts = np.zeros(len(rows), dtype=np.int64)
    for index, (left, right) in enumerate(zip(rows, cols)):
        first, second = neighbours[left], neighbours[right]
        if first.size and second.size:
            counts[index] = np.intersect1d(first, second, assume_unique=True).size
    return counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = sequences()
    directed = Counter()
    items = set()
    for sequence in seqs:
        if len(sequence) < 2:
            continue
        items.update(sequence)
        for left, right in zip(sequence, sequence[1:]):
            if left != right:
                directed[(left, right)] += 1

    undirected = defaultdict(int)
    for (left, right), count in directed.items():
        undirected[(left, right) if left < right else (right, left)] += count
    edges = sorted(undirected)
    rows = np.array([edge[0] for edge in edges], dtype=np.int64)
    cols = np.array([edge[1] for edge in edges], dtype=np.int64)
    weights = np.array([undirected[edge] for edge in edges], dtype=np.int64)

    n_nodes = int(pd.read_parquet(ITEMS).shape[0])
    degree = np.bincount(np.concatenate([rows, cols]), minlength=n_nodes)
    triangles = edge_triangles(rows, cols, n_nodes)
    af3 = 4.0 - degree[rows] - degree[cols] + 3.0 * triangles
    node_triangles = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(node_triangles, rows, triangles)
    np.add.at(node_triangles, cols, triangles)
    node_triangles //= 2

    pd.DataFrame({"source": rows, "target": cols, "weight": weights}).to_parquet(
        OUT / "edges.parquet", index=False)
    adjacency = sparse.coo_matrix((af3, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    np.savez_compressed(
        OUT / "graph.npz",
        nodes=np.array(n_nodes, dtype=np.int64),
        rows=rows, cols=cols, af3=af3,
        median=np.array(float(np.median(af3))), p90=np.array(float(np.percentile(af3, 90))),
        weight=weights,
        degree=degree[rows],
        triangles=triangles,
        triangles_per_node=node_triangles,
        indptr=adjacency.indptr, indices=adjacency.indices, data=adjacency.data,
        format=np.array("csr"))
    features = np.load(FEATURES)
    np.save(OUT / "node_features.npy", np.asarray(features, dtype=np.float32))

    metadata = {
        "schema": SCHEMA,
        "definition": "E = {{i_t, i_{t+1}} | (i_1..i_T) in S_train}; undirected, deduplicated",
        "node_id": "equals the stage 0 item id; node_features.npy row k is item k",
        "af3": "kappa_ij = 4 - d_i - d_j + 3 * T_ij, d and T on this graph",
        "sources": {str(TRAIN.relative_to(REPO)): sha256(TRAIN),
                    str(ITEMS.relative_to(REPO)): sha256(ITEMS),
                    str(FEATURES.relative_to(REPO)): sha256(FEATURES)},
        "counts": {"sequences": len(seqs), "items_with_edge": int(len(items)),
                   "items": n_nodes, "edges": int(len(rows)),
                   "directed_transitions_distinct": len(directed),
                   "directed_transitions_total": int(sum(directed.values())),
                   "triangles": int(triangles.sum() // 3),
                   "edges_with_both_directions": int(sum(
                       1 for left, right in edges if (left, right) in directed and (right, left) in directed))},
        "degree": {"min": int(degree[degree > 0].min()), "median": float(np.median(degree[degree > 0])),
                   "p90": float(np.percentile(degree[degree > 0], 90)),
                   "max": int(degree.max()), "nodes_without_edge": int((degree == 0).sum())},
        "af3_summary": {"min": float(af3.min()), "median": float(np.median(af3)),
                        "p90": float(np.percentile(af3, 90)), "max": float(af3.max())},
        "outputs": {},
    }
    for name in ("edges.parquet", "graph.npz", "node_features.npy"):
        path = OUT / name
        metadata["outputs"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=1))
    print(json.dumps(metadata, indent=1))


if __name__ == "__main__":
    main()
