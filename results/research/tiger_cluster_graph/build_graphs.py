"""Build behavior / pre-quantization / post-quantization graphs for TIGER.

Behaviour graph  : TRAIN user-session consecutive pairs.
Pre-quantization : KMeans (k=256) on TIGER encoder output z, then k nearest neighbours *inside each
                   cluster*, symmetrised and budgeted to the behaviour graph's edge count.
Post-quantization: the same KMeans + within-cluster kNN on the quantized readout q built from the
                   exported TIGER SID codes and codebooks.

The same clustering rule, the same k-selection rule and the same edge budget are used on both sides.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/build_graphs.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VECTORS = HERE / "tiger_vectors.npz"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
OUT = HERE / "cluster_graphs.npz"
META = HERE / "cluster_graphs_meta.json"

K_CLUSTERS = 256
SEED = 42
MAX_NEIGHBOURS = 64


def behaviour_edges() -> tuple[int, np.ndarray]:
    frame = pd.read_parquet(TRAIN, columns=["user", "seen_history", "target"])
    best: dict[int, tuple[int, list[int]]] = {}
    for user, history, target in zip(
        frame["user"].to_numpy(), frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)
    ):
        history = [] if history is None else [int(x) for x in history]
        user = int(user)
        if user not in best or len(history) > best[user][0]:
            best[user] = (len(history), [*history, int(target)])
    edges = set()
    for _, session in best.values():
        for left, right in zip(session, session[1:]):
            edges.add((left, right) if left < right else (right, left))
    array = np.array(sorted(edges), dtype=np.int32)
    nodes = int(max(array.max(), frame["target"].max())) + 1
    return nodes, array


def cluster_knn(vectors: np.ndarray, labels: np.ndarray, target_edges: int) -> tuple[np.ndarray, int, int]:
    n = len(vectors)
    labels = np.asarray(labels)
    groups = []
    max_cluster = 0
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        if len(members) < 2:
            continue
        max_cluster = max(max_cluster, len(members))
        points = vectors[members].astype(np.float64)
        squared = np.einsum("ij,ij->i", points, points)
        distance = squared[:, None] + squared[None, :] - 2.0 * points @ points.T
        np.fill_diagonal(distance, np.inf)
        order = np.argsort(distance, axis=1)
        groups.append((members.astype(np.int64), order.astype(np.int64)))

    max_m = min(MAX_NEIGHBOURS, max_cluster - 1)

    def edges_for(m: int) -> np.ndarray:
        codes = []
        for members, order in groups:
            take = min(m, len(members) - 1)
            if take <= 0:
                continue
            left = np.repeat(members, take)
            right = members[order[:, :take]].reshape(-1)
            low = np.minimum(left, right)
            high = np.maximum(left, right)
            codes.append(low * np.int64(n) + high)
        if not codes:
            return np.zeros((0, 2), dtype=np.int32)
        unique = np.unique(np.concatenate(codes))
        return np.stack([unique // n, unique % n], axis=1).astype(np.int32)

    low, high = 1, max_m
    best_m, best_edges = 1, edges_for(1)
    best_gap = abs(len(best_edges) - target_edges)
    while low <= high:
        middle = (low + high) // 2
        edges = edges_for(middle)
        gap = abs(len(edges) - target_edges)
        if gap < best_gap:
            best_m, best_edges, best_gap = middle, edges, gap
        if len(edges) < target_edges:
            low = middle + 1
        else:
            high = middle - 1
    for m in range(max(1, best_m - 2), min(max_m, best_m + 2) + 1):
        edges = edges_for(m)
        gap = abs(len(edges) - target_edges)
        if gap < best_gap:
            best_m, best_edges, best_gap = m, edges, gap
    return best_edges, best_m, max_m


def edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {(int(left), int(right)) for left, right in edges.tolist()}


def main() -> int:
    payload = np.load(VECTORS)
    z = payload["z"].astype(np.float32)
    q = payload["q"].astype(np.float32)
    nodes, behaviour = behaviour_edges()
    if nodes != len(z):
        raise SystemExit(f"node mismatch: graph {nodes}, vectors {len(z)}")
    behaviour_set = edge_set(behaviour)
    print(f"behaviour edges {len(behaviour_set)}", flush=True)

    labels_z = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=SEED).fit_predict(z)
    labels_q = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=SEED).fit_predict(q)
    print(f"clusters: z {len(np.unique(labels_z))}, q {len(np.unique(labels_q))}", flush=True)

    edges_z, m_z, max_m_z = cluster_knn(z, labels_z, len(behaviour_set))
    edges_q, m_q, max_m_q = cluster_knn(q, labels_q, len(behaviour_set))
    set_z, set_q = edge_set(edges_z), edge_set(edges_q)
    print(f"cluster-kNN: z {len(set_z)} edges m={m_z}; q {len(set_q)} edges m={m_q}", flush=True)

    def overlap(edges: set[tuple[int, int]]) -> dict:
        intersection = len(edges & behaviour_set)
        return {
            "edges": len(edges),
            "behaviour_recall": intersection / len(behaviour_set) if behaviour_set else 0.0,
            "precision": intersection / len(edges) if edges else 0.0,
            "jaccard": intersection / len(edges | behaviour_set) if edges | behaviour_set else 0.0,
        }

    def degrees(edges: np.ndarray) -> np.ndarray:
        degree = np.zeros(nodes, dtype=np.int64)
        if len(edges):
            np.add.at(degree, edges[:, 0], 1)
            np.add.at(degree, edges[:, 1], 1)
        return degree

    degree_b = degrees(behaviour)
    degree_z = degrees(edges_z)
    degree_q = degrees(edges_q)
    from scipy.stats import spearmanr
    meta = {
        "k_clusters": K_CLUSTERS,
        "seed": SEED,
        "nodes": nodes,
        "behaviour_edges": len(behaviour_set),
        "z": {
            "clusters": int(len(np.unique(labels_z))),
            "max_cluster": int(max(Counter(labels_z).values())),
            "within_cluster_m": int(m_z),
            "max_m": int(max_m_z),
            "graph": overlap(set_z),
            "degree_spearman_vs_behaviour": float(spearmanr(degree_b, degree_z).statistic),
        },
        "q": {
            "clusters": int(len(np.unique(labels_q))),
            "max_cluster": int(max(Counter(labels_q).values())),
            "within_cluster_m": int(m_q),
            "max_m": int(max_m_q),
            "graph": overlap(set_q),
            "degree_spearman_vs_behaviour": float(spearmanr(degree_b, degree_q).statistic),
        },
        "cluster_agreement_z_q": {
            "ari": float(adjusted_rand_score(labels_z, labels_q)),
            "nmi": float(normalized_mutual_info_score(labels_z, labels_q)),
        },
    }
    np.savez(
        OUT,
        behaviour=behaviour,
        z_edges=edges_z,
        q_edges=edges_q,
        z_labels=labels_z.astype(np.int32),
        q_labels=labels_q.astype(np.int32),
    )
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
