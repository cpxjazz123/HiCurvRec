"""Original sentence-T5 embedding kNN: K sweep, 4-node graphlets, semantic consistency.

Builds the directed global top-K cosine kNN for K in {5,10,15,20}, symmetrises the union,
and reports:
  * connected 4-node induced graphlet counts/shares (PGD global exact counts);
  * coarse (KMeans-16) and fine (KMeans-256) semantic cluster consistency;
  * expected random-cluster rates and lifts, plus edge cosine similarity.

The parquet `categories` field is null, so cluster partitions are explicit semantic proxies.
Running this script writes all artifacts under
`results/research/embedding_knn_k_sweep/`.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/embedding_knn_k_sweep.py
"""
from __future__ import annotations

import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
BEHAVIOUR_CSV = REPO / "results" / "research" / "tiger_cluster_graph" / "pgd_behavior_edges.csv"
FINE_CACHE = HERE / "semantic_clusters.npz"
COARSE_CACHE = HERE / "semantic_clusters_coarse16.npz"
PGD = REPO / "tools" / "research" / "third_party" / "PGD" / "pgd"
OUT_DIR = REPO / "results" / "research" / "embedding_knn_k_sweep"
OUT_JSON = OUT_DIR / "embedding_knn_k_sweep.json"
OUT_MD = OUT_DIR / "embedding_knn_k_sweep.md"
K_VALUES = (5, 10, 15, 20)
K_MAX = max(K_VALUES)
COARSE_K = 16
FINE_K = 256
SEED = 42

GRAPHLETS = ("K4", "diamond", "C4", "paw", "K1,3", "P4")
PGD_NAMES = {
    "4_clique": "K4",
    "4_chordcycle": "diamond",
    "4_cycle": "C4",
    "4_tailed_tris": "paw",
    "3_star": "K1,3",
    "4_path": "P4",
}
ALL_PGD_NAMES = (*PGD_NAMES, "3_tris", "2_star")


def write_edges(path: Path, edges: np.ndarray) -> None:
    with path.open("w") as handle:
        for left, right in edges.tolist():
            handle.write(f"{int(left)},{int(right)}\n")


def read_edges(path: Path) -> np.ndarray:
    output = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            left, right = (int(value) for value in line.replace(",", " ").split()[:2])
            if left != right:
                output.append((min(left, right), max(left, right)))
    return np.unique(np.asarray(output, dtype=np.int64), axis=0)


def run_pgd(edges: np.ndarray, name: str) -> dict:
    csv_path = OUT_DIR / f"{name}.csv"
    log_path = OUT_DIR / f"{name}.pgd.log"
    write_edges(csv_path, edges)
    with log_path.open("w") as handle:
        subprocess.run([str(PGD), "-f", str(csv_path), "-w", "8", "-v"],
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    text = log_path.read_text()
    counts = {key: int(value) for key, value in re.findall(r"total_(\w+) = (\d+)", text)}
    missing = [key for key in ALL_PGD_NAMES if key not in counts]
    if missing:
        raise ValueError(f"PGD {name} missing totals {missing}")
    return counts


def brute_force_graphlets(edges: np.ndarray, nodes: int) -> dict:
    adjacency = [set() for _ in range(nodes)]
    for left, right in edges.tolist():
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    counts = {name: 0 for name in GRAPHLETS}
    for subset in itertools.combinations(range(nodes), 4):
        degrees = sorted((sum(1 for other in subset if other in adjacency[vertex])
                          for vertex in subset))
        edge_count = sum(degrees) // 2
        if edge_count == 6:
            kind = "K4"
        elif edge_count == 5:
            kind = "diamond"
        elif edge_count == 4 and degrees == [2, 2, 2, 2]:
            kind = "C4"
        elif edge_count == 4 and degrees == [1, 2, 2, 3]:
            kind = "paw"
        elif edge_count == 3 and degrees == [1, 1, 1, 3]:
            kind = "K1,3"
        elif edge_count == 3 and degrees == [1, 1, 2, 2]:
            kind = "P4"
        else:
            continue
        counts[kind] += 1
    return counts


def pgd_selftest() -> dict:
    rng = np.random.default_rng(20261117)
    nodes = 14
    edges = [(left, right) for left in range(nodes) for right in range(left + 1, nodes)
             if rng.random() < 0.30]
    edges_array = np.asarray(edges, dtype=np.int64)
    counts = run_pgd(edges_array, "pgd_selftest")
    brute = brute_force_graphlets(edges_array, nodes)
    pgd = {PGD_NAMES[key]: counts[key] for key in PGD_NAMES}
    for name in GRAPHLETS:
        if int(pgd[name]) != int(brute[name]):
            raise AssertionError(f"PGD selftest {name}: {pgd[name]} != {brute[name]}")
    return {"nodes": nodes, "edges": len(edges), "pgd": pgd, "brute_force": brute}


def edge_cosine(embedding: np.ndarray, edges: np.ndarray) -> dict:
    left = embedding[edges[:, 0]]
    right = embedding[edges[:, 1]]
    cosine = np.einsum("ij,ij->i", left, right)
    return {
        "mean": float(cosine.mean()),
        "median": float(np.median(cosine)),
        "p05": float(np.quantile(cosine, 0.05)),
        "p95": float(np.quantile(cosine, 0.95)),
    }


def cluster_consistency(labels: np.ndarray, edges: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    counts = np.bincount(labels)
    probabilities = counts / n
    expected = float(np.sum(probabilities ** 2))
    rate = float(np.mean(labels[edges[:, 0]] == labels[edges[:, 1]]))
    return {
        "clusters": int(len(counts)),
        "rate": rate,
        "expected_random": expected,
        "lift": rate / expected if expected else float("nan"),
    }


def kmeans_labels(embedding: np.ndarray, clusters: int, name: str) -> np.ndarray:
    cache = FINE_CACHE if clusters == FINE_K else (COARSE_CACHE if clusters == COARSE_K else None)
    if cache is not None and cache.exists():
        payload = np.load(cache)
        if "labels" in payload and len(payload["labels"]) == len(embedding):
            return payload["labels"].astype(np.int32)
    model = KMeans(n_clusters=clusters, n_init=5, random_state=SEED, max_iter=300)
    labels = model.fit_predict(embedding).astype(np.int32)
    if cache is not None:
        np.savez_compressed(cache, labels=labels, inertia=float(model.inertia_))
    return labels


def graph_stats(edges: np.ndarray, nodes: int, counts: dict) -> dict:
    degree = np.bincount(edges.ravel(), minlength=nodes)
    return {
        "nodes": int(nodes),
        "undirected_edges": int(len(edges)),
        "mean_degree": float(degree.mean()),
        "max_degree": int(degree.max()),
        "triangles": int(counts["3_tris"]),
        "wedges": int(counts["2_star"]),
        "triangle_share_3node": float(counts["3_tris"] / (counts["3_tris"] + counts["2_star"]))
        if (counts["3_tris"] + counts["2_star"]) else float("nan"),
        "components_note": "not recomputed; graphlet counts are the primary structural result",
    }


def graphlet_distribution(counts: dict) -> dict:
    values = {name: int(counts[key]) for key, name in PGD_NAMES.items()}
    total = sum(values.values())
    shares = {name: (value / total if total else float("nan")) for name, value in values.items()}
    return {"counts": values, "shares": shares, "total_connected_4node": int(total)}


def build_knn(embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(n_neighbors=K_MAX + 1, metric="cosine", algorithm="auto", n_jobs=-1)
    model.fit(embedding)
    raw = model.kneighbors(embedding, return_distance=False)
    neighbors = np.empty((len(embedding), K_MAX), dtype=np.int64)
    for index, row in enumerate(raw):
        row = row[row != index]
        if len(row) < K_MAX:
            raise ValueError(f"only {len(row)} non-self neighbours for item {index}")
        neighbors[index] = row[:K_MAX]
    return neighbors


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    embedding = np.load(EMBEDDING).astype(np.float32)
    if embedding.shape != (24587, 768):
        raise ValueError(f"unexpected embedding shape {embedding.shape}")
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    nodes = len(embedding)

    print("PGD selftest", flush=True)
    selftest = pgd_selftest()
    print(json.dumps(selftest, indent=2), flush=True)

    print("KMeans semantic partitions", flush=True)
    fine = kmeans_labels(embedding, FINE_K, "fine256")
    coarse = kmeans_labels(embedding, COARSE_K, "coarse16")
    print("coarse sizes", np.bincount(coarse).tolist(), flush=True)

    print("cosine kNN query", flush=True)
    neighbors = build_knn(embedding)
    rows = []

    behavior_edges = read_edges(BEHAVIOUR_CSV)
    behavior_counts = run_pgd(behavior_edges, "behavior_reference")
    behavior_row = {
        "name": "behavior_reference",
        "graph_stats": graph_stats(behavior_edges, nodes, behavior_counts),
        "graphlets": graphlet_distribution(behavior_counts),
        "semantic": {
            "coarse": cluster_consistency(coarse, behavior_edges),
            "fine": cluster_consistency(fine, behavior_edges),
            "cosine": edge_cosine(embedding, behavior_edges),
        },
    }
    print("behavior reference", json.dumps(behavior_row, indent=2), flush=True)

    for k in K_VALUES:
        directed_pairs = np.column_stack([
            np.repeat(np.arange(nodes, dtype=np.int64), k),
            neighbors[:, :k].reshape(-1),
        ])
        reverse = set(map(tuple, directed_pairs[:, ::-1].tolist()))
        mutual = sum((int(left), int(right)) in reverse for left, right in directed_pairs.tolist())
        undirected = np.unique(np.sort(directed_pairs, axis=1), axis=0)
        counts = run_pgd(undirected, f"knn_k{k}")
        row = {
            "name": f"knn_k{k}",
            "k": k,
            "directed_edges": int(len(directed_pairs)),
            "mutual_directed_fraction": float(mutual / len(directed_pairs)),
            "graph_stats": graph_stats(undirected, nodes, counts),
            "graphlets": graphlet_distribution(counts),
            "semantic": {
                "coarse": cluster_consistency(coarse, undirected),
                "fine": cluster_consistency(fine, undirected),
                "cosine": edge_cosine(embedding, undirected),
            },
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    result = {
        "embedding": str(EMBEDDING.relative_to(REPO)),
        "embedding_shape": list(embedding.shape),
        "distance": "cosine",
        "graph_construction": "top-K directed kNN, symmetrised as undirected union",
        "k_values": list(K_VALUES),
        "semantic_partitions": {
            "coarse": {"method": "KMeans", "clusters": COARSE_K, "seed": SEED},
            "fine": {"method": "KMeans", "clusters": FINE_K, "seed": SEED},
            "note": "parquet categories are null; cluster partitions are semantic proxies",
        },
        "pgd_selftest": selftest,
        "behavior_reference": behavior_row,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        "# Sentence-T5 kNN K sweep: graphlets and semantic consistency",
        "",
        "Original sentence-T5 embedding, L2-normalised, global cosine top-K, directed edges",
        "symmetrised as the undirected union. K = 5, 10, 15, 20.",
        "",
        "PGD global 4-node counts are used; the PGD **local** 3-star column is not used.",
        "The PGD global-count implementation is checked against brute force on a random graph.",
        "",
        f"Categories in the parquet are null, so semantic consistency uses cluster proxies:",
        f"**coarse = KMeans({COARSE_K})**, **fine = KMeans({FINE_K})** on the same embedding.",
        "Expected random is the empirical within-cluster pair probability; lift = rate / expected.",
        "",
        "## Graphlet distribution",
        "",
        "Connected 4-node induced graphlets; shares sum to 1 across the six connected types.",
        "",
        "| graph | edges | mean deg | triangles | 3-node triangle share | K4 | diamond | C4 | paw | K1,3 | P4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [behavior_row, *rows]:
        stats = row["graph_stats"]
        shares = row["graphlets"]["shares"]
        lines.append(
            f"| {row['name']} | {stats['undirected_edges']} | {stats['mean_degree']:.2f} | "
            f"{stats['triangles']} | {stats['triangle_share_3node']:.4f} | " + " | ".join(
                f"{shares[name]:.4f}" for name in GRAPHLETS) + " |"
        )
    lines += [
        "",
        "## Semantic consistency",
        "",
        "| graph | mutual directed | coarse rate | coarse lift | fine rate | fine lift | edge cosine mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [behavior_row, *rows]:
        semantic = row["semantic"]
        mutual = row.get("mutual_directed_fraction", float("nan"))
        lines.append(
            f"| {row['name']} | {mutual:.4f} | "
            f"{semantic['coarse']['rate']:.4f} | {semantic['coarse']['lift']:.3f} | "
            f"{semantic['fine']['rate']:.4f} | {semantic['fine']['lift']:.3f} | "
            f"{semantic['cosine']['mean']:.4f} |"
        )
    lines += [
        "",
        "The behavior reference is not a KNN graph and has no reciprocal directed fraction.",
        "",
        "## Reading",
        "",
        "- Graphlet shares are conditional on connected 4-node subgraphs, so raw edge",
        "  density changes do not mechanically force a share change.",
        "- Coarse/fine rates are not directly comparable in absolute value; the lift over",
        "  the same clustering's random pair baseline is the comparable quantity.",
        "- This is an embedding-space diagnostic only: no SID, no RQ-VAE, no Stage 3.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON} and {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
