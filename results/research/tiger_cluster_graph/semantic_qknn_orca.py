"""Semantic clusters, quantized-vector kNN graphs, and ORCA role relations.

This is the experiment requested next:

1. cluster item text embeddings (sentence-T5) with KMeans;
2. keep the original behaviour graph and its ORCA roles;
3. build a fresh global kNN graph from each quantized vector `q` (pure TIGER and mixed F0);
4. run ORCA on the q-kNN graph;
5. compare ORCA roles and role-to-role relations on the same semantic clusters.

The quantized graph is **not** the previous within-cluster graph.  For every item, its k nearest
quantized vectors are connected globally, the directed kNN edges are symmetrised, and duplicates
are removed.  Euclidean distance in q-space is used for both arms; k=10.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/semantic_qknn_orca.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore", category=UserWarning)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cluster_orca_relations import (  # noqa: E402
    boundary_profiles,
    cluster_profiles,
    cluster_scalar,
    contingency,
    jsd,
    pair_stats,
    top_pair_deltas,
)
from sid_structural_fidelity import load_behavior_graph, purity  # noqa: E402

REPO = HERE.parents[2]
ORCA_GPU = REPO / "results" / "orca_build" / "orca" / "orca_gpu"
ORCA_RUNS = REPO / "results" / "orca_build" / "runs"
BEHAVIOUR_ORCA = ORCA_RUNS / "behaviour_gpu_5.out"
ORBIT_GROUPS = HERE / "orbit_groups.json"
SENTENCE_EMB = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
VECTORS = {
    "pure_euclidean_TIGER": HERE / "tiger_vectors.npz",
    "mixed_F0": HERE / "mixed_vectors.npz",
}
SEMANTIC_CACHE = HERE / "semantic_clusters.npz"
Q_GRAPH_CACHE = HERE / "qknn_edges.npz"
Q_ORCA_CACHE = HERE / "qknn_orca.npz"
OUT_JSON = HERE / "semantic_qknn_orca.json"
OUT_MD = HERE / "semantic_qknn_orca.md"
K_SEMANTIC = 256
K_Q = 256
K_NEIGHBOURS = 10
SEED = 42
DEPTHS = (1, 2, 3)


def semantic_labels(nodes: int) -> tuple[np.ndarray, float]:
    if SEMANTIC_CACHE.exists():
        payload = np.load(SEMANTIC_CACHE)
        labels = payload["labels"]
        if len(labels) == nodes:
            return labels.astype(np.int32), float(payload["inertia"]) if "inertia" in payload else float("nan")
    embedding = np.load(SENTENCE_EMB).astype(np.float32)
    if len(embedding) != nodes:
        raise ValueError(f"semantic embedding rows {len(embedding)} != nodes {nodes}")
    embedding = embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    model = KMeans(n_clusters=K_SEMANTIC, n_init=5, random_state=SEED, max_iter=300)
    labels = model.fit_predict(embedding).astype(np.int32)
    np.savez_compressed(SEMANTIC_CACHE, labels=labels, inertia=float(model.inertia_))
    return labels, float(model.inertia_)


def quantized_vectors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return payload["z"].astype(np.float32), payload["q"].astype(np.float32)


def q_clusters(q: np.ndarray) -> np.ndarray:
    model = KMeans(n_clusters=K_Q, n_init=5, random_state=SEED, max_iter=300)
    return model.fit_predict(q).astype(np.int32)


def qknn_edges(q: np.ndarray, k: int = K_NEIGHBOURS, chunk: int = 512) -> np.ndarray:
    n = len(q)
    squared = np.einsum("ij,ij->i", q, q)
    codes = []
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        block = q[start:end]
        distance = squared[start:end, None] + squared[None, :] - 2.0 * block @ q.T
        rows = np.arange(start, end)
        distance[np.arange(end - start), rows] = np.inf
        nearest = np.argpartition(distance, k, axis=1)[:, :k]
        left = np.repeat(rows, k)
        right = nearest.reshape(-1)
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        codes.append(low.astype(np.int64) * n + high.astype(np.int64))
    unique = np.unique(np.concatenate(codes))
    return np.stack([unique // n, unique % n], axis=1).astype(np.int32)


def write_graph(path: Path, nodes: int, edges: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write(f"{nodes} {len(edges)}\n")
        for left, right in edges.tolist():
            handle.write(f"{int(left)} {int(right)}\n")


def run_orca(nodes: int, edges: np.ndarray, name: str) -> np.ndarray:
    ORCA_RUNS.mkdir(parents=True, exist_ok=True)
    in_path = ORCA_RUNS / f"{name}_5.in"
    out_path = ORCA_RUNS / f"{name}_gpu_5.out"
    if not in_path.exists() or in_path.stat().st_size == 0:
        write_graph(in_path, nodes, edges)
    if not out_path.exists() or out_path.stat().st_size == 0:
        subprocess.run([str(ORCA_GPU), "node", "5", str(in_path), str(out_path)], check=True)
    table = np.loadtxt(out_path, dtype=np.float64)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape != (nodes, 73):
        raise ValueError(f"ORCA output {out_path} has shape {table.shape}, expected {(nodes, 73)}")
    return table


def roles_from_table(table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups = json.loads(ORBIT_GROUPS.read_text())
    names = list(groups)
    table = np.maximum(table, 0.0)
    row_sum = table.sum(axis=1, keepdims=True)
    table = table / np.where(row_sum > 0, row_sum, 1.0)
    scores = np.zeros((len(table), len(names)), dtype=np.float64)
    for index, columns in enumerate(groups.values()):
        scores[:, index] = table[:, columns].sum(axis=1)
    total = scores.sum(axis=1, keepdims=True)
    role_dist = scores / np.where(total > 0, total, 1.0)
    top = role_dist.argmax(axis=1)
    multi = (role_dist >= 0.25).sum(axis=1) >= 2
    return role_dist, top, multi, table


def role_agreement(before_dist: np.ndarray, before_top: np.ndarray, before_multi: np.ndarray,
                   after_dist: np.ndarray, after_top: np.ndarray, after_multi: np.ndarray,
                   bridge_edges: np.ndarray) -> dict:
    item_jsd = np.array([jsd(before_dist[i], after_dist[i]) for i in range(len(before_dist))])
    bridge_nodes = np.unique(bridge_edges.ravel()) if len(bridge_edges) else np.array([], dtype=int)
    return {
        "top_role_ari": float(adjusted_rand_score(before_top, after_top)),
        "top_role_nmi": float(normalized_mutual_info_score(before_top, after_top)),
        "top_role_agreement": float(np.mean(before_top == after_top)),
        "role_distribution_jsd_mean": float(np.nanmean(item_jsd)),
        "role_distribution_jsd_median": float(np.nanmedian(item_jsd)),
        "multi_role_agreement": float(np.mean(before_multi == after_multi)),
        "multi_role_before": float(before_multi.mean()),
        "multi_role_after": float(after_multi.mean()),
        "bridge_nodes": int(len(bridge_nodes)),
        "bridge_role_jsd": float(jsd(before_dist[bridge_nodes].mean(axis=0), after_dist[bridge_nodes].mean(axis=0))) if len(bridge_nodes) else float("nan"),
    }


def role_pair_dist(top: np.ndarray, edges: np.ndarray) -> np.ndarray:
    classes = 9
    left = top[edges[:, 0]]
    right = top[edges[:, 1]]
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    counts = np.bincount(low * classes + high, minlength=classes ** 2)
    return counts / max(counts.sum(), 1)


def semantic_pair_relation(top: np.ndarray, semantic: np.ndarray, edges: np.ndarray) -> dict:
    return pair_stats(top, semantic, edges)


def cluster_structure(labels: np.ndarray, edges: np.ndarray, triangles: np.ndarray, nodes: int) -> dict:
    k = labels.max() + 1
    same = labels[edges[:, 0]] == labels[edges[:, 1]]
    within = np.bincount(labels[edges[:, 0]][same], minlength=k)
    incident = np.bincount(labels[edges[:, 0]], minlength=k) + np.bincount(labels[edges[:, 1]], minlength=k)
    cross = incident - 2 * within
    tri_inside = np.zeros(k, dtype=np.float64)
    for value in range(k):
        members = np.flatnonzero(labels == value)
        tri_inside[value] = triangles[members].sum() / 3.0
    tri_per_edge = np.divide(tri_inside, within, out=np.zeros(k, dtype=float), where=within > 0)
    cross_rate = np.divide(cross, within + cross, out=np.zeros(k, dtype=float), where=(within + cross) > 0)
    return {
        "k": int(k),
        "within_edges": within,
        "cross_edges": cross,
        "triangles_inside": tri_inside,
        "triangles_per_internal_edge": tri_per_edge,
        "cross_edge_rate": cross_rate,
        "total_triangles": float(triangles.sum() / 3.0),
        "global_cross_rate": float(cross.sum() / max(within.sum() + cross.sum(), 1)),
    }


def compare_structure(before: dict, after: dict, semantic_sizes: np.ndarray) -> dict:
    mask = (before["within_edges"] > 0) & (after["within_edges"] > 0)
    from scipy.stats import spearmanr
    rho = float(spearmanr(before["triangles_per_internal_edge"][mask], after["triangles_per_internal_edge"][mask]).statistic) if mask.sum() > 2 else float("nan")
    return {
        "triangles_per_edge_spearman": rho,
        "triangles_per_edge_mae_weighted": float(np.average(np.abs(before["triangles_per_internal_edge"] - after["triangles_per_internal_edge"]), weights=semantic_sizes)),
        "cross_rate_gap": float(np.average(after["cross_edge_rate"] - before["cross_edge_rate"], weights=semantic_sizes)),
        "triangles_before": before["total_triangles"],
        "triangles_after": after["total_triangles"],
        "cross_rate_before": before["global_cross_rate"],
        "cross_rate_after": after["global_cross_rate"],
    }


def main() -> int:
    nodes, behavior_edges, behavior_adj = load_behavior_graph()
    semantic, semantic_inertia = semantic_labels(nodes)
    behavior_table = np.loadtxt(BEHAVIOUR_ORCA, dtype=np.float64)
    behavior_roles, behavior_top, behavior_multi, behavior_gdv = roles_from_table(behavior_table)
    semantic_sizes = np.bincount(semantic)
    print(f"nodes {nodes}, behavior edges {len(behavior_edges)}, semantic clusters {len(semantic_sizes)}", flush=True)

    behavior_graph = nx.Graph()
    behavior_graph.add_nodes_from(range(nodes))
    behavior_graph.add_edges_from(map(tuple, behavior_edges.tolist()))
    behavior_triangles = np.fromiter(nx.triangles(behavior_graph).values(), dtype=np.float64, count=nodes)
    behavior_structure = cluster_structure(semantic, behavior_edges, behavior_triangles, nodes)

    q_graphs: dict[str, np.ndarray] = {}
    q_orca: dict[str, np.ndarray] = {}
    q_roles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    q_clusters_dict: dict[str, np.ndarray] = {}
    q_structures = {}
    results = {}
    for name, path in VECTORS.items():
        _, q = quantized_vectors(path)
        q_labels = q_clusters(q)
        q_clusters_dict[name] = q_labels
        edges = qknn_edges(q)
        q_graphs[name] = edges
        graph = nx.Graph()
        graph.add_nodes_from(range(nodes))
        graph.add_edges_from(map(tuple, edges.tolist()))
        triangles = np.fromiter(nx.triangles(graph).values(), dtype=np.float64, count=nodes)
        structure = cluster_structure(semantic, edges, triangles, nodes)
        q_structures[name] = structure
        table = run_orca(nodes, edges, f"semantic_qknn_{name}")
        q_orca[name] = table
        q_roles[name] = roles_from_table(table)
        rd, top, multi, gdv = q_roles[name]
        bridge_mask = semantic[behavior_edges[:, 0]] != semantic[behavior_edges[:, 1]]
        bridge_edges = behavior_edges[bridge_mask]
        agreement = role_agreement(behavior_roles, behavior_top, behavior_multi, rd, top, multi, bridge_edges)
        # same semantic partition, behaviour vs q graph
        before_profile, _ = cluster_profiles(semantic, behavior_roles, len(semantic_sizes))
        after_profile, _ = cluster_profiles(semantic, rd, len(semantic_sizes))
        profile_jsd = np.array([jsd(before_profile[c], after_profile[c]) for c in range(len(semantic_sizes))])
        profile_valid = np.isfinite(profile_jsd)
        before_boundary = boundary_profiles(semantic, behavior_adj, behavior_roles, len(semantic_sizes))
        q_adjacency = [set(graph.neighbors(item)) for item in range(nodes)]
        after_boundary = boundary_profiles(semantic, q_adjacency, rd, len(semantic_sizes))
        boundary_jsd = np.array([
            jsd(before_boundary[c], after_boundary[c])
            if np.isfinite(before_boundary[c]).all() and np.isfinite(after_boundary[c]).all()
            else np.nan
            for c in range(len(semantic_sizes))
        ])
        before_pairs = semantic_pair_relation(behavior_top, semantic, behavior_edges)
        after_pairs = semantic_pair_relation(top, semantic, edges)
        before_all = role_pair_dist(behavior_top, behavior_edges)
        after_all = role_pair_dist(top, edges)
        before_matrix = before_all.reshape(9, 9)
        after_matrix = after_all.reshape(9, 9)
        before_multi_profile = cluster_scalar(semantic, behavior_multi, len(semantic_sizes))
        after_multi_profile = cluster_scalar(semantic, multi, len(semantic_sizes))
        top_clusters = []
        for cluster in np.argsort(-semantic_sizes)[:12]:
            top_clusters.append({
                "cluster": int(cluster),
                "size": int(semantic_sizes[cluster]),
                "top_before": list(json.loads(ORBIT_GROUPS.read_text()).keys())[int(before_profile[cluster].argmax())],
                "top_after": list(json.loads(ORBIT_GROUPS.read_text()).keys())[int(after_profile[cluster].argmax())],
                "profile_jsd": float(profile_jsd[cluster]),
                "multi_before": float(before_multi_profile[cluster]),
                "multi_after": float(after_multi_profile[cluster]),
                "cross_before": float(behavior_structure["cross_edge_rate"][cluster]),
                "cross_after": float(structure["cross_edge_rate"][cluster]),
            })
        # C_sem vs C_q contingency
        cont = contingency(semantic, q_labels)
        best_q = cont.argmax(axis=1)
        q_sizes = np.bincount(q_labels)
        coverage = cont.max(axis=1) / semantic_sizes
        profile_cross = np.array([
            jsd(before_profile[c], after_profile[best_q[c]]) for c in range(len(semantic_sizes))
        ])
        cross_valid = np.isfinite(profile_cross)
        results[name] = {
            "q_edges": int(len(edges)),
            "q_components": int(nx.number_connected_components(graph)),
            "q_clusters": int(len(np.bincount(q_labels))),
            "q_cluster_max_share": float(np.bincount(q_labels).max() / nodes),
            "q_cluster_sizes": np.bincount(q_labels).tolist(),
            "role_agreement": agreement,
            "semantic_cluster": {
                "profile_jsd_weighted": float(np.average(profile_jsd[profile_valid], weights=semantic_sizes[profile_valid])) if profile_valid.any() else float("nan"),
                "profile_jsd_median": float(np.median(profile_jsd[profile_valid])) if profile_valid.any() else float("nan"),
                "profile_jsd_valid_clusters": int(profile_valid.sum()),
                "top_role_match_rate": float(np.mean(before_profile.argmax(1)[profile_valid] == after_profile.argmax(1)[profile_valid])) if profile_valid.any() else float("nan"),
                "boundary_jsd_weighted": float(np.average(boundary_jsd[np.isfinite(boundary_jsd)], weights=semantic_sizes[np.isfinite(boundary_jsd)])) if np.isfinite(boundary_jsd).any() else float("nan"),
                "multi_role_mae": float(np.average(np.abs(cluster_scalar(semantic, behavior_multi, len(semantic_sizes)) - cluster_scalar(semantic, multi, len(semantic_sizes))), weights=semantic_sizes)),
            },
            "role_pair_relation": {
                "before_global_same": before_pairs["global_same"],
                "after_global_same": after_pairs["global_same"],
                "all_role_pair_jsd": jsd(before_all, after_all),
                "within_semantic_role_pair_jsd": jsd(before_pairs["within_dist"], after_pairs["within_dist"]),
                "cross_semantic_role_pair_jsd": jsd(before_pairs["cross_dist"], after_pairs["cross_dist"]),
                "before_within": before_pairs["within_dist"].tolist(),
                "after_within": after_pairs["within_dist"].tolist(),
                "top_pair_deltas": top_pair_deltas(before_pairs, after_pairs, list(json.loads(ORBIT_GROUPS.read_text()).keys())),
                "before_matrix": before_matrix.tolist(),
                "after_matrix": after_matrix.tolist(),
            },
            "semantic_profiles": {
                "before": before_profile.tolist(),
                "after": after_profile.tolist(),
                "top_clusters": top_clusters,
            },
            "structure": {
                "before": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in behavior_structure.items()},
                "after": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in structure.items()},
                "comparison": compare_structure(behavior_structure, structure, semantic_sizes),
            },
            "semantic_vs_q_clusters": {
                "ari": float(adjusted_rand_score(semantic, q_labels)),
                "nmi": float(normalized_mutual_info_score(semantic, q_labels)),
                "coverage_weighted": float(np.average(coverage, weights=semantic_sizes)),
                "coverage_median": float(np.median(coverage)),
                "q_cluster_sizes": q_sizes.tolist(),
                "profile_jsd_weighted": float(np.average(profile_cross[cross_valid], weights=semantic_sizes[cross_valid])) if cross_valid.any() else float("nan"),
                "profile_jsd_valid_clusters": int(cross_valid.sum()),
            },
        }
        print(f"{name}: q edges {len(edges)}, components {nx.number_connected_components(graph)}, q clusters {len(np.bincount(q_labels))}", flush=True)

    payload = {
        "behavior_graph": {
            "nodes": int(nodes),
            "edges": int(len(behavior_edges)),
            "semantic_clusters": int(len(semantic_sizes)),
            "semantic_inertia": semantic_inertia,
            "semantic_max_share": float(semantic_sizes.max() / nodes),
            "semantic_sizes": semantic_sizes.tolist(),
            "multi_role_fraction": float(behavior_multi.mean()),
            "role_groups": list(json.loads(ORBIT_GROUPS.read_text()).keys()),
        },
        "k_neighbours": K_NEIGHBOURS,
        "k_q_clusters": K_Q,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# Semantic clusters → quantized kNN graph ORCA relations",
        "",
        "Semantic clusters: KMeans(256) on L2-normalised sentence-T5 item embeddings. "
        f"Quantized graphs: global k={K_NEIGHBOURS} nearest neighbours of q (Euclidean), symmetrised. "
        "ORCA is recomputed on each q-kNN graph with exact 5-node orbits.",
        "",
        f"Behaviour graph: **{nodes}** nodes, **{len(behavior_edges)}** edges, "
        f"**{len(semantic_sizes)}** semantic clusters, multi-role **{behavior_multi.mean():.3%}**.",
        "",
        "## Role preservation (node level)",
        "",
        "| arm | q-kNN edges | components | top-role ARI | top-role NMI | same top role | role JSD mean | multi agreement | bridge nodes | bridge role JSD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results.items():
        a = row["role_agreement"]
        lines.append(
            f"| {name} | {row['q_edges']} | {row['q_components']} | {a['top_role_ari']:.4f} | "
            f"{a['top_role_nmi']:.4f} | {a['top_role_agreement']:.4f} | {a['role_distribution_jsd_mean']:.4f} | "
            f"{a['multi_role_agreement']:.4f} | {a['bridge_nodes']} | {a['bridge_role_jsd']:.4f} |"
        )
    lines += [
        "",
        "## Semantic-cluster role profiles and role-pair relations",
        "",
        "Same semantic clusters are used on the behaviour graph and on each q-kNN graph; only the graph changes. "
        "`profile JSD` compares the 9-group ORCA profile of each semantic cluster before/after. "
        "`within semantic` / `cross semantic` role-pair JSD compare role-pair distributions inside/outside semantic clusters.",
        "",
        "| arm | profile JSD | profile JSD median | top-role match | boundary JSD | multi-role MAE | all role-pair JSD | within semantic JSD | cross semantic JSD | before same | after same |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results.items():
        c = row["semantic_cluster"]
        p = row["role_pair_relation"]
        lines.append(
            f"| {name} | {c['profile_jsd_weighted']:.4f} | {c['profile_jsd_median']:.4f} | "
            f"{c['top_role_match_rate']:.4f} | {c['boundary_jsd_weighted']:.4f} | {c['multi_role_mae']:.4f} | "
            f"{p['all_role_pair_jsd']:.4f} | {p['within_semantic_role_pair_jsd']:.4f} | "
            f"{p['cross_semantic_role_pair_jsd']:.4f} | {p['before_global_same']:.4f} | {p['after_global_same']:.4f} |"
        )
    lines += [
        "",
        "## Semantic cluster structure: triangles and cross-cluster edges",
        "",
        "`triangles/internal edge` is the number of triangles fully inside a semantic cluster divided by "
        "the number of internal behaviour/q edges. `cross rate` is the fraction of incident edges that leave "
        "the semantic cluster.",
        "",
        "| arm | triangles before | triangles after | triangles/edge Spearman | triangles/edge MAE | cross rate before | cross rate after | cross-rate gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results.items():
        c = row["structure"]["comparison"]
        lines.append(
            f"| {name} | {c['triangles_before']:.0f} | {c['triangles_after']:.0f} | "
            f"{c['triangles_per_edge_spearman']:.3f} | {c['triangles_per_edge_mae_weighted']:.4f} | "
            f"{c['cross_rate_before']:.4f} | {c['cross_rate_after']:.4f} | {c['cross_rate_gap']:+.4f} |"
        )
    lines += [
        "",
        "## Semantic clusters vs quantized-vector clusters",
        "",
        "| arm | q clusters | q max share | ARI | NMI | forward coverage | profile JSD (semantic→q) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results.items():
        c = row["semantic_vs_q_clusters"]
        lines.append(
            f"| {name} | {row['q_clusters']} | {row['q_cluster_max_share']:.4f} | {c['ari']:.4f} | "
            f"{c['nmi']:.4f} | {c['coverage_weighted']:.4f} | {c['profile_jsd_weighted']:.4f} |"
        )
    lines += [
        "",
        "## Top semantic clusters by size",
        "",
        "| arm | cluster | size | top role before | top role after | profile JSD | multi before | multi after | cross rate before | cross rate after |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in results.items():
        for cluster in row["semantic_profiles"]["top_clusters"]:
            lines.append(
                f"| {name} | {cluster['cluster']} | {cluster['size']} | {cluster['top_before']} | "
                f"{cluster['top_after']} | {cluster['profile_jsd']:.4f} | {cluster['multi_before']:.3f} | "
                f"{cluster['multi_after']:.3f} | {cluster['cross_before']:.4f} | {cluster['cross_after']:.4f} |"
            )
    lines += [
        "",
        "## Top role-pair changes (same semantic clusters)",
        "",
        "| arm | left role | right role | edges | before within | after within | before lift | after lift | delta lift |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results.items():
        for pair in row["role_pair_relation"]["top_pair_deltas"][:6]:
            lines.append(
                f"| {name} | {pair['left']} | {pair['right']} | {pair['count']} | {pair['before_within']:.4f} | "
                f"{pair['after_within']:.4f} | {pair['before_lift']:.3f} | {pair['after_lift']:.3f} | {pair['delta_lift']:+.3f} |"
            )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
