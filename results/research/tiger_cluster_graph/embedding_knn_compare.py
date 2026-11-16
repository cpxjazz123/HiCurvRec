"""Pre-quantisation z-kNN vs post-quantisation q-kNN vs behaviour graph.

No semantic clustering; this is the direct global-kNN experiment:

    behaviour G_b  vs  z-kNN G_z  vs  q-kNN G_q

For each arm, both near-neighbour graphs use the same global k=10 candidate rule.  The graph with
more edges is trimmed by edge distance to the smaller graph's edge count, so the z and q graphs
have the same number of edges and can be compared without a density confound.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/embedding_knn_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cluster_orca_relations import jsd  # noqa: E402
from semantic_qknn_orca import (  # noqa: E402
    BEHAVIOUR_ORCA,
    ORBIT_GROUPS,
    qknn_edges,
    quantized_vectors,
    roles_from_table,
    run_orca,
)
from sid_structural_fidelity import load_behavior_graph  # noqa: E402

VECTORS = {
    "pure_euclidean_TIGER": HERE / "tiger_vectors.npz",
    "mixed_F0": HERE / "mixed_vectors.npz",
}
OUT_JSON = HERE / "embedding_knn_compare.json"
OUT_MD = HERE / "embedding_knn_compare.md"
K = 10
ROLE_NAMES = list(json.loads(ORBIT_GROUPS.read_text()).keys())


def edged(edges: np.ndarray) -> set[tuple[int, int]]:
    return {(int(left), int(right)) for left, right in edges.tolist()}


def edge_distances(vectors: np.ndarray, edges: np.ndarray) -> np.ndarray:
    left = vectors[edges[:, 0]]
    right = vectors[edges[:, 1]]
    return np.linalg.norm(left - right, axis=1)


def trim_edges(vectors: np.ndarray, edges: np.ndarray, target: int) -> tuple[np.ndarray, int]:
    if len(edges) <= target:
        return edges, 0
    distance = edge_distances(vectors, edges)
    keep = np.argsort(distance)[:target]
    return edges[keep], int(len(edges) - target)


def graph_stats(edges: np.ndarray, behavior_edges: np.ndarray, nodes: int) -> dict:
    graph_set = edged(edges)
    behavior_set = edged(behavior_edges)
    intersection = len(graph_set & behavior_set)
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes))
    graph.add_edges_from(map(tuple, edges.tolist()))
    degree = np.bincount(edges.ravel(), minlength=nodes) if len(edges) else np.zeros(nodes)
    behavior_degree = np.bincount(behavior_edges.ravel(), minlength=nodes)
    triangles = sum(nx.triangles(graph).values()) / 3.0
    return {
        "edges": int(len(graph_set)),
        "behavior_recall": float(intersection / len(behavior_set)) if behavior_set else float("nan"),
        "precision": float(intersection / len(graph_set)) if graph_set else float("nan"),
        "jaccard": float(intersection / len(graph_set | behavior_set)) if graph_set | behavior_set else float("nan"),
        "degree_spearman": float(spearmanr(behavior_degree, degree).statistic),
        "components": int(nx.number_connected_components(graph)),
        "triangles": float(triangles),
    }


def role_pair_dist(top: np.ndarray, edges: np.ndarray) -> np.ndarray:
    left = top[edges[:, 0]]
    right = top[edges[:, 1]]
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    counts = np.bincount(low * 9 + high, minlength=81)
    return counts / max(counts.sum(), 1)


def role_compare(table_a: np.ndarray, edges_a: np.ndarray, table_b: np.ndarray, edges_b: np.ndarray) -> dict:
    roles_a, top_a, multi_a, _ = roles_from_table(table_a)
    roles_b, top_b, multi_b, _ = roles_from_table(table_b)
    item_jsd = np.array([jsd(roles_a[i], roles_b[i]) for i in range(len(roles_a))])
    return {
        "top_role_ari": float(adjusted_rand_score(top_a, top_b)),
        "top_role_nmi": float(normalized_mutual_info_score(top_a, top_b)),
        "top_role_agreement": float(np.mean(top_a == top_b)),
        "role_distribution_jsd_mean": float(np.nanmean(item_jsd)),
        "role_distribution_jsd_median": float(np.nanmedian(item_jsd)),
        "multi_before": float(multi_a.mean()),
        "multi_after": float(multi_b.mean()),
        "multi_agreement": float(np.mean(multi_a == multi_b)),
        "role_pair_jsd": float(jsd(role_pair_dist(top_a, edges_a), role_pair_dist(top_b, edges_b))),
    }


def compare_graphs(edges_a: np.ndarray, edges_b: np.ndarray) -> dict:
    set_a = edged(edges_a)
    set_b = edged(edges_b)
    return {
        "edges_a": len(set_a),
        "edges_b": len(set_b),
        "intersection": len(set_a & set_b),
        "jaccard": len(set_a & set_b) / max(len(set_a | set_b), 1),
        "only_a": len(set_a - set_b),
        "only_b": len(set_b - set_a),
    }


def main() -> int:
    nodes, behavior_edges, _ = load_behavior_graph()
    behavior_table = np.loadtxt(BEHAVIOUR_ORCA, dtype=np.float64)
    behavior_roles, behavior_top, behavior_multi, _ = roles_from_table(behavior_table)
    behavior_stats = graph_stats(behavior_edges, behavior_edges, nodes)
    print(f"behaviour edges {len(behavior_edges)}", flush=True)

    result = {
        "nodes": int(nodes),
        "behaviour_edges": int(len(behavior_edges)),
        "k_candidates": K,
        "behaviour": {
            "graph": behavior_stats,
            "multi_role_fraction": float(behavior_multi.mean()),
        },
        "arms": {},
    }

    for name, path in VECTORS.items():
        z, q = quantized_vectors(path)
        z_raw = qknn_edges(z, k=K)
        q_raw = qknn_edges(q, k=K)
        target = min(len(z_raw), len(q_raw))
        z_edges, z_removed = trim_edges(z, z_raw, target)
        q_edges, q_removed = trim_edges(q, q_raw, target)
        q_vectors = q
        q_orca = run_orca(nodes, q_edges, f"embedding_q_{name}_matched")
        z_orca = run_orca(nodes, z_edges, f"embedding_z_{name}_matched")
        z_stats = graph_stats(z_edges, behavior_edges, nodes)
        q_stats = graph_stats(q_edges, behavior_edges, nodes)
        z_vs_behavior = role_compare(behavior_table, behavior_edges, z_orca, z_edges)
        q_vs_behavior = role_compare(behavior_table, behavior_edges, q_orca, q_edges)
        z_vs_q_edges = compare_graphs(z_edges, q_edges)
        z_vs_q_roles = role_compare(z_orca, z_edges, q_orca, q_edges)
        result["arms"][name] = {
            "z_edges_raw": int(len(z_raw)),
            "q_edges_raw": int(len(q_raw)),
            "matched_edges": int(target),
            "z_trimmed": int(z_removed),
            "q_trimmed": int(q_removed),
            "z": {"graph": z_stats, "roles_vs_behavior": z_vs_behavior},
            "q": {"graph": q_stats, "roles_vs_behavior": q_vs_behavior},
            "z_vs_q": {"graph": z_vs_q_edges, "roles": z_vs_q_roles},
        }
        print(
            f"{name}: z {len(z_raw)}->{len(z_edges)}, q {len(q_raw)}->{len(q_edges)}, "
            f"behavior recall z/q {z_stats['behavior_recall']:.4f}/{q_stats['behavior_recall']:.4f}, "
            f"role JSD z/q {z_vs_behavior['role_distribution_jsd_mean']:.4f}/"
            f"{q_vs_behavior['role_distribution_jsd_mean']:.4f}",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# Pre-quantisation z-kNN vs post-quantisation q-kNN vs behaviour graph",
        "",
        "No semantic clustering. Both z and q graphs use global k=10 candidates and are trimmed "
        "to the smaller graph's edge count by shortest edge distance, so the z/q comparison has "
        "the same number of edges.",
        "",
        "## Graph recovery of behaviour edges",
        "",
        "| arm | graph | edges | behavior recall | precision | Jaccard | degree Spearman | components | triangles |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in result["arms"].items():
        for label, value in (("z-kNN", arm["z"]["graph"]), ("q-kNN", arm["q"]["graph"])):
            lines.append(
                f"| {name} | {label} | {value['edges']} | {value['behavior_recall']:.4f} | "
                f"{value['precision']:.4f} | {value['jaccard']:.4f} | {value['degree_spearman']:.4f} | "
                f"{value['components']} | {value['triangles']:.0f} |"
            )
    lines += [
        "",
        "## ORCA roles: behavior vs z / behavior vs q",
        "",
        "| arm | pair | top ARI | top NMI | same top | role JSD mean | role JSD median | multi before | multi after | role-pair JSD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in result["arms"].items():
        for label, value in (("behavior vs z", arm["z"]["roles_vs_behavior"]),
                             ("behavior vs q", arm["q"]["roles_vs_behavior"])):
            lines.append(
                f"| {name} | {label} | {value['top_role_ari']:.4f} | {value['top_role_nmi']:.4f} | "
                f"{value['top_role_agreement']:.4f} | {value['role_distribution_jsd_mean']:.4f} | "
                f"{value['role_distribution_jsd_median']:.4f} | {value['multi_before']:.4f} | "
                f"{value['multi_after']:.4f} | {value['role_pair_jsd']:.4f} |"
            )
    lines += [
        "",
        "## Quantisation step: z-kNN vs q-kNN",
        "",
        "| arm | z edges | q edges | edge intersection | edge Jaccard | top ARI | top NMI | same top | role JSD mean | multi agreement | role-pair JSD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in result["arms"].items():
        edge = arm["z_vs_q"]["graph"]
        role = arm["z_vs_q"]["roles"]
        lines.append(
            f"| {name} | {edge['edges_a']} | {edge['edges_b']} | {edge['intersection']} | "
            f"{edge['jaccard']:.4f} | {role['top_role_ari']:.4f} | {role['top_role_nmi']:.4f} | "
            f"{role['top_role_agreement']:.4f} | {role['role_distribution_jsd_mean']:.4f} | "
            f"{role['multi_agreement']:.4f} | {role['role_pair_jsd']:.4f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
