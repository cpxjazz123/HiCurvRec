"""Cluster-level 73-d ORCA: full-graph aggregation (A) vs internal-subgraph ORCA (B).

Fixed semantic clusters from sentence-T5 KMeans(256).  For the behaviour graph and each q-kNN
graph:

A. aggregate the full-graph node ORCA table over each semantic cluster;
B. keep only edges internal to each semantic cluster, run ORCA on that internal graph, and
   aggregate the internal node ORCA table over the same clusters.

Also keep external information: boundary nodes (full-graph ORCA), cross-cluster edge rate, and the
cross-cluster 9x9 role-pair matrix.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/cluster_internal_orca.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cluster_orca_relations import jsd  # noqa: E402
from semantic_qknn_orca import (  # noqa: E402
    BEHAVIOUR_ORCA,
    ORBIT_GROUPS,
    qknn_edges,
    quantized_vectors,
    run_orca,
    semantic_labels,
)
from sid_structural_fidelity import load_behavior_graph  # noqa: E402

REPO = HERE.parents[2]
VECTORS = {
    "pure_euclidean_TIGER": HERE / "tiger_vectors.npz",
    "mixed_F0": HERE / "mixed_vectors.npz",
}
OUT_JSON = HERE / "cluster_internal_orca.json"
OUT_MD = HERE / "cluster_internal_orca.md"
ORCA_GROUPS = list(json.loads(ORBIT_GROUPS.read_text()).keys())
DEPTH = 73


def aggregate_vectors(table: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.log1p(np.maximum(table, 0.0))
    sums = np.zeros((k, values.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, values)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    means = sums / np.maximum(counts[:, None], 1.0)
    norm = means.sum(axis=1, keepdims=True)
    return means / np.where(norm > 0, norm, 1.0), counts


def compare_vectors(before: np.ndarray, after: np.ndarray, sizes: np.ndarray) -> dict:
    valid = (before.sum(axis=1) > 0) & (after.sum(axis=1) > 0)
    jsds = np.array([
        jsd(before[c], after[c]) if valid[c] else np.nan for c in range(len(before))
    ])
    cosine = np.array([
        float(np.dot(before[c], after[c]) / max(np.linalg.norm(before[c]) * np.linalg.norm(after[c]), 1e-12))
        if valid[c] else np.nan
        for c in range(len(before))
    ])
    valid_jsd = np.isfinite(jsds)
    valid_cos = np.isfinite(cosine)
    return {
        "valid_clusters": int(valid.sum()),
        "skipped_clusters": int((~valid).sum()),
        "jsd_weighted": float(np.average(jsds[valid_jsd], weights=sizes[valid_jsd])) if valid_jsd.any() else float("nan"),
        "jsd_median": float(np.median(jsds[valid_jsd])) if valid_jsd.any() else float("nan"),
        "cosine_weighted": float(np.average(cosine[valid_cos], weights=sizes[valid_cos])) if valid_cos.any() else float("nan"),
        "cosine_median": float(np.median(cosine[valid_cos])) if valid_cos.any() else float("nan"),
        "jsd": jsds,
        "cosine": cosine,
    }


def boundary_vectors(table: np.ndarray, labels: np.ndarray, adjacency: list[set[int]], k: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.log1p(np.maximum(table, 0.0))
    out = np.zeros((k, values.shape[1]), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    members: list[list[int]] = [[] for _ in range(k)]
    for item, label in enumerate(labels):
        members[int(label)].append(item)
    for label, nodes in enumerate(members):
        if not nodes:
            continue
        inside = set(nodes)
        external: set[int] = set()
        for node in nodes:
            external.update(adjacency[node])
        external.difference_update(inside)
        if external:
            out[label] = values[list(external)].mean(axis=0)
            counts[label] = len(external)
    norm = out.sum(axis=1, keepdims=True)
    return out / np.where(norm > 0, norm, 1.0), counts


def cluster_edges(labels: np.ndarray, edges: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    same = labels[edges[:, 0]] == labels[edges[:, 1]]
    within = np.bincount(labels[edges[:, 0]][same], minlength=k)
    incident = np.bincount(labels[edges[:, 0]], minlength=k) + np.bincount(labels[edges[:, 1]], minlength=k)
    cross = incident - 2 * within
    return within, cross, same


def cross_role_matrix(top: np.ndarray, labels: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cross = labels[edges[:, 0]] != labels[edges[:, 1]]
    left = top[edges[cross, 0]]
    right = top[edges[cross, 1]]
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    counts = np.bincount(low * 9 + high, minlength=81)
    matrix = counts.reshape(9, 9)
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    total = matrix.sum()
    return (matrix / total if total else matrix), counts


def top_changed(jsds: np.ndarray, sizes: np.ndarray, within_before: np.ndarray, within_after: np.ndarray,
                cross_before: np.ndarray, cross_after: np.ndarray, limit: int = 10) -> list[dict]:
    rows = []
    for cluster in np.argsort(-np.nan_to_num(jsds, nan=-1.0))[:limit]:
        rows.append({
            "cluster": int(cluster),
            "size": int(sizes[cluster]),
            "jsd": float(jsds[cluster]),
            "internal_edges_before": int(within_before[cluster]),
            "internal_edges_after": int(within_after[cluster]),
            "cross_rate_before": float(cross_before[cluster]),
            "cross_rate_after": float(cross_after[cluster]),
        })
    return rows


def main() -> int:
    nodes, behavior_edges, behavior_adj = load_behavior_graph()
    semantic, semantic_inertia = semantic_labels(nodes)
    k = int(semantic.max() + 1)
    sizes = np.bincount(semantic)
    behavior_table = np.loadtxt(BEHAVIOUR_ORCA, dtype=np.float64)
    behavior_full, _ = aggregate_vectors(behavior_table, semantic, k)
    behavior_internal_edges = behavior_edges[semantic[behavior_edges[:, 0]] == semantic[behavior_edges[:, 1]]]
    print(f"behaviour internal edges {len(behavior_internal_edges)} / {len(behavior_edges)}", flush=True)
    behavior_internal_table = run_orca(nodes, behavior_internal_edges, "cluster_internal_behaviour")
    behavior_internal, _ = aggregate_vectors(behavior_internal_table, semantic, k)
    behavior_boundary, behavior_boundary_counts = boundary_vectors(behavior_table, semantic, behavior_adj, k)
    behavior_within, behavior_cross, _ = cluster_edges(semantic, behavior_edges, k)
    behavior_cross_rate = np.divide(behavior_cross, behavior_within + behavior_cross,
                                    out=np.zeros(k), where=(behavior_within + behavior_cross) > 0)
    behavior_top = behavior_table.argmax(axis=1)  # replaced below with 9-group top roles
    from semantic_qknn_orca import roles_from_table
    _, behavior_top, behavior_multi, _ = roles_from_table(behavior_table)
    behavior_cross_matrix, behavior_cross_counts = cross_role_matrix(behavior_top, semantic, behavior_edges)

    behavior_internal_vs_full = compare_vectors(behavior_full, behavior_internal, sizes)

    results = {
        "semantic_clusters": int(k),
        "semantic_inertia": semantic_inertia,
        "semantic_max_share": float(sizes.max() / nodes),
        "behaviour_internal_edges": int(len(behavior_internal_edges)),
        "behaviour_total_edges": int(len(behavior_edges)),
        "behaviour_any_internal_edges": int((behavior_within > 0).sum()),
        "behaviour_cross_role_matrix": behavior_cross_matrix.tolist(),
        "behaviour_cross_role_counts": behavior_cross_counts.tolist(),
        "behaviour_cross_edge_rate": float(behavior_cross.sum() / max(behavior_cross.sum() + behavior_within.sum(), 1)),
        "behaviour_internal_vs_full": {
            key: value for key, value in behavior_internal_vs_full.items() if not isinstance(value, np.ndarray)
        },
        "arms": {},
    }

    for name, path in VECTORS.items():
        _, q = quantized_vectors(path)
        edges = qknn_edges(q)
        q_adjacency = [set() for _ in range(nodes)]
        for left, right in edges:
            q_adjacency[int(left)].add(int(right))
            q_adjacency[int(right)].add(int(left))
        q_table = run_orca(nodes, edges, f"semantic_qknn_{name}")
        q_full, _ = aggregate_vectors(q_table, semantic, k)
        q_internal_edges = edges[semantic[edges[:, 0]] == semantic[edges[:, 1]]]
        q_internal_table = run_orca(nodes, q_internal_edges, f"cluster_internal_{name}")
        q_internal, _ = aggregate_vectors(q_internal_table, semantic, k)
        q_boundary, q_boundary_counts = boundary_vectors(q_table, semantic, q_adjacency, k)
        q_within, q_cross, _ = cluster_edges(semantic, edges, k)
        q_cross_rate = np.divide(q_cross, q_within + q_cross, out=np.zeros(k), where=(q_within + q_cross) > 0)
        _, q_top, q_multi, _ = roles_from_table(q_table)
        q_cross_matrix, q_cross_counts = cross_role_matrix(q_top, semantic, edges)

        full_compare = compare_vectors(behavior_full, q_full, sizes)
        internal_compare = compare_vectors(behavior_internal, q_internal, sizes)
        boundary_compare = compare_vectors(behavior_boundary, q_boundary, sizes)
        cross_jsd = jsd(behavior_cross_matrix.ravel(), q_cross_matrix.ravel())
        internal_valid = (behavior_internal.sum(axis=1) > 0) & (q_internal.sum(axis=1) > 0)
        full_valid = (behavior_full.sum(axis=1) > 0) & (q_full.sum(axis=1) > 0)
        results["arms"][name] = {
            "q_edges": int(len(edges)),
            "q_internal_edges": int(len(q_internal_edges)),
            "q_any_internal_edges": int((q_within > 0).sum()),
            "q_cross_edge_rate": float(q_cross.sum() / max(q_cross.sum() + q_within.sum(), 1)),
            "q_cross_role_matrix": q_cross_matrix.tolist(),
            "q_cross_role_counts": q_cross_counts.tolist(),
            "cross_role_matrix_jsd": float(cross_jsd),
            "method_a_full": {key: value for key, value in full_compare.items() if not isinstance(value, np.ndarray)},
            "method_b_internal": {key: value for key, value in internal_compare.items() if not isinstance(value, np.ndarray)},
            "boundary": {key: value for key, value in boundary_compare.items() if not isinstance(value, np.ndarray)},
            "boundary_counts_before": behavior_boundary_counts.tolist(),
            "boundary_counts_after": q_boundary_counts.tolist(),
            "top_changed_internal": top_changed(
                internal_compare["jsd"], sizes, behavior_within, q_within,
                behavior_cross_rate, q_cross_rate,
            ),
            "top_changed_full": top_changed(
                full_compare["jsd"], sizes, behavior_within, q_within,
                behavior_cross_rate, q_cross_rate,
            ),
            "method_a_valid": int(full_valid.sum()),
            "method_b_valid": int(internal_valid.sum()),
        }
        print(
            f"{name}: internal edges {len(q_internal_edges)}, "
            f"A valid {full_valid.sum()}, B valid {internal_valid.sum()}, cross JSD {cross_jsd:.4f}",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# Cluster-level 73-d ORCA: full graph vs internal subgraph",
        "",
        "Fixed semantic clusters: KMeans(256) on L2-normalised sentence-T5 embeddings. "
        "Method A aggregates the full-graph node ORCA over each cluster. "
        "Method B runs ORCA on the union of internal semantic-cluster edges and aggregates that. "
        "Vectors are `mean_i log1p(ORCA_i)` normalised to sum 1; JSD and cosine are computed per cluster.",
        "",
        f"Behaviour graph: **{nodes}** nodes, **{len(behavior_edges)}** edges, **{k}** semantic clusters.",
        f"Behaviour internal edges: **{len(behavior_internal_edges)} / {len(behavior_edges)}**; "
        f"clusters with at least one internal edge: **{(behavior_within > 0).sum()} / {k}**.",
        "",
        "## Method A: full-graph ORCA aggregated per semantic cluster",
        "",
        "| arm | valid clusters | JSD weighted | JSD median | cosine weighted | cosine median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in results["arms"].items():
        value = row["method_a_full"]
        lines.append(
            f"| {name} | {value['valid_clusters']} | {value['jsd_weighted']:.4f} | {value['jsd_median']:.4f} | "
            f"{value['cosine_weighted']:.4f} | {value['cosine_median']:.4f} |"
        )
    lines += [
        "",
        "## Method B: internal-subgraph ORCA per semantic cluster",
        "",
        "| arm | q internal edges | clusters with internal edges | valid clusters | JSD weighted | JSD median | cosine weighted | cosine median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results["arms"].items():
        value = row["method_b_internal"]
        lines.append(
            f"| {name} | {row['q_internal_edges']} | {row['q_any_internal_edges']} | {row['method_b_valid']} | "
            f"{value['jsd_weighted']:.4f} | {value['jsd_median']:.4f} | {value['cosine_weighted']:.4f} | "
            f"{value['cosine_median']:.4f} |"
        )
    lines += [
        "",
        "## External structure: boundary ORCA and cross-cluster role matrix",
        "",
        "Boundary vectors use the full-graph ORCA of unique external neighbours. "
        "`cross role JSD` compares the 9x9 role-pair matrix of cross-cluster edges before/after.",
        "",
        "| arm | boundary JSD weighted | boundary JSD median | boundary cosine weighted | cross rate before | cross rate after | cross role JSD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results["arms"].items():
        value = row["boundary"]
        lines.append(
            f"| {name} | {value['jsd_weighted']:.4f} | {value['jsd_median']:.4f} | "
            f"{value['cosine_weighted']:.4f} | {results['behaviour_cross_edge_rate']:.4f} | "
            f"{row['q_cross_edge_rate']:.4f} | {row['cross_role_matrix_jsd']:.4f} |"
        )
    lines += [
        "",
        "## How much does dropping cross-cluster edges change the behaviour cluster vector?",
        "",
        "Behaviour graph: full-graph aggregation (A) vs internal-only ORCA (B).",
        "",
        f"- valid clusters: **{results['behaviour_internal_vs_full']['valid_clusters']} / {k}**",
        f"- JSD weighted / median: **{results['behaviour_internal_vs_full']['jsd_weighted']:.4f} / "
        f"{results['behaviour_internal_vs_full']['jsd_median']:.4f}**",
        f"- cosine weighted / median: **{results['behaviour_internal_vs_full']['cosine_weighted']:.4f} / "
        f"{results['behaviour_internal_vs_full']['cosine_median']:.4f}**",
        "",
        "This is the direct cost of using only internal subgraphs on the behaviour side.",
        "",
        "## Top changed semantic clusters, internal ORCA",
        "",
        "| arm | cluster | size | JSD | internal edges before | internal edges after | cross rate before | cross rate after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results["arms"].items():
        for item in row["top_changed_internal"]:
            lines.append(
                f"| {name} | {item['cluster']} | {item['size']} | {item['jsd']:.4f} | "
                f"{item['internal_edges_before']} | {item['internal_edges_after']} | "
                f"{item['cross_rate_before']:.4f} | {item['cross_rate_after']:.4f} |"
            )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
