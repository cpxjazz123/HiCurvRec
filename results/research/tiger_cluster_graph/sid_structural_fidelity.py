"""Behavior-graph -> quantized-SID structural fidelity.

Compares two exported SID tables directly against the TRAIN behaviour graph:

    pure  results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json
    mixed results/stage2_RQ-VAE/TIGER_mixed_rq_F0/item_sids.json

The evaluation uses the actual SID prefixes (L1/L2/L3), never KMeans on z/q.  Ground truth is the
same undirected behaviour graph for both arms, so every number is a function of the SID table only.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/sid_structural_fidelity.py

Writes sid_structural_fidelity.json and sid_structural_fidelity.md.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
ORCA = REPO / "results" / "orca_build" / "runs" / "behaviour_gpu_5.out"
ORBIT_GROUPS = HERE / "orbit_groups.json"
GRAPHLETS = HERE / "graphlet_instances.npz"
SIDS = {
    "pure_euclidean_TIGER": REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json",
    "mixed_F0": REPO / "results" / "stage2_RQ-VAE" / "TIGER_mixed_rq_F0" / "item_sids.json",
}
OUT_JSON = HERE / "sid_structural_fidelity.json"
OUT_MD = HERE / "sid_structural_fidelity.md"
DEPTHS = (1, 2, 3)
MULTI_ROLE_THRESHOLD = 0.25


def load_behavior_graph() -> tuple[int, np.ndarray, list[set[int]]]:
    frame = pd.read_parquet(TRAIN, columns=["user", "seen_history", "target"])
    best: dict[int, tuple[int, list[int]]] = {}
    for user, history, target in zip(
        frame["user"].to_numpy(), frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)
    ):
        history = [] if history is None else [int(x) for x in history]
        user = int(user)
        if user not in best or len(history) > best[user][0]:
            best[user] = (len(history), [*history, int(target)])
    pairs: set[tuple[int, int]] = set()
    for _, session in best.values():
        for left, right in zip(session, session[1:]):
            pairs.add((left, right) if left < right else (right, left))
    edges = np.array(sorted(pairs), dtype=np.int64)
    nodes = int(edges.max()) + 1
    adjacency = [set() for _ in range(nodes)]
    for left, right in edges:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    return nodes, edges, adjacency


def load_sid_tokens(nodes: int, path: Path) -> np.ndarray:
    raw = json.loads(path.read_text())
    tokens = np.full((nodes, 3), -1, dtype=np.int64)
    for key, value in raw.items():
        item = int(key)
        if item < nodes:
            tokens[item] = [int(x) for x in value[:3]]
    if (tokens < 0).any():
        raise ValueError(f"{path} does not cover all {nodes} nodes")
    return tokens


def prefix_labels(tokens: np.ndarray, depth: int) -> np.ndarray:
    return np.unique(tokens[:, :depth], axis=0, return_inverse=True)[1]


def lcp(tokens: np.ndarray, edges: np.ndarray) -> np.ndarray:
    left = tokens[edges[:, 0]]
    right = tokens[edges[:, 1]]
    same1 = left[:, 0] == right[:, 0]
    same2 = same1 & (left[:, 1] == right[:, 1])
    same3 = same2 & (left[:, 2] == right[:, 2])
    return same1.astype(np.int8) + same2.astype(np.int8) + same3.astype(np.int8)


def prefix_masks(tokens: np.ndarray, edges: np.ndarray) -> dict[int, np.ndarray]:
    left = tokens[edges[:, 0]]
    right = tokens[edges[:, 1]]
    same1 = left[:, 0] == right[:, 0]
    same2 = same1 & (left[:, 1] == right[:, 1])
    same3 = same2 & (left[:, 2] == right[:, 2])
    return {1: same1, 2: same2, 3: same3}


def edge_summary(tokens: np.ndarray, edges: np.ndarray, n: int) -> dict:
    masks = prefix_masks(tokens, edges)
    values = lcp(tokens, edges)
    out = {
        "n_edges": int(len(edges)),
        "global_edge_density": float(2.0 * len(edges) / (n * (n - 1))),
        "lcp_mean": float(values.mean()),
        "lcp_distribution": np.bincount(values, minlength=4).astype(int).tolist(),
    }
    for depth, mask in masks.items():
        rate = float(mask.mean())
        out[f"prefix{depth}"] = rate
        out[f"prefix{depth}_se"] = math.sqrt(rate * (1.0 - rate) / len(edges)) if len(edges) else float("nan")
    return out


def cluster_summary(tokens: np.ndarray, edges: np.ndarray, n: int) -> dict:
    out = {}
    global_density = float(2.0 * len(edges) / (n * (n - 1)))
    for depth in DEPTHS:
        labels = prefix_labels(tokens, depth)
        counts = np.bincount(labels)
        within = labels[edges[:, 0]] == labels[edges[:, 1]]
        incident = np.bincount(labels[edges[:, 0]], minlength=len(counts)) + np.bincount(
            labels[edges[:, 1]], minlength=len(counts)
        )
        within_count = np.bincount(labels[edges[:, 0]][within], minlength=len(counts))
        denominator = incident - within_count
        shares = np.divide(
            within_count,
            denominator,
            out=np.zeros(len(counts), dtype=float),
            where=denominator > 0,
        )
        pairs = float(np.sum(counts.astype(np.float64) * (counts - 1.0) / 2.0))
        out[f"L{depth}"] = {
            "clusters": int(len(counts)),
            "max_cluster_share": float(counts.max() / len(labels)),
            "entropy": float(-np.sum((counts / counts.sum()) * np.log2(counts / counts.sum()))),
            "within_edge_share_weighted": float(within_count.sum() / denominator.sum()),
            "within_edge_share_max": float(shares[denominator > 0].max()) if (denominator > 0).any() else float("nan"),
            "within_edges": int(within.sum()),
            "within_edge_density": float(within.sum() / pairs) if pairs > 0 else float("nan"),
            "within_density_lift": float(within.sum() / pairs / global_density) if pairs > 0 and global_density > 0 else float("nan"),
            "random_prefix": float(np.sum((counts / len(labels)) ** 2)),
        }
    return out


def purity(labels: np.ndarray, target: np.ndarray) -> dict:
    rows = labels.max() + 1
    cols = target.max() + 1
    counts = np.zeros((rows, cols), dtype=np.float64)
    np.add.at(counts, (labels, target), 1.0)
    sizes = counts.sum(axis=1)
    maxima = counts.max(axis=1) / np.maximum(sizes, 1.0)
    weighted = float((maxima * sizes).sum() / sizes.sum()) if sizes.sum() else float("nan")
    probabilities = counts / np.maximum(sizes[:, None], 1.0)
    log_probabilities = np.where(probabilities > 0, np.log2(np.maximum(probabilities, 1e-300)), 0.0)
    entropy = -np.sum(probabilities * log_probabilities, axis=1)
    marginal = counts.sum(axis=0) / counts.sum()
    baseline = float(np.sum(marginal ** 2))
    return {
        "purity": weighted,
        "purity_baseline": baseline,
        "purity_lift": float(weighted / baseline) if baseline > 0 else float("nan"),
        "normalized_purity": float((weighted - baseline) / (1.0 - baseline)) if baseline < 1.0 else float("nan"),
        "entropy": float((entropy * sizes).sum() / sizes.sum()),
        "n_target_classes": int(cols),
    }


def community_summary(tokens: np.ndarray, edges: np.ndarray, community: np.ndarray) -> dict:
    same = community[edges[:, 0]] == community[edges[:, 1]]
    masks = prefix_masks(tokens, edges)
    out = {
        "n_communities": int(community.max() + 1),
        "edge_same_community_rate": float(same.mean()),
        "within": {"n_edges": int(same.sum())},
        "cross": {"n_edges": int((~same).sum())},
    }
    for depth, mask in masks.items():
        labels = prefix_labels(tokens, depth)
        counts = np.bincount(labels)
        random = float(np.sum((counts / len(labels)) ** 2))
        within = float(mask[same].mean()) if same.any() else float("nan")
        cross = float(mask[~same].mean()) if (~same).any() else float("nan")
        out["within"][f"prefix{depth}"] = within
        out["cross"][f"prefix{depth}"] = cross
        out["within"][f"prefix{depth}_lift"] = within / random if random > 0 else float("nan")
        out["cross"][f"prefix{depth}_lift"] = cross / random if random > 0 else float("nan")
        out[f"random_prefix{depth}"] = random
    out["gap_prefix1"] = out["within"]["prefix1"] - out["cross"]["prefix1"]
    out["gap_prefix2"] = out["within"]["prefix2"] - out["cross"]["prefix2"]
    out["gap_prefix3"] = out["within"]["prefix3"] - out["cross"]["prefix3"]
    out["gap_lift1"] = out["within"]["prefix1_lift"] - out["cross"]["prefix1_lift"]
    out["gap_lift2"] = out["within"]["prefix2_lift"] - out["cross"]["prefix2_lift"]
    out["gap_lift3"] = out["within"]["prefix3_lift"] - out["cross"]["prefix3_lift"]
    out["L1"] = {
        "community": purity(prefix_labels(tokens, 1), community),
        "nmi": float(normalized_mutual_info_score(community, prefix_labels(tokens, 1))),
        "ari": float(adjusted_rand_score(community, prefix_labels(tokens, 1))),
    }
    return out


def load_roles() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    groups = json.loads(ORBIT_GROUPS.read_text())
    names = list(groups)
    gdv = np.loadtxt(ORCA)
    gdv = np.maximum(gdv, 0.0)
    gdv = gdv / np.where(gdv.sum(axis=1, keepdims=True) > 0, gdv.sum(axis=1, keepdims=True), 1.0)
    scores = np.zeros((gdv.shape[0], len(names)), dtype=np.float64)
    for index, columns in enumerate(groups.values()):
        scores[:, index] = gdv[:, columns].sum(axis=1)
    role_dist = scores / np.where(scores.sum(axis=1, keepdims=True) > 0, scores.sum(axis=1, keepdims=True), 1.0)
    top = role_dist.argmax(axis=1)
    multi = (role_dist >= MULTI_ROLE_THRESHOLD).sum(axis=1) >= 2
    return role_dist, top, multi, names


def role_summary(tokens: np.ndarray, edges: np.ndarray, top: np.ndarray, multi: np.ndarray, names: list[str]) -> dict:
    masks = prefix_masks(tokens, edges)
    multi_edge = multi[edges[:, 0]] | multi[edges[:, 1]]
    both_multi = multi[edges[:, 0]] & multi[edges[:, 1]]
    out = {
        "multi_role_fraction": float(multi.mean()),
        "multi_role_edges": {
            "n_edges": int(multi_edge.sum()),
            "both_multi_n_edges": int(both_multi.sum()),
        },
        "single_role_edges": {"n_edges": int((~multi_edge).sum())},
    }
    for depth, mask in masks.items():
        out["multi_role_edges"][f"prefix{depth}"] = float(mask[multi_edge].mean()) if multi_edge.any() else float("nan")
        out["single_role_edges"][f"prefix{depth}"] = float(mask[~multi_edge].mean()) if (~multi_edge).any() else float("nan")
        out[f"top_role_nmi_L{depth}"] = float(normalized_mutual_info_score(top, prefix_labels(tokens, depth)))
        out[f"top_role_ari_L{depth}"] = float(adjusted_rand_score(top, prefix_labels(tokens, depth)))
        out[f"role_L{depth}"] = purity(prefix_labels(tokens, depth), top)
    # Role-pair table: how often do edges between two top-role groups share a prefix?
    left_role = top[edges[:, 0]]
    right_role = top[edges[:, 1]]
    low = np.minimum(left_role, right_role)
    high = np.maximum(left_role, right_role)
    pair = low * len(names) + high
    counts = np.bincount(pair, minlength=len(names) ** 2)
    rows = []
    for value in np.argsort(-counts):
        count = int(counts[value])
        if count == 0 or len(rows) >= 12:
            break
        left, right = divmod(int(value), len(names))
        selected = pair == value
        rows.append({
            "left": names[left],
            "right": names[right],
            "count": count,
            "prefix1": float(masks[1][selected].mean()),
            "prefix2": float(masks[2][selected].mean()),
            "prefix3": float(masks[3][selected].mean()),
        })
    out["top_role_pairs"] = rows
    return out


def graphlet_summary(tokens: np.ndarray, adjacency: list[set[int]], random_prefix: dict[int, float]) -> dict:
    if not GRAPHLETS.exists():
        raise FileNotFoundError(f"{GRAPHLETS} missing; run sample_graphlets.py first")
    archive = np.load(GRAPHLETS)
    out = {}
    for name in archive.files:
        instances = archive[name]
        per_instance = {1: [], 2: [], 3: []}
        all_prefix = {1: [], 2: [], 3: []}
        edge_count = 0
        for instance in instances:
            induced = []
            for left_index in range(len(instance)):
                for right_index in range(left_index + 1, len(instance)):
                    left, right = int(instance[left_index]), int(instance[right_index])
                    if right in adjacency[left]:
                        induced.append((left, right))
            if not induced:
                continue
            edge_count += len(induced)
            for depth in DEPTHS:
                shares = [
                    1.0 if np.array_equal(tokens[left, :depth], tokens[right, :depth]) else 0.0
                    for left, right in induced
                ]
                per_instance[depth].append(float(np.mean(shares)))
                all_prefix[depth].append(
                    1.0 if np.all(tokens[instance, :depth] == tokens[instance[0], :depth]) else 0.0
                )
        row = {"instances": int(len(instances)), "induced_edges": edge_count}
        for depth in DEPTHS:
            values = np.array(per_instance[depth], dtype=float)
            row[f"prefix{depth}"] = float(values.mean()) if len(values) else float("nan")
            row[f"prefix{depth}_se"] = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")
            row[f"all_prefix{depth}"] = float(np.mean(all_prefix[depth])) if all_prefix[depth] else float("nan")
            row[f"prefix{depth}_lift"] = row[f"prefix{depth}"] / random_prefix[depth] if random_prefix[depth] > 0 else float("nan")
        out[name] = row
    return out


def arm_summary(tokens: np.ndarray, edges: np.ndarray, adjacency: list[set[int]], community: np.ndarray, top: np.ndarray, multi: np.ndarray, names: list[str], n: int) -> dict:
    edges_out = edge_summary(tokens, edges, n)
    clusters = cluster_summary(tokens, edges, n)
    random_prefix = {depth: clusters[f"L{depth}"]["random_prefix"] for depth in DEPTHS}
    for depth in DEPTHS:
        observed = edges_out[f"prefix{depth}"]
        edges_out[f"random_prefix{depth}"] = random_prefix[depth]
        edges_out[f"prefix{depth}_lift"] = observed / random_prefix[depth] if random_prefix[depth] > 0 else float("nan")
        edges_out[f"prefix{depth}_excess"] = observed - random_prefix[depth]
        clusters[f"L{depth}"]["prefix_lift"] = edges_out[f"prefix{depth}_lift"]
    return {
        "edges": edges_out,
        "clusters": clusters,
        "community": community_summary(tokens, edges, community),
        "roles": role_summary(tokens, edges, top, multi, names),
        "graphlets": graphlet_summary(tokens, adjacency, random_prefix),
    }


def paired_delta(tokens_a: np.ndarray, tokens_b: np.ndarray, edges: np.ndarray) -> dict:
    masks_a = prefix_masks(tokens_a, edges)
    masks_b = prefix_masks(tokens_b, edges)
    out = {}
    for depth in DEPTHS:
        difference = masks_b[depth].astype(np.int8) - masks_a[depth].astype(np.int8)
        out[f"prefix{depth}"] = {
            "delta": float(difference.mean()),
            "se": float(difference.std(ddof=1) / math.sqrt(len(edges))) if len(edges) > 1 else float("nan"),
        }
    return out


def main() -> int:
    nodes, edges, adjacency = load_behavior_graph()
    print(f"behavior graph: {nodes} nodes, {len(edges)} edges", flush=True)
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes))
    graph.add_edges_from(map(tuple, edges.tolist()))
    communities = nx.community.louvain_communities(graph, seed=42, resolution=1.0)
    community = np.full(nodes, -1, dtype=np.int64)
    for index, members in enumerate(communities):
        community[list(members)] = index
    print(f"Louvain: {len(communities)} communities", flush=True)
    role_dist, top, multi, role_names = load_roles()
    tokens = {name: load_sid_tokens(nodes, path) for name, path in SIDS.items()}
    arms = {
        name: arm_summary(tokens[name], edges, adjacency, community, top, multi, role_names, nodes)
        for name in SIDS
    }
    pure = tokens["pure_euclidean_TIGER"]
    mixed = tokens["mixed_F0"]
    result = {
        "behavior_graph": {
            "nodes": int(nodes),
            "edges": int(len(edges)),
            "communities": int(len(communities)),
            "community_sizes": sorted([len(x) for x in communities], reverse=True),
            "multi_role_fraction": float(multi.mean()),
            "role_groups": role_names,
        },
        "arms": arms,
        "paired_mixed_minus_pure": paired_delta(pure, mixed, edges),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)
    lines = [
        "# Behavior Graph → Quantized SID Structural Fidelity",
        "",
        "Ground truth: TRAIN behaviour graph; both arms use the same graph and ORCA table.",
        "SID clusters are the actual L1/L2/L3 prefixes from item_sids.json (no KMeans).",
        "",
        f"Behaviour graph: **{nodes}** nodes, **{len(edges)}** edges, **{len(communities)}** Louvain communities.",
        f"Multi-role items: **{multi.mean():.3%}**.",
        "",
        "## Behaviour-edge prefix rates",
        "",
        "Raw prefix rates are not comparable across arms by themselves: a collapsed codebook makes "
        "prefix sharing easier. `lift@k = prefix@k / E_random@k`, where `E_random@k = sum_c (n_c/n)^2` "
        "over that arm's own Lk clusters; lift controls for cluster granularity and imbalance.",
        "",
        "| arm | prefix@1 | lift@1 | prefix@2 | lift@2 | prefix@3 | lift@3 | mean LCP | LCP 0/1/2/3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, arm in arms.items():
        edge = arm["edges"]
        lines.append(
            f"| {name} | {edge['prefix1']:.4f} | {edge['prefix1_lift']:.2f} | "
            f"{edge['prefix2']:.4f} | {edge['prefix2_lift']:.2f} | "
            f"{edge['prefix3']:.4f} | {edge['prefix3_lift']:.2f} | "
            f"{edge['lcp_mean']:.3f} | {edge['lcp_distribution']} |"
        )
    delta = result["paired_mixed_minus_pure"]
    lines += [
        "",
        "Paired mixed − pure (same edges): "
        f"prefix@1 {delta['prefix1']['delta']:+.4f} ± {1.96 * delta['prefix1']['se']:.4f}; "
        f"prefix@2 {delta['prefix2']['delta']:+.4f} ± {1.96 * delta['prefix2']['se']:.4f}; "
        f"prefix@3 {delta['prefix3']['delta']:+.4f} ± {1.96 * delta['prefix3']['se']:.4f} (95% CI).",
        "",
        "## SID prefix clusters",
        "",
        "| arm | depth | clusters | max share | random prefix | within-edge share | within density | density lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in arms.items():
        for depth in DEPTHS:
            row = arm["clusters"][f"L{depth}"]
            lines.append(
                f"| {name} | L{depth} | {row['clusters']} | {row['max_cluster_share']:.3f} | "
                f"{row['random_prefix']:.4f} | {row['within_edge_share_weighted']:.4f} | "
                f"{row['within_edge_density']:.5f} | {row['within_density_lift']:.2f} |"
            )
    lines += [
        "",
        "## Communities: within vs cross prefix rates",
        "",
        "| arm | within prefix@1 | cross prefix@1 | gap | within lift@1 | cross lift@1 | gap lift | within prefix@2 | cross prefix@2 | L1 community purity | normalized purity | L1 NMI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in arms.items():
        row = arm["community"]
        lines.append(
            f"| {name} | {row['within']['prefix1']:.4f} | {row['cross']['prefix1']:.4f} | "
            f"{row['gap_prefix1']:+.4f} | {row['within']['prefix1_lift']:.2f} | {row['cross']['prefix1_lift']:.2f} | "
            f"{row['gap_lift1']:+.2f} | {row['within']['prefix2']:.4f} | {row['cross']['prefix2']:.4f} | "
            f"{row['L1']['community']['purity']:.4f} | {row['L1']['community']['normalized_purity']:.4f} | {row['L1']['nmi']:.4f} |"
        )
    lines += [
        "",
        "## ORCA role alignment",
        "",
        "| arm | role purity L1 | normalized role purity L1 | role purity L2 | role NMI L1 | role ARI L1 | multi-role edge prefix@1 | single-role edge prefix@1 | multi-role edge prefix@2 | single-role edge prefix@2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in arms.items():
        row = arm["roles"]
        lines.append(
            f"| {name} | {row['role_L1']['purity']:.4f} | {row['role_L1']['normalized_purity']:.4f} | "
            f"{row['role_L2']['purity']:.4f} | {row['top_role_nmi_L1']:.4f} | {row['top_role_ari_L1']:.4f} | "
            f"{row['multi_role_edges']['prefix1']:.4f} | {row['single_role_edges']['prefix1']:.4f} | "
            f"{row['multi_role_edges']['prefix2']:.4f} | {row['single_role_edges']['prefix2']:.4f} |"
        )
    lines += [
        "",
        "## Graphlet instances",
        "",
        "Values are the fraction of induced behaviour edges whose endpoints share the prefix; "
        "`all@k` is the fraction of instances where all five nodes share the prefix.",
        "",
        "| arm | pattern | instances | induced edges | prefix@1 | lift@1 | prefix@2 | lift@2 | prefix@3 | lift@3 | all@1 | all@2 | all@3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in arms.items():
        for pattern, row in arm["graphlets"].items():
            lines.append(
                f"| {name} | {pattern} | {row['instances']} | {row['induced_edges']} | "
                f"{row['prefix1']:.4f} | {row['prefix1_lift']:.2f} | {row['prefix2']:.4f} | {row['prefix2_lift']:.2f} | "
                f"{row['prefix3']:.4f} | {row['prefix3_lift']:.2f} | "
                f"{row['all_prefix1']:.4f} | {row['all_prefix2']:.4f} | {row['all_prefix3']:.4f} |"
            )
    lines += [
        "",
        "## Top ORCA role pairs by behaviour-edge count",
        "",
        "| arm | left role | right role | edges | prefix@1 | prefix@2 | prefix@3 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name, arm in arms.items():
        for row in arm["roles"]["top_role_pairs"]:
            lines.append(
                f"| {name} | {row['left']} | {row['right']} | {row['count']} | "
                f"{row['prefix1']:.4f} | {row['prefix2']:.4f} | {row['prefix3']:.4f} |"
            )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
