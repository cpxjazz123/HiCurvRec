"""Compare cluster-level ORCA relations before and after quantisation.

Before: Louvain communities of the TRAIN behaviour graph.
After:  actual L1/L2/L3 prefix clusters of an exported SID table.

The comparison is cluster-to-cluster, not item-to-item:
* role profile of a cluster (9 ORCA orbit groups);
* boundary role profile (roles of unique external behaviour neighbours);
* for every ORCA role pair, the fraction of behaviour edges whose endpoints share a cluster;
* within/cross role-pair distributions;
* multi-role composition of matched clusters.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/cluster_orca_relations.py
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore", category=UserWarning)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sid_structural_fidelity import SIDS, load_behavior_graph, load_roles, load_sid_tokens, prefix_labels  # noqa: E402

OUT_JSON = HERE / "cluster_orca_relations.json"
OUT_MD = HERE / "cluster_orca_relations.md"
DEPTHS = (1, 2, 3)
MIN_PAIR_COUNT = 100


def jsd(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.sum() <= 0 or right.sum() <= 0:
        return float("nan")
    left = left / left.sum()
    right = right / right.sum()
    value = float(jensenshannon(left, right, base=2.0))
    return value * value


def contingency(community: np.ndarray, labels: np.ndarray) -> np.ndarray:
    table = np.zeros((community.max() + 1, labels.max() + 1), dtype=np.int64)
    np.add.at(table, (community, labels), 1)
    return table


def cluster_profiles(labels: np.ndarray, values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((k, values.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, values)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    return sums / np.maximum(counts[:, None], 1.0), counts


def cluster_scalar(labels: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    sums = np.bincount(labels, weights=values.astype(np.float64), minlength=k)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    return sums / np.maximum(counts, 1.0)


def boundary_profiles(labels: np.ndarray, adjacency: list[set[int]], role_dist: np.ndarray, k: int) -> np.ndarray:
    members: list[list[int]] = [[] for _ in range(k)]
    for item, label in enumerate(labels):
        members[int(label)].append(item)
    out = np.full((k, role_dist.shape[1]), np.nan, dtype=np.float64)
    for label, nodes in enumerate(members):
        if not nodes:
            continue
        inside = set(nodes)
        external: set[int] = set()
        for node in nodes:
            external.update(adjacency[node])
        external.difference_update(inside)
        if external:
            out[label] = role_dist[list(external)].mean(axis=0)
    return out


def matched_metrics(community: np.ndarray, sid_labels: np.ndarray, community_profile: np.ndarray,
                    sid_profile: np.ndarray, community_boundary: np.ndarray | None,
                    sid_boundary: np.ndarray | None, multi: np.ndarray) -> dict:
    table = contingency(community, sid_labels)
    sizes = table.sum(axis=1)
    best_sid = table.argmax(axis=1)
    overlap = table.max(axis=1)
    coverage = overlap / sizes
    role = np.array([jsd(community_profile[c], sid_profile[s]) for c, s in enumerate(best_sid)])
    boundary = np.array([
        jsd(community_boundary[c], sid_boundary[s])
        if community_boundary is not None and sid_boundary is not None
        and np.isfinite(community_boundary[c]).all() and np.isfinite(sid_boundary[s]).all()
        else np.nan
        for c, s in enumerate(best_sid)
    ])
    role_valid = np.isfinite(role)
    boundary_valid = np.isfinite(boundary)
    multi_community = cluster_scalar(community, multi, community.max() + 1)
    multi_sid = cluster_scalar(sid_labels, multi, sid_labels.max() + 1)
    multi_delta = np.abs(multi_community - multi_sid[best_sid])
    return {
        "best_sid": best_sid,
        "overlap": overlap,
        "coverage_weighted": float(np.average(coverage, weights=sizes)),
        "coverage_median": float(np.median(coverage)),
        "role_jsd_weighted": float(np.average(role[role_valid], weights=sizes[role_valid])) if role_valid.any() else float("nan"),
        "role_jsd_valid_communities": int(role_valid.sum()),
        "role_jsd_skipped_items": int(sizes[~role_valid].sum()),
        "boundary_jsd_weighted": float(np.average(boundary[boundary_valid], weights=sizes[boundary_valid])) if boundary_valid.any() else float("nan"),
        "boundary_jsd_valid_communities": int(boundary_valid.sum()),
        "multi_role_mae_weighted": float(np.average(multi_delta, weights=sizes)),
        "role_jsd": role,
        "boundary_jsd": boundary,
        "multi_role_mae": multi_delta,
    }


def reverse_metrics(community: np.ndarray, sid_labels: np.ndarray, community_profile: np.ndarray,
                    sid_profile: np.ndarray) -> dict:
    table = contingency(community, sid_labels)
    sizes = table.sum(axis=0)
    best_community = table.argmax(axis=0)
    overlap = table.max(axis=0)
    coverage = overlap / np.maximum(sizes, 1)
    role = np.array([
        jsd(sid_profile[s], community_profile[c]) for s, c in enumerate(best_community)
    ])
    role_valid = np.isfinite(role)
    return {
        "best_community": best_community,
        "overlap": overlap,
        "coverage_weighted": float(np.average(coverage, weights=sizes)),
        "coverage_median": float(np.median(coverage)),
        "role_jsd_weighted": float(np.average(role[role_valid], weights=sizes[role_valid])) if role_valid.any() else float("nan"),
        "role_jsd_valid_clusters": int(role_valid.sum()),
        "role_jsd_skipped_items": int(sizes[~role_valid].sum()),
        "role_jsd": role,
    }


def pair_stats(top: np.ndarray, labels: np.ndarray, edges: np.ndarray, classes: int = 9) -> dict:
    left = top[edges[:, 0]]
    right = top[edges[:, 1]]
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    pair = low * classes + high
    total = np.bincount(pair, minlength=classes ** 2)
    same = labels[edges[:, 0]] == labels[edges[:, 1]]
    within = np.bincount(pair[same], minlength=classes ** 2)
    cross = total - within
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = within / total
        global_same = float(within.sum() / total.sum()) if total.sum() else float("nan")
        lift = fraction / global_same if global_same > 0 else np.full_like(fraction, np.nan)
    matrix = fraction.reshape(classes, classes)
    # Symmetrise the matrix for display; the lower triangle is the same count.
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    return {
        "total": total,
        "within": within,
        "cross": cross,
        "fraction": fraction,
        "lift": lift,
        "global_same": global_same,
        "matrix": matrix,
        "within_dist": within / max(within.sum(), 1),
        "cross_dist": cross / max(cross.sum(), 1),
        "n_edges": int(len(edges)),
    }


def pair_comparison(before: dict, after: dict) -> dict:
    counts = before["total"]
    mask = counts > 0
    strong = (counts >= MIN_PAIR_COUNT) & mask
    weighted_mae = float(
        np.sum(counts[mask] * np.abs(after["fraction"][mask] - before["fraction"][mask])) / np.sum(counts[mask])
    )
    lift_valid = mask & np.isfinite(before["lift"]) & np.isfinite(after["lift"])
    lift_mae = float(
        np.sum(counts[lift_valid] * np.abs(after["lift"][lift_valid] - before["lift"][lift_valid]))
        / np.sum(counts[lift_valid])
    ) if lift_valid.any() else float("nan")
    strong_lift = strong & lift_valid
    if strong_lift.sum() > 2:
        rho = float(spearmanr(before["lift"][strong_lift], after["lift"][strong_lift]).statistic)
    else:
        rho = float("nan")
    return {
        "weighted_mae": weighted_mae,
        "lift_mae": lift_mae,
        "spearman_pairs": rho,
        "pairs_used": int(strong_lift.sum()),
        "zero_count_pairs": int((~mask).sum()),
        "before_global_same": float(before["global_same"]),
        "after_global_same": float(after["global_same"]),
        "within_role_pair_jsd": jsd(before["within_dist"], after["within_dist"]),
        "cross_role_pair_jsd": jsd(before["cross_dist"], after["cross_dist"]),
        "within_edges": int(before["within"].sum()),
        "after_within_edges": int(after["within"].sum()),
    }


def top_pair_deltas(before: dict, after: dict, names: list[str], limit: int = 10) -> list[dict]:
    counts = before["total"]
    rows = []
    for value in np.argsort(-np.abs(after["lift"] - before["lift"]) * counts):
        count = int(counts[value])
        if count == 0 or len(rows) >= limit:
            break
        left, right = divmod(int(value), len(names))
        rows.append({
            "left": names[left],
            "right": names[right],
            "count": count,
            "before_within": float(before["fraction"][value]),
            "after_within": float(after["fraction"][value]),
            "before_lift": float(before["lift"][value]),
            "after_lift": float(after["lift"][value]),
            "delta_fraction": float(after["fraction"][value] - before["fraction"][value]),
            "delta_lift": float(after["lift"][value] - before["lift"][value]),
        })
    return rows


def arm_partition(tokens: np.ndarray, comm: np.ndarray, edges: np.ndarray, role_dist: np.ndarray,
                  top: np.ndarray, multi: np.ndarray, names: list[str], adjacency: list[set[int]],
                  active: np.ndarray) -> dict:
    k_comm = comm.max() + 1
    comm_profile, comm_sizes = cluster_profiles(comm, role_dist, k_comm)
    comm_boundary = boundary_profiles(comm, adjacency, role_dist, k_comm)
    before_pairs = pair_stats(top, comm, edges)
    out = {}
    for depth in DEPTHS:
        labels = prefix_labels(tokens, depth)
        k = labels.max() + 1
        sid_profile, sid_sizes = cluster_profiles(labels, role_dist, k)
        sid_boundary = boundary_profiles(labels, adjacency, role_dist, k)
        after_pairs = pair_stats(top, labels, edges)
        forward = matched_metrics(comm, labels, comm_profile, sid_profile, comm_boundary, sid_boundary, multi)
        reverse = reverse_metrics(comm, labels, comm_profile, sid_profile)
        out[f"L{depth}"] = {
            "sid_clusters": int(k),
            "ari": float(adjusted_rand_score(comm[active], labels[active])),
            "nmi": float(normalized_mutual_info_score(comm[active], labels[active])),
            "forward": {key: value for key, value in forward.items() if not isinstance(value, np.ndarray)},
            "reverse": {key: value for key, value in reverse.items() if not isinstance(value, np.ndarray)},
            "pairs": pair_comparison(before_pairs, after_pairs),
            "top_pair_deltas": top_pair_deltas(before_pairs, after_pairs, names),
            "after_pairs": {key: value for key, value in after_pairs.items() if key not in ("total", "within", "cross", "fraction", "lift", "matrix", "within_dist", "cross_dist")},
        }
    return {
        "before": {
            "communities": int(k_comm),
            "sizes": comm_sizes.tolist(),
            "pairs": pair_comparison(before_pairs, before_pairs),
        },
        "depths": out,
    }


def main() -> int:
    nodes, edges, adjacency = load_behavior_graph()
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes))
    graph.add_edges_from(map(tuple, edges.tolist()))
    communities = nx.community.louvain_communities(graph, seed=42, resolution=1.0)
    comm = np.full(nodes, -1, dtype=np.int64)
    for index, members in enumerate(communities):
        comm[list(members)] = index
    active = np.bincount(edges.ravel(), minlength=nodes) > 0
    role_dist, top, multi, role_names = load_roles()
    tokens = {name: load_sid_tokens(nodes, path) for name, path in SIDS.items()}
    result = {
        "behavior_graph": {
            "nodes": int(nodes),
            "edges": int(len(edges)),
            "communities": int(len(communities)),
            "active_communities": int(len(np.unique(comm[active]))),
            "active_nodes": int(active.sum()),
            "isolated_nodes": int((~active).sum()),
            "community_sizes": sorted([len(x) for x in communities], reverse=True),
            "multi_role_fraction": float(multi.mean()),
            "role_groups": role_names,
        },
        "arms": {
            name: arm_partition(tokens[name], comm, edges, role_dist, top, multi, role_names, adjacency, active)
            for name in SIDS
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# Cluster-level ORCA relations before vs after quantisation",
        "",
        "Before: Louvain communities of the TRAIN behaviour graph. "
        "After: actual L1/L2/L3 SID prefix clusters. No KMeans.",
        "",
        f"Behaviour graph: **{nodes}** nodes, **{len(edges)}** edges, **{len(communities)}** Louvain labels: "
        f"**{len(np.unique(comm[active]))}** active communities plus **{int((~active).sum())}** isolated nodes. "
        f"Role JSD / boundary JSD are computed on the active communities only. "
        f"Multi-role items: **{multi.mean():.3%}**.",
        "",
        "## Partition agreement and matched-cluster role profiles",
        "",
        "`coverage` is the largest fraction of a behaviour community captured by one SID cluster. "
        "`role JSD` compares mean ORCA role profiles of matched clusters; `boundary JSD` compares "
        "the role profiles of their unique external behaviour neighbours.",
        "",
        "| arm | depth | SID clusters | ARI | NMI | forward coverage | forward role JSD | role valid | reverse coverage | reverse role JSD | boundary JSD | boundary valid | multi-role MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in result["arms"].items():
        for depth in DEPTHS:
            row = arm["depths"][f"L{depth}"]
            forward = row["forward"]
            reverse = row["reverse"]
            lines.append(
                f"| {name} | L{depth} | {row['sid_clusters']} | {row['ari']:.4f} | {row['nmi']:.4f} | "
                f"{forward['coverage_weighted']:.4f} | {forward['role_jsd_weighted']:.4f} | "
                f"{forward['role_jsd_valid_communities']} | {reverse['coverage_weighted']:.4f} | "
                f"{reverse['role_jsd_weighted']:.4f} | {forward['boundary_jsd_weighted']:.4f} | "
                f"{forward['boundary_jsd_valid_communities']} | {forward['multi_role_mae_weighted']:.4f} |"
            )
    lines += [
        "",
        "## Per-role-pair within-cluster fraction",
        "",
        "For each unordered ORCA role pair, the fraction of its behaviour edges whose endpoints share a cluster. "
        "`pair MAE` is weighted by the number of edges of that role pair. `pair Spearman` is computed on "
        "role pairs with at least 100 edges; there are only three such pairs in these data, so it is not "
        "informative and is reported only for completeness.",
        "",
        "`pair MAE` is the raw within-fraction difference; `lift MAE` uses the same-fraction lift "
        "`within_fraction / global_same_cluster_fraction`, which controls the overall cluster granularity. "
        "`pairs_used` is small when only a few ORCA role pairs have enough edges.",
        "",
        "| arm | depth | global same before | global same after | pair MAE | lift MAE | pair Spearman | pairs used | within role-pair JSD | cross role-pair JSD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in result["arms"].items():
        for depth in DEPTHS:
            row = arm["depths"][f"L{depth}"]
            pair = row["pairs"]
            lines.append(
                f"| {name} | L{depth} | {pair['before_global_same']:.4f} | {pair['after_global_same']:.4f} | "
                f"{pair['weighted_mae']:.4f} | {pair['lift_mae']:.4f} | {pair['spearman_pairs']:.3f} | "
                f"{pair['pairs_used']} | {pair['within_role_pair_jsd']:.4f} | {pair['cross_role_pair_jsd']:.4f} |"
            )
    lines += [
        "",
        "## Top role-pair changes at L1",
        "",
        "| arm | left role | right role | edges | before within | after within | before lift | after lift | delta lift |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in result["arms"].items():
        for row in arm["depths"]["L1"]["top_pair_deltas"]:
            lines.append(
                f"| {name} | {row['left']} | {row['right']} | {row['count']} | "
                f"{row['before_within']:.4f} | {row['after_within']:.4f} | "
                f"{row['before_lift']:.3f} | {row['after_lift']:.3f} | {row['delta_lift']:+.3f} |"
            )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
