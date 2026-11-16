"""Graphlet-edge recovery vs Stage3 recall across standard-VQ TIGER arms.

For each of the 48 standard 3x(256,32) VQ arms:
1. rebuild q from codebooks + exported SID tokens;
2. build a global k=10 q-kNN graph;
3. compute recovery of real behaviour edges belonging to:
   - exact triangle and K4 edge sets;
   - strict sampled K1,4, P5, T5, C5 graphlet instances;
   - inside-community and cross-community edges;
4. correlate each recovery rate with test_recall@10, and test incremental predictive
   value over overall recovery, q-kNN edge count, L3 collision and degree-stratified
   expected recovery.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/qknn_graphlet_recall.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import f as f_dist

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "tiger_cluster_graph"))
import analysis  # noqa: E402
from qknn_community_correlation import checkpoint_vq, qknn_edges  # noqa: E402
from sid_structural_fidelity import load_behavior_graph  # noqa: E402

R2 = HERE / "r2"
GRAPHLETS = REPO / "results" / "research" / "tiger_cluster_graph" / "graphlet_instances.npz"
OUT_JSON = HERE / "results" / "qknn_graphlet_recall.json"
OUT_MD = HERE / "results" / "qknn_graphlet_recall.md"
K = 10
PERMUTATIONS = 5000
N_ITEMS = 24587
STRUCTURES = ("K1_4", "P5", "T5", "C5", "triangle", "K4", "inside", "cross")


def edge_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def sampled_edges(name: str) -> set[tuple[int, int]]:
    archive = np.load(GRAPHLETS)
    if name not in archive.files:
        return set()
    edges: set[tuple[int, int]] = set()
    for instance in archive[name]:
        for left in range(5):
            for right in range(left + 1, 5):
                edges.add(edge_key(int(instance[left]), int(instance[right])))
    return edges


def exact_frontier(behavior_edges: np.ndarray, adjacency: list[set[int]]) -> tuple[set, set, set]:
    triangle: set[tuple[int, int]] = set()
    k4: set[tuple[int, int]] = set()
    for left, right in behavior_edges.tolist():
        shared = adjacency[int(left)] & adjacency[int(right)]
        if shared:
            key = edge_key(int(left), int(right))
            triangle.add(key)
            if len(shared) >= 2:
                values = list(shared)
                for first in range(len(values)):
                    for second in range(first + 1, len(values)):
                        if values[second] in adjacency[values[first]]:
                            k4.add(key)
                            break
                    if key in k4:
                        break
    return triangle, k4


def edge_indices(behavior_edges: np.ndarray) -> dict[tuple[int, int], int]:
    return {edge_key(int(left), int(right)): index for index, (left, right) in enumerate(behavior_edges.tolist())}


def ols_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    total = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum(residual ** 2) / total) if total > 0 else float("nan"), residual


def rank_residual(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    ranked = np.column_stack([analysis.average_ranks(values[:, i]) for i in range(values.shape[1])])
    design = np.column_stack([np.ones(len(ranked)), controls])
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def correlate(values: np.ndarray, recall: np.ndarray, controls: np.ndarray) -> dict:
    usable = np.isfinite(values) & np.isfinite(recall) & np.isfinite(controls).all(axis=1)
    first = values[usable]
    second = recall[usable]
    control = controls[usable]
    if len(first) < 5:
        return {"n": int(len(first)), "spearman": None}
    rho = analysis.spearman(first, second)
    pearson = analysis.pearson(first, second)
    # Partial on all controls through two residualisations.
    ranked_x = rank_residual(np.column_stack([first]), control)[:, 0]
    ranked_y = rank_residual(np.column_stack([second]), control)[:, 0]
    partial_spearman = analysis.pearson(ranked_x, ranked_y)
    partial_pearson = analysis.partial(first, second, control[:, 0], rank=False) if control.shape[1] == 1 else float("nan")
    if control.shape[1] > 1:
        design = np.column_stack([np.ones(len(first)), control])
        res_x = first - design @ np.linalg.lstsq(design, first, rcond=None)[0]
        res_y = second - design @ np.linalg.lstsq(design, second, rcond=None)[0]
        partial_pearson = analysis.pearson(res_x, res_y)
    rng = np.random.default_rng(42)
    beats_s = beats_r = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(second)
        beats_s += abs(analysis.spearman(first, shuffled)) >= abs(rho)
        beats_r += abs(analysis.pearson(first, shuffled)) >= abs(pearson)
    return {
        "n": int(len(first)),
        "mean": float(np.mean(first)),
        "spearman": rho,
        "pearson": pearson,
        "p_spearman": (beats_s + 1) / (PERMUTATIONS + 1),
        "p_pearson": (beats_r + 1) / (PERMUTATIONS + 1),
        "partial_spearman": partial_spearman,
        "partial_pearson": partial_pearson,
    }


def main() -> int:
    nodes, behavior_edges, adjacency = load_behavior_graph()
    edge_to_index = edge_indices(behavior_edges)
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes))
    graph.add_edges_from(map(tuple, behavior_edges.tolist()))
    communities = nx.community.louvain_communities(graph, seed=42, resolution=1.0)
    comm = np.full(nodes, -1, dtype=np.int64)
    for index, members in enumerate(communities):
        comm[list(members)] = index
    within = comm[behavior_edges[:, 0]] == comm[behavior_edges[:, 1]]

    triangle, k4 = exact_frontier(behavior_edges, adjacency)
    sampled = {
        "K1_4": sampled_edges("K1_4"),
        "P5": sampled_edges("P5_centre") | sampled_edges("P5_hops"),
        "T5": sampled_edges("T5"),
        "C5": sampled_edges("C5"),
    }
    structures: dict[str, set[tuple[int, int]]] = {
        "K1_4": sampled["K1_4"],
        "P5": sampled["P5"],
        "T5": sampled["T5"],
        "C5": sampled["C5"],
        "triangle": triangle,
        "K4": k4,
        "inside": {edge_key(int(left), int(right)) for left, right in behavior_edges[within].tolist()},
        "cross": {edge_key(int(left), int(right)) for left, right in behavior_edges[~within].tolist()},
    }
    structure_indices = {
        name: np.array([edge_to_index[edge] for edge in edges if edge in edge_to_index], dtype=np.int64)
        for name, edges in structures.items()
    }
    degree = np.array([len(neighbours) for neighbours in adjacency], dtype=np.int64)
    dmax = np.maximum(degree[behavior_edges[:, 0]], degree[behavior_edges[:, 1]])
    degree_bin = np.asarray(pd.qcut(dmax, 5, labels=False, duplicates="drop"), dtype=np.int64)
    structure_bins = {name: degree_bin[indices] for name, indices in structure_indices.items()}
    structure_info = {
        name: {
            "edges": int(len(indices)),
            "definition": "exact" if name in ("triangle", "K4", "inside", "cross") else "strict sampled instances",
            "sample_instances": 300 if name in ("K1_4", "P5", "T5", "C5") else None,
        }
        for name, indices in structure_indices.items()
    }

    rows = []
    skipped = []
    for directory in sorted((path for path in R2.iterdir() if path.is_dir()), key=lambda path: path.name):
        checkpoint = directory / "rqvae_best.pth"
        sid_path = directory / "item_sids.json"
        recall_path = directory / "recall.json"
        if not (checkpoint.is_file() and sid_path.is_file() and recall_path.is_file()):
            continue
        packed = checkpoint_vq(checkpoint)
        if packed is None:
            skipped.append({"number": directory.name, "reason": "not standard 3x(256,32) vq_layers"})
            continue
        codebooks, payload = packed
        tokens = analysis.tokens_of(sid_path, nodes)
        if (tokens < 0).any() or (tokens > 255).any():
            skipped.append({"number": directory.name, "reason": "invalid SID tokens"})
            continue
        q = sum(codebooks[level][tokens[:, level]] for level in range(3)).astype(np.float32)
        edges = qknn_edges(q)
        edge_set = {(int(left), int(right)) for left, right in edges.tolist()}
        recovered = np.array([(int(left), int(right)) in edge_set for left, right in behavior_edges], dtype=bool)
        recall = json.loads(recall_path.read_text()).get("test_recall@10")
        if recall is None:
            skipped.append({"number": directory.name, "reason": "no test_recall@10"})
            continue
        # Arm-specific degree-bin recovery; used to compute a degree-adjusted excess.
        bin_rate = np.array([
            recovered[degree_bin == value].mean() if (degree_bin == value).any() else np.nan
            for value in range(5)
        ])
        row = {
            "number": directory.name,
            "variant": payload.get("arm_variant"),
            "recall": float(recall),
            "q_edges": int(len(edges)),
            "collision": analysis.collision_of(tokens),
            "overall_recovery": float(recovered.mean()),
        }
        for name, indices in structure_indices.items():
            observed = float(recovered[indices].mean()) if len(indices) else float("nan")
            expected = float(np.nanmean(bin_rate[structure_bins[name]])) if len(indices) else float("nan")
            row[f"{name}_rate"] = observed
            row[f"{name}_excess"] = observed - expected
        rows.append(row)
        print(f"{directory.name} recall {recall:.5f} q_edges {len(edges)} "
              f"overall {row['overall_recovery']:.5f} P5 {row['P5_rate']:.5f} K4 {row['K4_rate']:.5f}",
              flush=True)

    recall = np.array([row["recall"] for row in rows])
    q_edges = np.array([row["q_edges"] for row in rows], dtype=float)
    collision = np.array([row["collision"] for row in rows], dtype=float)
    overall = np.array([row["overall_recovery"] for row in rows], dtype=float)
    inside = np.array([row["inside_rate"] for row in rows], dtype=float)
    cross = np.array([row["cross_rate"] for row in rows], dtype=float)
    base_controls = np.column_stack([overall, q_edges, collision])
    # Graphlet structures must additionally beat the arm's inside/cross recovery.
    community_controls = np.column_stack([overall, q_edges, collision, inside, cross])

    def controls_for(name: str) -> np.ndarray:
        return base_controls if name in ("inside", "cross") else community_controls

    direct = {}
    degree_adjusted = {}
    incremental_rate = {}
    incremental_excess = {}
    for name in STRUCTURES:
        values = np.array([row[f"{name}_rate"] for row in rows])
        excess = np.array([row[f"{name}_excess"] for row in rows])
        control = controls_for(name)
        direct[name] = correlate(values, recall, control)
        degree_adjusted[name] = correlate(excess, recall, control)
        for label, candidate, store in (("rate", values, incremental_rate), ("excess", excess, incremental_excess)):
            usable = np.isfinite(candidate) & np.isfinite(recall) & np.isfinite(control).all(axis=1)
            base_r2, _ = ols_r2(control[usable], recall[usable])
            with_r2, _ = ols_r2(np.column_stack([control[usable], candidate[usable]]), recall[usable])
            n = int(usable.sum())
            delta = with_r2 - base_r2
            p_value = (
                float(f_dist.sf((delta / 1.0) / ((1.0 - with_r2) / (n - control.shape[1] - 2)), 1, n - control.shape[1] - 2))
                if n > control.shape[1] + 2 and delta > 0 else float("nan")
            )
            store[name] = {
                "n": n,
                "base_r2": float(base_r2),
                "with_candidate_r2": float(with_r2),
                "delta_r2": float(delta),
                "p_delta": p_value,
            }

    result = {
        "nodes": int(nodes),
        "behavior_edges": int(len(behavior_edges)),
        "inside_edges": int(within.sum()),
        "cross_edges": int((~within).sum()),
        "k": K,
        "arms": len(rows),
        "structures": structure_info,
        "direct": direct,
        "degree_adjusted": degree_adjusted,
        "incremental_r2_rate": incremental_rate,
        "incremental_r2_excess": incremental_excess,
        "rows": rows,
        "skipped": skipped,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# Graphlet-edge recovery vs Stage3 recall (standard VQ arms)",
        "",
        "48 standard 3x(256,32) VQ arms; q rebuilt from codebooks + exported SID tokens; global k=10 "
        "q-kNN graph. Triangle/K4/inside/cross are exact edge sets. K1,4 / P5 / T5 / C5 use the "
        "existing **strict sampled 5-node instances** (300 each); edges are deduplicated.",
        "",
        f"Behaviour graph: **{nodes}** nodes, **{len(behavior_edges)}** edges; "
        f"inside **{int(within.sum())}**, cross **{int((~within).sum())}**.",
        "",
        "## Recovery rates and correlations",
        "",
        "`partial` controls overall recovery, q-kNN edge count and L3 collision. "
        "`excess` subtracts the arm's degree-bin expected recovery for the same edge set; "
        "the degree-adjusted correlation uses that excess.",
        "",
        "| structure | edges | mean rate | Spearman | p | Pearson | p | partial Spearman | partial Pearson | degree-excess rho | ΔR² rate | p(rate) | ΔR² excess | p(excess) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in STRUCTURES:
        info = structure_info[name]
        d = direct[name]
        adj = degree_adjusted[name]
        inc_rate = incremental_rate[name]
        inc_excess = incremental_excess[name]
        lines.append(
            f"| {name} | {info['edges']} | {d['mean']:.5f} | {d['spearman']:+.4f} | {d['p_spearman']:.4f} | "
            f"{d['pearson']:+.4f} | {d['p_pearson']:.4f} | {d['partial_spearman']:+.4f} | "
            f"{d['partial_pearson']:+.4f} | {adj['spearman']:+.4f} | {inc_rate['delta_r2']:+.4f} | "
            f"{inc_rate['p_delta']:.4f} | {inc_excess['delta_r2']:+.4f} | {inc_excess['p_delta']:.4f} |"
        )
    lines += [
        "",
        "## Per-arm rates",
        "",
        "| arm | recall | q edges | overall | K1_4 | P5 | T5 | C5 | triangle | K4 | inside | cross |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['number']} | {row['recall']:.6f} | {row['q_edges']} | {row['overall_recovery']:.5f} | "
            + " | ".join(f"{row[f'{name}_rate']:.5f}" for name in STRUCTURES)
            + " |"
        )
    lines += [
        "",
        "`excess` values, degree-adjusted correlations and incremental R² are in "
        "`qknn_graphlet_recall.json`.",
        "",
        "Skipped archives: " + ", ".join(item["number"] for item in skipped) + ".",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
