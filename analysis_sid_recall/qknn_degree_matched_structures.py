"""Degree-matched AF3/local-topology edge classes, no Louvain.

The behaviour graph is fixed.  Edges are classified by AF3 within dmax quintile and local
topology (common neighbours, degree asymmetry, approximate edge betweenness), with **no
community/louvain information**.  Every class is compared with a degree-matched control set:
for each edge, take the cell (dmax quintile, dmin quintile), estimate recovery from non-class
edges in the same cell, and compute excess = class recovery - matched recovery.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/qknn_degree_matched_structures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import analysis  # noqa: E402
from qknn_community_correlation import checkpoint_vq, qknn_edges  # noqa: E402
from qknn_graphlet_cv import cv_scores  # noqa: E402
from sid_structural_fidelity import load_behavior_graph  # noqa: E402

R2 = HERE / "r2"
OUT_JSON = HERE / "results" / "qknn_degree_matched_structures.json"
OUT_MD = HERE / "results" / "qknn_degree_matched_structures.md"
K = 10
N_ITEMS = 24587
CLASSES = ("dense", "bridge", "star", "chain")


def edge_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def af3_rank_within_degree(af3: np.ndarray, degree_bin: np.ndarray) -> np.ndarray:
    output = np.zeros(len(af3), dtype=float)
    for value in np.unique(degree_bin):
        mask = degree_bin == value
        order = np.argsort(af3[mask], kind="mergesort")
        sorted_values = af3[mask][order]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_values) != 0) + 1])
        ends = np.concatenate([starts[1:], [len(sorted_values)]]) - 1
        ranks = np.empty(mask.sum(), dtype=float)
        for start, end in zip(starts, ends):
            ranks[order[start:end + 1]] = (start + end) / 2.0
        output[mask] = ranks / max(len(ranks) - 1, 1)
    return output


def direct(values: np.ndarray, recall: np.ndarray, controls: np.ndarray) -> dict:
    usable = np.isfinite(values) & np.isfinite(recall) & np.isfinite(controls).all(axis=1)
    first, second, control = values[usable], recall[usable], controls[usable]
    if len(first) < 5:
        return {"n": int(len(first)), "spearman": None}
    rho = analysis.spearman(first, second)
    pear = analysis.pearson(first, second)
    design = np.column_stack([np.ones(len(first)), control])
    res_x = first - design @ np.linalg.lstsq(design, first, rcond=None)[0]
    res_y = second - design @ np.linalg.lstsq(design, second, rcond=None)[0]
    ranked_first = analysis.average_ranks(first)
    ranked_second = analysis.average_ranks(second)
    ranked_control = np.column_stack([analysis.average_ranks(control[:, i]) for i in range(control.shape[1])])
    ranked_design = np.column_stack([np.ones(len(first)), ranked_control])
    ranked_res_x = ranked_first - ranked_design @ np.linalg.lstsq(ranked_design, ranked_first, rcond=None)[0]
    ranked_res_y = ranked_second - ranked_design @ np.linalg.lstsq(ranked_design, ranked_second, rcond=None)[0]
    rng = np.random.default_rng(42)
    beats_s = beats_r = 0
    for _ in range(5000):
        shuffled = rng.permutation(second)
        beats_s += abs(analysis.spearman(first, shuffled)) >= abs(rho)
        beats_r += abs(analysis.pearson(first, shuffled)) >= abs(pear)
    return {
        "n": int(len(first)),
        "mean": float(first.mean()),
        "spearman": rho,
        "pearson": pear,
        "p_spearman": (beats_s + 1) / 5001,
        "p_pearson": (beats_r + 1) / 5001,
        "partial_spearman": analysis.pearson(ranked_res_x, ranked_res_y),
        "partial_pearson": analysis.pearson(res_x, res_y),
    }


def incremental(values: np.ndarray, recall: np.ndarray, controls: np.ndarray) -> dict:
    usable = np.isfinite(values) & np.isfinite(recall) & np.isfinite(controls).all(axis=1)
    base_design = np.column_stack([np.ones(usable.sum()), controls[usable]])
    base_beta = np.linalg.lstsq(base_design, recall[usable], rcond=None)[0]
    base_resid = recall[usable] - base_design @ base_beta
    total = np.sum((recall[usable] - recall[usable].mean()) ** 2)
    base_r2 = 1.0 - np.sum(base_resid ** 2) / total
    full_design = np.column_stack([base_design, values[usable]])
    full_beta = np.linalg.lstsq(full_design, recall[usable], rcond=None)[0]
    full_resid = recall[usable] - full_design @ full_beta
    full_r2 = 1.0 - np.sum(full_resid ** 2) / total
    n = int(usable.sum())
    delta = full_r2 - base_r2
    p_value = (
        float(f_dist.sf((delta / 1.0) / ((1.0 - full_r2) / (n - controls.shape[1] - 2)),
                        1, n - controls.shape[1] - 2))
        if n > controls.shape[1] + 2 and delta > 0 else float("nan")
    )
    return {"n": n, "base_r2": float(base_r2), "with_r2": float(full_r2),
            "delta_r2": float(delta), "p_delta": p_value}


def main() -> int:
    nodes, behavior_edges, adjacency = load_behavior_graph()
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes))
    graph.add_edges_from(map(tuple, behavior_edges.tolist()))
    degree = np.array([len(neighbours) for neighbours in adjacency], dtype=np.int64)
    left, right = behavior_edges[:, 0], behavior_edges[:, 1]
    du, dv = degree[left], degree[right]
    dmax, dmin = np.maximum(du, dv), np.minimum(du, dv)
    asym = (dmax - dmin) / np.maximum(dmax, 1)
    common = np.array([len(adjacency[int(u)] & adjacency[int(v)]) for u, v in behavior_edges.tolist()], dtype=np.int64)
    af3 = 4 - du - dv + 3 * common
    dmax_bin = np.asarray(pd.qcut(dmax, 5, labels=False, duplicates="drop"), dtype=np.int64)
    dmin_bin = np.asarray(pd.qcut(dmin, 5, labels=False, duplicates="drop"), dtype=np.int64)
    cell = dmax_bin * 5 + dmin_bin
    af3_rank = af3_rank_within_degree(af3, dmax_bin)

    print("edge betweenness (approx k=100)...", flush=True)
    betweenness = nx.edge_betweenness_centrality(graph, k=100, seed=42)
    between = np.array([
        betweenness.get((int(u), int(v)), betweenness.get((int(v), int(u)), 0.0))
        for u, v in behavior_edges.tolist()
    ])
    q75_af3 = float(np.quantile(af3, 0.75))
    q25_af3 = float(np.quantile(af3, 0.25))
    q75_dmax = float(np.quantile(dmax, 0.75))
    q25_dmin = float(np.quantile(dmin, 0.25))
    q50_dmax = float(np.quantile(dmax, 0.50))
    q25_asym = float(np.quantile(asym, 0.25))
    q75_asym = float(np.quantile(asym, 0.75))
    q75_between = float(np.quantile(between, 0.75))

    classes = {
        "dense": (common >= 2) & (af3_rank >= 0.75),
        "bridge": (common == 0) & (between >= q75_between) & (af3_rank <= 0.30),
        "star": (common == 0) & (asym >= q75_asym) & (dmax >= q75_dmax) & (dmin <= q25_dmin) & (af3_rank <= 0.30),
        "chain": (common == 0) & (dmax <= q50_dmax) & (asym <= q25_asym) & (af3_rank <= 0.50),
    }
    class_info = {
        name: {
            "edges": int(mask.sum()),
            "share": float(mask.mean()),
            "af3_median": float(np.median(af3[mask])) if mask.any() else None,
            "dmax_median": float(np.median(dmax[mask])) if mask.any() else None,
            "dmin_median": float(np.median(dmin[mask])) if mask.any() else None,
            "asym_median": float(np.median(asym[mask])) if mask.any() else None,
            "common_median": float(np.median(common[mask])) if mask.any() else None,
            "between_median": float(np.median(between[mask])) if mask.any() else None,
        }
        for name, mask in classes.items()
    }
    print("classes", json.dumps(class_info, indent=2), flush=True)

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
            skipped.append({"number": directory.name, "reason": "invalid SID"})
            continue
        q = sum(codebooks[level][tokens[:, level]] for level in range(3)).astype(np.float32)
        edges = qknn_edges(q)
        edge_set = {(int(a), int(b)) for a, b in edges.tolist()}
        recovered = np.array([(int(u), int(v)) in edge_set for u, v in behavior_edges], dtype=bool)
        recall = json.loads(recall_path.read_text()).get("test_recall@10")
        if recall is None:
            skipped.append({"number": directory.name, "reason": "no recall"})
            continue
        row = {"number": directory.name, "recall": float(recall), "q_edges": int(len(edges)),
               "collision": analysis.collision_of(tokens), "overall": float(recovered.mean())}
        for name, mask in classes.items():
            observed = float(recovered[mask].mean())
            # Degree-matched control: non-class edges in the same (dmax,dmin) cell.
            cell_rate = np.full(25, np.nan)
            for value in np.unique(cell):
                control = (~mask) & (cell == value)
                if control.any():
                    cell_rate[value] = recovered[control].mean()
            expected = float(np.nanmean(cell_rate[cell[mask]]))
            row[f"{name}_rate"] = observed
            row[f"{name}_matched"] = expected
            row[f"{name}_excess"] = observed - expected
            row[f"{name}_lift"] = observed / expected if expected > 0 else float("nan")
        rows.append(row)
        print(f"{directory.name} recall {recall:.5f} q_edges {len(edges)} "
              f"dense {row['dense_rate']:.5f} bridge {row['bridge_rate']:.5f} "
              f"star {row['star_rate']:.5f} chain {row['chain_rate']:.5f}", flush=True)

    recall = np.array([row["recall"] for row in rows])
    q_edges = np.array([row["q_edges"] for row in rows], dtype=float)
    collision = np.array([row["collision"] for row in rows], dtype=float)
    overall = np.array([row["overall"] for row in rows], dtype=float)
    controls = np.column_stack([overall, q_edges, collision])

    results = {}
    for name in CLASSES:
        rate = np.array([row[f"{name}_rate"] for row in rows])
        excess = np.array([row[f"{name}_excess"] for row in rows])
        lift = np.array([row[f"{name}_lift"] for row in rows])
        results[name] = {
            "rate": direct(rate, recall, controls),
            "excess": direct(excess, recall, controls),
            "lift": direct(lift, recall, controls),
            "incremental_rate": incremental(rate, recall, controls),
            "incremental_excess": incremental(excess, recall, controls),
            "cv_base": cv_scores(controls, recall),
            "cv_rate": cv_scores(np.column_stack([controls, rate]), recall),
            "cv_excess": cv_scores(np.column_stack([controls, excess]), recall),
        }

    output = {
        "nodes": int(nodes),
        "behavior_edges": int(len(behavior_edges)),
        "k": K,
        "arms": len(rows),
        "af3_sign_counts": {"positive": int((af3 > 0).sum()), "zero": int((af3 == 0).sum()),
                             "negative": int((af3 < 0).sum())},
        "thresholds": {
            "af3_q75": q75_af3, "af3_q25": q25_af3,
            "dmax_q75": q75_dmax, "dmin_q25": q25_dmin, "dmax_q50": q50_dmax,
            "asym_q25": q25_asym, "asym_q75": q75_asym, "betweenness_q75": q75_between,
        },
        "class_info": class_info,
        "results": results,
        "rows": rows,
        "skipped": skipped,
    }
    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# Degree-matched AF3/local-topology edge classes (no Louvain)",
        "",
        "48 standard VQ arms; q rebuilt from codebooks + SID; global k=10 q-kNN. "
        "No community label is used. Classes use degree-normalised AF3 and local topology. "
        "Each class is compared with non-class edges in the same (dmax,dmin) quintile cell.",
        "",
        f"Behaviour edges **{len(behavior_edges)}**; AF3 sign: positive **{(af3 > 0).sum()}**, "
        f"zero **{(af3 == 0).sum()}**, negative **{(af3 < 0).sum()}**.",
        "",
        "## Classes",
        "",
        "| class | edges | share | AF3 median | dmax median | dmin median | asym median | common median | betweenness median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in class_info.items():
        lines.append(
            f"| {name} | {value['edges']} | {value['share']:.4f} | {value['af3_median']:.1f} | "
            f"{value['dmax_median']:.1f} | {value['dmin_median']:.1f} | {value['asym_median']:.3f} | "
            f"{value['common_median']:.1f} | {value['between_median']:.3e} |"
        )
    lines += [
        "",
        "## Recall relations",
        "",
        "Partial controls overall recovery, q-kNN edge count and L3 collision. "
        "`+rate`, `+excess`, `+lift` are CV R² differences over the base controls.",
        "",
        "| class | mean rate | direct rho | direct r | partial rho | partial r | ΔR² rate | p(rate) | CV +rate | CV +excess |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CLASSES:
        value = results[name]
        rate = value["rate"]
        inc = value["incremental_rate"]
        base = value["cv_base"]["r2_mean"]
        lines.append(
            f"| {name} | {rate['mean']:.5f} | {rate['spearman']:+.4f} | {rate['pearson']:+.4f} | "
            f"{rate['partial_spearman']:+.4f} | {rate['partial_pearson']:+.4f} | {inc['delta_r2']:+.4f} | "
            f"{inc['p_delta']:.4f} | {value['cv_rate']['r2_mean'] - base:+.4f} | "
            f"{value['cv_excess']['r2_mean'] - base:+.4f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
