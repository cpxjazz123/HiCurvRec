"""AF3/local-topology edge classes vs q-kNN recovery and Stage3 recall.

Fixed behaviour graph (24,587 nodes, 315,391 undirected TRAIN edges).  Every edge gets:
  AF3 = 4 - du - dv + 3*T, where T is the number of common neighbours;
  degree asymmetry, common neighbours, Louvain inside/cross, and approximate edge betweenness.

Classes are quantile/degree based, not hard absolute AF3 cuts:
  dense   : T >= 2 and AF3 percentile within its dmax quintile >= 75%;
  bridge  : T = 0, cross-community, edge-betweenness >= 75th percentile, low within-degree AF3;
  star    : T = 0, high degree asymmetry, high dmax, low dmin, low AF3;
  chain   : T = 0, low dmax, low degree asymmetry, low AF3.
The classes may overlap; overlap counts are reported.

For 48 standard 3x(256,32) VQ arms: rebuild q, build the global k=10 q-kNN graph, compute
class recovery, correlate with test_recall@10, and test incremental/CV prediction.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/qknn_af3_structures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import analysis  # noqa: E402
from qknn_community_correlation import checkpoint_vq, qknn_edges  # noqa: E402
from qknn_graphlet_cv import cv_scores  # noqa: E402
from sid_structural_fidelity import load_behavior_graph  # noqa: E402

R2 = HERE / "r2"
OUT_JSON = HERE / "results" / "qknn_af3_structures.json"
OUT_MD = HERE / "results" / "qknn_af3_structures.md"
K = 10
N_ITEMS = 24587
CLASSES = ("dense", "bridge", "star", "chain", "inside", "cross")


def edge_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def edge_to_index(edges: np.ndarray) -> dict[tuple[int, int], int]:
    return {edge_key(int(left), int(right)): index for index, (left, right) in enumerate(edges.tolist())}


def class_indices(classes: dict[str, np.ndarray], mapping: dict[tuple[int, int], int]) -> dict[str, np.ndarray]:
    return {
        name: np.flatnonzero(values) for name, values in classes.items()
    }


def af3_rank_within_degree(af3: np.ndarray, degree_bin: np.ndarray) -> np.ndarray:
    output = np.zeros(len(af3), dtype=float)
    for value in np.unique(degree_bin):
        mask = degree_bin == value
        order = np.argsort(af3[mask], kind="mergesort")
        ranks = np.empty(mask.sum(), dtype=float)
        sorted_values = af3[mask][order]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_values) != 0) + 1])
        ends = np.concatenate([starts[1:], [len(sorted_values)]]) - 1
        for start, end in zip(starts, ends):
            ranks[order[start:end + 1]] = (start + end) / 2.0
        output[mask] = ranks / max(len(ranks) - 1, 1)
    return output


def ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    total = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum(residual ** 2) / total) if total > 0 else float("nan")


def incremental(values: np.ndarray, recall: np.ndarray, controls: np.ndarray) -> dict:
    usable = np.isfinite(values) & np.isfinite(recall) & np.isfinite(controls).all(axis=1)
    base = ols_r2(controls[usable], recall[usable])
    full = ols_r2(np.column_stack([controls[usable], values[usable]]), recall[usable])
    n = int(usable.sum())
    delta = full - base
    p_value = (
        float(f_dist.sf((delta / 1.0) / ((1.0 - full) / (n - controls.shape[1] - 2)),
                        1, n - controls.shape[1] - 2))
        if n > controls.shape[1] + 2 and delta > 0 else float("nan")
    )
    return {"n": n, "base_r2": base, "with_r2": full, "delta_r2": delta, "p_delta": p_value}


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
    return {
        "n": int(len(first)),
        "mean": float(np.mean(first)),
        "spearman": rho,
        "pearson": pear,
        "partial_spearman": analysis.partial(first, second, control[0]) if control.shape[1] == 1 else analysis.pearson(
            first - design @ np.linalg.lstsq(design, first, rcond=None)[0],
            second - design @ np.linalg.lstsq(design, second, rcond=None)[0],
        ),
        "partial_pearson": analysis.pearson(res_x, res_y),
    }


def main() -> int:
    nodes, behavior_edges, adjacency = load_behavior_graph()
    edge_map = edge_to_index(behavior_edges)
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes))
    graph.add_edges_from(map(tuple, behavior_edges.tolist()))
    communities = nx.community.louvain_communities(graph, seed=42, resolution=1.0)
    comm = np.full(nodes, -1, dtype=np.int64)
    for index, members in enumerate(communities):
        comm[list(members)] = index
    inside = comm[behavior_edges[:, 0]] == comm[behavior_edges[:, 1]]

    degree = np.array([len(neighbours) for neighbours in adjacency], dtype=np.int64)
    left = behavior_edges[:, 0]
    right = behavior_edges[:, 1]
    du, dv = degree[left], degree[right]
    dmax = np.maximum(du, dv)
    dmin = np.minimum(du, dv)
    asym = (dmax - dmin) / np.maximum(dmax, 1)
    common = np.array([len(adjacency[int(u)] & adjacency[int(v)]) for u, v in behavior_edges.tolist()], dtype=np.int64)
    af3 = 4 - du - dv + 3 * common
    degree_bin = np.asarray(pd.qcut(dmax, 5, labels=False, duplicates="drop"), dtype=np.int64)
    af3_rank = af3_rank_within_degree(af3, degree_bin)
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
        "bridge": (common == 0) & (~inside) & (between >= q75_between) & (af3_rank <= 0.30),
        "star": (common == 0) & (asym >= q75_asym) & (dmax >= q75_dmax) & (dmin <= q25_dmin) & (af3_rank <= 0.30),
        "chain": (common == 0) & (dmax <= q50_dmax) & (asym <= q25_asym) & (af3_rank <= 0.50),
        "af3_pos": af3 > 0,
        "af3_zero": af3 == 0,
        "af3_neg": af3 < 0,
        "inside": inside,
        "cross": ~inside,
    }
    indices = class_indices(classes, edge_map)
    class_info = {}
    for name, values in classes.items():
        class_info[name] = {
            "edges": int(indices[name].size),
            "share": float(values.mean()),
            "af3_mean": float(np.mean(af3[indices[name]])) if indices[name].size else None,
            "af3_median": float(np.median(af3[indices[name]])) if indices[name].size else None,
            "dmax_median": float(np.median(dmax[indices[name]])) if indices[name].size else None,
            "common_median": float(np.median(common[indices[name]])) if indices[name].size else None,
            "between_median": float(np.median(between[indices[name]])) if indices[name].size else None,
            "inside_share": float(inside[indices[name]].mean()) if indices[name].size else None,
        }
    print("classes", json.dumps(class_info, indent=2), flush=True)

    af3_by_degree = {}
    for value in range(int(degree_bin.max()) + 1):
        mask = degree_bin == value
        af3_by_degree[f"Q{value + 1}"] = {
            "edges": int(mask.sum()),
            "dmax_min": int(dmax[mask].min()),
            "dmax_max": int(dmax[mask].max()),
            "af3_mean": float(af3[mask].mean()),
            "af3_median": float(np.median(af3[mask])),
            "af3_p05": float(np.quantile(af3[mask], 0.05)),
            "af3_p95": float(np.quantile(af3[mask], 0.95)),
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
        collision = analysis.collision_of(tokens)
        bin_rate = np.array([
            recovered[degree_bin == value].mean() if (degree_bin == value).any() else np.nan
            for value in range(5)
        ])
        row = {"number": directory.name, "recall": float(recall), "q_edges": int(len(edges)),
               "collision": float(collision), "overall": float(recovered.mean())}
        for name, values in indices.items():
            observed = float(recovered[values].mean()) if len(values) else float("nan")
            expected = float(np.nanmean(bin_rate[degree_bin[values]])) if len(values) else float("nan")
            row[f"{name}_rate"] = observed
            row[f"{name}_excess"] = observed - expected
        rows.append(row)
        print(f"{directory.name} recall {recall:.5f} q_edges {len(edges)} dense {row['dense_rate']:.5f} "
              f"bridge {row['bridge_rate']:.5f}", flush=True)

    recall = np.array([row["recall"] for row in rows])
    q_edges = np.array([row["q_edges"] for row in rows], dtype=float)
    collision = np.array([row["collision"] for row in rows], dtype=float)
    overall = np.array([row["overall"] for row in rows], dtype=float)
    inside_rate = np.array([row["inside_rate"] for row in rows], dtype=float)
    cross_rate = np.array([row["cross_rate"] for row in rows], dtype=float)
    base_controls = np.column_stack([overall, q_edges, collision])
    community_controls = np.column_stack([overall, q_edges, collision, inside_rate, cross_rate])

    def controls_for(name: str) -> np.ndarray:
        return base_controls if name in ("inside", "cross") else community_controls

    results = {}
    for name in CLASSES:
        values = np.array([row[f"{name}_rate"] for row in rows])
        excess = np.array([row[f"{name}_excess"] for row in rows])
        control = controls_for(name)
        results[name] = {
            "direct": direct(values, recall, control),
            "excess": direct(excess, recall, control),
            "incremental_rate": incremental(values, recall, control),
            "incremental_excess": incremental(excess, recall, control),
            "cv_rate": cv_scores(np.column_stack([control, values]), recall),
            "cv_excess": cv_scores(np.column_stack([control, excess]), recall),
            "cv_base": cv_scores(control, recall),
        }

    output = {
        "af3_sign_counts": {
            "positive": int((af3 > 0).sum()),
            "zero": int((af3 == 0).sum()),
            "negative": int((af3 < 0).sum()),
        },
        "nodes": int(nodes),
        "behavior_edges": int(len(behavior_edges)),
        "k": K,
        "arms": len(rows),
        "class_info": class_info,
        "af3_by_degree": af3_by_degree,
        "thresholds": {
            "af3_q75": q75_af3,
            "af3_q25": q25_af3,
            "dmax_q75": q75_dmax,
            "dmin_q25": q25_dmin,
            "dmax_q50": q50_dmax,
            "asym_q25": q25_asym,
            "asym_q75": q75_asym,
            "betweenness_q75": q75_between,
            "degree_bins": int(degree_bin.max() + 1),
        },
        "results": results,
        "rows": rows,
        "skipped": skipped,
    }
    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# AF3 / local-topology edge classes vs q-kNN recovery and recall",
        "",
        "AF3 = 4 - du - dv + 3*T, on the fixed TRAIN behaviour graph. "
        "Dense/bridge/star/chain thresholds are degree/quantile based and may overlap. "
        "q is rebuilt from standard 3x(256,32) VQ codebooks; q-kNN is global k=10.",
        "",
        f"Behaviour edges: **{len(behavior_edges)}**; arms: **{len(rows)}**.",
        "",
        "## AF3 distribution by dmax quintile",
        "",
        "| bin | dmax range | edges | AF3 mean | AF3 median | p05 | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in af3_by_degree.items():
        lines.append(
            f"| {name} | {value['dmax_min']}–{value['dmax_max']} | {value['edges']} | "
            f"{value['af3_mean']:.3f} | {value['af3_median']:.3f} | {value['af3_p05']:.3f} | {value['af3_p95']:.3f} |"
        )
    lines += [
        "",
        "## Class definitions and sizes",
        "",
        "| class | edges | share | AF3 mean | AF3 median | dmax median | common median | edge-betweenness median | inside share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in class_info.items():
        if value["edges"] == 0:
            lines.append(f"| {name} | 0 | 0.0000 | NA | NA | NA | NA | NA | NA |")
        else:
            lines.append(
                f"| {name} | {value['edges']} | {value['share']:.4f} | {value['af3_mean']:.3f} | "
                f"{value['af3_median']:.3f} | {value['dmax_median']:.1f} | {value['common_median']:.1f} | "
                f"{value['between_median']:.3e} | {value['inside_share']:.4f} |"
            )
    lines += [
        "",
        "Raw AF3 sign counts: positive **{}**, zero **{}**, negative **{}**. "
        "Because almost every edge is negative, raw AF3 sign is not a useful class; the classes "
        "above use AF3 rank within the dmax quintile plus local topology.".format(
            int((af3 > 0).sum()), int((af3 == 0).sum()), int((af3 < 0).sum())
        ),
        "",
        "## Direct and incremental recall relations",
        "",
        "Partial controls overall recovery, q-kNN edge count, L3 collision, and inside/cross recovery. "
        "ΔR² is in-sample; CV is repeated 5-fold Ridge(alpha=1).",
        "",
        "| class | mean rate | direct Spearman | direct Pearson | partial Spearman | partial Pearson | ΔR² rate | p(rate) | CV ΔR² rate | CV ΔR² excess |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CLASSES:
        value = results[name]
        d = value["direct"]
        rate = value["incremental_rate"]
        cv_rate = value["cv_rate"]["r2_mean"] - value["cv_base"]["r2_mean"]
        cv_excess = value["cv_excess"]["r2_mean"] - value["cv_base"]["r2_mean"]
        lines.append(
            f"| {name} | {d['mean']:.5f} | {d['spearman']:+.4f} | {d['pearson']:+.4f} | "
            f"{d['partial_spearman']:+.4f} | {d['partial_pearson']:+.4f} | {rate['delta_r2']:+.4f} | "
            f"{rate['p_delta']:.4f} | {cv_rate:+.4f} | {cv_excess:+.4f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
