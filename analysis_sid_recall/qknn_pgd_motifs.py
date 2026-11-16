"""PGD 4-node edge motifs vs q-kNN recovery and Stage3 recall.

PGD micro output is an edge x motif matrix:
  triangle, 2-star, 4-clique, 4-chordal-cycle, 4-tailed-triangle, 4-cycle, 3-star, 4-path.
This is a 4-node substitute, NOT 68-d Edge-ORCA; classes may overlap.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/qknn_pgd_motifs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "tiger_cluster_graph"))
import analysis  # noqa: E402
from qknn_community_correlation import checkpoint_vq, qknn_edges  # noqa: E402
from qknn_graphlet_cv import cv_scores  # noqa: E402
from sid_structural_fidelity import load_behavior_graph  # noqa: E402

R2 = HERE / "r2"
MICRO = REPO / "results" / "research" / "tiger_cluster_graph" / "pgd_behavior.micro"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT_JSON = HERE / "results" / "qknn_pgd_motifs.json"
OUT_MD = HERE / "results" / "qknn_pgd_motifs.md"
K = 10
N_ITEMS = 24587
MOTIFS = ("triangle", "2-star", "4-clique", "4-chordal-cycle",
          "4-tailed-triangle", "4-cycle", "3-star", "4-path")


def rank_within_bins(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=float)
    for value in np.unique(bins):
        mask = bins == value
        order = np.argsort(values[mask], kind="mergesort")
        sorted_values = values[mask][order]
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
    rank_x = analysis.average_ranks(first)
    rank_y = analysis.average_ranks(second)
    rank_control = np.column_stack([analysis.average_ranks(control[:, i]) for i in range(control.shape[1])])
    rank_design = np.column_stack([np.ones(len(first)), rank_control])
    rank_res_x = rank_x - rank_design @ np.linalg.lstsq(rank_design, rank_x, rcond=None)[0]
    rank_res_y = rank_y - rank_design @ np.linalg.lstsq(rank_design, rank_y, rcond=None)[0]
    return {
        "n": int(len(first)), "mean": float(first.mean()),
        "spearman": rho, "pearson": pear,
        "partial_spearman": analysis.pearson(rank_res_x, rank_res_y),
        "partial_pearson": analysis.pearson(res_x, res_y),
    }


def incremental(values: np.ndarray, recall: np.ndarray, controls: np.ndarray) -> dict:
    usable = np.isfinite(values) & np.isfinite(recall) & np.isfinite(controls).all(axis=1)
    base = np.column_stack([np.ones(usable.sum()), controls[usable]])
    base_resid = recall[usable] - base @ np.linalg.lstsq(base, recall[usable], rcond=None)[0]
    total = np.sum((recall[usable] - recall[usable].mean()) ** 2)
    base_r2 = 1.0 - np.sum(base_resid ** 2) / total
    full = np.column_stack([base, values[usable]])
    full_resid = recall[usable] - full @ np.linalg.lstsq(full, recall[usable], rcond=None)[0]
    full_r2 = 1.0 - np.sum(full_resid ** 2) / total
    n = int(usable.sum())
    delta = full_r2 - base_r2
    p = (f_dist.sf((delta / 1.0) / ((1.0 - full_r2) / (n - controls.shape[1] - 2)),
                   1, n - controls.shape[1] - 2)
         if n > controls.shape[1] + 2 and delta > 0 else float("nan"))
    return {"n": n, "base_r2": float(base_r2), "with_r2": float(full_r2),
            "delta_r2": float(delta), "p_delta": p}


def main() -> int:
    nodes, behavior_edges, adjacency = load_behavior_graph()
    edge_map = {(int(a), int(b)): i for i, (a, b) in enumerate(behavior_edges.tolist())}
    raw = pd.read_csv(MICRO, skiprows=1, header=None, names=["src", "dst", *MOTIFS])
    # PGD relabels vertices in first-appearance order and does not write that mapping.
    input_csv = REPO / "results" / "research" / "tiger_cluster_graph" / "pgd_behavior_edges.csv"
    lookup: dict[int, int] = {}
    internal_to_original: list[int] = []
    with input_csv.open() as handle:
        for line in handle:
            left, right = (int(x) for x in line.strip().split(","))
            for original in (left, right):
                if original not in lookup:
                    lookup[original] = len(internal_to_original)
                    internal_to_original.append(original)
    print("PGD input relabel entries", len(internal_to_original), flush=True)
    motif = np.zeros((len(behavior_edges), len(MOTIFS)), dtype=np.float64)
    matched = 0
    src = raw["src"].to_numpy(dtype=np.int64)
    dst = raw["dst"].to_numpy(dtype=np.int64)
    values = raw[list(MOTIFS)].to_numpy(dtype=np.float64)
    for left, right, value in zip(src, dst, values):
        original_left = internal_to_original[int(left) - 1]
        original_right = internal_to_original[int(right) - 1]
        key = (original_left, original_right) if original_left < original_right else (original_right, original_left)
        index = edge_map.get(key)
        if index is None:
            continue
        motif[index] = value
        matched += 1
    if matched != len(behavior_edges):
        raise ValueError(f"PGD matched {matched} edges, expected {len(behavior_edges)}")
    print("PGD motif matrix", motif.shape, flush=True)

    degree = np.array([len(x) for x in adjacency], dtype=np.int64)
    left, right = behavior_edges[:, 0], behavior_edges[:, 1]
    du, dv = degree[left], degree[right]
    dmax, dmin = np.maximum(du, dv), np.minimum(du, dv)
    dmax_bin = np.asarray(pd.qcut(dmax, 5, labels=False, duplicates="drop"), dtype=np.int64)
    dmin_bin = np.asarray(pd.qcut(dmin, 5, labels=False, duplicates="drop"), dtype=np.int64)
    embedding = np.load(EMBEDDING).astype(np.float32)
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    semantic = np.einsum("ij,ij->i", embedding[left], embedding[right])
    semantic_bin = np.asarray(pd.qcut(semantic, 5, labels=False, duplicates="drop"), dtype=np.int64)
    cell = dmax_bin * 25 + dmin_bin * 5 + semantic_bin

    classes = {}
    for name in MOTIFS:
        ranks = rank_within_bins(motif[:, MOTIFS.index(name)], dmax_bin)
        classes[name] = ranks >= 0.75
    class_info = {
        name: {
            "edges": int(mask.sum()),
            "share": float(mask.mean()),
            "motif_median": float(np.median(motif[mask, MOTIFS.index(name)])),
            "dmax_median": float(np.median(dmax[mask])),
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
            cell_rate = np.full(125, np.nan)
            for value in np.unique(cell):
                control = (~mask) & (cell == value)
                if control.any():
                    cell_rate[value] = recovered[control].mean()
            expected = float(np.nanmean(cell_rate[cell[mask]]))
            row[f"{name}_rate"] = observed
            row[f"{name}_excess"] = observed - expected
        rows.append(row)
        print(f"{directory.name} recall {recall:.5f} q_edges {len(edges)}", flush=True)

    recall = np.array([row["recall"] for row in rows])
    q_edges = np.array([row["q_edges"] for row in rows], dtype=float)
    collision = np.array([row["collision"] for row in rows], dtype=float)
    overall = np.array([row["overall"] for row in rows], dtype=float)
    controls = np.column_stack([overall, q_edges, collision])
    results = {}
    for name in MOTIFS:
        rate = np.array([row[f"{name}_rate"] for row in rows])
        excess = np.array([row[f"{name}_excess"] for row in rows])
        results[name] = {
            "rate": direct(rate, recall, controls),
            "excess": direct(excess, recall, controls),
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
        "motifs": list(MOTIFS),
        "arms": len(rows),
        "class_info": class_info,
        "results": results,
        "rows": rows,
        "skipped": skipped,
    }
    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# PGD 4-node edge motifs vs q-kNN recovery and recall",
        "",
        "PGD micro output: triangle, 2-star, 4-clique, 4-chordal-cycle, 4-tailed-triangle, "
        "4-cycle, 3-star, 4-path. This is **4-node PGD**, not 68-d Edge-ORCA and not K1,4/P5/T5.",
        "",
        f"Behaviour edges **{len(behavior_edges)}**; arms **{len(rows)}**.",
        "",
        "## Motif-high classes (degree-normalised top quartile)",
        "",
        "| motif | edges | share | motif median | dmax median |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in class_info.items():
        lines.append(
            f"| {name} | {value['edges']} | {value['share']:.4f} | {value['motif_median']:.1f} | "
            f"{value['dmax_median']:.1f} |"
        )
    lines += [
        "",
        "## Recall relations",
        "",
        "Partial controls overall recovery, q-kNN edge count, L3 collision. "
        "CV is repeated 5-fold Ridge; excess uses the same (dmax,dmin,semantic) matching cell.",
        "",
        "| motif | mean rate | mean excess | direct rho | direct r | partial rho | ΔR² rate | p(rate) | CV +rate | CV +excess |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MOTIFS:
        value = results[name]
        if value["rate"].get("spearman") is None:
            continue
        base = value["cv_base"]["r2_mean"]
        lines.append(
            f"| {name} | {value['rate']['mean']:.5f} | {value['excess']['mean']:+.5f} | "
            f"{value['rate']['spearman']:+.4f} | {value['rate']['pearson']:+.4f} | "
            f"{value['rate']['partial_spearman']:+.4f} | {value['incremental_rate']['delta_r2']:+.4f} | "
            f"{value['incremental_rate']['p_delta']:.4f} | {value['cv_rate']['r2_mean'] - base:+.4f} | "
            f"{value['cv_excess']['r2_mean'] - base:+.4f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
