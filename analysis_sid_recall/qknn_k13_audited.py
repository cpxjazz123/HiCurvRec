"""Exact per-edge K1,3 recovery and Recall@10 relation.

The K1,3 counts are produced by `k13_exact.cpp` (exact, synthetic + random tested),
not by PGD's broken 3-star column.  Primary class is the top quartile of the exact
K1,3 count within degree-max bins, matching the earlier PGD class definition.
Also reports the recovery of all K1,3-positive edges for completeness.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/qknn_k13_audited.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "tiger_cluster_graph"))
import analysis  # noqa: E402
from qknn_community_correlation import checkpoint_vq, qknn_edges  # noqa: E402
from qknn_pgd_audited import series_of  # noqa: E402
from sid_structural_fidelity import load_behavior_graph  # noqa: E402

R2 = HERE / "r2"
K13 = REPO / "results" / "research" / "tiger_cluster_graph" / "k13_edges.csv"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT_JSON = HERE / "results" / "qknn_k13_audited.json"
OUT_MD = HERE / "results" / "qknn_k13_audited.md"
MOTIF = "k13"


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


def partial(values: np.ndarray, recall: np.ndarray, controls: np.ndarray) -> dict:
    usable = np.isfinite(values) & np.isfinite(recall) & np.isfinite(controls).all(axis=1)
    first, second, control = values[usable], recall[usable], controls[usable]
    rho, rho_p = stats.spearmanr(first, second)
    pearson, pearson_p = stats.pearsonr(first, second)
    design = np.column_stack([np.ones(len(first)), control])
    res_x = first - design @ np.linalg.lstsq(design, first, rcond=None)[0]
    res_y = second - design @ np.linalg.lstsq(design, second, rcond=None)[0]
    rank_x, rank_y = stats.rankdata(first), stats.rankdata(second)
    rank_control = np.column_stack([stats.rankdata(control[:, i]) for i in range(control.shape[1])])
    rank_design = np.column_stack([np.ones(len(first)), rank_control])
    rank_res_x = rank_x - rank_design @ np.linalg.lstsq(rank_design, rank_x, rcond=None)[0]
    rank_res_y = rank_y - rank_design @ np.linalg.lstsq(rank_design, rank_y, rcond=None)[0]
    prho, prho_p = stats.pearsonr(rank_res_x, rank_res_y)
    pr, pr_p = stats.pearsonr(res_x, res_y)
    return {
        "n": int(len(first)), "mean": float(first.mean()),
        "spearman": float(rho), "spearman_p": float(rho_p),
        "pearson": float(pearson), "pearson_p": float(pearson_p),
        "partial_spearman": float(prho), "partial_spearman_p": float(prho_p),
        "partial_pearson": float(pr), "partial_pearson_p": float(pr_p),
    }


def cv_folds(features: np.ndarray, target: np.ndarray, groups: np.ndarray,
             repeats: int = 30, test_size: float = 0.2, seed: int = 42) -> list[float]:
    usable = np.isfinite(features).all(axis=1) & np.isfinite(target)
    features, target, groups = features[usable], target[usable], groups[usable]
    splitter = GroupShuffleSplit(n_splits=repeats, test_size=test_size, random_state=seed)
    output = []
    for train, test in splitter.split(features, target, groups):
        if len(np.unique(groups[train])) < 2:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(features[train], target[train])
        output.append(float(r2_score(target[test], model.predict(features[test]))))
    return output


def paired(base: list[float], feature: list[float]) -> dict:
    base = np.asarray(base, dtype=float)
    feature = np.asarray(feature, dtype=float)
    difference = feature - base
    wins = int((difference > 0).sum())
    losses = int((difference < 0).sum())
    p_value = float(stats.wilcoxon(feature, base, alternative="greater").pvalue) \
        if len(difference) >= 5 and np.any(difference != 0) else 1.0
    return {
        "n_folds": int(len(difference)),
        "base_r2_mean": float(base.mean()), "feature_r2_mean": float(feature.mean()),
        "delta_mean": float(difference.mean()),
        "delta_sd": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
        "wins": wins, "losses": losses, "wilcoxon_greater_p": p_value,
    }


def main() -> int:
    nodes, behavior_edges, adjacency = load_behavior_graph()
    exact = pd.read_csv(K13)
    edge_map = {(int(a), int(b)): i for i, (a, b) in enumerate(behavior_edges.tolist())}
    k13 = np.zeros(len(behavior_edges), dtype=float)
    for left, right, value in zip(exact["src"], exact["dst"], exact["k13"]):
        pair = (int(left), int(right)) if int(left) < int(right) else (int(right), int(left))
        k13[edge_map[pair]] = float(value)
    if len(exact) != len(behavior_edges):
        raise ValueError(f"K1,3 edge count differs: {len(exact)} vs {len(behavior_edges)}")

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

    ranks = rank_within_bins(k13, dmax_bin)
    top = ranks >= 0.75
    positive = k13 > 0
    class_info = {
        "top_edges": int(top.sum()), "top_share": float(top.mean()),
        "positive_edges": int(positive.sum()), "positive_share": float(positive.mean()),
        "k13_min": float(k13.min()), "k13_median": float(np.median(k13)),
        "k13_max": float(k13.max()),
        "top_min_count": float(k13[top].min()), "top_median_count": float(np.median(k13[top])),
        "top_dmax_median": float(np.median(dmax[top])),
    }
    print("K1,3 classes", json.dumps(class_info, indent=2), flush=True)

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
        codebooks, _ = packed
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
        tree = next((child for child in directory.iterdir() if child.is_dir()), directory).name
        row = {
            "number": directory.name, "series": series_of(tree), "tree": tree,
            "recall": float(recall), "q_edges": int(len(edges)),
            "collision": analysis.collision_of(tokens),
            "overall": float(recovered.mean()),
        }

        indices = np.flatnonzero(top)
        observed = float(recovered[indices].mean())
        cell_rate = np.full(125, np.nan)
        for value in np.unique(cell):
            control = (~top) & (cell == value)
            if control.any():
                cell_rate[value] = recovered[control].mean()
        covered = np.isfinite(cell_rate[cell[indices]])
        expected = float(np.nanmean(cell_rate[cell[indices]])) if covered.any() else float("nan")
        remaining = len(behavior_edges) - len(indices)
        row["k13_rate"] = observed
        row["k13_excess"] = observed - expected if covered.any() else float("nan")
        row["k13_coverage"] = float(covered.mean())
        row["k13_overall_loo"] = float((recovered.sum() - recovered[indices].sum()) / remaining)
        row["k13_positive_rate"] = float(recovered[positive].mean()) if positive.any() else float("nan")
        rows.append(row)
        print(f"{directory.name} recall {recall:.5f} q_edges {len(edges)} "
              f"k13_rate {observed:.5f} excess {row['k13_excess']:+.5f} "
              f"coverage {row['k13_coverage']:.4f}", flush=True)

    recall = np.array([row["recall"] for row in rows])
    q_edges = np.array([row["q_edges"] for row in rows], dtype=float)
    collision = np.array([row["collision"] for row in rows], dtype=float)
    groups = np.array([row["series"] for row in rows])
    controls = np.column_stack([
        np.array([row["k13_overall_loo"] for row in rows], dtype=float), q_edges, collision
    ])
    rate = np.array([row["k13_rate"] for row in rows], dtype=float)
    excess = np.array([row["k13_excess"] for row in rows], dtype=float)
    base_folds = cv_folds(controls, recall, groups)
    rate_folds = cv_folds(np.column_stack([controls, rate]), recall, groups)
    excess_folds = cv_folds(np.column_stack([controls, excess]), recall, groups)
    main_result = {
        "coverage_mean": float(np.mean([row["k13_coverage"] for row in rows])),
        "coverage_min": float(np.min([row["k13_coverage"] for row in rows])),
        "rate": partial(rate, recall, controls),
        "excess": partial(excess, recall, controls),
        "cv_rate": paired(base_folds, rate_folds),
        "cv_excess": paired(base_folds, excess_folds),
    }

    low = np.array([row["collision"] < 0.90 for row in rows])
    low_result = {}
    if low.sum() >= 10:
        low_controls = controls[low]
        low_groups = groups[low]
        low_recall = recall[low]
        low_base = cv_folds(low_controls, low_recall, low_groups)
        low_result = {
            "arms": int(low.sum()),
            "rate": partial(rate[low], low_recall, low_controls),
            "excess": partial(excess[low], low_recall, low_controls),
            "cv_rate": paired(low_base, cv_folds(np.column_stack([low_controls, rate[low]]), low_recall, low_groups)),
            "cv_excess": paired(low_base, cv_folds(np.column_stack([low_controls, excess[low]]), low_recall, low_groups)),
        }

    result = {
        "nodes": int(nodes), "behavior_edges": int(len(behavior_edges)), "arms": len(rows),
        "k13_source": str(K13.relative_to(REPO)),
        "class_info": class_info,
        "main": main_result, "collision_below_0_9": low_result,
        "rows": rows, "skipped": skipped,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    lines = [
        "# Exact K1,3 edge recovery and Recall@10",
        "",
        "K1,3 counts are exact per-edge induced star counts from `k13_exact.cpp`, not PGD's",
        "broken `3-star` column. Primary class: top quartile of exact count within degree-max bins;",
        "matched controls use degree-max + degree-min + semantic bins. `positive` uses every edge",
        "with a nonzero exact count. CV is grouped by model series.",
        "",
        f"Behavior edges **{len(behavior_edges)}**; arms **{len(rows)}**; "
        f"K1,3 edges **{class_info['positive_edges']}**; top class **{class_info['top_edges']}**.",
        "",
        "## Class",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| K1,3 positive edges | {class_info['positive_edges']} |",
        f"| K1,3 positive share | {class_info['positive_share']:.6f} |",
        f"| count median / max | {class_info['k13_median']:.0f} / {class_info['k13_max']:.0f} |",
        f"| top-class count median | {class_info['top_median_count']:.0f} |",
        f"| top-class dmax median | {class_info['top_dmax_median']:.0f} |",
        "",
        "## Recovery and recall",
        "",
        "| quantity | n | mean | direct rho (p) | partial rho (p) | CV ΔR² (p) | wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, key in (("K1,3 rate", "rate"), ("K1,3 matched excess", "excess")):
        value = main_result[key]
        cv_key = "cv_rate" if key == "rate" else "cv_excess"
        cv = main_result[cv_key]
        lines.append(
            f"| {name} | {value['n']} | {value['mean']:.6f} | "
            f"{value['spearman']:+.4f} ({value['spearman_p']:.3g}) | "
            f"{value['partial_spearman']:+.4f} ({value['partial_spearman_p']:.3g}) | "
            f"{cv['delta_mean']:+.4f} ({cv['wilcoxon_greater_p']:.3g}) | "
            f"{cv['wins']}/{cv['n_folds']} |"
        )
    lines += [
        "",
        f"## Sensitivity: collision < 0.90 ({low_result.get('arms', 0)} arms)",
        "",
        "| quantity | direct rho | partial rho (p) | CV ΔR² (p) |",
        "|---|---:|---:|---:|",
    ]
    for name, key, cv_key in (
        ("K1,3 rate", "rate", "cv_rate"),
        ("K1,3 matched excess", "excess", "cv_excess"),
    ):
        if not low_result:
            continue
        value = low_result[key]
        cv = low_result[cv_key]
        lines.append(
            f"| {name} | {value['spearman']:+.4f} | "
            f"{value['partial_spearman']:+.4f} ({value['partial_spearman_p']:.3g}) | "
            f"{cv['delta_mean']:+.4f} ({cv['wilcoxon_greater_p']:.3g}) |"
        )
    lines += [
        "",
        "Coverage for the primary class: "
        f"mean {main_result['coverage_mean']:.4f}, min {main_result['coverage_min']:.4f}.",
        "The all-positive subset is nearly the whole graph; its recovery is not a discriminative",
        "class and is reported only for completeness.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON} and {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
