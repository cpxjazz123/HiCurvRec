"""Paired significance for the audited PGD/q-kNN relations, plus collision sensitivity.

Reads the already computed rows (no q-graph recomputation), repeats the grouped CV with
fold-level output, and reports paired Wilcoxon / sign-test p-values. The direct and partial
correlations are reported separately from the CV gain so a positive CV does not get read as
a positive direction.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/qknn_pgd_significance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results" / "qknn_pgd_audited.json"
OUT_JSON = HERE / "results" / "qknn_pgd_significance.json"
OUT_MD = HERE / "results" / "qknn_pgd_significance.md"
MOTIFS = ("triangle", "4-clique", "4-chordal-cycle", "4-cycle", "4-path")


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


def partial_correlation(values: np.ndarray, recall: np.ndarray, controls: np.ndarray) -> dict:
    usable = np.isfinite(values) & np.isfinite(recall) & np.isfinite(controls).all(axis=1)
    first, second, control = values[usable], recall[usable], controls[usable]
    design = np.column_stack([np.ones(len(first)), control])
    res_x = first - design @ np.linalg.lstsq(design, first, rcond=None)[0]
    res_y = second - design @ np.linalg.lstsq(design, second, rcond=None)[0]
    rank_x, rank_y = stats.rankdata(first), stats.rankdata(second)
    rank_control = np.column_stack([stats.rankdata(control[:, i]) for i in range(control.shape[1])])
    rank_design = np.column_stack([np.ones(len(first)), rank_control])
    rank_res_x = rank_x - rank_design @ np.linalg.lstsq(rank_design, rank_x, rcond=None)[0]
    rank_res_y = rank_y - rank_design @ np.linalg.lstsq(rank_design, rank_y, rcond=None)[0]
    rho, rho_p = stats.spearmanr(first, second)
    pearson, pearson_p = stats.pearsonr(first, second)
    partial_rho, partial_rho_p = stats.pearsonr(rank_res_x, rank_res_y)
    partial_pearson, partial_pearson_p = stats.pearsonr(res_x, res_y)
    return {
        "n": int(len(first)),
        "spearman": float(rho), "spearman_p": float(rho_p),
        "pearson": float(pearson), "pearson_p": float(pearson_p),
        "partial_spearman": float(partial_rho), "partial_spearman_p": float(partial_rho_p),
        "partial_pearson": float(partial_pearson), "partial_pearson_p": float(partial_pearson_p),
    }


def paired(base: list[float], feature: list[float]) -> dict:
    base = np.asarray(base, dtype=float)
    feature = np.asarray(feature, dtype=float)
    difference = feature - base
    wins = int((difference > 0).sum())
    losses = int((difference < 0).sum())
    if len(difference) >= 5 and np.any(difference != 0):
        wilcoxon = stats.wilcoxon(feature, base, alternative="greater")
        p_value = float(wilcoxon.pvalue)
    else:
        p_value = 1.0
    sign_p = float(stats.binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue) \
        if wins + losses else 1.0
    return {
        "n_folds": int(len(difference)),
        "base_r2_mean": float(base.mean()), "feature_r2_mean": float(feature.mean()),
        "delta_mean": float(difference.mean()), "delta_sd": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
        "wins": wins, "losses": losses, "wilcoxon_greater_p": p_value, "sign_test_greater_p": sign_p,
    }


def analyze(rows: list[dict]) -> dict:
    recall = np.array([row["recall"] for row in rows], dtype=float)
    q_edges = np.array([row["q_edges"] for row in rows], dtype=float)
    collision = np.array([row["collision"] for row in rows], dtype=float)
    groups = np.array([row["series"] for row in rows])
    output = {}
    for name in MOTIFS:
        rate = np.array([row[f"{name}_rate"] for row in rows], dtype=float)
        excess = np.array([row[f"{name}_excess"] for row in rows], dtype=float)
        controls = np.column_stack([
            np.array([row[f"{name}_overall_loo"] for row in rows], dtype=float), q_edges, collision
        ])
        base_folds = cv_folds(controls, recall, groups)
        rate_folds = cv_folds(np.column_stack([controls, rate]), recall, groups)
        excess_folds = cv_folds(np.column_stack([controls, excess]), recall, groups)
        output[name] = {
            "direct": partial_correlation(rate, recall, controls),
            "excess": partial_correlation(excess, recall, controls),
            "cv_base_folds": base_folds,
            "cv_rate_folds": rate_folds,
            "cv_excess_folds": excess_folds,
            "cv_rate_paired": paired(base_folds, rate_folds),
            "cv_excess_paired": paired(base_folds, excess_folds),
        }
    return output


def main() -> int:
    data = json.loads(SOURCE.read_text())
    rows = data["rows"]
    main_result = analyze(rows)
    low_collision = [row for row in rows if row["collision"] < 0.90]
    sensitivity = analyze(low_collision)
    result = {
        "source": str(SOURCE.relative_to(HERE.parent)),
        "arms": len(rows), "low_collision_arms": len(low_collision),
        "note": "CV gain is predictive, not a directional claim; direct and partial signs can differ.",
        "main": main_result, "collision_below_0_9": sensitivity,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    lines = [
        "# Paired significance and collision sensitivity",
        "",
        "Grouped CV (GroupShuffleSplit by model series, 30 splits, Ridge alpha=1). "
        "`ΔR²` is feature minus base; p is one-sided Wilcoxon on the paired folds. "
        "Significance of a predictive gain is **not** evidence that higher retention raises recall: "
        "the partial correlation is reported separately.",
        "",
        "## Main (48 arms)",
        "",
        "| motif | n | partial rho (p) | partial r (p) | CV ΔR² rate (p) | wins | CV ΔR² excess (p) | wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MOTIFS:
        value = main_result[name]
        d, e = value["direct"], value["excess"]
        r, x = value["cv_rate_paired"], value["cv_excess_paired"]
        lines.append(
            f"| {name} | {d['n']} | {d['partial_spearman']:+.4f} ({d['partial_spearman_p']:.3f}) | "
            f"{d['partial_pearson']:+.4f} ({d['partial_pearson_p']:.3f}) | "
            f"{r['delta_mean']:+.4f} ({r['wilcoxon_greater_p']:.3f}) | {r['wins']}/{r['n_folds']} | "
            f"{x['delta_mean']:+.4f} ({x['wilcoxon_greater_p']:.3f}) | {x['wins']}/{x['n_folds']} |"
        )
    lines += [
        "",
        f"## Sensitivity: collision < 0.90 ({len(low_collision)} arms)",
        "",
        "| motif | partial rho | CV ΔR² rate (p) | CV ΔR² excess (p) |",
        "|---|---:|---:|---:|",
    ]
    for name in MOTIFS:
        value = sensitivity[name]
        d = value["direct"]
        r, x = value["cv_rate_paired"], value["cv_excess_paired"]
        lines.append(
            f"| {name} | {d['partial_spearman']:+.4f} | {r['delta_mean']:+.4f} ({r['wilcoxon_greater_p']:.3f}) | "
            f"{x['delta_mean']:+.4f} ({x['wilcoxon_greater_p']:.3f}) |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "main": {name: {"rate": main_result[name]["cv_rate_paired"],
                        "excess": main_result[name]["cv_excess_paired"],
                        "partial_rho": main_result[name]["direct"]["partial_spearman"],
                        "partial_rho_p": main_result[name]["direct"]["partial_spearman_p"]}
                 for name in MOTIFS},
        "low_collision": len(low_collision),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
