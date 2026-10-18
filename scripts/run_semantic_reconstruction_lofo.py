"""Phase 2 LOFO prediction for Issue #50: does semantic reconstruction quality
complement L1 Hard ER for Stage 3 Recall?

Four prespecified Ridge feature sets on the same 38 frozen models, same five
family folds and same targets as #47/#49, with M0 (log_ER_L1) recomputed on
exactly the same subset for each challenger so increments are comparable.

Protocol (identical to #49):
- Ridge with train-fold-only standardization
- Train-only grouped inner CV for alpha selection
- Outer CV: leave-one-family-out
- Paired 95% family-cluster bootstrap (2000 draws, fixed seed 42)
- Recall@10 primary, NDCG@10 secondary
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import joint_prefix_lofo as L47

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
OUT_DIR = REPO_ROOT / "results/stage3_T5Train/monotonicity/semantic_reconstruction_er_predictors"
FEATURES_CSV = OUT_DIR / "per_model_semantic_metrics.csv"
CV_SEED = 42
BOOT_REPLICATES = 2000
TOP_K = 10

TARGETS = {"recall@10": "recall@10", "ndcg@10": "ndcg@10"}

# Feature sets prespecified in Issue #50 Phase 2.
# M0: log(ER_L1) — canonical ER-only baseline
# M8: log(ER_L1) + R_cos — decoder reconstruction quality
# M9: log(ER_L1) + N_sem — checkpoint-free prefix neighborhood preservation
# M10: log(ER_L1) + R_cos + log(ER_L1)*R_cos — interaction ("both high")
FEATURE_SETS_FULL: dict[str, list[str]] = {
    "M0": ["log_er_l1"],
    "M9": ["log_er_l1", "N_sem"],
}

# For M8/M10 we refit M0 on exactly the same subset (32 recon-available models).
FEATURE_SETS_RECON: dict[str, list[str]] = {
    "M0": ["log_er_l1"],
    "M8": ["log_er_l1", "R_cos_uniform"],
    "M10": ["log_er_l1", "R_cos_uniform", "log_er_l1:R_cos"],
}


def load_rows() -> list[dict[str, Any]]:
    """Load per_model_semantic_metrics.csv and add derived features."""
    rows: list[dict[str, Any]] = []
    with FEATURES_CSV.open() as f:
        for r in csv.DictReader(f):
            # log(ER_L1): recompute from pos_share/neg_share? No — we use the ER
            # from the inventory which is already validated. But per_model_semantic_metrics.csv
            # doesn't have ER directly; we need to load it from the #49 features CSV or recompute.
            # Actually, let's load ER from the #49 per_model_features.csv (fix48).
            rows.append(r)
    return rows


def load_er_from_49() -> dict[str, float]:
    """Load ER_L1 from the #49 fix48 per_model_features.csv."""
    er_map: dict[str, float] = {}
    p49 = REPO_ROOT / "results/stage3_T5Train/monotonicity/history_hierarchy_predictors_fix48/per_model_features.csv"
    with p49.open() as f:
        for r in csv.DictReader(f):
            er_map[r["model"]] = float(r["er_l1_recheck"])
    return er_map


def prepare_rows() -> list[dict[str, Any]]:
    """Merge ER from #49 into semantic metrics and add log_er_l1 + interaction."""
    er_map = load_er_from_49()
    rows: list[dict[str, Any]] = []
    with FEATURES_CSV.open() as f:
        for r in csv.DictReader(f):
            mid = r["model"]
            er = er_map.get(mid)
            if er is None:
                raise ValueError(f"Model {mid} not found in #49 features — cannot proceed")
            log_er = float(np.log(er))
            r["log_er_l1"] = log_er
            r["family"] = r["family"]
            # Interaction feature
            r_cos = float(r["R_cos_uniform"]) if r["recon_available"] == "True" else float("nan")
            r["log_er_l1:R_cos"] = log_er * r_cos
            rows.append(r)
    return rows


def run_lofo_on_subset(
    rows: list[dict[str, Any]],
    target: str,
    feature_sets: dict[str, list[str]],
    seed: int,
) -> dict[str, Any]:
    """Run LOFO on a subset of rows with the given feature sets."""
    out: dict[str, Any] = {}
    oof: dict[str, np.ndarray] = {}
    for name, feats in feature_sets.items():
        res = L47.lofo_predictions(rows, target, feats, seed=seed)
        oof[name] = res["preds"]
        out[name] = {
            "features": feats,
            "metrics": L47.metrics(res["y"], res["preds"]),
            "folds": [
                {"held_family": f["held_family"], "n_train": f["n_train"],
                 "n_test": f["n_test"], "alpha": f["alpha"]}
                for f in res["folds"]
            ],
        }
    y = np.array([float(r[target]) for r in rows])
    families = [r["family"] for r in rows]
    base_err = oof["M0"] - y
    deltas: dict[str, Any] = {}
    for name in feature_sets:
        if name == "M0":
            continue
        err = oof[name] - y
        boot = L47.family_cluster_bootstrap_delta_mae(
            base_err, err, families, None, replicates=BOOT_REPLICATES, seed=seed
        )
        base_mae = out["M0"]["metrics"]["mae"]
        boot["relative_improvement"] = boot["delta_mae"] / base_mae if base_mae else None
        boot["family_deltas"] = {
            f: float(
                np.abs(base_err[[i for i, g in enumerate(families) if g == f]]).mean()
                - np.abs(err[[i for i, g in enumerate(families) if g == f]]).mean()
            )
            for f in sorted(set(families))
        }
        boot["leave_one_family_out"] = {
            f: float(boot["delta_mae"] - v)
            for f, v in boot["family_deltas"].items()
        }
        deltas[name] = boot
    return {"sets": out, "deltas": deltas, "_oof": oof, "_y": y, "_families": families}


def run_all() -> dict[str, Any]:
    """Run both the full-38 (M0, M9) and recon-32 (M0, M8, M10) analyses."""
    all_rows = prepare_rows()
    print(f"Loaded {len(all_rows)} models with ER + semantic features")

    # --- Full 38 models: M0 vs M9 ---
    print("\n=== Full 38 models: M0 vs M9 (N_sem) ===")
    full_results = run_lofo_on_subset(all_rows, "recall@10", FEATURE_SETS_FULL, CV_SEED)
    print(f"  M0 MAE = {full_results['sets']['M0']['metrics']['mae']:.6f}")
    print(f"  M9 MAE = {full_results['sets']['M9']['metrics']['mae']:.6f}")
    if "M9" in full_results["deltas"]:
        d = full_results["deltas"]["M9"]
        print(f"  M9 dMAE = {d['delta_mae']:+.6f}, CI95 = [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]")

    # --- Recon-available 32 models: M0 vs M8 vs M10 ---
    recon_rows = [r for r in all_rows if r["recon_available"] == "True"]
    print(f"\n=== Recon-available {len(recon_rows)} models: M0 vs M8 vs M10 (R_cos) ===")
    recon_results = run_lofo_on_subset(recon_rows, "recall@10", FEATURE_SETS_RECON, CV_SEED)
    print(f"  M0 (subset) MAE = {recon_results['sets']['M0']['metrics']['mae']:.6f}")
    for m in ["M8", "M10"]:
        print(f"  {m} MAE = {recon_results['sets'][m]['metrics']['mae']:.6f}")
        if m in recon_results["deltas"]:
            d = recon_results["deltas"][m]
            print(f"  {m} dMAE = {d['delta_mae']:+.6f}, CI95 = [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]")

    # --- NDCG@10 secondary target ---
    print("\n=== NDCG@10 secondary target ===")
    full_ndcg = run_lofo_on_subset(all_rows, "ndcg@10", FEATURE_SETS_FULL, CV_SEED)
    recon_ndcg = run_lofo_on_subset(recon_rows, "ndcg@10", FEATURE_SETS_RECON, CV_SEED)

    # --- TopK screening: compare only rankings over the same full 38-model pool ---
    print("\n=== Top-10 candidate-model screening (full 38; raw ER vs OOF) ===")
    topk: dict[str, Any] = {}
    raw_er = np.array([float(r["log_er_l1"]) for r in all_rows], dtype=float)
    topk["raw_ER"] = L47.topk_selection(
        full_results["_y"], raw_er, full_results["_families"], TOP_K
    )
    for name in ["M0", "M9"]:
        topk[name] = L47.topk_selection(
            full_results["_y"], full_results["_oof"][name],
            full_results["_families"], TOP_K
        )
    for name, result in topk.items():
        print(f"  {name}: P@{TOP_K} = {result['precision_at_k']:.2f} ({result['hits']}/{TOP_K})")
    topk["scope"] = "All rankings use the same full 38-model pool; recon-subset TopK is not compared."
    # --- Write OOF predictions for both targets on each common subset ---
    oof_path = OUT_DIR / "oof_predictions.csv"
    with oof_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "family", "subset", "target", "model_set", "y_true", "y_pred_oof"])
        for target_name, full_run, recon_run in [
            ("recall@10", full_results, recon_results),
            ("ndcg@10", full_ndcg, recon_ndcg),
        ]:
            for i, r in enumerate(all_rows):
                for name in ["M0", "M9"]:
                    writer.writerow([
                        r["model"], r["family"], "full_38", target_name, name,
                        full_run["_y"][i], full_run["_oof"][name][i],
                    ])
            for i, r in enumerate(recon_rows):
                for name in ["M0", "M8", "M10"]:
                    writer.writerow([
                        r["model"], r["family"], "recon_32", target_name, name,
                        recon_run["_y"][i], recon_run["_oof"][name][i],
                    ])

    # --- Assemble ablation_summary ---
    summary: dict[str, Any] = {
        "n_models_total": len(all_rows),
        "n_models_recon": len(recon_rows),
        "n_models_full": len(all_rows),
        "feature_sets_full": {k: v for k, v in FEATURE_SETS_FULL.items()},
        "feature_sets_recon": {k: v for k, v in FEATURE_SETS_RECON.items()},
        "targets": {},
        "topk_screening": topk,
    }
    for target_name, target_col in [("recall@10", "recall@10"), ("ndcg@10", "ndcg@10")]:
        if target_name == "recall@10":
            fr = full_results
            rr = recon_results
        else:
            fr = full_ndcg
            rr = recon_ndcg
        summary["targets"][target_name] = {
            "full_38": {
                "sets": fr["sets"],
                "deltas_vs_M0": fr["deltas"],
            },
            "recon_32": {
                "sets": rr["sets"],
                "deltas_vs_M0": rr["deltas"],
            },
        }


    # --- Write ablation_summary.json ---
    summary_path = OUT_DIR / "ablation_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {summary_path}")

    # --- Write ablation_summary.csv ---
    csv_path = OUT_DIR / "ablation_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "target", "model_set", "mae", "rmse", "oof_r2", "oof_spearman",
                          "delta_mae", "ci95_lower", "ci95_upper", "sig"])
        for target_name in ["recall@10", "ndcg@10"]:
            for subset_label, subset_key in [("full_38", "full_38"), ("recon_32", "recon_32")]:
                tdata = summary["targets"][target_name][subset_key]
                for ms_name in tdata["sets"]:
                    m = tdata["sets"][ms_name]["metrics"]
                    d = tdata["deltas_vs_M0"].get(ms_name, {})
                    writer.writerow([
                        subset_label, target_name, ms_name,
                        m["mae"], m["rmse"], m.get("oof_r2", ""), m.get("oof_spearman", ""),
                        d.get("delta_mae", ""), d.get("ci95", ["", ""])[0],
                        d.get("ci95", ["", ""])[1],
                        d.get("lower_bound_gt_zero", "")
                    ])
    print(f"Wrote {csv_path}")

    return summary


if __name__ == "__main__":
    run_all()
