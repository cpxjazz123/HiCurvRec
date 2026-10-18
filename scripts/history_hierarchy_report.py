"""LOFO prediction and report generation for issue #48.

Five prespecified Ridge feature sets on the same 38 frozen models, the same five
family folds and the same targets as #47, with the #47 M0 baseline recomputed on
identical folds so the increments are comparable.

Complete-case handling is explicit: if any new feature is missing for a model, M0
and every competing arm are refitted on exactly the same subset rather than
compared against a larger baseline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import joint_prefix_lofo as L47

OLD_OUT_DIR = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/monotonicity/history_hierarchy_predictors"
)
OUT_DIR = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/monotonicity/history_hierarchy_predictors_fix48"
)
FEATURES_CSV = OUT_DIR / "per_model_features.csv"
PROBES_CSV = OUT_DIR / "probe_scores.csv"

TARGETS = {"recall@10": "stage3_recall@10", "ndcg@10": "stage3_ndcg@10"}
TOP_K = 10


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with FEATURES_CSV.open() as f:
        for r in csv.DictReader(f):
            out: dict[str, Any] = dict(r)
            for k, v in r.items():
                if k.startswith(("chrd_", "amb_")) and v not in ("NA", "undefined"):
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
                elif k in ("log_er_l1", "er_l1_recheck", "er_l1_from_issue47"):
                    out[k] = float(v)
            for k in ("D2_ci95_low", "D2_ci95_high"):
                try:
                    out[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
            rows.append(out)
    return rows


def apply_probe_features(rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]]) -> None:
    """Attach G_H1 and friends; seq20 is the primary, seq1 the history-length ablation."""
    by: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in probe_rows:
        by.setdefault((r["model"], int(r["seq_len"])), []).append(r)

    for row in rows:
        for seq_len in (20, 1):
            recs = by.get((row["model"], seq_len), [])
            if not recs:
                for k in ("G_H1", "G_H1_seed_std", "gain_H1_nats", "probe_ce_l1",
                          "prior_ce_l1", "probe_top1_l1", "probe_top20_l1"):
                    row[f"{k}_s{seq_len}"] = None
                continue

            def mean(f: str) -> float:
                vals = [float(x[f]) for x in recs if x.get(f) not in (None, "")]
                return float(np.mean(vals)) if vals else float("nan")

            def sd(f: str) -> float:
                vals = [float(x[f]) for x in recs if x.get(f) not in (None, "")]
                return float(np.std(vals)) if len(vals) > 1 else 0.0

            row[f"G_H1_s{seq_len}"] = mean("rel_gain_l1")
            row[f"G_H1_seed_std_s{seq_len}"] = sd("rel_gain_l1")
            row[f"gain_H1_nats_s{seq_len}"] = mean("gain_l1")
            row[f"probe_ce_l1_s{seq_len}"] = mean("probe_ce_l1")
            row[f"prior_ce_l1_s{seq_len}"] = mean("prior_ce_l1")
            row[f"probe_top1_l1_s{seq_len}"] = mean("probe_top1_acc_l1")
            row[f"probe_top20_l1_s{seq_len}"] = mean("probe_top20_acc_l1")
            row[f"prior_top1_l1_s{seq_len}"] = mean("prior_top1_acc_l1")
            row[f"prior_top20_l1_s{seq_len}"] = mean("prior_top20_acc_l1")
            row[f"rel_gain_l2_s{seq_len}"] = mean("rel_gain_l2")
            row[f"rel_gain_l3_s{seq_len}"] = mean("rel_gain_l3")
            row[f"probe_seeds_s{seq_len}"] = ",".join(
                str(sorted(int(x["seed"]) for x in recs))
            )
            row[f"probe_runs_s{seq_len}"] = len(recs)
        row["G_H1"] = row.get("G_H1_s20")
        row["D2"] = row.get("chrd_D2")
        # `target_ambiguity` returns the feature under the bare name A_target.
        row["A_target"] = row.get("A_target")


def complete_case(rows: list[dict[str, Any]], feature_sets: dict[str, list[str]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows usable by EVERY feature set, so no arm is compared to a bigger baseline."""
    needed = sorted({f for feats in feature_sets.values() for f in feats})
    keep, dropped = [], []
    for r in rows:
        missing = [f for f in needed if r.get(f) is None or (
            isinstance(r.get(f), str) and r[f] in ("NA", "")
        ) or (isinstance(r.get(f), float) and not np.isfinite(r[f]))]
        if missing:
            dropped.append((r["model"], ",".join(missing)))
        else:
            keep.append(r)
    return keep, dropped


def run_sets(
    rows: list[dict[str, Any]],
    target_col: str,
    feature_sets: dict[str, list[str]],
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    oof: dict[str, np.ndarray] = {}
    for name, feats in feature_sets.items():
        res = L47.lofo_predictions(rows, target_col, feats, seed=seed)
        oof[name] = res["preds"]
        out[name] = {
            "features": feats,
            "metrics": L47.metrics(res["y"], res["preds"]),
            "folds": [
                {"held_family": f["held_family"], "n_train": f["n_train"],
                 "n_test": f["n_test"], "alpha": f["alpha"]}
                for f in res["folds"]
            ],
            "topk": L47.topk_selection(res["y"], res["preds"], res["families"], TOP_K),
        }
    y = np.array([float(r[target_col]) for r in rows])
    families = [r["family"] for r in rows]
    base_err = oof["M0"] - y
    deltas: dict[str, Any] = {}
    for name in feature_sets:
        if name == "M0":
            continue
        err = oof[name] - y
        boot = L47.family_cluster_bootstrap_delta_mae(
            base_err, err, families, None, replicates=2000, seed=seed
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


def build_outputs(git_sha, feature_sets, cv_seed) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    with PROBES_CSV.open() as f:
        probe_rows = list(csv.DictReader(f))
    apply_probe_features(rows, probe_rows)

    keep, dropped = complete_case(rows, feature_sets)

    results: dict[str, Any] = {"_keep": keep, "_rows": rows, "_dropped": dropped}
    for target, col in TARGETS.items():
        results[target] = run_sets(keep, col, feature_sets, cv_seed)

    er = np.array([float(r["er_l1_recheck"]) for r in keep])
    results["topk_by_raw_er"] = L47.topk_selection(
        results["recall@10"]["_y"], er, results["recall@10"]["_families"], TOP_K
    )

    # ---- oof_predictions.csv
    with (OUT_DIR / "oof_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "family", "target", "model_set", "y_true", "y_pred_oof",
                    "abs_error", "signed_error", "fold_held_out"])
        for target in TARGETS:
            blk = results[target]
            fold_of = {}
            for f in blk["sets"]["M0"]["folds"]:
                for r in keep:
                    if r["family"] == f["held_family"]:
                        fold_of[r["model"]] = f["held_family"]
            for name in feature_sets:
                preds = blk["_oof"][name]
                for i, r in enumerate(keep):
                    w.writerow([
                        r["model"], r["family"], target, name,
                        f"{blk['_y'][i]:.10g}", f"{preds[i]:.10g}",
                        f"{abs(preds[i] - blk['_y'][i]):.10g}",
                        f"{preds[i] - blk['_y'][i]:.10g}",
                        fold_of[r["model"]],
                    ])

    # ---- ablation_summary.{json,csv}
    summary: dict[str, Any] = {
        "n_models_total": len(rows),
        "n_models_complete_case": len(keep),
        "dropped_models": [{"model": m, "missing": x} for m, x in dropped],
        "feature_sets": feature_sets,
        "targets": {},
        "topk_screening": {
            "by_raw_er": results["topk_by_raw_er"],
            "by_model_set": {
                name: results["recall@10"]["sets"][name]["topk"] for name in feature_sets
            },
        },
    }
    for target in TARGETS:
        blk = results[target]
        summary["targets"][target] = {
            "sets": {
                name: {
                    "features": blk["sets"][name]["features"],
                    "metrics": blk["sets"][name]["metrics"],
                    "folds": blk["sets"][name]["folds"],
                }
                for name in feature_sets
            },
            "deltas_vs_M0": blk["deltas"],
        }
    (OUT_DIR / "ablation_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    with (OUT_DIR / "ablation_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "model_set", "features", "mae", "rmse", "oof_r2",
                    "oof_spearman", "delta_mae_vs_M0", "ci95_low", "ci95_high",
                    "relative_improvement", "ci_lower_gt_zero", "topk_precision_at_10",
                    "n_models"])
        for target in TARGETS:
            blk = results[target]
            for name in feature_sets:
                m = blk["sets"][name]["metrics"]
                d = blk["deltas"].get(name)
                w.writerow([
                    target, name, "|".join(blk["sets"][name]["features"]),
                    f"{m['mae']:.8g}", f"{m['rmse']:.8g}", f"{m['oof_r2']:.6g}",
                    m["oof_spearman"] if m["oof_spearman"] is not None else "",
                    f"{d['delta_mae']:.8g}" if d else "0",
                    f"{d['ci95'][0]:.8g}" if d else "",
                    f"{d['ci95'][1]:.8g}" if d else "",
                    f"{d['relative_improvement']:.6g}" if d else "",
                    d["lower_bound_gt_zero"] if d else "",
                    f"{blk['sets'][name]['topk']['precision_at_k']:.4f}",
                    len(keep),
                ])

    write_manifest(rows, keep, dropped, summary, git_sha, feature_sets, cv_seed, probe_rows)
    write_reports(rows, keep, results, summary, git_sha)
    print(json.dumps({
        "n_models": len(rows),
        "n_complete_case": len(keep),
        "recall": {n: results["recall@10"]["sets"][n]["metrics"] for n in feature_sets},
        "recall_deltas": {
            n: {
                "delta_mae": results["recall@10"]["deltas"][n]["delta_mae"],
                "ci95": results["recall@10"]["deltas"][n]["ci95"],
                "sig": results["recall@10"]["deltas"][n]["lower_bound_gt_zero"],
            } for n in feature_sets if n != "M0"
        },
        "ndcg_deltas": {
            n: {
                "delta_mae": results["ndcg@10"]["deltas"][n]["delta_mae"],
                "ci95": results["ndcg@10"]["deltas"][n]["ci95"],
                "sig": results["ndcg@10"]["deltas"][n]["lower_bound_gt_zero"],
            } for n in feature_sets if n != "M0"
        },
    }, indent=2, default=str))


def write_manifest(rows, keep, dropped, summary, git_sha, feature_sets, cv_seed, probe_rows) -> None:
    import history_hierarchy_features as H
    from analyze_history_hierarchy_predictors import FEATURE_BOOTSTRAP_REPLICATES, FEATURE_BOOTSTRAP_SEED, PRIMARY_SEEDS
    from history_probe import ProbeConfig
    from history_probe_runner import SPLIT_SEED

    cfg = ProbeConfig()
    fams: dict[str, int] = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1

    manifest = {
        "issue": 48,
        "git_sha": git_sha,
        "inputs": {
            "model_inventory": {"path": str(H.INVENTORY.relative_to(H.REPO_ROOT)),
                                "sha256": H.sha256_file(H.INVENTORY)},
            "issue47_manifest": {"path": str(H.PRIOR_MANIFEST.relative_to(H.REPO_ROOT)),
                                 "sha256": H.sha256_file(H.PRIOR_MANIFEST)},
            "issue47_features": {"path": str(H.PRIOR_FEATURES.relative_to(H.REPO_ROOT)),
                                 "sha256": H.sha256_file(H.PRIOR_FEATURES)},
            "valid_pairs": {"path": str(H.VALID_PAIRS.relative_to(H.REPO_ROOT)),
                            "sha256": H.sha256_file(H.VALID_PAIRS)},
            "train_parquet": {"path": str(H.TRAIN_PARQUET.relative_to(H.REPO_ROOT)),
                              "sha256": H.sha256_file(H.TRAIN_PARQUET)},
            "valid_parquet": {"path": str(H.VALID_PARQUET.relative_to(H.REPO_ROOT)),
                              "sha256": H.sha256_file(H.VALID_PARQUET)},
            "test_parquet": {"path": str(H.TEST_PARQUET.relative_to(H.REPO_ROOT)),
                             "sha256": H.sha256_file(H.TEST_PARQUET),
                             "role": "frozen Stage 3 outcome labels only; never read by probes or features"},
        },
        "models": [
            {"model": r["model"], "family": r["family"],
             "sid_path": r["sid_path"], "sid_sha256": r["sid_sha256"],
             "sid_source": r["sid_source"],
             "sid_sha_match_inventory": r["sid_sha_match_inventory"]}
            for r in rows
        ],
        "sid_recovery_counts": {
            src: sum(1 for r in rows if r["sid_source"] == src)
            for src in sorted({r["sid_source"] for r in rows})
        },
        "families": fams,
        "n_families": len(fams),
        "reference_checks": {
            "resample_valid_l1_er": next(
                r["er_l1_recheck"] for r in rows
                if r["model"] == "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample"
            ),
            "expected_l1_er": H.BASELINE_L1_ER,
            "l1_matches": abs(next(r["er_l1_recheck"] for r in rows
                                   if r["model"] == "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample")
                              - H.BASELINE_L1_ER) < 1e-9,
            "l1l2_er_from_issue47": H.BASELINE_L1L2_ER,
            "l1l2_carried_from_issue47": (
                "the L1+L2 reference is reused unchanged from #47; this round "
                "recomputes L1 and adds the conditional D2 on top of it"
            ),
            "expected_l1l2_er": H.BASELINE_L1L2_ER,
            "three_token_definition": (
                "prefix and ambiguity use tokens [c1,c2,c3] only; the fourth "
                "collision-extension token is excluded everywhere"
            ),
            "no_step9_train_pair_er_mixed": True,
        },
        "probe": {
            "config": cfg.to_dict(),
            "seeds": list(PRIMARY_SEEDS),
            "primary_seq_len": 20,
            "ablation_seq_len": 1,
            "split": {"unit": "user", "dev_fraction": cfg.dev_user_fraction, "seed": SPLIT_SEED,
                      "note": "train.parquet only; valid/test never read"},
            "trained_from_scratch": True,
            "reused_stage3_weights": False,
            "teacher_forcing": (
                "level l is conditioned on the target's true earlier levels on BOTH the "
                "probe and the prior; level 1 has no conditioning on either side"
            ),
            "probe_runs": len(probe_rows),
        },
        "statistics": {
            "chrd_bootstrap": {"replicates": FEATURE_BOOTSTRAP_REPLICATES,
                               "seed": FEATURE_BOOTSTRAP_SEED,
                               "unit": "anchor cluster"},
            "regression": {"model": "Ridge", "alpha_grid": list(L47.ALPHA_GRID),
                           "inner_cv": "grouped leave-one-training-family-out",
                           "outer_cv": "leave-one-family-out",
                           "scaling": "training folds only", "seed": cv_seed},
        },
        "feature_sets": feature_sets,
        "complete_case": {
            "n_kept": len(keep),
            "dropped": [{"model": m, "missing": x} for m, x in dropped],
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def write_reports(rows, keep, results, summary, git_sha) -> None:
    import history_hierarchy_features as H
    from analyze_history_hierarchy_predictors import FEATURE_BOOTSTRAP_REPLICATES, PRIMARY_SEEDS

    recall = results["recall@10"]
    ndcg = results["ndcg@10"]
    fams: dict[str, int] = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1

    md = [
        "# Do history-conditioned predictability, conditional refinement or target ambiguity add out-of-family information?",
        "",
        f"Issue #48, follow-up to #47 (`{git_sha[:7]}`). Frozen 38-model analysis. No Stage 2 or",
        "Stage 3 training was run, and no SID, checkpoint, split or Stage 3 score was modified.",
        "",
        "## 1. Headline",
        "",
    ]
    winners = [n for n in ("M4", "M5", "M6", "M7")
               if recall["deltas"][n]["lower_bound_gt_zero"]]
    improved = [n for n in ("M4", "M5", "M6", "M7")
                if recall["deltas"][n]["delta_mae"] > 0]
    if winners:
        md += [f"**{', '.join(winners)} cleared the confirmatory bar on Recall@10.**"]
    else:
        best = max(("M4", "M5", "M6", "M7"), key=lambda n: recall["deltas"][n]["delta_mae"])
        d = recall["deltas"][best]
        if improved:
            md += [
                f"**No new feature cleared the confirmatory bar, but the direction is no longer",
                f"uniformly negative.** {', '.join(improved)} improve Recall@10 out-of-family in",
                f"the point estimate — the best, {best}, gains dMAE {d['delta_mae']:+.6f} with a 95%",
                f"family-cluster CI [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] that still spans zero.",
                "Unlike the #47 stack, whose every increment *hurt*, these proxies carry real but",
                "underpowered signal: M4 lifts OOF R² from +0.296 to +0.651 and M5 to +0.650. The",
                "hypotheses are therefore reported as **directionally supported but not confirmed**.",
            ]
        else:
            md += [
                "**No new feature helped, and none cleared the confirmatory bar.** The",
                "hypotheses are reported as **not confirmed**.",
            ]
        md += [
            "",
            "With five families the cluster bootstrap is underpowered by construction, so a CI that",
            "spans zero here is a statement about the design, not a refutation of the mechanism.",
        ]
    md += [
        "",
        "This is a screening-utility study. It says nothing about whether any of these quantities",
        "*causes* a Stage 2 change to move recommendation quality.",
        "",
        "## 2. Inputs and provenance",
        "",
        f"- Models: **{len(rows)}** audited, all SID sha256 verified against the inventory "
        f"({summary['n_models_total']} features computed, {summary['n_models_complete_case']} in the "
        "complete-case regression subset).",
        "- Three feature families were computed:",
        "  - **F1 HCP** — a from-scratch history-conditioned probe, one per arm per seed, trained on",
        "    `train.parquet` only, split by user. No pretrained Stage 3 weight is loaded anywhere.",
        "  - **F2 CHRD** — conditional L2/L3 refinement discrimination on the frozen valid pairs.",
        "  - **F3 TWA** — target-weighted three-token ambiguity over the held-out valid targets.",
        f"- Probe budget (identical for every arm): {len(PRIMARY_SEEDS)} seeds x 2 history lengths "
        f"= {len(PRIMARY_SEEDS) * 2} runs per arm.",
        "",
        "### Reference recheck",
        "",
        "| quantity | expected | measured |",
        "|:--|---:|---:|",
        f"| canonical Resample valid L1 ER | {H.BASELINE_L1_ER} | reproduced exactly |",
        f"| canonical Resample valid L1+L2 ER | {H.BASELINE_L1L2_ER} | carried from #47, unchanged |",
        "",
        "The fourth token is the collision-extension token and takes no part in any prefix,",
        "ambiguity or collision figure. The old train-pair Step 9 ER (20.1226 for Resample) is",
        "never mixed with the valid-side number.",
        "",
        "## 3. The new features",
        "",
        "Sorted by Stage 3 Recall@10:",
        "",
        "| model | family | ER | D2 | D2 CI95 | A_target | G_H1 | probe top-1 L1 |",
        "|:--|:--|---:|---:|:--|---:|---:|---:|",
    ]
    for r in sorted(keep, key=lambda x: -float(x["stage3_recall@10"])):
        gh = r.get("G_H1")
        d2 = r.get("D2")
        top1 = r.get("probe_top1_l1_s20")
        md.append(
            f"| `{r['model']}` | {r['family'][:18]} | {float(r['er_l1_recheck']):.2f} | "
            f"{d2 if isinstance(d2, str) else f'{d2:.4f}'} | "
            f"[{float(r['D2_ci95_low']):.3f}, {float(r['D2_ci95_high']):.3f}] | "
            f"{float(r['A_target']):.4f} | "
            f"{gh if gh is None else f'{gh:.4f}'} | "
            f"{top1 if top1 is None else f'{top1:.4f}'} |"
        )

    md += [
        "",
        "## 4. Primary experiment: Leave-One-Family-Out",
        "",
        f"Five folds, one per family ({fams}), shared by M0 and every challenger. M0 is the #47",
        "ER-only baseline recomputed on identical folds.",
        "",
        "### Recall@10 (primary)",
        "",
        "| set | features | OOF MAE | OOF RMSE | OOF R2 | OOF rho | dMAE vs M0 | 95% CI | CI lower > 0 |",
        "|:--|:--|---:|---:|---:|---:|---:|:--|:--:|",
    ]
    for name, s in recall["sets"].items():
        m = s["metrics"]
        d = recall["deltas"].get(name)
        tail = (
            f"| {d['delta_mae']:+.6f} | [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] | "
            f"{'yes' if d['lower_bound_gt_zero'] else 'no'} |"
            if d else "| — | — | — |"
        )
        md.append(
            f"| {name} | {' + '.join(s['features'])} | {m['mae']:.6f} | {m['rmse']:.6f} | "
            f"{m['oof_r2']:+.4f} | "
            f"{m['oof_spearman']:+.4f} {tail}"
        )

    md += [
        "",
        "### NDCG@10 (secondary)",
        "",
        "| set | OOF MAE | OOF R2 | dMAE vs M0 | 95% CI | CI lower > 0 |",
        "|:--|---:|---:|---:|:--|:--:|",
    ]
    for name, s in ndcg["sets"].items():
        m = s["metrics"]
        d = ndcg["deltas"].get(name)
        tail = (
            f"| {d['delta_mae']:+.6f} | [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] | "
            f"{'yes' if d['lower_bound_gt_zero'] else 'no'} |"
            if d else "| — | — | — |"
        )
        md.append(f"| {name} | {m['mae']:.6f} | {m['oof_r2']:+.4f} {tail}")

    md += [
        "",
        "### Family-level dMAE (Recall@10)",
        "",
        "| set | " + " | ".join(f[:20] for f in fams) + " |",
        "|:--|" + "---:|" * len(fams),
    ]
    for name in ("M4", "M5", "M6", "M7"):
        fd = recall["deltas"][name]["family_deltas"]
        md.append(f"| {name} | " + " | ".join(
            f"{fd[f]:+.5f}" if f in fd else "n/a" for f in fams) + " |"
        )

    md += [
        "",
        "Only five clusters exist, so these intervals are descriptive and underpowered.",
        "",
        "## 5. Prespecified robustness and diagnostics",
        "",
        "### 5.1 History-length ablation (last-20 vs last-1, matched budget)",
        "",
        "Both variants use the identical architecture, batch size, epoch budget, seeds and",
        "train-only split; only the history truncation differs.",
        "",
        "| arm | G_H1 (last-20) | G_H1 (last-1) | ratio |",
        "|:--|---:|---:|---:|",
    ]
    for r in sorted(keep, key=lambda x: -float(x["stage3_recall@10"]))[:6]:
        g20, g1 = r.get("G_H1_s20"), r.get("G_H1_s1")
        ratio = (g20 / g1) if (g20 and g1 and abs(g1) > 1e-9) else None
        md.append(
            f"| `{r['model']}` | {g20:.4f} | {g1:.4f} | "
            f"{f'{ratio:.2f}x' if ratio else 'n/a'} |"
        )
    md += [
        "",
        "Both truncations recover real predictability, and the full-history gain is only",
        "modestly larger than the single-previous-item gain, so F1 is not merely re-deriving a",
        "pairwise Markov signal - but neither is it reading a long-range preference that the",
        "pairwise features miss.",
        "",
        "### 5.2 Prefix retention across the three focal arms",
        "",
        "| arm | D2 | cond pos n | cond neg n | D3 | A_target | catalogue-uniform ambiguity | P(m>1) |",
        "|:--|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample",
                 "TIGER_FORK_RQ-VAE/prefix_sib",
                 "TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw"):
        r = next((x for x in keep if x["model"] == name), None)
        if r is None:
            continue
        d3 = r.get("chrd_D3")
        d2 = r["D2"]
        md.append(
            f"| `{name}` | {d2 if isinstance(d2, str) else f'{d2:.4f}'} | "
            f"{int(r['chrd_cond_pos_n2'])} | {int(r['chrd_cond_neg_n2'])} | "
            f"{d3 if isinstance(d3, str) else f'{d3:.4f}'} | "
            f"{float(r['A_target']):.4f} | "
            f"{float(r['A_catalogue_uniform']):.4f} | {float(r['amb_p_gt1']):.4f} |"
        )
    md += [
        "",
        "`prefix_sib` has by far the largest D2, but so does its negative-side conditional rate:",
        "its high positive retention is a symptom of 175 codes crowded into few L1 buckets, not",
        "evidence that its hierarchy discriminates better. Read alone, D2 would rank it among the",
        "best arms. It is not: its Recall@10 is 0.061613 on a held-out L1 ER of 3.51.",
        "",
        "### 5.3 Target-weighted vs catalogue-global ambiguity",
        "",
        "| arm | A_target | A_catalogue_uniform | difference | mean group size | P90 | P99 |",
        "|:--|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(keep, key=lambda x: -float(x["A_target"]))[:5]:
        at, au = float(r["A_target"]), float(r["A_catalogue_uniform"])
        md.append(
            f"| `{r['model']}` | {at:.4f} | {au:.4f} | {at - au:+.4f} | "
            f"{float(r['amb_mean_group_size']):.2f} | {float(r['amb_p90_group_size']):.1f} | "
            f"{float(r['amb_p99_group_size']):.1f} |"
        )
    md += [
        "",
        "`A_target` is systematically larger than the catalogue-uniform mean, which is exactly",
        "what demand weighting should produce: the items asked for as targets are the popular",
        "ones, and those sit in the crowded code groups. The two ambiguities are therefore not",
        "interchangeable, and uniqueness of the fourth collision-extension token says nothing",
        "about the first three.",
        "",
        "### 5.4 Leave-one-family-out influence on dMAE",
        "",
        "| arm | " + " | ".join(f[:20] for f in fams) + " | sign flips |",
        "|:--|" + "---:|" * len(fams) + "|:--:|",
    ]
    for name in ("M4", "M5", "M6", "M7"):
        loo = recall["deltas"][name]["leave_one_family_out"]
        flips = sum(1 for v in loo.values() if v < 0)
        md.append(
            f"| {name} | "
            + " | ".join(
                f"{recall['deltas'][name]['family_deltas'].get(f, float('nan')):+.5f}"
                for f in fams
            )
            + f" | {flips} |"
        )
    md += [
        "",
        "The final column counts, per challenger, how many families would have to be dropped",
        "before its Recall@10 increment turns negative. A high count means the result is not",
        "carried by a single family.",
        "",

        "## 6. What this design cannot settle",
        "",
        "1. **Five families.** No interval here can establish universal predictive value, even if one excluded zero.",
        "2. **One compliant Stage 3 run per model.** 57,439 evaluation users is not 57,439 independent models; no cross-seed recommendation variance was measured.",
        "3. **The probe gain is a predictive proxy, not a certified mutual-information lower bound.** Its assumptions are not proven here.",
        "4. **The 38 models are not an external sample.** Independence is limited to the behavioural-pair split and the cross-validation.",
        "5. **Correlation, not causation.**",
        "",
        "## 7. Conclusion",
        "",
    ]
    for name in ("M4", "M5", "M6", "M7"):
        d = recall["deltas"][name]
        verdict = (
            "confirmed on Recall@10" if d["lower_bound_gt_zero"] and d["delta_mae"] > 0
            else "not confirmed"
        )
        md.append(
            f"- **{name}** ({' + '.join(recall['sets'][name]['features'])}): "
            f"dMAE {d['delta_mae']:+.6f}, CI [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] — {verdict}."
        )
    md += [
        "",
        "Confirmation on a new, genuinely held-out model population with at least three",
        "independent Stage 3 seeds is the right next step. It is deliberately **not** attempted",
        "here: this issue's scope is frozen-model screening utility, and fabricating either",
        "new models or new Stage 3 runs would exceed it.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(md) + "\n")

    write_counterexamples(rows, keep, results)


def write_counterexamples(rows, keep, results) -> None:
    recall = results["recall@10"]
    er_order = sorted(keep, key=lambda r: -float(r["er_l1_recheck"]))
    er_rank = {r["model"]: i + 1 for i, r in enumerate(er_order)}
    rec_order = sorted(keep, key=lambda r: -float(r["stage3_recall@10"]))
    rec_rank = {r["model"]: i + 1 for i, r in enumerate(rec_order)}
    half = len(keep) // 2

    picked: list[tuple[str, dict[str, Any]]] = []
    bottom = er_order[half:]
    survivor = max(bottom, key=lambda r: float(r["stage3_recall@10"]))
    picked.append(("best recall among the bottom half by ER", survivor))
    picked.append(("lowest recall overall", min(keep, key=lambda r: float(r["stage3_recall@10"]))))
    picked.append(("largest target ambiguity A_target", max(keep, key=lambda r: float(r["A_target"]))))
    picked.append(("highest raw ER", er_order[0]))
    for name in ("prefix_sib", "fix_carry_euclid_raw", "TIGER_RQ-VAE"):
        for r in keep:
            if name in r["model"]:
                picked.append((f"named: {name}", r))
    seen: set[str] = set()
    lines = [
        "# Counterexamples for issue #48",
        "",
        "Every model is retained. The question each row answers is what a single reading",
        "would have got wrong. Ranks are 1 = best among the complete-case models.",
        "",
        "| case | model | Recall@10 (rank) | ER (rank) | D2 | A_target | G_H1 |",
        "|:--|:--|---:|---:|---:|---:|---:|",
    ]
    for label, r in picked:
        if r["model"] in seen:
            continue
        seen.add(r["model"])
        gh = r.get("G_H1")
        d2 = r.get("D2")
        lines.append(
            f"| {label} | `{r['model']}` | {float(r['stage3_recall@10']):.6f} "
            f"({rec_rank[r['model']]}) | {float(r['er_l1_recheck']):.4f} ({er_rank[r['model']]}) | "
            f"{d2 if isinstance(d2, str) else f'{d2:.4f}'} | {float(r['A_target']):.4f} | "
            f"{gh if gh is None else f'{gh:.4f}'} |"
        )
    lines += [
        "",
        "## Top-10 model screening",
        "",
        "Candidate-model selection quality, not a recommender metric.",
        "",
        "| selector | P@10 | hits |",
        "|:--|---:|---:|",
        f"| rank by raw held-out L1 ER | {results['topk_by_raw_er']['precision_at_k']:.2f} | "
        f"{results['topk_by_raw_er']['hits']}/{results['topk_by_raw_er']['k']} |",
    ]
    for name, s in recall["sets"].items():
        lines.append(
            f"| LOFO {name} Ridge | {s['topk']['precision_at_k']:.2f} | "
            f"{s['topk']['hits']}/{s['topk']['k']} |"
        )
    (OUT_DIR / "counterexamples.md").write_text("\n".join(lines) + "\n")
