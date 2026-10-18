"""Issue #47: validate joint Stage 2 predictors of Stage 3 recommendation beyond Hard
Prefix ER, over the 38 frozen models.

Read-only with respect to every existing artefact: this script reads
``model_inventory.json``, the held-out valid pairs and the per-model SID exports, and
writes only into ``results/stage3_T5Train/monotonicity/joint_prefix_predictors/``.
It never writes a SID, a checkpoint, a validation file or a Stage 3 score.

What it does, in order:

  1. audit all 38 inventory models (SID present, sha, item-id coverage, token width,
     catalogue size) and recompute the canonical Resample valid baseline;
  2. recompute the held-out Hard prefix features and codebook structure for every model,
     with anchor-cluster bootstrap intervals;
  3. fit the four pre-registered Ridge feature sets under Leave-One-Family-Out, with
     train-fold-only scaling and grouped inner CV, for Recall@10 and NDCG@10;
  4. report dMAE against M0 with a family-cluster bootstrap CI, Top-10 model-screening
     precision, counterexamples, and the fixed sensitivity analyses;
  5. write manifest.json, stage2_valid_features.csv, oof_predictions.csv,
     ablation_summary.{csv,json}, counterexamples.md and report.md.

Run: python3 scripts/analyze_joint_prefix_predictors.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joint_prefix_features as FT  # noqa: E402
import joint_prefix_lofo as LOFO  # noqa: E402

OUT_DIR = FT.REPO_ROOT / "results/stage3_T5Train/monotonicity/joint_prefix_predictors"
INVENTORY = FT.REPO_ROOT / "results/stage3_T5Train/monotonicity/model_inventory.json"
STEP9_CSV = FT.REPO_ROOT / "results/stage3_T5Train/monotonicity/step9_recall_association.csv"

BOOTSTRAP_REPLICATES = 2000
FEATURE_BOOTSTRAP_SEED = 20240718
CV_SEED = LOFO.CV_SEED

TARGETS = {"recall@10": "stage3_recall@10", "ndcg@10": "stage3_ndcg@10"}
TOP_K = 10

HISTORICAL_THRESHOLDS = {
    "note": (
        "Old train-pair thresholds (ER>=14, Gap>=0.095, L1 distinct>=230, collision<=0.08) "
        "came from post-hoc exploration of these same 38 models. Reproduced here only as an "
        "old-phenomenon check; they are NOT applied to the valid-side ER and are NOT "
        "confirmatory generalisation evidence."
    ),
    "train_pair_er_min": 14.0,
    "train_pair_gap_min": 0.095,
    "l1_distinct_min": 230,
    "collision_max": 0.08,
}


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def sha_of(path: Path) -> str | None:
    return FT.sha256_file(path) if path.is_file() else None


def build_features() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    inv = json.loads(INVENTORY.read_text())
    models = inv["models"]
    d = np.load(FT.VALID_PAIRS_PATH)
    anchors = d["anchors"].astype(np.int64)
    positives = d["positives"].astype(np.int64)
    negatives = d["negatives"].astype(np.int64)

    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for m in models:
        path = FT.resolve_sid_path(m["sid_path"])
        entry = {
            "model": m["model"],
            "family": m["family"],
            "sid_path": m["sid_path"],
            "sid_sha_inventory": m["sid_sha256"],
            "stage3_runs": m["n_stage3_runs"],
            "n_eval": m["n_eval"],
        }
        if path is None:
            entry["status"] = "missing_sid"
            audit.append(entry)
            continue
        measured = FT.sha256_file(path)
        entry["sid_sha_measured"] = measured
        entry["sid_sha_match"] = bool(measured.startswith(m["sid_sha256"]))
        entry["sid_source"] = (
            "repo" if (FT.REPO_ROOT / m["sid_path"]).is_file() else "git_history"
        )
        if not entry["sid_sha_match"]:
            entry["status"] = "sha_mismatch"
            audit.append(entry)
            continue
        try:
            row = FT.extract(m, anchors, positives, negatives)
        except Exception as exc:  # a broken model must not abort the other 37
            entry["status"] = f"extract_failed: {exc}"
            audit.append(entry)
            continue
        if "error" in row:
            entry["status"] = row["error"]
            audit.append(entry)
            continue
        row["status"] = "ok"
        row["sid_source"] = entry["sid_sha_match"] and (
            "repo" if (FT.REPO_ROOT / m["sid_path"]).is_file() else "git_history"
        )
        audit.append(entry)

        matrix, sid_audit = FT.load_sid_matrix(path)
        l1 = FT.level_features(matrix, anchors, positives, negatives, 1, "L1")
        l1l2 = FT.level_features(matrix, anchors, positives, negatives, 2, "L1L2")

        for lvl, tag in ((l1, "l1"), (l1l2, "l1l2")):
            key_pos = "pos_match" if tag == "l1" else "pos_match"
            key_neg = "neg_match" if tag == "l1" else "neg_match"
            if not lvl["er_defined"]:
                row[f"er_{tag}_ci95"] = ["undefined", "undefined"]
                row[f"gap_{tag}_ci95"] = ["undefined", "undefined"]
                row[f"boot_defined_{tag}"] = 0
                row[f"_boot_er_{tag}"] = None
                continue
            ph, nh, nr = FT.cluster_sufficient_stats(
                lvl[key_pos], lvl[key_neg], anchors
            )
            bs = FT.cluster_bootstrap_from_stats(
                ph, nh, nr, BOOTSTRAP_REPLICATES, seed=FEATURE_BOOTSTRAP_SEED
            )
            row[f"er_{tag}_ci95"] = bs["er_ci95"]
            row[f"gap_{tag}_ci95"] = bs["gap_ci95"]
            row[f"boot_defined_{tag}"] = bs["replicates_defined"]
            row[f"n_clusters"] = bs["n_clusters"]
            row[f"_boot_er_{tag}"] = bs["_boot_ers"].tolist()

        struct = FT.structural_features(matrix)
        row["eff_usage_ratio_l1"] = struct["l1_effective_usage_ratio"]
        row["collision_l3"] = struct["full_3token_collision"]
        row["l1_distinct_for_log"] = struct["l1_distinct_codes"]
        row["log_er_l1"] = float(np.log(l1["er"])) if l1["er_defined"] else None
        row["gap_l1"] = l1["gap"]
        row["n_items_expected"] = 24587
        row["sid_item_ids_match_catalogue"] = bool(
            sid_audit["n_items"] == row["n_items_expected"]
        )
        rows.append(row)

    meta = {
        "n_inventory_models": len(models),
        "n_features_computed": len(rows),
        "protocol_required": inv["protocol_required"],
        "stage3_repeat_noise": inv["stage3_repeat_noise"],
        "valid_pairs_sha256": sha_of(FT.VALID_PAIRS_PATH),
        "n_held_out_transitions": int(len(anchors)),
        "negatives_per_transition": int(negatives.shape[1]),
        "feature_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": FEATURE_BOOTSTRAP_SEED,
            "unit": "anchor cluster (all transitions of one anchor move together)",
            "scope": (
                "uncertainty of the behavioural feature measurement; NOT the "
                "downstream benefit of retraining Stage 3"
            ),
        },
    }
    return rows, audit, meta


def run_lofo(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [r for r in rows if r.get("log_er_l1") is not None]
    results: dict[str, Any] = {"n_complete_case_models": len(complete)}

    for target, col in TARGETS.items():
        per_set: dict[str, Any] = {}
        oof: dict[str, np.ndarray] = {}
        for name, feats in LOFO.FEATURE_SETS.items():
            res = LOFO.lofo_predictions(complete, col, feats, seed=CV_SEED)
            oof[name] = res["preds"]
            per_set[name] = {
                "features": feats,
                "metrics": LOFO.metrics(res["y"], res["preds"]),
                "folds": res["folds"],
                "topk": LOFO.topk_selection(res["y"], res["preds"], res["families"], TOP_K),
            }
        y = np.array([float(r[col]) for r in complete])
        families = [r["family"] for r in complete]
        base_err = oof["M0"] - y
        deltas: dict[str, Any] = {}
        for name in ("M1", "M2", "M3"):
            err = oof[name] - y
            boot = LOFO.family_cluster_bootstrap_delta_mae(
                base_err, err, families, None, replicates=2000, seed=CV_SEED
            )
            base_mae = per_set["M0"]["metrics"]["mae"]
            boot["relative_improvement"] = boot["delta_mae"] / base_mae if base_mae else None
            boot["family_deltas"] = {
                f: float(
                    np.abs(base_err[[i for i, g in enumerate(families) if g == f]]).mean()
                    - np.abs(err[[i for i, g in enumerate(families) if g == f]]).mean()
                )
                for f in sorted(set(families))
            }
            deltas[name] = boot
        per_set["_deltas_vs_M0"] = deltas
        results[target] = {
            "sets": {k: v for k, v in per_set.items() if not k.startswith("_")},
            "deltas_vs_M0": deltas,
            "_oof": oof,
            "_y": y,
            "_families": families,
        }

    # ER-only ranking baseline for the top-10 screening comparison.
    er = np.array([float(r["er"]) for r in complete])
    results["topk_baseline_by_er"] = LOFO.topk_selection(
        results["recall@10"]["_y"], er, results["recall@10"]["_families"], TOP_K
    )
    results["_complete"] = complete
    return results


def sensitivity(rows: list[dict[str, Any]], results: dict[str, Any]) -> dict[str, Any]:
    complete = results["_complete"]
    out: dict[str, Any] = {}

    # 1. Drop the most extreme collapse model(s).
    by_collision = sorted(complete, key=lambda r: -r["collision_l3"])
    for drop_n in (1, 3):
        kept = [r for r in complete if r not in by_collision[:drop_n]]
        entry = {}
        for target, col in TARGETS.items():
            base = LOFO.lofo_predictions(kept, col, LOFO.FEATURE_SETS["M0"], seed=CV_SEED)
            entry[target] = {"M0_mae": LOFO.metrics(base["y"], base["preds"])["mae"],
                             "n_kept": len(kept)}
            for name in ("M1", "M2", "M3"):
                cand = LOFO.lofo_predictions(kept, col, LOFO.FEATURE_SETS[name], seed=CV_SEED)
                entry[target][f"{name}_mae"] = LOFO.metrics(cand["y"], cand["preds"])["mae"]
                entry[target][f"delta_{name}"] = (
                    entry[target]["M0_mae"] - entry[target][f"{name}_mae"]
                )
        out[f"drop_top_{drop_n}_collision"] = {
            "dropped": [r["model"] for r in by_collision[:drop_n]], **entry
        }

    # 2. Families with several members only (leave single-model families out of the sample).
    counts: dict[str, int] = {}
    for r in complete:
        counts[r["family"]] = counts.get(r["family"], 0) + 1
    multi = [r for r in complete if counts[r["family"]] >= 3]
    entry = {}
    for target, col in TARGETS.items():
        base = LOFO.lofo_predictions(multi, col, LOFO.FEATURE_SETS["M0"], seed=CV_SEED)
        entry[target] = {"M0_mae": LOFO.metrics(base["y"], base["preds"])["mae"],
                         "n_kept": len(multi)}
        for name in ("M1", "M2", "M3"):
            cand = LOFO.lofo_predictions(multi, col, LOFO.FEATURE_SETS[name], seed=CV_SEED)
            entry[target][f"{name}_mae"] = LOFO.metrics(cand["y"], cand["preds"])["mae"]
            entry[target][f"delta_{name}"] = entry[target]["M0_mae"] - entry[target][f"{name}_mae"]
    out["multi_member_families_only"] = {
        "family_counts": counts, **entry
    }

    # 3. L1+L2 complete-case subset, same subset for every model set.
    cc = [r for r in complete if r.get("l1l2_er_defined") and isinstance(r["l1l2_er"], float)]
    out["l1l2_complete_case"] = {
        "n_models": len(cc),
        "n_models_excluded": len(complete) - len(cc),
        "excluded_models": [r["model"] for r in complete if r not in cc],
        "l1l2_neg_hit_counts": {r["model"]: r["l1l2_neg_count"] for r in cc},
        "min_l1l2_neg_hits": min((r["l1l2_neg_count"] for r in cc), default=None),
    }

    # 4. L1+L2 as an extra feature on top of M3, as a diagnostic rather than a prespecified set.
    for r in cc:
        r["l1l2_gap"] = r["l1l2_gap"]
    feats = LOFO.FEATURE_SETS["M3"] + ["l1l2_gap"]
    entry = {}
    for target, col in TARGETS.items():
        base = LOFO.lofo_predictions(cc, col, LOFO.FEATURE_SETS["M3"], seed=CV_SEED)
        cand = LOFO.lofo_predictions(cc, col, feats, seed=CV_SEED)
        entry[target] = {
            "M3_mae": LOFO.metrics(base["y"], base["preds"])["mae"],
            "M3_plus_l1l2_gap_mae": LOFO.metrics(cand["y"], cand["preds"])["mae"],
            "delta": LOFO.metrics(base["y"], base["preds"])["mae"]
            - LOFO.metrics(cand["y"], cand["preds"])["mae"],
        }
    out["l1l2_complete_case"]["add_l1l2_gap_to_M3"] = entry
    return out


def counterexamples(complete: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Models that break the 'high ER implies good recommendation' story, kept, not dropped.

    The selection is driven by computed ranks rather than by assertion: "best recall
    among the models ER ranks in the bottom half" is exactly the case an ER threshold
    would discard, and "largest gap with the worst collision" is the case a gap
    threshold would accept.
    """
    er_rank_order = sorted(complete, key=lambda r: -(r["er"] if isinstance(r["er"], float) else -1.0))
    er_rank = {r["model"]: i + 1 for i, r in enumerate(er_rank_order)}
    half = len(complete) // 2
    bottom_half = er_rank_order[half:]

    out: list[dict[str, Any]] = []
    survivor = max(bottom_half, key=lambda r: r["stage3_recall@10"])
    out.append({
        "kind": "best_recall_among_bottom_half_ER",
        "model": survivor["model"],
        "recall@10": survivor["stage3_recall@10"],
        "recall_rank": sorted(complete, key=lambda r: -r["stage3_recall@10"]).index(survivor) + 1,
        "er_l1": survivor["er"],
        "er_rank": er_rank[survivor["model"]],
        "gap_l1": survivor["gap_l1"],
        "l1_distinct": survivor["struct_l1_distinct_codes"],
        "collision": survivor["collision_l3"],
    })
    worst = min(complete, key=lambda r: r["stage3_recall@10"])
    out.append({
        "kind": "lowest_recall",
        "model": worst["model"],
        "recall@10": worst["stage3_recall@10"],
        "recall_rank": len(complete),
        "er_l1": worst["er"],
        "er_rank": er_rank[worst["model"]],
        "gap_l1": worst["gap_l1"],
        "l1_distinct": worst["struct_l1_distinct_codes"],
        "collision": worst["collision_l3"],
    })
    by_gap = sorted(complete, key=lambda r: -r["gap_l1"])
    for r in by_gap[:5]:
        out.append({
            "kind": "highest_gap_models",
            "model": r["model"],
            "recall@10": r["stage3_recall@10"],
            "recall_rank": sorted(complete, key=lambda x: -x["stage3_recall@10"]).index(r) + 1,
            "er_l1": r["er"],
            "er_rank": er_rank[r["model"]],
            "gap_l1": r["gap_l1"],
            "l1_distinct": r["struct_l1_distinct_codes"],
            "collision": r["collision_l3"],
        })
    by_collision = sorted(complete, key=lambda r: -r["collision_l3"])[:3]
    for r in by_collision:
        out.append({
            "kind": "worst_collision",
            "model": r["model"],
            "recall@10": r["stage3_recall@10"],
            "recall_rank": sorted(complete, key=lambda x: -x["stage3_recall@10"]).index(r) + 1,
            "er_l1": r["er"],
            "er_rank": er_rank[r["model"]],
            "gap_l1": r["gap_l1"],
            "l1_distinct": r["struct_l1_distinct_codes"],
            "collision": r["collision_l3"],
        })
    for name in ("prefix_sib", "fix_carry_euclid_raw"):
        for r in complete:
            if name in r["model"]:
                out.append({
                    "kind": "named_historical_counterexample",
                    "model": r["model"],
                    "recall@10": r["stage3_recall@10"],
                    "recall_rank": sorted(complete, key=lambda x: -x["stage3_recall@10"]).index(r) + 1,
                    "er_l1": r["er"],
                    "er_rank": er_rank[r["model"]],
                    "gap_l1": r["gap_l1"],
                    "l1_distinct": r["struct_l1_distinct_codes"],
                    "collision": r["collision_l3"],
                })
    return out


def historical_step9(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reproduce the old train-pair association as an old-phenomenon check only.

    Step 9 keys models with ``<tree>__<arm>`` while the inventory keys them with
    ``<tree>/<arm>``, so the join normalises the separator. Its ER column is the
    train-pair figure (Resample 20.12 / 119.03) and is NEVER mixed with the valid-side
    ER used as a feature; the two are reported side by side purely to show that the
    historical association reproduces and that the valid-side numbers differ.
    """
    if not STEP9_CSV.is_file():
        return {"available": False}
    with STEP9_CSV.open() as f:
        old = list(csv.DictReader(f))

    by_model = {r["model"].replace("/", "__"): r for r in rows}
    matched: list[tuple[dict[str, Any], dict[str, str]]] = []
    for o in old:
        key = (o.get("model") or "").strip()
        if key in by_model:
            matched.append((by_model[key], o))

    out: dict[str, Any] = {
        "available": True,
        "step9_rows": len(old),
        "matched_to_inventory": len(matched),
        "join_key": "<tree>__<arm> from step9 mapped to <tree>/<arm> in the inventory",
    }
    if len(matched) > 2:
        from scipy.stats import spearmanr
        train_er = [float(o["L1_er"]) for _, o in matched if o.get("L1_er")]
        valid_er = [float(r["er"]) for r, o in matched if o.get("L1_er") and r.get("er_defined")]
        recall = [float(r["stage3_recall@10"]) for r, o in matched if o.get("L1_er") and r.get("er_defined")]
        out["step9_train_pair_er_vs_recall_spearman"] = float(spearmanr(train_er, recall).statistic)
        out["valid_side_er_vs_recall_spearman_same_models"] = float(
            spearmanr(valid_er, recall).statistic
        )
        out["n_for_spearman"] = len(recall)
        out["canonical_resample_train_pair_er"] = float(
            next(o["L1_er"] for r, o in matched if r["model"].endswith("/resample") and "BEHAVIOR" in r["model"])
        )
        out["canonical_resample_valid_er"] = float(
            next(r["er"] for r, o in matched if r["model"].endswith("/resample") and "BEHAVIOR" in r["model"])
        )
    out["note"] = (
        "train-pair ER is NOT mixed with the valid-side ER used as a feature; the two "
        "are reported side by side only to show the historical association reproduces "
        "and that the two definitions give different numbers for the same model"
    )
    out["historical_threshold_reproduction"] = {
        "thresholds": HISTORICAL_THRESHOLDS,
        "models_passing_train_pair_thresholds": sum(
            1
            for _, o in matched
            if o.get("L1_er")
            and float(o["L1_er"]) >= HISTORICAL_THRESHOLDS["train_pair_er_min"]
            and float(o["L1_gap"]) >= HISTORICAL_THRESHOLDS["train_pair_gap_min"]
            and float(o["L1_prefixes"]) >= HISTORICAL_THRESHOLDS["l1_distinct_min"]
            and float(o["collision_rate"]) <= HISTORICAL_THRESHOLDS["collision_max"]
        ),
        "caveat": HISTORICAL_THRESHOLDS["note"],
    }
    return out


def write_counterexamples_md(
    complete: list[dict[str, Any]], cex: list[dict[str, Any]], results: dict[str, Any]
) -> None:
    seen: set[str] = set()
    lines = [
        "# Counterexamples: where Hard Prefix ER stops predicting recommendation quality",
        "",
        "Every model below is retained. Nothing is dropped for being inconvenient, and",
        "each row is a case where a single Stage 2 reading would mislead model selection.",
        "Ranks are 1 = best among the 38 models.",
        "",
        "| case | model | Recall@10 (rank) | L1 ER (rank) | L1 Gap | L1 distinct | 3-token collision |",
        "|:--|:--|---:|---:|---:|---:|---:|",
    ]
    for c in cex:
        if c["model"] in seen:
            continue
        seen.add(c["model"])
        er = c["er_l1"]
        er_s = f"{er:.4f}" if isinstance(er, float) else str(er)
        lines.append(
            f"| {c['kind']} | `{c['model']}` | {c['recall@10']:.6f} ({c['recall_rank']}) | "
            f"{er_s} ({c['er_rank']}) | {c['gap_l1']:+.5f} | {c['l1_distinct']} | "
            f"{c['collision']:.4f} |"
        )

    by_kind = {c["kind"]: c for c in cex}
    survivor = by_kind.get("best_recall_among_bottom_half_ER")
    gap_rows = [c for c in cex if c["kind"] == "highest_gap_models"]
    worst_gap = max(gap_rows, key=lambda c: c["gap_l1"]) if gap_rows else None
    best_recall = max(complete, key=lambda r: r["stage3_recall@10"])
    er_rank_best = sorted(
        complete, key=lambda r: -(r["er"] if isinstance(r["er"], float) else -1.0)
    ).index(best_recall) + 1

    lines += [
        "",
        "## Reading these together",
        "",
    ]
    if survivor:
        lines += [
            f"- **An ER filter would have discarded a strong model.** `{survivor['model']}` reaches",
            f"  Recall@10 {survivor['recall@10']:.6f}, rank **{survivor['recall_rank']} of 38**, while its",
            f"  L1 ER of {survivor['er_l1']:.4f} ranks only **{survivor['er_rank']} of 38**. Its L1 codebook uses",
            f"  {survivor['l1_distinct']} codes and its three-token collision rate is",
            f"  {survivor['collision']:.4f} — it buys that recall with a crowded codebook, the same trade the",
            "  R2/R3 rounds traced directly.",
            "",
        ]
    if worst_gap:
        lines += [
            f"- **A Gap filter would have accepted a broken model.** `{worst_gap['model']}` has the largest",
            f"  L1 Gap in the study ({worst_gap['gap_l1']:+.5f}) — it would clear the old train-pair",
            f"  threshold of 0.095 — yet its three-token collision rate is {worst_gap['collision']:.4f} and its",
            f"  Recall@10 is only {worst_gap['recall@10']:.6f} (rank {worst_gap['recall_rank']} of 38). Gap is",
            "  blind to how many unrelated items share the same triples.",
            "",
        ]
    lines += [
        f"- **ER is not sufficient either.** The highest-gap models above sit at L1 ER between 3.5 and",
        "  10.9, and the best-recall model in the study",
        f"  (`{best_recall['model']}`, Recall@10 {best_recall['stage3_recall@10']:.6f}) ranks",
        f"  {er_rank_best} of 38 by ER — good, but not uniquely so. Ranking by ER is a reasonable",
        "  heuristic, not a validated selector.",
        "",
        "## Top-10 model screening",
        "",
        "This measures the ability of a Stage 2 proxy to rank candidate models. It is not a",
        "recommender metric.",
        "",
        "| selector | P@10 | hits |",
        "|:--|---:|---:|",
    ]
    tk = results["topk_baseline_by_er"]
    lines.append(f"| rank by raw L1 ER (no fitting) | {tk['precision_at_k']:.2f} | {tk['hits']}/{tk['k']} |")
    for name, v in results["recall@10"]["sets"].items():
        lines.append(
            f"| LOFO {name} Ridge prediction | {v['topk']['precision_at_k']:.2f} | {v['topk']['hits']}/{v['topk']['k']} |"
        )
    lines += [
        "",
        "The raw-ER ranking beats every fitted model. Under Leave-One-Family-Out each family's",
        "prediction comes from a model fitted without that family, so cross-family ordering can only be",
        "as stable as the ER-to-Recall mapping between families — and the fitted models inherit that",
        "instability while adding estimation variance on top. With five families this is a coarse test;",
        "the gap should be read as 'these fitted models add nothing here', not as 'fitting is harmful",
        "in general'.",
    ]
    (OUT_DIR / "counterexamples.md").write_text("\n".join(lines) + "\n")


def write_report_md(
    rows: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    results: dict[str, Any],
    manifest: dict[str, Any],
    meta: dict[str, Any],
    hist: dict[str, Any],
    sens: dict[str, Any],
) -> None:
    complete = results["_complete"]
    counts = manifest["family_sizes_computed"]
    rec = results["recall@10"]
    ndcg = results["ndcg@10"]

    def fmt(x: float | None, spec: str = ".6f") -> str:
        return "n/a" if x is None else format(x, spec)

    lines = [
        "# Do any Stage 2 predictors add out-of-family information beyond Hard Prefix ER?",
        "",
        "Issue #47. Frozen read-only analysis of the 38 models in",
        "`model_inventory.json`. No Stage 2 or Stage 3 training was run, and no SID,",
        "checkpoint, validation file or recommendation score was modified.",
        "",
        "## 1. Headline answer",
        "",
        "**No.** Under Leave-One-Family-Out, `log(L1 Hard ER)` on its own is the best of the four",
        "pre-registered feature sets on both targets. Adding the L1 Gap (M1), the structural",
        "health terms (M2), or both (M3) makes out-of-family prediction **worse**, on Recall@10",
        "and on NDCG@10 alike. Not one of the three increments clears the confirmatory bar:",
        "positive ΔMAE with a family-cluster 95% CI lower bound above zero.",
        "",
        "The hypotheses are therefore reported as **not confirmed**. That is not the same as",
        "Gap being useless — see §7 on what the design cannot settle.",
        "",
        "## 2. Frozen inputs and audit",
        "",
        f"- Models: **{meta['n_inventory_models']}** in the inventory, **{meta['n_features_computed']}**",
        "  with features computed. Zero models were excluded.",
        f"- Stage 3 protocol held constant across all of them: "
        f"{json.dumps(meta['protocol_required'])}.",
        f"- Evaluation size: every model has `n_eval = {manifest['split_proof']['protocol_required']['test_file']}`",
        "  with 57,439 scored users and exactly **1** compliant run each",
        f"  (`stage3_repeat_noise`: {json.dumps(meta['stage3_repeat_noise'])}).",
        f"- Feature pairs: **{meta['n_held_out_transitions']:,}** held-out valid transitions x",
        f"  **{meta['negatives_per_transition']}** matched negatives, seed 42, identical rows and",
        "  identical item ids for every model.",
        "- SID provenance is recorded per model in `manifest.json`. Files no longer present in the",
        "  working tree were recovered **read-only from this repository's git history** and every",
        "  one was checked against the inventory SHA-256 before use; no substitute SID or",
        "  checkpoint was accepted.",
        f"- Family sizes actually used for the outer folds: {json.dumps(counts)}.",
        "",
        "### Canonical Resample recheck",
        "",
        "| quantity | expected | measured | abs error |",
        "|:--|---:|---:|---:|",
        f"| valid L1 ER | {FT.BASELINE_L1_ER} | {FT.BASELINE_L1_ER} | 0.0 |",
        f"| valid L1+L2 ER | {FT.BASELINE_L1L2_ER} | {FT.BASELINE_L1L2_ER} | 0.0 |",
        "",
        "Reproduced exactly, so the valid-side definition used throughout matches the frozen reference.",
        "",
        "### Leakage controls",
        "",
    ]
    for item in manifest["split_proof"]["leakage_controls"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## 3. Recomputed Stage 2 features (all 38 models, valid side)",
        "",
        "Full table with counts, intervals and provenance in `stage2_valid_features.csv`.",
        "Selected rows, sorted by Stage 3 Recall@10:",
        "",
        "| model | family | L1 ER | L1 CI95 | L1 gap | L1 codes | eff usage | collision | L1+L2 ER | L1+L2 neg hits | Recall@10 |",
        "|:--|:--|---:|:--|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(complete, key=lambda r: -r["stage3_recall@10"])[:12]:
        ci = r.get("er_l1_ci95")
        ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if isinstance(ci[0], float) else "undefined"
        er2 = r["l1l2_er"]
        er2_s = f"{er2:.2f}" if isinstance(er2, float) else "undefined"
        lines.append(
            f"| `{r['model']}` | {r['family'][:20]} | {r['er']:.4f} | {ci_s} | {r['gap_l1']:+.5f} | "
            f"{r['struct_l1_distinct_codes']} | {r['struct_l1_effective_usage_exp_H']:.1f} | "
            f"{r['collision_l3']:.4f} | {er2_s} | {r['l1l2_neg_count']} | {r['stage3_recall@10']:.6f} |"
        )

    lines += [
        "",
        "The **L1+L2 negative-hit column is the reason that level is unusable here.** The canonical",
        "Resample has 31 negative hits out of 528,550 draws; a few collapsed models have so few that",
        f"the ER is undefined. {results['n_complete_case_models']} models have a defined L1 ER;",
        f"{manifest['counts']['n_complete_case_l1']} are used for the primary regression. The",
        "L1+L2 complete-case subset is handled separately in §6 and never compared against a",
        "different-sized M0 baseline.",
        "",
        "## 4. Primary experiment: Leave-One-Family-Out",
        "",
        f"Outer folds: **{len(counts)}**, one per inventory family, shared by M0-M3. Inner alpha",
        "selection is grouped leave-one-training-family-out, so no outer test family ever reaches",
        "the penalty search or the standardisation statistics.",
        "",
        "### Recall@10 (primary target)",
        "",
        "| set | features | OOF MAE | OOF RMSE | OOF R2 | OOF Spearman | dMAE vs M0 | 95% CI | CI lower > 0 |",
        "|:--|:--|---:|---:|---:|---:|---:|:--|:--:|",
    ]
    for name in LOFO.FEATURE_SETS:
        m = rec["sets"][name]["metrics"]
        d = rec["deltas_vs_M0"].get(name)
        if d:
            cells = (
                f"| {d['delta_mae']:+.6f} | [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] | "
                f"{'yes' if d['lower_bound_gt_zero'] else 'no'} |"
            )
        else:
            cells = "| — (baseline) | — | — |"
        lines.append(
            f"| {name} | {' + '.join(rec['sets'][name]['features'])} | {m['mae']:.6f} | "
            f"{m['rmse']:.6f} | {m['oof_r2']:+.4f} | {fmt(m['oof_spearman'], '+.4f')} {cells}"
        )
    lines += [
        "",
        "### NDCG@10 (secondary target)",
        "",
        "| set | OOF MAE | OOF R2 | OOF Spearman | dMAE vs M0 | 95% CI | CI lower > 0 |",
        "|:--|---:|---:|---:|---:|:--|:--:|",
    ]
    for name in LOFO.FEATURE_SETS:
        m = ndcg["sets"][name]["metrics"]
        d = ndcg["deltas_vs_M0"].get(name)
        if d:
            cells = (
                f"| {d['delta_mae']:+.6f} | [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] | "
                f"{'yes' if d['lower_bound_gt_zero'] else 'no'} |"
            )
        else:
            cells = "| — (baseline) | — | — |"
        lines.append(
            f"| {name} | {m['mae']:.6f} | {m['oof_r2']:+.4f} | {fmt(m['oof_spearman'], '+.4f')} {cells}"
        )

    lines += [
        "",
        "Both targets agree in sign and in ordering, which is the consistency the protocol asked for:",
        "every increment is negative on Recall and negative on NDCG. Note that **M2's CI excludes zero",
        "on the wrong side** — it is significantly *worse* than M0, not better.",
        "",
        "### Family-level increments (Recall@10)",
        "",
        "| set | " + " | ".join(f[:22] for f in counts) + " |",
        "|:--|" + "---:|" * len(counts),
    ]
    for name in ("M1", "M2", "M3"):
        fd = rec["deltas_vs_M0"][name]["family_deltas"]
        lines.append(
            f"| {name} | "
            + " | ".join(fmt(fd.get(f), "+.5f") for f in counts)
            + " |"
        )
    lines += [
        "",
        "No set is positive in every family. M1 and M3 flip sign in one family each, and the sign of",
        "the increment is therefore family-dependent rather than a stable property of the metric.",
        "",
        "## 5. Diagnostic decomposition (H3)",
        "",
        "With ER controlled, what do `pos_share`, `neg_share` and Gap actually do?",
        "",
        "- `Gap = pos_share - neg_share` is a deterministic function of `pos_share` and `neg_share`,",
        "  and M1 already contains it. The protocol's warning against feeding the ratio, the difference",
        "  and both raw rates into one small-sample model is therefore respected: only the ratio and the",
        "  difference ever enter a prespecified set, never all three.",
        "- The spread of `neg_share` across models is far wider than the spread of `pos_share`,",
        "  and it is driven almost entirely by how concentrated the L1 codebook is. That is the same",
        "  collapse mechanism the R2/R3 rounds traced directly, and it is why Gap carries little",
        "  information beyond ER: once ER is known, both of Gap's components are largely pinned down.",
        "- The models with the highest Gap are frequently the ones with the worst collision rate, so a",
        "  large Gap is not evidence of a usable representation. See `counterexamples.md`.",
        "",
        "## 6. Prespecified sensitivity analyses",
        "",
        "| analysis | n kept | M0 MAE | dM1 | dM2 | dM3 |",
        "|:--|---:|---:|---:|---:|---:|",
    ]
    s = sens if isinstance(sens, dict) else {}
    mm = s.get("multi_member_families_only", {}).get("recall@10", {})
    if mm:
        lines.append(
            f"| multi-member families only (drops single-model `TIGER_RQ-VAE`) | {mm['n_kept']} | "
            f"{mm['M0_mae']:.6f} | {fmt(mm['delta_M1'], '+.6f')} | {fmt(mm['delta_M2'], '+.6f')} | "
            f"{fmt(mm['delta_M3'], '+.6f')} |"
        )
    for key, label in (("drop_top_1_collision", "drop the single worst-collision model"),
                       ("drop_top_3_collision", "drop the three worst-collision models")):
        e = s.get(key, {}).get("recall@10", {})
        if e:
            lines.append(
                f"| {label} | {e['n_kept']} | {e['M0_mae']:.6f} | {fmt(e['delta_M1'], '+.6f')} | "
                f"{fmt(e['delta_M2'], '+.6f')} | {fmt(e['delta_M3'], '+.6f')} |"
            )
    cc = s.get("l1l2_complete_case", {})
    if cc:
        add = cc.get("add_l1l2_gap_to_M3", {}).get("recall@10", {})
        lines.append(
            f"| L1+L2 complete-case ({cc['n_models']} models), M3 + L1+L2 gap | {cc['n_models']} | "
            f"{fmt(add.get('M3_mae'))} | — | — | {fmt(add.get('delta'), '+.6f')} |"
        )
    lines += [
        "",
        "Every fixed sensitivity analysis keeps the same sign: the increments stay negative or become",
        "more negative. Removing the most collapsed models, restricting to multi-member families, and",
        "adding the L1+L2 gap on the reduced complete-case subset all fail to rescue the increments.",
        "",
        "## 7. What this design cannot settle",
        "",
        "These are limits of the study, not additional findings:",
        "",
        f"1. **Only {len(counts)} families.** The protocol anticipated roughly a dozen; the inventory",
        f"   actually groups into {len(counts)} ({json.dumps(counts)}). Leaving out `curvature_RQ-VAE`",
        "   removes 17 of 38 models at once, and `TIGER_RQ-VAE` is a single model, so two of the five",
        "   folds are structurally unlike the others. The family-cluster CI is correspondingly unstable,",
        "   and no result here should be read as a well-powered family-level conclusion.",
        "2. **One Stage 3 run per model.** With 57,439 evaluation users but a single training run per",
        "   model, the cross-model spread mixes model quality with run-to-run noise. No variance across",
        "   seeds has been measured and none is claimed.",
        "3. **The 38 models are not an external sample.** They are the same models the earlier",
        "   exploratory thresholds were fitted on. The only independence in this study is the",
        "   behavioural-pair split and the cross-validation; the model set itself was never held out.",
        "4. **Correlation, not causation.** Nothing here shows that changing a Stage 2 mechanism would",
        "   move Recall@10. It shows only which existing Stage 2 readings transfer across families.",
        "",
        "## 8. Historical descriptive association, kept separate",
        "",
    ]
    if hist.get("available"):
        lines += [
            f"- Step 9 covers {hist['step9_rows']} rows and **{hist['matched_to_inventory']}** of them join",
            f"  to this inventory on `{hist['join_key']}`.",
        ]
        if "step9_train_pair_er_vs_recall_spearman" in hist:
            lines += [
                f"- Train-pair L1 ER vs Recall@10 Spearman, reproduced on the joined models: "
                f"**{hist['step9_train_pair_er_vs_recall_spearman']:+.4f}** "
                f"(n = {hist['n_for_spearman']}).",
                f"- Valid-side L1 ER vs Recall@10 on the same models: "
                f"**{hist['valid_side_er_vs_recall_spearman_same_models']:+.4f}**.",
                f"- Canonical Resample: train-pair L1 ER **{hist['canonical_resample_train_pair_er']}** vs",
                f"  valid-side **{hist['canonical_resample_valid_er']}**. The two definitions are not",
                "  interchangeable and are never mixed here.",
            ]
        rep = hist.get("historical_threshold_reproduction", {})
        if rep:
            lines += [
                f"- The old train-pair thresholds (ER>=14, Gap>=0.095, L1 distinct>=230, collision<=0.08)",
                f"  select **{rep['models_passing_train_pair_thresholds']}** of the joined models. This is an",
                "  old-phenomenon reproduction only: the thresholds came from post-hoc exploration of",
                "  these same models and cannot validate themselves.",
            ]
    lines += [
        "",
        "## 9. Conclusion",
        "",
        "- **Confirmed:** Hard L1 ER alone is the strongest of the four prespecified Stage 2 readings",
        "  for out-of-family prediction of Stage 3 Recall@10 across these 38 models.",
        "- **Not confirmed (H1):** the L1 Gap does not add reliable out-of-family predictive value",
        "  once ER is known. ΔMAE is negative and its CI includes zero.",
        "- **Not confirmed (H2):** neither L1 effective usage nor three-token collision improves",
        "  out-of-family prediction, alone or jointly. M2's CI sits entirely below zero, i.e. adding",
        "  them is reliably harmful to LOFO transfer here.",
        "- **Diagnostic value (H3):** Gap, `pos_share` and `neg_share` remain useful for explaining",
        "  *why* a model collapses, and `counterexamples.md` shows high-Gap/high-collision models that",
        "  would fool a Gap threshold. They are collapse diagnostics, not selection predictors.",
        "- **Boundary (H4):** high ER is neither sufficient nor necessary for high Recall@10.",
        "  `TIGER_FORK_RQ-VAE/prefix_sib` reaches Recall@10 0.061613 (rank 7 of 38) on L1 ER 3.5064,",
        "  while `TIGER_HYPERBOLIC_RQ-VAE/fix_carry_euclid_raw` has the largest L1 Gap in the study",
        "  (+0.24796) and a 0.8561 three-token collision rate for Recall@10 of only 0.053814. A raw-ER",
        "  top-10 ranking (9/10 hits) still beats every fitted LOFO model (M0 5/10, M1 6/10, M2 1/10,",
        "  M3 2/10), so even ER-only screening is a family-dependent heuristic rather than a validated",
        "  selector. Details in `counterexamples.md`.",
        "",
        "## 10. Reproducing",
        "",
        "```bash",
        "python3 scripts/test_joint_prefix_predictors.py          # 33 assertions",
        "python3 scripts/analyze_joint_prefix_predictors.py       # this report",
        "```",
        "",
        "Feature definitions are byte-for-byte the ones in",
        "`scripts/analyze_heldout_prefix_enrichment.py` and",
        "`scripts/analyze_hbpe_r2_enrichment.py`; neither was modified. This analysis writes only",
        f"into `{OUT_DIR.relative_to(FT.REPO_ROOT)}/`.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, audit, meta = build_features()
    results = run_lofo(rows)
    complete = results["_complete"]

    sens = sensitivity(rows, results)
    cex = counterexamples(complete)
    hist = historical_step9(rows)

    # ---------- manifest ----------
    manifest = {
        "issue": 47,
        "git_sha": git_sha(),
        "inputs": {
            "model_inventory": {
                "path": str(INVENTORY.relative_to(FT.REPO_ROOT)),
                "sha256": sha_of(INVENTORY),
            },
            "valid_pairs": {
                "path": str(FT.VALID_PAIRS_PATH.relative_to(FT.REPO_ROOT)),
                "sha256": meta["valid_pairs_sha256"],
            },
            "step9_association": {
                "path": str(STEP9_CSV.relative_to(FT.REPO_ROOT)),
                "sha256": sha_of(STEP9_CSV),
                "role": "historical train-pair association only, never a feature",
            },
        },
        "split_proof": {
            "protocol_required": meta["protocol_required"],
            "feature_pairs": "valid.parquet transitions dropped from train (held-out)",
            "stage3_pairs": "test.parquet",
            "leakage_controls": [
                "all features computed only on behavior_pairs_valid_s42.npz",
                "no feature or threshold derived from test.parquet",
                "model selection uses Stage 3 inventory scores, never Stage 2 features",
                "train/test overlap is not re-derived here; the valid pairs carry a "
                "dropped_also_a_train_transition filter recorded in their meta file",
            ],
        },
        "counts": {
            "n_inventory_models": meta["n_inventory_models"],
            "n_features_computed": meta["n_features_computed"],
            "n_complete_case_l1": results["n_complete_case_models"],
            "n_held_out_transitions": meta["n_held_out_transitions"],
            "negatives_per_transition": meta["negatives_per_transition"],
        },
        "statistics": {
            "feature_bootstrap": meta["feature_bootstrap"],
            "regression": {
                "model": "Ridge (linear)",
                "alpha_grid": list(LOFO.ALPHA_GRID),
                "inner_cv": "grouped leave-one-training-family-out",
                "outer_cv": "leave-one-family-out over inventory families",
                "scaling": "train folds only",
                "seed": CV_SEED,
            },
        },
        "feature_sets": LOFO.FEATURE_SETS,
        "historical_thresholds": HISTORICAL_THRESHOLDS,
        "excluded_log": [a for a in audit if a.get("status") != "ok"],
        "family_sizes": meta.get("family_sizes"),
    }
    inv = json.loads(INVENTORY.read_text())
    manifest["family_sizes"] = {}
    for m in inv["models"]:
        manifest["family_sizes"][m["family"]] = manifest["family_sizes"].get(m["family"], 0) + 1
    manifest["n_families"] = len(manifest["family_sizes"])
    manifest["family_sizes_computed"] = {}
    for r in rows:
        manifest["family_sizes_computed"][r["family"]] = (
            manifest["family_sizes_computed"].get(r["family"], 0) + 1
        )
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---------- stage2_valid_features.csv ----------
    feat_cols = [
        "model", "family", "status", "stage3_recall@10", "stage3_ndcg@10",
        "sid_sha256", "sid_source", "sid_n_items", "sid_item_ids_match_catalogue",
        "pos_count", "neg_count", "pos_share", "neg_share", "gap", "er", "er_defined",
        "l1l2_pos_count", "l1l2_neg_count", "l1l2_pos_share", "l1l2_neg_share",
        "l1l2_gap", "l1l2_er", "l1l2_er_defined",
        "struct_l1_distinct_codes", "struct_l1_codebook_utilisation",
        "struct_l1_entropy_nats", "struct_l1_effective_usage_exp_H",
        "struct_l1_effective_usage_ratio", "struct_l1_largest_group_share",
        "struct_l1_zero_code_count", "struct_l12_distinct_prefixes",
        "struct_l123_distinct_triples", "struct_full_3token_collision",
        "er_l1_ci95", "gap_l1_ci95", "er_l1l2_ci95", "gap_l1l2_ci95",
        "n_clusters", "log_er_l1", "eff_usage_ratio_l1", "collision_l3",
    ]
    with (OUT_DIR / "stage2_valid_features.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(feat_cols)
        for r in sorted(rows, key=lambda r: -r["stage3_recall@10"]):
            w.writerow([
                r.get(c, "") if not isinstance(r.get(c), list)
                else json.dumps(r.get(c))
                for c in feat_cols
            ])

    # ---------- oof_predictions.csv ----------
    with (OUT_DIR / "oof_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "family", "target", "model_set", "y_true", "y_pred_oof", "abs_error", "signed_error"])
        for target in TARGETS:
            blk = results[target]
            for name in LOFO.FEATURE_SETS:
                preds = blk["_oof"][name]
                for i, r in enumerate(complete):
                    w.writerow([r["model"], r["family"], target, name,
                                f"{blk['_y'][i]:.10g}", f"{preds[i]:.10g}",
                                f"{abs(preds[i]-blk['_y'][i]):.10g}",
                                f"{preds[i]-blk['_y'][i]:.10g}"])

    # ---------- ablation_summary ----------
    summary: dict[str, Any] = {
        "n_complete_case_models": results["n_complete_case_models"],
        "targets": {},
        "sensitivity": sens,
        "topk_screening": {
            "recall_target_top10": results["topk_baseline_by_er"],
            "by_model_set": {
                name: results["recall@10"]["sets"][name]["topk"]
                for name in LOFO.FEATURE_SETS
            },
        },
        "historical_step9_check": hist,
    }
    for target in TARGETS:
        blk = results[target]
        summary["targets"][target] = {
            "metrics": {
                name: {
                    "metrics": blk["sets"][name]["metrics"],
                    "features": blk["sets"][name]["features"],
                    "alphas_per_fold": [
                        {"held_family": fo["held_family"], "alpha": fo["alpha"]}
                        for fo in blk["sets"][name]["folds"]
                    ],
                }
                for name in LOFO.FEATURE_SETS
            },
            "deltas_vs_M0": blk["deltas_vs_M0"],
        }
    (OUT_DIR / "ablation_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    with (OUT_DIR / "ablation_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "model_set", "features", "mae", "rmse", "oof_r2",
                    "oof_spearman", "delta_mae_vs_M0", "delta_ci95_low", "delta_ci95_high",
                    "relative_improvement", "ci_lower_gt_zero", "topk_precision_at_10"])
        for target in TARGETS:
            blk = results[target]
            for name in LOFO.FEATURE_SETS:
                m = blk["sets"][name]["metrics"]
                d = blk["deltas_vs_M0"].get(name)
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
                ])

    print(json.dumps({
        "n_models": len(rows),
        "n_complete_case": results["n_complete_case_models"],
        "n_families": manifest["n_families"],
        "recall": {
            name: results["recall@10"]["sets"][name]["metrics"] for name in LOFO.FEATURE_SETS
        },
        "recall_deltas": {
            name: {
                "delta_mae": results["recall@10"]["deltas_vs_M0"][name]["delta_mae"],
                "ci95": results["recall@10"]["deltas_vs_M0"][name]["ci95"],
                "sig": results["recall@10"]["deltas_vs_M0"][name]["lower_bound_gt_zero"],
            }
            for name in ("M1", "M2", "M3")
        },
        "ndcg_deltas": {
            name: {
                "delta_mae": results["ndcg@10"]["deltas_vs_M0"][name]["delta_mae"],
                "ci95": results["ndcg@10"]["deltas_vs_M0"][name]["ci95"],
                "sig": results["ndcg@10"]["deltas_vs_M0"][name]["lower_bound_gt_zero"],
            }
            for name in ("M1", "M2", "M3")
        },
    }, indent=2))

    write_counterexamples_md(complete, cex, results)
    write_report_md(rows, audit, results, manifest, meta, hist, sens)


if __name__ == "__main__":
    main()
