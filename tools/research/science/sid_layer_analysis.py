"""Frozen third-token (L3) diagnostics for Stage2 semantic IDs.

The analysis is read-only and predictive: it asks whether an L3-specific
statistic of an exported SID adds information about downstream Stage3
Recall@10 beyond the independent-validation Hard-L1 ER that is already a hard
admission gate.

Three L3 statistics are frozen here:

* ``l3_collision`` -- share of items whose first three codes are shared with
  another item, i.e. what the third code fails to separate.
* ``l3_refinement_gain`` -- log2 of the bucket the third code splits, averaged
  over items: how much the third token narrows the candidate set inside its own
  level-two prefix.
* ``conditional_l3_gain`` -- log2 likelihood ratio of the true next item's third
  code under a train-transition-conditioned predictor versus a prefix-only
  predictor, averaged over the frozen validation pairs. Positive means the
  observed history predicts the third code better than the prefix alone.

Every number is computed from files on disk with the paths frozen in the
constants below; nothing is trained, sampled or inferred from Stage3 outputs.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from math import log2
from pathlib import Path

import numpy as np

from .behavior import recover_train_sequences

REPO = Path(__file__).resolve().parents[3]
TRAIN_PARQUET = REPO / "results/stage0_build_parquet/train.parquet"
PAIRS_NPZ = REPO / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
STAGE2_ROOT = REPO / "results/stage2_RQ-VAE"
STAGE3_ROOT = REPO / "results/stage3_T5Train"
GATE_PATH = REPO / "tools/genrec-mcp/stage2_quality_gate.py"
REFERENCE_SID = STAGE2_ROOT / "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json"
OUTPUT = REPO / "results/research/issue117/diagnostics/l3_predictive_analysis.json"

L3_ALPHA = 1.0          # Dirichlet mass mixing the history term into the prefix prior
PROB_FLOOR = 1e-6
THREE_TOKENS = slice(0, 3)   # the three semantic codes; any trailing token is a sentinel


# ---------------------------------------------------------------- SID metrics
def load_sid(path: Path) -> dict[int, tuple[int, ...]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): tuple(int(x) for x in v) for k, v in raw.items()}


def l3_collision(sids: dict[int, tuple[int, ...]]) -> dict:
    """How many items still share the first three codes."""
    codes = [v[THREE_TOKENS] for v in sids.values()]
    counts = Counter(codes)
    shared = sum(1 for c in codes if counts[c] > 1)
    return {
        "items": len(codes),
        "distinct_three_token_sids": len(counts),
        "colliding_items": shared,
        "l3_collision": shared / len(codes) if codes else float("nan"),
        "max_bucket": max(counts.values()) if counts else 0,
    }


def l3_refinement_gain(sids: dict[int, tuple[int, ...]]) -> dict:
    """Mean log2 reduction of the candidate bucket that the third code buys."""
    by_prefix2 = Counter()
    by_prefix3 = Counter()
    for v in sids.values():
        by_prefix2[v[:2]] += 1
        by_prefix3[v[:3]] += 1
    gains = []
    for v in sids.values():
        bucket2 = by_prefix2[v[:2]]
        bucket3 = by_prefix3[v[:3]]
        gains.append(log2(bucket2 / bucket3))
    return {
        "mean_gain_bits": float(np.mean(gains)) if gains else float("nan"),
        "median_gain_bits": float(np.median(gains)) if gains else float("nan"),
        "distinct_prefix2": len(by_prefix2),
        "distinct_prefix3": len(by_prefix3),
    }


def conditional_l3_gain(sids, anchors, positives, successors) -> dict:
    """log2(history-conditioned likelihood / prefix-only likelihood) of the true L3.

    The prefix-only predictor knows the target item's own level-two prefix and
    the population of items inside that prefix; the conditioning predictor adds
    the training successors of the observed anchor item. Both are smoothed, so
    the ratio is finite for every pair.
    """
    by_prefix2 = defaultdict(Counter)
    for item, code in sids.items():
        by_prefix2[code[:2]][code[2]] += 1
    deltas, hit_hist, hit_pref, covered = [], 0, 0, 0
    for anchor, positive in zip(anchors, positives):
        code = sids.get(int(positive))
        if code is None:
            continue
        prefix2, level3 = code[:2], code[2]
        bucket = by_prefix2[prefix2]
        total_pref = sum(bucket.values()) - 1  # leave the target item out
        if total_pref <= 0:
            continue
        # Laplace smoothing over all 256 third-level codes. A leave-one-out count
        # of zero would otherwise collapse to the probability floor and make the
        # ratio insensitive exactly in the singleton buckets that matter most.
        p_pref = max((bucket[level3] - 1 + L3_ALPHA) / (total_pref + L3_ALPHA * 256), PROB_FLOOR)
        succ = successors.get(int(anchor)) or Counter()
        if succ:
            covered += 1
        num = sum(n for item, n in succ.items()
                  if item in sids and sids[item][:2] == prefix2 and sids[item][2] == level3)
        den = sum(n for item, n in succ.items() if item in sids and sids[item][:2] == prefix2)
        p_hist = max((num + L3_ALPHA * p_pref) / (den + L3_ALPHA), PROB_FLOOR)
        deltas.append(log2(p_hist / p_pref))
        hit_hist += int(p_hist >= p_pref)
        hit_pref += int(p_pref >= p_hist)
    n = len(deltas)
    return {
        "pairs_scored": n,
        "pairs_with_train_successors": covered,
        "mean_log2_likelihood_ratio": float(np.mean(deltas)) if n else float("nan"),
        "median_log2_likelihood_ratio": float(np.median(deltas)) if n else float("nan"),
        "share_pairs_history_helps": hit_hist / n if n else float("nan"),
        "share_pairs_prefix_helps": hit_pref / n if n else float("nan"),
    }


# ---------------------------------------------------------------- admission ER
def hard_l1_er(sid_path: Path, pairs: Path, meta: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location("stage2_quality_gate", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verdict = module.evaluate(Path(sid_path), REFERENCE_SID, pairs, meta)
    return dict(verdict["candidate"])


# ---------------------------------------------------------------- prediction
def family_of(arm: str) -> str:
    name = arm.split("/")[-1]
    for stem in ("resample_proj", "resample", "crowd", "cap"):
        if name.startswith(stem):
            return stem
    return name


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def predict(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return coef[0] + x @ coef[1:]


def leave_one_family_out(rows, features):
    """OOF residuals for a feature list; returns metrics and per-family residuals."""
    oof = np.zeros(len(rows))
    y = np.array([r["stage3_recall@10"] for r in rows])
    for family in sorted({r["family"] for r in rows}):
        train_idx = [i for i, r in enumerate(rows) if r["family"] != family]
        test_idx = [i for i, r in enumerate(rows) if r["family"] == family]
        if len(train_idx) <= len(features) + 1:
            oof[test_idx] = y[train_idx].mean() if train_idx else y.mean()
            continue
        x_tr = np.array([[rows[i][f] for f in features] for i in train_idx])
        x_te = np.array([[rows[i][f] for f in features] for i in test_idx])
        coef = fit_ols(x_tr, y[train_idx])
        oof[test_idx] = predict(coef, x_te)
    resid = oof - y
    return {
        "features": list(features),
        "oof_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "oof_mae": float(np.mean(np.abs(resid))),
        "oof_max_abs": float(np.max(np.abs(resid))),
        "oof_r2": float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)),
        "residuals": {r["arm"]: float(e) for r, e in zip(rows, resid)},
    }


def run(output: Path | None = None) -> Path:
    import pandas as pd

    output = Path(output or OUTPUT)
    frame = pd.read_parquet(TRAIN_PARQUET, columns=["user", "history", "target"])
    sequences = recover_train_sequences(frame.to_dict("records"))
    successors = defaultdict(Counter)
    for seq in sequences.values():
        for src, dst in zip(seq, seq[1:]):
            successors[src][dst] += 1

    pairs = np.load(PAIRS_NPZ)
    anchors, positives = pairs["anchors"], pairs["positives"]
    meta = PAIRS_NPZ.with_suffix(".meta.json")

    stage3 = _stage3_results()
    rows = []
    for arm, (sid_path, recall) in sorted(stage3.items()):
        sids = load_sid(sid_path)
        er = hard_l1_er(sid_path, PAIRS_NPZ, meta)["valid_l1_hard_er"]
        coll = l3_collision(sids)
        ref = l3_refinement_gain(sids)
        cond = conditional_l3_gain(sids, anchors, positives, successors)
        rows.append({
            "arm": arm,
            "family": family_of(arm),
            "sid": str(sid_path.relative_to(REPO)),
            "stage3_recall@10": recall,
            "valid_l1_hard_er": er,
            "l3_collision": coll["l3_collision"],
            "l3_refinement_gain": ref["mean_gain_bits"],
            "conditional_l3_gain": cond["mean_log2_likelihood_ratio"],
            **{f"detail_{k}": v for k, v in coll.items()},
            **{f"detail_ref_{k}": v for k, v in ref.items()},
            **{f"detail_cond_{k}": v for k, v in cond.items()},
        })

    models = {
        "M0_er_only": ["valid_l1_hard_er"],
        "M1_er_plus_collision": ["valid_l1_hard_er", "l3_collision"],
        "M2_er_plus_refinement": ["valid_l1_hard_er", "l3_refinement_gain"],
        "M3_er_plus_conditional": ["valid_l1_hard_er", "conditional_l3_gain"],
    }
    comparison = {name: leave_one_family_out(rows, feats) for name, feats in models.items()}
    base = comparison["M0_er_only"]["oof_rmse"]
    for name, result in comparison.items():
        result["delta_rmse_vs_M0"] = result["oof_rmse"] - base

    oof = {name: leave_one_family_out(rows, feats) for name, feats in models.items()}
    corr = _correlations(rows)
    per_family = _per_family_errors(rows, oof)
    bootstrap = _paired_bootstrap(rows, oof, "M0_er_only", "M2_er_plus_refinement")
    bootstrap_g3 = _paired_bootstrap(rows, oof, "M0_er_only", "M3_er_plus_conditional")

    payload = {
        "analysis": "l3_predictive_analysis",
        "frozen_inputs": {
            "train_parquet": str(TRAIN_PARQUET.relative_to(REPO)),
            "pairs_npz": str(PAIRS_NPZ.relative_to(REPO)),
            "reference_sid": str(REFERENCE_SID.relative_to(REPO)),
            "l3_alpha": L3_ALPHA,
            "prob_floor": PROB_FLOOR,
            "tokens_used": "first three codes; a trailing sentinel token is ignored",
        },
        "population": {
            "stage3_runs_with_sid_on_disk": len(rows),
            "families": sorted({r["family"] for r in rows}),
            "note": "Only arms whose Stage2 item_sids.json still exists on disk and whose "
                    "Stage3 run recorded test_recall@10 can enter a predictive comparison.",
        },
        "rows": rows,
        "models": comparison,
        "feature_correlations": corr,
        "per_family_oof_rmse": per_family,
        "paired_bootstrap_abs_error": {
            "M2_vs_M0": bootstrap,
            "M3_vs_M0": bootstrap_g3,
        },
        "lofo_scheme": "leave-one-family-out; families are name stems of the Stage2 arm",
        "power_warning": "n=13 arms over 7 families; every OOF comparison here is a direction, "
                         "not a significance test.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _stage3_results() -> dict[str, tuple[Path, float]]:
    """arm -> (existing SID path, best recorded Stage3 test_recall@10)."""
    import re

    best: dict[str, tuple[Path, float]] = {}
    for log in STAGE3_ROOT.glob("*/logs/*/*/training_metrics.jsonl"):
        run_dir = log.parent
        final = run_dir / "test_final.json"
        if not final.is_file():
            continue
        sid = None
        with log.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = re.search(r"(/[^\"' ]*item_sids\.json)", line)
                if match:
                    sid = Path(match.group(1))
                    break
        if sid is None:
            continue
        local = _local(sid)
        if not local.is_file():
            continue
        arm = str(local.relative_to(STAGE2_ROOT)).replace("/item_sids.json", "")
        recall = json.loads(final.read_text(encoding="utf-8"))["test_recall@10"]
        if arm not in best or recall > best[arm][1]:
            best[arm] = (local, recall)
    return best


def _local(path: Path) -> Path:
    text = str(path)
    for prefix in ("/home/wlia0047/ar57/wenyu/GeneRec", "/fs04/ar57/wenyu/GeneRec"):
        if text.startswith(prefix):
            return REPO / text[len(prefix):].lstrip("/")
    return path


# ----------------------------------------------------------- per-candidate record
def build_single_sid_record(sids, gate_result, anchors, positives, successors) -> dict:
    """Complete, numeric Stage2 layer diagnostics independent of Stage3 scores.

    ER and L2 prefix counts are diagnostics only, never promotion thresholds;
    both prefix-hit rates and all three L3 metrics are diagnostic only.
    """
    from math import isfinite

    if "candidate" not in gate_result or "baseline" not in gate_result:
        raise ValueError("quality-gate source/provenance could not be verified")
    candidate = gate_result["candidate"]
    collision = l3_collision(sids)
    refinement = l3_refinement_gain(sids)
    conditional = conditional_l3_gain(sids, anchors, positives, successors)
    metrics = {
        "valid_l1_hard_er": candidate["valid_l1_hard_er"],
        "valid_l1_positive_hit_rate": candidate["valid_l1_positive_hit_rate"],
        "valid_l1_negative_hit_rate": candidate["valid_l1_negative_hit_rate"],
        "level2_distinct_prefixes": candidate["level2_distinct_prefixes"],
        "level3_distinct_triples": collision["distinct_three_token_sids"],
        "l3_collision": collision["l3_collision"],
        "l3_refinement_gain_bits": refinement["mean_gain_bits"],
        "conditional_l3_gain": conditional["mean_log2_likelihood_ratio"],
    }
    if (conditional["pairs_scored"] < 1 or
            conditional["pairs_scored"] > len(anchors) or
            any(isinstance(x, bool) or not isinstance(x, (float, int)) or
                not isfinite(x) for x in metrics.values())):
        raise ValueError("incomplete/nonfinite layer metrics or no eligible G3 validation pairs")
    return {
        "schema": "stage2_sid_layer_metrics_v1",
        "status": "complete",
        "metrics": metrics,
        # Issue #120: there is no quality floor any more. Every number here is a
        # diagnostic; only the technical validity of the SID decides admission.
        "diagnostics": {key: metrics[key] for key in sorted(metrics)},
        "admission_policy": "technical validity only; no quality floor",
        "technically_valid_sid": bool(gate_result["passed"]),
        "diagnostics_only": [
            "valid_l1_positive_hit_rate", "valid_l1_negative_hit_rate",
            "level3_distinct_triples", "l3_collision", "l3_refinement_gain_bits",
            "conditional_l3_gain",
        ],
        "methodology": {
            "L3": "first three semantic codes; fourth collision-extension token ignored",
            "collision": "item-weighted share of repeated three-token SIDs",
            "refinement": "item-weighted mean log2(size of L2 bucket / size of L3 bucket)",
            "conditional_G3": "mean log2(train-successor likelihood / prefix-only likelihood), conditioned on true target L1+L2",
            "l3_alpha": L3_ALPHA,
            "prob_floor": PROB_FLOOR,
            "validation_pairs_total": len(anchors),
            "validation_pairs": conditional["pairs_scored"],
            "validation_pairs_excluded": len(anchors) - conditional["pairs_scored"],
            "validation_pairs_with_train_successors": conditional["pairs_with_train_successors"],
        },
        "quality_source": gate_result["sha256"],
        "baseline_metrics": gate_result["baseline"],
    }


def record_single_sid(sid_path: Path, output: Path | None = None) -> Path:
    """Record all required numbers once per exported SID, even if ER/L2 fail.

    Called after Stage2 SID export, BEFORE any Stage3 result is used for model
    selection. Writes a versioned, atomic report under the Stage2 result tree.
    Missing inputs are errors, not artificial zero measurements.
    """
    import hashlib
    import importlib.util
    import os
    import pandas as pd

    sid_path = Path(sid_path).resolve(strict=True)
    if sid_path.name != "item_sids.json" or STAGE2_ROOT.resolve() not in sid_path.parents:
        raise ValueError("SID must be an item_sids.json inside the Stage2 result tree")
    spec = importlib.util.spec_from_file_location("stage2_quality_gate", GATE_PATH)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    verdict = gate.evaluate(
        sid_path, REFERENCE_SID, PAIRS_NPZ, PAIRS_NPZ.with_suffix(".meta.json"))
    if "candidate" not in verdict:
        raise ValueError("Stage2 quality evidence unavailable: " +
                         str(verdict.get("refused")))
    sids = load_sid(sid_path)
    pairs = np.load(PAIRS_NPZ)
    anchors, positives = pairs["anchors"], pairs["positives"]
    train = pd.read_parquet(TRAIN_PARQUET, columns=["user", "history", "target"])
    sequences = recover_train_sequences(train.to_dict("records"))
    successors = defaultdict(Counter)
    for seq in sequences.values():
        for src, dst in zip(seq, seq[1:]):
            successors[src][dst] += 1
    record = build_single_sid_record(sids, verdict, anchors, positives, successors)
    def digest(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    record["provenance"] = {
        "sid": str(sid_path),
        "candidate_arm": str(sid_path.parent.relative_to(STAGE2_ROOT.resolve())),
        "train_parquet": str(TRAIN_PARQUET),
        "validation_npz": str(PAIRS_NPZ),
        "resample_sid": str(REFERENCE_SID),
        "sid_sha256": digest(sid_path),
        "train_parquet_sha256": digest(TRAIN_PARQUET),
        "analysis_source_sha256": digest(Path(__file__)),
        "resample_sid_sha256": digest(REFERENCE_SID),
        "validation_npz_sha256": digest(PAIRS_NPZ),
        "stage3_results_used": False,
    }
    if output is None:
        output = sid_path.parent / "stage2_metrics" / (record["provenance"]["sid_sha256"] + ".json")
    output = Path(output)
    record["report_path"] = str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


# ---------------------------------------------------------------- robustness
_FEATURES = ["valid_l1_hard_er", "l3_collision", "l3_refinement_gain", "conditional_l3_gain"]


def _correlations(rows):
    y = np.array([r["stage3_recall@10"] for r in rows])
    out = {}
    for name in _FEATURES:
        x = np.array([r[name] for r in rows])
        out[name] = {
            "pearson_vs_recall": float(np.corrcoef(x, y)[0, 1]),
            "pearson_vs_er": float(np.corrcoef(x, np.array([r["valid_l1_hard_er"] for r in rows]))[0, 1]),
        }
    return out


def _per_family_errors(rows, oof):
    out = {}
    for family in sorted({r["family"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["family"] == family]
        entry = {}
        for name, result in oof.items():
            res = np.array([result["residuals"][rows[i]["arm"]] for i in idx])
            entry[name] = {"n": len(idx), "rmse": float(np.sqrt(np.mean(res ** 2)))}
        out[family] = entry
    return out


def _paired_bootstrap(rows, oof, base, other, draws=2000, seed=42):
    """Bootstrap the per-arm absolute-error difference, arms resampled."""
    rng = np.random.default_rng(seed)
    base_err = np.array([abs(oof[base]["residuals"][r["arm"]]) for r in rows])
    other_err = np.array([abs(oof[other]["residuals"][r["arm"]]) for r in rows])
    delta = other_err - base_err
    samples = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(draws)])
    return {
        "mean_delta_abs_error": float(delta.mean()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "share_draws_better": float(np.mean(samples < 0)),
        "draws": draws,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: sid_layer_analysis.py <path/to/item_sids.json>", file=sys.stderr)
        raise SystemExit(2)
    print(record_single_sid(Path(sys.argv[1])))
