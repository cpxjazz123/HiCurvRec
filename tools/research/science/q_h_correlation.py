#!/usr/bin/env python3
"""Q_H against Stage 3 recall over the confirmed SID+recall pairs.

The pairs come from ``analysis_sid_recall/sid_recall.csv`` -- the collection in
which every run's own vocabulary vouches for the table it is paired with. For
each of them the Issue #126 score is computed with the FROZEN reference and the
held-out validation probe, and the association with ``test_recall@10`` is
reported with the sample size it deserves.

Nothing here is tuned: the reference, its temperature, the probe and the score
are the ones Issue #126 froze.
"""
from __future__ import annotations

# One BLAS thread per worker: the score is many small array operations, and a pool
# of workers that each open a full thread team simply thrashes the machine (load
# average 130 with 24 workers).
import os as _os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_name, "1")

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
COLLECTION = REPO / "analysis_sid_recall"
REPORT = REPO / "results" / "research" / "issue126" / "correlation_report_confirmed.json"


def spearman(x: np.ndarray, y: np.ndarray) -> tuple:
    from scipy import stats
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def kendall(x: np.ndarray, y: np.ndarray) -> tuple:
    from scipy import stats
    tau, p = stats.kendalltau(x, y)
    return float(tau), float(p)


def pearson(x: np.ndarray, y: np.ndarray) -> tuple:
    from scipy import stats
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, draws: int = 4000, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    values = []
    n = len(x)
    for _ in range(draws):
        index = rng.integers(0, n, n)
        if len(np.unique(x[index])) < 3:
            continue
        values.append(spearman(x[index], y[index])[0])
    if not values:
        return []
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]



# The score of one table is independent of every other table, so the collection is
# scored in parallel. The reference and the geometry are loaded once and inherited
# by the workers through fork; each worker writes only its own record files.
_REFERENCE = None
_GEOMETRY = None
_LEVELS = (1,)


def _score_one(row: dict) -> dict:
    import sys
    sys.path.insert(0, str(REPO))
    import torch
    torch.set_num_threads(1)
    from tools.research.science import hyperbolic_sid_quality as quality

    sid_path = COLLECTION / "sids" / row["sid_file"]
    record = {
        "sid_sha256": row["sid_sha256"],
        "sid_file": row["sid_file"],
        "variant": row["variant"],
        "run_dir": row["run_dir"],
        "pair_confidence": row["pair_confidence"],
        "test_recall@10": float(row["test_recall@10"]),
        "test_recall@5": float(row["test_recall@5"]),
    }
    for level in _LEVELS:
        scored = quality.score(sid_path, reference=_REFERENCE, geometry=_GEOMETRY,
                               levels=level)
        key = "L%d" % level
        record[key] = {
            "q_h": float(scored["q_h"]),
            "buckets": scored["buckets"]["count"],
            "same_bucket_ratio": scored["same_bucket_ratio"],
            "delta_mean": scored["delta_mean"],
            "negatives_ranked_above_positive":
                scored["negatives_ranked_above_positive"],
        }
        record["metric_record_%s" % key] = str(
            quality.record(sid_path, suffix=".L%d" % level, reference=_REFERENCE,
                           geometry=_GEOMETRY, levels=level))
    return record


def main() -> int:
    import sys
    sys.path.insert(0, str(REPO))
    from tools.research.science import hyperbolic_sid_quality as quality
    from tools.research.science import hyperbolic_behavior_reference as reference_module

    levels = [int(value) for value in
              (os.environ.get("Q_H_LEVELS") or "1").split(",") if value.strip()]
    rows = list(csv.DictReader((COLLECTION / "sid_recall.csv").open()))
    reference = reference_module.load()
    geometry = quality.BallGeometry()

    global _REFERENCE, _GEOMETRY, _LEVELS
    _REFERENCE, _GEOMETRY, _LEVELS = reference, geometry, tuple(levels)
    todo = [row for row in rows
            if (COLLECTION / "sids" / row["sid_file"]).is_file()]
    workers = int(os.environ.get("Q_H_WORKERS") or 0) or min(16, os.cpu_count() or 1)
    if workers <= 1:
        records = [_score_one(row) for row in todo]
    else:
        from concurrent.futures import ProcessPoolExecutor
        records = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, record in enumerate(pool.map(_score_one, todo), start=1):
                records.append(record)
                if index % 10 == 0:
                    print(f"  scored {index}/{len(todo)} (levels {levels}, "
                          f"{workers} workers)", flush=True)

    recall = np.array([r["test_recall@10"] for r in records])
    first = "L%d" % levels[0]
    q_h = np.array([r[first]["q_h"] for r in records])
    buckets = np.array([r[first]["buckets"] for r in records], dtype=float)
    same = np.array([r[first]["same_bucket_ratio"] for r in records])
    per_level = {}
    for level in levels:
        key = "L%d" % level
        values = np.array([r[key]["q_h"] for r in records])
        entry = {"spearman": spearman(values, recall)[0],
                 "p": spearman(values, recall)[1],
                 "kendall": kendall(values, recall)[0],
                 "pearson": pearson(values, recall)[0],
                 "bootstrap_ci95": bootstrap_ci(values, recall),
                 "range": [float(values.min()), float(values.max())],
                 "mean": float(values.mean())}
        if level == levels[0]:
            loo = []
            for index in range(len(records)):
                keep = [i for i in range(len(records)) if i != index]
                loo.append(spearman(values[keep], recall[keep])[0])
            entry["leave_one_out_min"] = float(min(loo))
            entry["leave_one_out_max"] = float(max(loo))
        per_level[key] = entry

    # how much each level adds over the structural diagnostics, one at a time
    def r2(features):
        design = np.column_stack([np.ones(len(recall))] + features)
        beta, *_ = np.linalg.lstsq(design, recall, rcond=None)
        residual = recall - design @ beta
        return float(1 - (residual ** 2).sum() / ((recall - recall.mean()) ** 2).sum())

    structure = [buckets, same]
    incremental = {"structure_only": r2(structure)}
    for level in levels:
        key = "L%d" % level
        values = np.array([r[key]["q_h"] for r in records])
        incremental["structure_plus_%s" % key] = r2(structure + [values])
        incremental["%s_only" % key] = r2([values])

    report = {
        "pairs": len(records),
        "source": "analysis_sid_recall/sid_recall.csv (the confirmed collection)",
        "metric": {
            "name": "Q_H, the hyperbolic prefix ranking score of Issue #126",
            "reference_sha256": reference["record"]["arrays"]["sha256"],
            "temperature": reference["record"]["temperature"],
            "probe": str(reference_module.VALID_PAIRS),
            "levels": levels,
            "scope": "the first `levels` SID codes form the prefix; levels=1 is the "
                     "score Issue #126 froze",
        },
        "per_level": per_level,
        "incremental_r2": incremental,
        "q_h_vs_bucket_count": dict(zip(("spearman", "p"), spearman(q_h, buckets))),
        "q_h_vs_same_bucket_ratio": dict(zip(("spearman", "p"), spearman(q_h, same))),
        "bucket_count_vs_recall@10": dict(zip(("spearman", "p"), spearman(buckets, recall))),
        "same_bucket_ratio_vs_recall@10": dict(zip(("spearman", "p"), spearman(same, recall))),
        "records": records,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\npairs: {len(records)}")
    for key, entry in per_level.items():
        print(f"{key}: Spearman {entry['spearman']:+.4f} (p={entry['p']:.3g}) | "
              f"Kendall {entry['kendall']:+.4f} | Pearson {entry['pearson']:+.4f} | "
              f"CI [{entry['bootstrap_ci95'][0]:+.3f}, {entry['bootstrap_ci95'][1]:+.3f}] | "
              f"Q_H {entry['range'][0]:+.4f}..{entry['range'][1]:+.4f}")
    print("R^2:", {k: round(v, 4) for k, v in incremental.items()})
    print(f"\nwritten: {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
