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

import csv
import json
import math
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


def main() -> int:
    import sys
    sys.path.insert(0, str(REPO))
    from tools.research.science import hyperbolic_sid_quality as quality
    from tools.research.science import hyperbolic_behavior_reference as reference_module

    rows = list(csv.DictReader((COLLECTION / "sid_recall.csv").open()))
    reference = reference_module.load()
    geometry = quality.BallGeometry()

    records = []
    for index, row in enumerate(rows, start=1):
        sid_path = COLLECTION / "sids" / row["sid_file"]
        if not sid_path.is_file():
            continue
        report = quality.score(sid_path, reference=reference, geometry=geometry)
        records.append({
            "sid_sha256": row["sid_sha256"],
            "sid_file": row["sid_file"],
            "variant": row["variant"],
            "run_dir": row["run_dir"],
            "pair_confidence": row["pair_confidence"],
            "test_recall@10": float(row["test_recall@10"]),
            "test_recall@5": float(row["test_recall@5"]),
            "q_h": float(report["q_h"]),
            "buckets": report["buckets"]["count"],
            "same_bucket_ratio": report["same_bucket_ratio"],
            "delta_mean": report["delta_mean"],
            "negatives_ranked_above_positive": report["negatives_ranked_above_positive"],
            "metric_record": str(quality.record(sid_path, reference=reference,
                                                geometry=geometry)),
        })
        if index % 10 == 0:
            print(f"  scored {index}/{len(rows)}", flush=True)

    q_h = np.array([r["q_h"] for r in records])
    recall = np.array([r["test_recall@10"] for r in records])
    buckets = np.array([r["buckets"] for r in records], dtype=float)
    same = np.array([r["same_bucket_ratio"] for r in records])

    rho, rho_p = spearman(q_h, recall)
    tau, tau_p = kendall(q_h, recall)
    pear, pear_p = pearson(q_h, recall)
    loo = []
    for index in range(len(records)):
        keep = [i for i in range(len(records)) if i != index]
        loo.append(spearman(q_h[keep], recall[keep])[0])

    report = {
        "pairs": len(records),
        "source": "analysis_sid_recall/sid_recall.csv (the confirmed collection)",
        "metric": {
            "name": "Q_H, the hyperbolic prefix ranking score of Issue #126",
            "reference_sha256": reference["record"]["arrays"]["sha256"],
            "temperature": reference["record"]["temperature"],
            "probe": str(reference_module.VALID_PAIRS),
            "scope": "level-1 prefix only",
        },
        "q_h_vs_recall@10": {
            "spearman": rho, "spearman_p": rho_p,
            "kendall": tau, "kendall_p": tau_p,
            "pearson": pear, "pearson_p": pear_p,
            "bootstrap_ci95": bootstrap_ci(q_h, recall),
            "leave_one_out_min": float(min(loo)),
            "leave_one_out_max": float(max(loo)),
        },
        "q_h_vs_bucket_count": dict(zip(("spearman", "p"), spearman(q_h, buckets))),
        "q_h_vs_same_bucket_ratio": dict(zip(("spearman", "p"), spearman(q_h, same))),
        "bucket_count_vs_recall@10": dict(zip(("spearman", "p"), spearman(buckets, recall))),
        "same_bucket_ratio_vs_recall@10": dict(zip(("spearman", "p"), spearman(same, recall))),
        "q_h_range": [float(q_h.min()), float(q_h.max())],
        "q_h_mean": float(q_h.mean()),
        "records": records,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\npairs                         : {len(records)}")
    print(f"Q_H range                     : {q_h.min():.4f} .. {q_h.max():.4f} "
          f"(mean {q_h.mean():.4f})")
    print(f"Spearman(Q_H, R@10)           : {rho:+.4f}  p={rho_p:.3g}")
    print(f"Kendall (Q_H, R@10)           : {tau:+.4f}  p={tau_p:.3g}")
    print(f"Pearson (Q_H, R@10)           : {pear:+.4f}  p={pear_p:.3g}")
    print(f"bootstrap 95% CI (Spearman)   : "
          f"[{report['q_h_vs_recall@10']['bootstrap_ci95'][0]:+.4f}, "
          f"{report['q_h_vs_recall@10']['bootstrap_ci95'][1]:+.4f}]")
    print(f"leave-one-out Spearman range  : [{min(loo):+.4f}, {max(loo):+.4f}]")
    print(f"Spearman(Q_H, bucket count)   : {report['q_h_vs_bucket_count']['spearman']:+.4f}")
    print(f"Spearman(Q_H, same-bucket)    : {report['q_h_vs_same_bucket_ratio']['spearman']:+.4f}")
    print(f"Spearman(buckets, R@10)       : {report['bucket_count_vs_recall@10']['spearman']:+.4f}")
    print(f"\nwritten: {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
