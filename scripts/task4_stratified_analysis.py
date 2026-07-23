"""
task4_stratified_analysis.py — 5-dimension ambiguity stratification of free-form
predictions across the 3 quantization algorithms (RQ-VAE / RKMeans / RVQ).

5 dimensions (all computed from per-user top-10 SID predictions only):
  D1. top10_diversity:   # of unique SIDs in user's top-10 (range 1..10)
  D2. digit0_entropy:    Shannon entropy of digit-0 values in top-10
  D3. digit0_dominance:  fraction of top-10 SIDs sharing the most common digit-0
  D4. depth_saturation:  # of hierarchy levels (1..4) where all 10 SIDs are equal
  D5. same_prefix_count: # of pairs of top-10 SIDs sharing all 4 digits (i.e. duplicates)

For each dimension, stratify users into 3-4 bins and report the population of
each bin per algorithm. Then compute the cross-algorithm "ambiguity signature"
table — does RKMeans produce more concentrated predictions than RQ-VAE? Etc.

This is a *descriptive* analysis: no per-user R@K yet. Per-user R@K would
require joining to ground-truth SIDs from the testing tfrecords (deferred).

Usage:
    cd /home/wlia0047/ar57/wenyu/GeneRec/GRID
    python /home/wlia0047/ar57/wenyu/GeneRec/scripts/task4_stratified_analysis.py
"""

import json
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

GRID_ROOT = Path("/fs04/ar57/wenyu/GeneRec/GRID")
os.chdir(str(GRID_ROOT))
sys.path.insert(0, str(GRID_ROOT))

ALGO_RUNS = {
    "rqvae": "2026-07-08/13-50-55",
    "rkmeans": "2026-07-08/14-04-02",
    "rvq": "2026-07-08/14-48-59",
}
PICKLE_DIR = GRID_ROOT / "logs/inference/runs"


def load_algo(algo: str):
    pkl_path = PICKLE_DIR / ALGO_RUNS[algo] / "pickle/merged_predictions.pkl"
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    sids = np.array([r["semantic_ids"] for r in records], dtype=np.int64)  # (N, 10, 4)
    user_ids = np.array([r["user_id"] for r in records], dtype=np.int64)
    return user_ids, sids


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a count distribution (base e)."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def per_user_metrics(sids: np.ndarray) -> dict:
    """Compute 5 ambiguity metrics for each user.

    sids: (N, 10, 4) int64 array.

    Returns dict[str, np.ndarray], each (N,).
    """
    n = sids.shape[0]
    out = {
        "top10_diversity": np.zeros(n, dtype=np.int64),
        "digit0_entropy": np.zeros(n, dtype=np.float64),
        "digit0_dominance": np.zeros(n, dtype=np.float64),
        "depth_saturation": np.zeros(n, dtype=np.int64),
        "same_prefix_count": np.zeros(n, dtype=np.int64),
    }
    for u in range(n):
        s = sids[u]  # (10, 4)
        # D1: unique SIDs across top-10
        unique_sids = {tuple(row) for row in s}
        out["top10_diversity"][u] = len(unique_sids)
        # D2: entropy of digit-0 values
        d0 = s[:, 0]
        c0 = np.bincount(d0, minlength=256)
        out["digit0_entropy"][u] = shannon_entropy(c0[c0 > 0])
        # D3: fraction of SIDs sharing the most-common digit-0
        out["digit0_dominance"][u] = c0.max() / 10.0
        # D4: # of hierarchy levels where all 10 SIDs are equal
        sat = 0
        for d in range(4):
            if len(np.unique(s[:, d])) == 1:
                sat += 1
        out["depth_saturation"][u] = sat
        # D5: # of duplicate SID rows in top-10
        c_full = Counter(tuple(row) for row in s)
        out["same_prefix_count"][u] = sum(v - 1 for v in c_full.values() if v > 1)
    return out


def stratify(values: np.ndarray, bins):
    """Stratify into bins. bins = [(name, low, high_inclusive), ...]."""
    out = {}
    for name, lo, hi in bins:
        if hi is None:
            mask = values >= lo
        else:
            mask = (values >= lo) & (values <= hi)
        out[name] = int(mask.sum())
    return out


# Bin definitions per dimension (lo, hi)
BINS = {
    "top10_diversity": [
        ("narrow_1-3", 1, 3),
        ("medium_4-7", 4, 7),
        ("wide_8-10", 8, 10),
    ],
    "digit0_entropy": [
        ("low_0.0-0.5", 0.0, 0.5),
        ("mid_0.5-1.5", 0.5, 1.5),
        ("high_1.5+", 1.5, None),
    ],
    "digit0_dominance": [
        ("concentrated_0.7+", 0.7, 1.0),
        ("mixed_0.4-0.7", 0.4, 0.7),
        ("dispersed_-0.4", 0.0, 0.4),
    ],
    "depth_saturation": [
        ("none_0", 0, 0),
        ("partial_1-2", 1, 2),
        ("full_3-4", 3, 4),
    ],
    "same_prefix_count": [
        ("distinct_0", 0, 0),
        ("few_dups_1-2", 1, 2),
        ("many_dups_3+", 3, None),
    ],
}


def main():
    summary = {}
    for algo in ["rqvae", "rkmeans", "rvq"]:
        user_ids, sids = load_algo(algo)
        metrics = per_user_metrics(sids)
        algo_summary = {"n_users": int(sids.shape[0])}
        # Means
        for dim, vals in metrics.items():
            algo_summary[f"{dim}_mean"] = float(vals.mean())
            algo_summary[f"{dim}_std"] = float(vals.std())
        # Stratification counts
        for dim, bins in BINS.items():
            algo_summary[f"strat_{dim}"] = stratify(metrics[dim], bins)
        summary[algo] = algo_summary
        print(f"\n=== {algo.upper()} (n={algo_summary['n_users']}) ===")
        for dim, bins in BINS.items():
            print(f"  {dim}:")
            for name, lo, hi in bins:
                count = algo_summary[f"strat_{dim}"][name]
                pct = 100 * count / algo_summary["n_users"]
                print(f"    {name:25s}: {count:6d} ({pct:5.1f}%)")

    # Cross-algorithm comparison table
    print("\n\n=== Cross-algorithm signature (mean per dimension) ===")
    print(f"{'Dimension':25s} {'RQ-VAE':>10s} {'RKMeans':>10s} {'RVQ':>10s}")
    for dim in BINS.keys():
        line = f"{dim:25s}"
        for algo in ["rqvae", "rkmeans", "rvq"]:
            line += f" {summary[algo][f'{dim}_mean']:>10.3f}"
        print(line)

    out_path = "/home/wlia0047/ar57/wenyu/GeneRec/task4_stratified.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
