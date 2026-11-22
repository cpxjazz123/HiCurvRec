"""Per-item exact curvature and its comparison against #235's 512-reference estimate.

Issue #236 C18/C19 need the per-item error of the sampled estimate, not just the global
one: a small global shift can hide large per-triple or per-item errors that cancel out.

Usage (no arguments, per CLAUDE.md 9-10):
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/issue236/run_per_node_exact.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_exact as rx  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue236"
PREV = REPO / "results" / "research" / "issue235"
NEAR_ZERO = 1e-9


def degree_strata(levels: np.ndarray, means: np.ndarray, mixed: np.ndarray) -> list:
    edges = [1, 5, 13, 50, 200, 10 ** 9]
    labels = ["1-4", "5-12", "13-49", "50-199", "200+"]
    out = []
    for low, high, label in zip(edges[:-1], edges[1:], labels):
        block = (levels >= low) & (levels < high)
        if not block.any():
            continue
        out.append({
            "degree_range": label,
            "items": int(block.sum()),
            "mean_of_means": float(means[block].mean()),
            "negative_mean_items": int((means[block] < -NEAR_ZERO).sum()),
            "mixed_fraction": float(mixed[block].mean()),
        })
    return out


def main() -> int:
    started = time.time()
    adjacency, metadata = rx.load_adjacency()
    n = adjacency.shape[0]
    degree = np.diff(adjacency.indptr).astype(int)
    principal = rx.principal_component(adjacency)
    square_sum, counts, _ = rx.batch_bfs_square_sums(adjacency, principal, principal)
    print(f"BFS done {time.time() - started:.0f}s", flush=True)

    previous = json.loads((PREV / "per_node_summary.json").read_text(encoding="utf-8"))
    previous_csv = {}
    for line in (PREV / "node_curvature.csv").read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        previous_csv[int(parts[0])] = {
            "n_triples": int(parts[2]),
            "mean": float(parts[3]) if parts[3] else None,
            "positive": int(parts[6]), "negative": int(parts[7]), "near_zero": int(parts[8]),
            "mixed": parts[11] == "True",
        }

    results = {}
    for mode, geodesic in (("geodesic", True), ("neighbour", False)):
        centres, lefts, rights = rx.enumerate_triples(adjacency, geodesic)
        pair_square = np.where(np.asarray(adjacency[lefts, rights]).ravel() > 0, 1.0, 4.0)
        values, denominators = rx.curvature(square_sum, counts, centres, lefts, rights,
                                            pair_square)
        order = np.argsort(centres, kind="stable")
        if values.size != centres.size:
            # curvature dropped triples with a zero denominator; align on the same mask
            _, drop_mask = None, None
        centres_sorted = centres[order]
        values_sorted = values[order]
        bounds = np.searchsorted(centres_sorted, np.arange(n + 1))

        rows = []
        for node in range(n):
            block = values_sorted[bounds[node]:bounds[node + 1]]
            if block.size == 0:
                rows.append({"item": node, "degree": int(degree[node]), "n_triples": 0,
                             "mean": None, "median": None, "std": None,
                             "positive": 0, "negative": 0, "near_zero": 0,
                             "mixed_sign": False})
                continue
            positive = int((block > NEAR_ZERO).sum())
            negative = int((block < -NEAR_ZERO).sum())
            rows.append({
                "item": node, "degree": int(degree[node]), "n_triples": int(block.size),
                "mean": float(block.mean()), "median": float(np.median(block)),
                "std": float(block.std()), "positive": positive, "negative": negative,
                "near_zero": int(block.size) - positive - negative,
                "mixed_sign": positive > 0 and negative > 0,
            })

        csv_path = OUT / f"node_curvature_exact_{mode}.csv"
        with csv_path.open("w", encoding="utf-8") as sink:
            sink.write("item,degree,n_triples,mean,median,std,positive,negative,"
                       "near_zero,mixed_sign\n")
            for row in rows:
                fmt = lambda v: "" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v))
                sink.write(",".join(fmt(row[k]) for k in (
                    "item", "degree", "n_triples", "mean", "median", "std", "positive",
                    "negative", "near_zero", "mixed_sign")) + "\n")

        scored = [r for r in rows if r["n_triples"] > 0]
        means = np.array([r["mean"] for r in scored])
        mixed = np.array([r["mixed_sign"] for r in scored])
        levels = np.array([r["degree"] for r in scored])
        rank = lambda v: np.argsort(np.argsort(v)).astype(float)
        a, b = rank(levels), rank(means)
        a -= a.mean(); b -= b.mean()
        spearman = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

        # per-item comparison against the 512-reference estimate
        paired = [(r, previous_csv[r["item"]]) for r in scored
                  if r["item"] in previous_csv and previous_csv[r["item"]]["mean"] is not None]
        errors = np.array([r["mean"] - p["mean"] for r, p in paired])
        flips = sum(1 for r, p in paired
                    if (r["mean"] > NEAR_ZERO) != (p["mean"] > NEAR_ZERO))
        results[mode] = {
            "items_total": len(rows),
            "items_with_triples": len(scored),
            "items_without_triples": len(rows) - len(scored),
            "mixed_sign_items": int(mixed.sum()),
            "mixed_sign_fraction": float(mixed.mean()),
            "node_mean_mean": float(means.mean()),
            "node_mean_median": float(np.median(means)),
            "node_mean_std": float(means.std()),
            "node_positive_mean_items": int((means > NEAR_ZERO).sum()),
            "node_negative_mean_items": int((means < -NEAR_ZERO).sum()),
            "spearman_degree_vs_mean": spearman,
            "degree_strata": degree_strata(levels, means, mixed),
            "vs_issue235": {
                "paired_items": len(paired),
                "mean_abs_error": float(np.abs(errors).mean()),
                "max_abs_error": float(np.abs(errors).max()),
                "mean_signed_error": float(errors.mean()),
                "sign_flip_items": int(flips),
                "sign_flip_fraction": flips / max(len(paired), 1),
                "mixed_235": previous["mixed_sign_fraction_of_scored"],
            },
            "csv": str(csv_path.relative_to(REPO)),
        }
        print(f"{mode}: items={len(scored)} mixed={results[mode]['mixed_sign_fraction']:.4f} "
              f"spearman={spearman:.4f} item_MAE={results[mode]['vs_issue235']['mean_abs_error']:.4f} "
              f"flips={flips}/{len(paired)}", flush=True)

    results["timing"] = {"total_s": round(time.time() - started, 1)}
    (OUT / "per_node_exact.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {OUT / 'per_node_exact.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
