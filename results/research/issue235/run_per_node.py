"""Per-item curvature statistics on the frozen behaviour graph.

The global distribution says the graph leans negative; this pass asks whether a
single item can be both positively and negatively curved, which is the property a
per-item mixed-curvature method would need.  Everything is computed on the same
seeded reference set as the global analysis, so the two are directly comparable.

Usage (no arguments, per CLAUDE.md 9-10):
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/issue235/run_per_node.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import run_analysis as ra

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue235"
NEAR_ZERO = ra.NEAR_ZERO


def main() -> int:
    started = time.time()
    adjacency, metadata = ra.load_graph()
    degree = np.diff(adjacency.indptr).astype(int)
    distances, reachable = ra.reference_tables(adjacency, ra.SEED)
    centres, lefts, rights, coverage = ra.enumerate_triples(
        adjacency, ra.TRIPLES_PER_CENTRE, ra.SEED)
    keep = ra.geodesic_mask(adjacency, lefts, rights)
    centres, lefts, rights = centres[keep], lefts[keep], rights[keep]
    values, counts = ra.curvature(centres, lefts, rights, distances, reachable)
    print(f"{values.size} geodesic triples over {np.unique(centres).size} centres "
          f"({time.time() - started:.0f}s)", flush=True)

    order = np.argsort(centres, kind="stable")
    centres_sorted = centres[order]
    values_sorted = values[order]
    bounds = np.searchsorted(centres_sorted, np.arange(adjacency.shape[0] + 1))

    rows = []
    for node in range(adjacency.shape[0]):
        block = values_sorted[bounds[node]:bounds[node + 1]]
        if block.size == 0:
            rows.append({"item": node, "degree": int(degree[node]), "n_triples": 0,
                         "mean": None, "median": None, "std": None,
                         "positive": 0, "negative": 0, "near_zero": 0,
                         "positive_fraction": None, "negative_fraction": None,
                         "mixed_sign": False})
            continue
        positive = int((block > NEAR_ZERO).sum())
        negative = int((block < -NEAR_ZERO).sum())
        rows.append({
            "item": node,
            "degree": int(degree[node]),
            "n_triples": int(block.size),
            "mean": float(block.mean()),
            "median": float(np.median(block)),
            "std": float(block.std()),
            "positive": positive,
            "negative": negative,
            "near_zero": int(block.size) - positive - negative,
            "positive_fraction": positive / block.size,
            "negative_fraction": negative / block.size,
            "mixed_sign": positive > 0 and negative > 0,
        })

    scored = [row for row in rows if row["n_triples"] > 0]
    node_path = OUT / "node_curvature.csv"
    with node_path.open("w", encoding="utf-8") as sink:
        sink.write("item,degree,n_triples,mean,median,std,positive,negative,near_zero,"
                   "positive_fraction,negative_fraction,mixed_sign\n")
        for row in rows:
            fmt = lambda value: "" if value is None else (
                f"{value:.6f}" if isinstance(value, float) else str(value))
            sink.write(",".join(fmt(row[key]) for key in (
                "item", "degree", "n_triples", "mean", "median", "std", "positive",
                "negative", "near_zero", "positive_fraction", "negative_fraction",
                "mixed_sign")) + "\n")

    means = np.array([row["mean"] for row in scored])
    mixed = np.array([row["mixed_sign"] for row in scored])
    summary = {
        "items_total": len(rows),
        "items_with_triples": len(scored),
        "items_without_triples": len(rows) - len(scored),
        "mixed_sign_items": int(mixed.sum()),
        "mixed_sign_fraction_of_scored": float(mixed.mean()),
        "node_mean_mean": float(means.mean()),
        "node_mean_median": float(np.median(means)),
        "node_mean_std": float(means.std()),
        "node_positive_mean_items": int((means > NEAR_ZERO).sum()),
        "node_negative_mean_items": int((means < -NEAR_ZERO).sum()),
        "node_mean_quantiles": {
            "p05": float(np.percentile(means, 5)),
            "p25": float(np.percentile(means, 25)),
            "p75": float(np.percentile(means, 75)),
            "p95": float(np.percentile(means, 95)),
            "min": float(means.min()),
            "max": float(means.max()),
        },
        "degree_strata": _degree_strata(scored),
        "examples": _examples(scored),
        "triple_coverage": coverage,
        "near_zero_tolerance": NEAR_ZERO,
        "seed": ra.SEED,
        "elapsed_s": round(time.time() - started, 1),
    }
    (OUT / "per_node_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=1))
    print(f"-> {node_path}")
    return 0


def _degree_strata(scored: list[dict]) -> list[dict]:
    edges = [0, 5, 13, 50, 200, 10 ** 9]
    labels = ["1-4", "5-12", "13-49", "50-199", "200+"]
    out = []
    for low, high, label in zip(edges[:-1], edges[1:], labels):
        block = [row for row in scored if low <= row["degree"] < high]
        if not block:
            continue
        means = np.array([row["mean"] for row in block])
        out.append({
            "degree_range": label,
            "items": len(block),
            "mean_of_means": float(means.mean()),
            "median_of_means": float(np.median(means)),
            "negative_mean_items": int((means < -NEAR_ZERO).sum()),
            "mixed_sign_items": int(sum(1 for row in block if row["mixed_sign"])),
        })
    return out


def _examples(scored: list[dict], count: int = 5) -> dict:
    def take(pool, key, reverse):
        return [{"item": row["item"], "degree": row["degree"], "n_triples": row["n_triples"],
                 "mean": round(row["mean"], 4),
                 "positive": row["positive"], "negative": row["negative"]}
                for row in sorted(pool, key=lambda r: r[key], reverse=reverse)[:count]]
    mixed = [row for row in scored if row["mixed_sign"]]
    return {
        "most_negative": take(scored, "mean", False),
        "most_positive": take(scored, "mean", True),
        "most_mixed": take(mixed, "n_triples", True) if mixed else [],
    }


if __name__ == "__main__":
    raise SystemExit(main())
