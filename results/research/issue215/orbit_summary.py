"""Same-item orbit similarity versus recall, with the two controls the review demanded.

Metrics: the whole-vector readings (cosine, per-orbit rank correlation) and the per-graphlet readings
(the median per-orbit rank correlation inside that graphlet's orbit group, and the agreement of the
argmax orbit inside the group).  Controls: the arm's global behaviour-edge retention and its mean degree,
both of which the earlier tables showed can stand in for the effect.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
GROUPS = ("K5", "K4_tail", "K2_3", "C5", "T5", "P5", "K1_4", "K4", "triangle")


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    change = np.flatnonzero(np.diff(sorted_values) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(values)]]) - 1
    ranks = np.empty(len(values))
    ranks[order] = np.repeat((starts + ends) / 2.0, ends - starts + 1)
    return ranks


def spearman(first, second):
    a = average_ranks(first) - (len(first) - 1) / 2
    b = average_ranks(second) - (len(second) - 1) / 2
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def partial(first, second, *controls):
    a = average_ranks(first)
    b = average_ranks(second)
    design = np.column_stack([average_ranks(control) for control in controls] + [np.ones(len(a))])
    a = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
    b = b - design @ np.linalg.lstsq(design, b, rcond=None)[0]
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def permutation_p(first, second, observed, permutations=10000, seed=42):
    rng = np.random.default_rng(seed)
    hits = sum(1 for _ in range(permutations)
               if abs(spearman(first, rng.permutation(second))) >= abs(observed))
    return (hits + 1) / (permutations + 1)


def bootstrap_partial(first, second, control, draws=1000, seed=7):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        pick = rng.integers(0, len(first), len(first))
        values.append(partial(first[pick], second[pick], control[pick]))
    values = np.asarray([value for value in values if np.isfinite(value)])
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> int:
    data = json.loads((OUT / "orbit_similarity.json").read_text())
    rows = [row for row in data["rows"] if "recall" in row]
    recall = np.array([row["recall"] for row in rows])
    edge_recall = np.array([row["edge_recall"] for row in rows])
    degree = np.array([row["mean_degree"] for row in rows])

    metrics = ["node_orbit_cos", "node_orbit_rho", "node_orbit_rho_mean"]
    metrics += [f"rho_{name}" for name in GROUPS] + [f"role_{name}" for name in GROUPS]
    summary = {"n_arms": len(rows), "metrics": {}}
    print(f"{len(rows)} arms")
    print(f"  {'metric':<22}{'mean':>9}{'rho':>8}{'p':>8}{'|edge':>9}{'ci':>20}{'|+deg':>8}")
    for metric in metrics:
        if not all(metric in row for row in rows):
            continue
        values = np.array([row[metric] for row in rows], dtype=float)
        keep = np.isfinite(values)
        if keep.sum() < 10 or np.ptp(values[keep]) == 0:
            continue
        observed = spearman(values[keep], recall[keep])
        p_value = permutation_p(values[keep], recall[keep], observed)
        with_edge = partial(values[keep], recall[keep], edge_recall[keep])
        interval = bootstrap_partial(values[keep], recall[keep], edge_recall[keep])
        with_both = partial(values[keep], recall[keep], edge_recall[keep], degree[keep])
        summary["metrics"][metric] = {"mean": float(values[keep].mean()), "rho": observed,
                                      "p": p_value, "partial_edge": with_edge,
                                      "partial_edge_ci": interval, "partial_both": with_both,
                                      "n": int(keep.sum())}
        print(f"  {metric:<22}{values[keep].mean():>9.4f}{observed:>+8.3f}{p_value:>8.4f}"
              f"{with_edge:>+9.3f}[{interval[0]:+.3f},{interval[1]:+.3f}]{with_both:>+8.3f}")
    (OUT / "orbit_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
