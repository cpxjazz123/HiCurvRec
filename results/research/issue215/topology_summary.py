"""Which topology-preservation metrics track recall, once overall edge retention is controlled.

The per-structure table showed that raw correlations mostly measured how many behaviour edges an arm
keeps.  The same control is applied here: every metric gets its raw rank correlation, its partial rank
correlation controlling the arm's behaviour-edge retention, and a paired bootstrap interval.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
METRICS = ("orbit_total_cos", "community_ari", "community_nmi", "degree_rho", "clustering_rho",
           "core_rho", "hub_jaccard_1pct", "star_centre_top10", "star_centre_sparse")
LOWER_BETTER = ("orbit_total_l1",)


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


def partial(first, second, control):
    a, b = average_ranks(first), average_ranks(second)
    c = average_ranks(control)
    design = np.column_stack([c, np.ones(len(c))])
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
    data = json.loads((OUT / "topology_preservation.json").read_text())
    rows = [row for row in data["rows"] if "recall" in row]
    recall = np.array([row["recall"] for row in rows])
    behaviour = json.loads((OUT / "quantised_structure_all.json").read_text())
    edge_recall = {row["number"]: row["edge_recall"] for row in behaviour["rows"] if "recall" in row}
    control = np.array([edge_recall[row["number"]] for row in rows])

    summary = {"n_arms": len(rows), "metrics": {}}
    print(f"{len(rows)} arms; control (global edge recall) mean {control.mean():.4f}")
    print(f"  {'metric':<22}{'mean':>9}{'rho':>8}{'p':>8}{'|edge':>9}{'ci':>20}")
    for metric in METRICS:
        values = np.array([row[metric] for row in rows], dtype=float)
        keep = np.isfinite(values)
        observed = spearman(values[keep], recall[keep])
        p_value = permutation_p(values[keep], recall[keep], observed)
        with_edge = partial(values[keep], recall[keep], control[keep])
        interval = bootstrap_partial(values[keep], recall[keep], control[keep])
        summary["metrics"][metric] = {"mean": float(values[keep].mean()), "rho": observed,
                                      "p": p_value, "partial_edge": with_edge,
                                      "partial_edge_ci": interval, "n": int(keep.sum())}
        print(f"  {metric:<22}{values[keep].mean():>9.4f}{observed:>+8.3f}{p_value:>8.4f}"
              f"{with_edge:>+9.3f}[{interval[0]:+.3f},{interval[1]:+.3f}]")
    (OUT / "topology_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
