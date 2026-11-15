"""Pair each graph-level structural metric with the arm's recall.

Lower is better for the distances (GCD, DeltaCon), higher is better for the overlap and role
measures.  The direction is fixed here, before the numbers are read, so a sign cannot be chosen after
the fact; the raw and the collision-controlled rank correlations are both reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"

DISTANCES = ("gcd73", "gcd15", "deltacon")
OVERLAPS = ("edge_recall", "edge_jaccard", "edge_precision")
ROLES = ("orbit_rho_median", "profile_cosine", "degree_rank", "star_role_kept",
         "star_leaves_recovered", "star_full_kept")


def ranks(values):
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values))
    out[order] = np.arange(len(values))
    return out


def spearman(first, second):
    a, b = ranks(first) - (len(first) - 1) / 2, ranks(second) - (len(second) - 1) / 2
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def partial(first, second, control):
    """Rank correlation of first/second after removing the rank of the control from both."""
    a, b, c = ranks(first), ranks(second), ranks(control)
    c = c - c.mean()
    a = a - a.mean() - (a - a.mean()).dot(c) / c.dot(c) * c
    b = b - b.mean() - (b - b.mean()).dot(c) / c.dot(c) * c
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def permutation_p(first, second, observed, permutations=2000, seed=42):
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        if abs(spearman(first, rng.permutation(second))) >= abs(observed):
            count += 1
    return (count + 1) / (permutations + 1)


def main() -> int:
    data = json.loads((OUT / "quantised_structure_all.json").read_text())
    rows = [row for row in data["rows"] if "recall" in row]
    report = json.loads((REPO / "analysis_sid_recall" / "results" / "report.json").read_text())
    collision = {pair["number"]: pair.get("l3_collision") for pair in report["pairs"]}
    recall = np.array([row["recall"] for row in rows])
    control = np.array([collision.get(row["number"], np.nan) for row in rows])
    usable = np.isfinite(control)

    print(f"{len(rows)} arms; {int(usable.sum())} carry a collision rate")
    print(f"  {'metric':<24}{'mean':>10}{'rho':>9}{'p':>8}{'partial':>10}")
    summary = {"n_arms": len(rows), "metrics": {}}
    for metric, sign in ([(name, -1) for name in DISTANCES] + [(name, +1) for name in OVERLAPS]
                         + [(name, +1) for name in ROLES]):
        values = np.array([row[metric] if row.get(metric) is not None else np.nan for row in rows])
        keep = np.isfinite(values)
        if keep.sum() < 10:
            continue
        observed = spearman(values[keep], recall[keep])
        p_value = permutation_p(values[keep], recall[keep], observed, permutations=500)
        partial_rho = partial(values[keep & usable], recall[keep & usable], control[keep & usable])
        summary["metrics"][metric] = {"direction": sign, "mean": float(values[keep].mean()),
                                      "rho": observed, "p": p_value, "partial": partial_rho,
                                      "n": int(keep.sum())}
        print(f"  {metric:<24}{values[keep].mean():>10.3f}{observed:>+9.3f}{p_value:>8.3f}"
              f"{partial_rho:>+10.3f}")
    (OUT / "graph_metrics_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
