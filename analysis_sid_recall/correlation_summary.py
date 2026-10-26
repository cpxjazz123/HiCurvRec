"""One correlation survey over every metric of all three batteries.

The three batteries are merged on the SID's sha256, so every row pairs one
SID's metrics with that same SID's Stage 3 recall. For each metric the survey
reports

* the Spearman rank correlation with ``test_recall@10`` and its bootstrap
  interval,
* a permutation p-value for that rank correlation,
* the Benjamini-Hochberg adjusted p-value over the metrics that were actually
  tested, which is what says whether a metric survives the fact that dozens of
  metrics were looked at,
* the best out-of-sample R2 of the non-linear shape analysis, which separates
  "ranks the tables correctly" from "predicts a new table's recall".

Duplicates are collapsed before the correction: the lift of a metric is that
metric divided by a constant, so it carries exactly the same information and
must not count as a second test.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 42
DRAWS = 2000
LABELS = {"sid_sha256", "variant", "recall@10", "recall@5", "n_eval", "sid_resolution"}


def load() -> list[dict]:
    """Merge the three batteries on the SID's identity."""
    merged: dict[str, dict] = {}
    for name in ("sid_behaviour_metrics.csv", "sid_behaviour_metrics2.csv",
                 "sid_behaviour_metrics3.csv"):
        path = HERE / name
        if not path.is_file():
            continue
        for row in csv.DictReader(path.open()):
            key = (row.get("sid_sha256") or "").strip()
            if not key:
                raise ValueError(f"{name} carries no sid_sha256")
            entry = merged.setdefault(key, {"sid_sha256": key})
            for field, value in row.items():
                if field in ("variant", "sid_sha256"):
                    continue
                try:
                    entry[field] = float(value)
                except (TypeError, ValueError):
                    pass
    return list(merged.values())


def ranks(values) -> np.ndarray:
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(len(values), dtype=float)
    # average the ranks of ties
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def bootstrap_ci(x, y, *, draws: int = DRAWS, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        index = rng.integers(0, len(x), len(x))
        value = spearman([x[i] for i in index], [y[i] for i in index])
        if not np.isnan(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def permutation_p(x, y, *, draws: int = DRAWS, seed: int = SEED) -> float:
    observed = abs(spearman(x, y))
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    extreme = 0
    for _ in range(draws):
        if abs(spearman(x, rng.permutation(y))) >= observed:
            extreme += 1
    return (extreme + 1) / (draws + 1)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Adjusted p-values, monotone and capped at one."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 1.0
    for rank in range(count - 1, -1, -1):
        index = order[rank]
        value = p_values[index] * count / (rank + 1)
        running = min(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


def duplicate_of(metric: str) -> str:
    """The metric a lift is the same information as.

    A lift is the metric divided by the random-pair baseline, a constant, so it
    must not count as a second test. The two families name their base metric
    differently: ``cooccur_lift_L3`` belongs to ``cooccur_overlap_L3`` while
    ``adjacency_lift_L3`` belongs to ``adjacency_L3``.
    """
    for level in ("L1", "L2", "L3"):
        if metric == f"cooccur_lift_{level}":
            return f"cooccur_overlap_{level}"
        if metric == f"adjacency_lift_{level}":
            return f"adjacency_{level}"
    return metric


def main() -> int:
    rows = load()
    shapes: dict[str, dict] = {}
    shapes_path = HERE / "nonlinear_shapes.csv"
    if shapes_path.is_file():
        shapes = {row["metric"]: row for row in csv.DictReader(shapes_path.open())}

    candidates: dict[str, list[float]] = {}
    for row in rows:
        for field, value in row.items():
            if field in LABELS or not isinstance(value, float):
                continue
            candidates.setdefault(field, []).append(value)

    tested: list[dict] = []
    seen_families: dict[str, str] = {}
    for metric, values in sorted(candidates.items()):
        column = [row[metric] for row in rows if metric in row]
        target = [row["recall@10"] for row in rows if metric in row]
        if len(column) < 30:
            continue
        if len(set(column)) < 5:
            continue
        rho = spearman(column, target)
        low, high = bootstrap_ci(column, target)
        p_value = permutation_p(column, target)
        tested.append({
            "metric": metric, "n": len(column), "spearman": rho,
            "ci_low": low, "ci_high": high, "p_value": p_value,
            "best_shape": (shapes.get(metric) or {}).get("best_shape", ""),
            "best_r2": (shapes.get(metric) or {}).get("best_r2", ""),
            "duplicate_of": duplicate_of(metric),
        })

    # Collapse the duplicates: a lift is its metric divided by a constant.
    unique: list[dict] = []
    for row in sorted(tested, key=lambda item: item["metric"]):
        parent = row["duplicate_of"]
        if parent != row["metric"] and any(other["metric"] == parent for other in tested):
            continue
        unique.append(row)

    adjusted = benjamini_hochberg([row["p_value"] for row in unique])
    for row, value in zip(unique, adjusted):
        row["p_adjusted"] = value

    fields = ["metric", "n", "spearman", "ci_low", "ci_high", "p_value", "p_adjusted",
              "best_shape", "best_r2"]
    with (HERE / "correlation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(unique, key=lambda item: -abs(item["spearman"])):
            writer.writerow(row)

    print("%d SIDs, %d metrics tested (%d after collapsing the lifts)\n" % (
        len(rows), len(tested), len(unique)))
    print("%-28s %3s %7s %-16s %9s %9s %-9s %6s" % (
        "metric", "n", "rho", "CI95", "p", "p(BH)", "best shape", "R2"))
    for row in sorted(unique, key=lambda item: -abs(item["spearman"])):
        print("%-28s %3d %7.3f [%6.3f, %6.3f] %9.4f %9.4f %-9s %6s" % (
            row["metric"], row["n"], row["spearman"], row["ci_low"], row["ci_high"],
            row["p_value"], row["p_adjusted"], row["best_shape"],
            ("%.3f" % float(row["best_r2"])) if row["best_r2"] else "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
