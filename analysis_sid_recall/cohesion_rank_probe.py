"""Is the level-1 partition blind to the behaviour cohesion rank?

The mechanism of round_20261106_020541_888 puts a bounded cohesion rank rho_i
into the level-1 cost, with a learned per-code stratum s_k = sigmoid(a_k): an
item is cheaper at a code whose stratum is close to its own rank. The mechanism
therefore assumes that the level-1 codes can be ORDERED by the cohesion rank -
if the partition already separates the ranks perfectly, there is nothing to
gain; if it is blind to them, the ordering is the mechanism's whole effect.

This probe measures that, on the arms that exist today:

    rho_i   = mid-rank of the item's cohesion over the training items
    eta^2   = share of the variance of rho that the level-1 partition explains

A partition that is blind to the rank has eta^2 near 0; one that already
stratifies it has eta^2 near 1. The training split is the only source of the
graph, and no validation or test interaction is read.

Usage:
    python3 analysis_sid_recall/cohesion_rank_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"

ARMS = {
    "TIGER_RQ-VAE": REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json",
    "TIGER_BEHAVIOR_SAMPLE_RQ-VAE": (REPO / "results" / "stage2_RQ-VAE" /
                                     "TIGER_BEHAVIOR_SAMPLE_RQ-VAE" / "resample" /
                                     "item_sids.json"),
    "TIGER_pmi-geomargin-l1-rq": (REPO / "results" / "stage2_RQ-VAE" /
                                   "TIGER_pmi-geomargin-l1-rq" / "item_sids.json"),
}

#: The neighbourhood width is the mean degree, rounded up, as the mechanism states.
TOP_FRACTION = 1.0


def adjacency(frame) -> tuple:
    """The training split's undirected co-occurrence counts (train only)."""
    counts: dict[tuple[int, int], float] = {}
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for first, second in zip(values[:-1], values[1:]):
            if first == second:
                continue
            key = (first, second) if first < second else (second, first)
            counts[key] = counts.get(key, 0.0) + 1.0
    return counts


def cohesion(counts: dict) -> dict:
    """coh(i) = the mean Jaccard-style overlap of i's neighbourhood, in [0,1]."""
    neighbours: dict[int, set] = {}
    for (first, second) in counts:
        neighbours.setdefault(first, set()).add(second)
        neighbours.setdefault(second, set()).add(first)
    degree = {item: len(hood) for item, hood in neighbours.items()}
    width = max(1, int(np.ceil(np.mean(list(degree.values())) * TOP_FRACTION)))
    scores = {}
    for item, hood in neighbours.items():
        kept = sorted(hood, key=lambda other: (-counts.get(
            (item, other) if item < other else (other, item), 0.0), other))[:width]
        total = 0.0
        for other in kept:
            shared = len(set(kept) & set(sorted(
                neighbours.get(other, ()), key=lambda third: (-counts.get(
                    (other, third) if other < third else (third, other), 0.0), third))[:width]))
            total += shared / np.sqrt(max(1, degree[item]) * max(1, degree[other]))
        scores[item] = total / max(1, len(kept))
    return scores


def midrank(values: dict) -> dict:
    """The mid-rank empirical CDF of a per-item statistic, scaled to (0,1)."""
    items = sorted(values)
    ordered = sorted(values[item] for item in items)
    n = len(ordered)
    position = {}
    index = 0
    while index < n:
        end = index
        while end + 1 < n and ordered[end + 1] == ordered[index]:
            end += 1
        rank = (index + end) / 2.0
        for spot in range(index, end + 1):
            position[ordered[spot]] = rank
        index = end + 1
    return {item: (position[values[item]] + 0.5) / n for item in items}


def eta_squared(rank: dict, sid_path: Path) -> dict:
    rows = json.loads(Path(sid_path).read_text())
    groups: dict[int, list] = {}
    for key, value in rows.items():
        item = int(key)
        if item not in rank:
            continue
        groups.setdefault(int(list(value)[0]), []).append(rank[item])
    samples = [value for values in groups.values() for value in values]
    if not samples:
        return {"codes": 0}
    overall = float(np.mean(samples))
    between = sum(len(values) * (float(np.mean(values)) - overall) ** 2
                  for values in groups.values())
    within = sum(float(np.sum((np.array(values) - np.mean(values)) ** 2))
                 for values in groups.values())
    total = between + within
    means = sorted((float(np.mean(values)), code, len(values))
                   for code, values in groups.items())
    return {
        "codes": len(groups),
        "items": len(samples),
        "eta_squared": (between / total) if total > 0 else 0.0,
        "lowest_code_median_rank": means[0][0] if means else None,
        "highest_code_median_rank": means[-1][0] if means else None,
        "spread_between_code_means": (means[-1][0] - means[0][0]) if means else None,
    }


def main() -> int:
    frame = pd.read_parquet(TRAIN)
    counts = adjacency(frame)
    scores = cohesion(counts)
    rank = midrank(scores)
    report = {"train_items_with_a_neighbourhood": len(rank),
              "cohesion": {"min": float(min(scores.values())),
                           "median": float(np.median(list(scores.values()))),
                           "max": float(max(scores.values()))}}
    for arm, path in ARMS.items():
        if not Path(path).is_file():
            report[arm] = {"status": "NOT_AVAILABLE"}
            continue
        report[arm] = eta_squared(rank, Path(path))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
