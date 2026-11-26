"""Independent score and rank-statistics reference for Issue #266.

Pure Python; it does not import the production aggregation module and does not
use scipy. The score is evaluated from the issue's frozen formula; the rank
statistics are implemented from their definitions so the aggregate output has a
second mathematical path.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


LO_C = 0.38207152485847473
HI_C = 3.225719451904297
LO_G = 1.6841177967230931
HI_G = 32.588647951083324
WEIGHT_C = 0.8
WEIGHT_G = 0.2


def score(c: float, g1: float) -> dict:
    """Frozen S, with unclipped components and no fitted constants."""
    c_component = WEIGHT_C * (float(c) - LO_C) / (HI_C - LO_C)
    g_component = WEIGHT_G * (float(g1) - LO_G) / (HI_G - LO_G)
    return {"c_component": c_component, "g_component": g_component,
            "S": c_component + g_component}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _rank_average(values: Sequence[float]) -> list[float]:
    """1-based average ranks; equal values share the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = _mean(x), _mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    sx = math.sqrt(sum(value * value for value in dx))
    sy = math.sqrt(sum(value * value for value in dy))
    if sx == 0 or sy == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    return pearson(_rank_average(x), _rank_average(y))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall tau-b with standard tie corrections; None for a constant axis."""
    if len(x) != len(y) or len(x) < 2:
        return None
    n = len(x)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0:
                ties_x += 1
            if dy == 0:
                ties_y += 1
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) // 2
    denominator = math.sqrt((n0 - ties_x) * (n0 - ties_y))
    if denominator == 0:
        return None
    if concordant + discordant == 0:
        return None
    return (concordant - discordant) / denominator


def rank_order(rows: Iterable[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row["recall"], row["index"]))


def adjacent_monotonic(rows: Iterable[dict]) -> dict:
    ordered = rank_order(rows)
    pairs = []
    for left, right in zip(ordered, ordered[1:]):
        pairs.append({
            "left": left["index"], "right": right["index"],
            "left_recall": left["recall"], "right_recall": right["recall"],
            "left_S": left["S"], "right_S": right["S"],
            "strict": bool(left["S"] < right["S"]),
            "tie": bool(left["S"] == right["S"]),
        })
    return {"order": [row["index"] for row in ordered], "pairs": pairs,
            "all_strict": all(pair["strict"] for pair in pairs),
            "violations": [pair for pair in pairs if not pair["strict"]]}


def inversions(rows: Iterable[dict]) -> dict:
    ordered = rank_order(rows)
    pairs = []
    tied_recall = []
    tied_score = []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            left, right = ordered[i], ordered[j]
            if left["recall"] == right["recall"]:
                tied_recall.append([left["index"], right["index"]])
                continue
            if left["S"] == right["S"]:
                tied_score.append([left["index"], right["index"]])
                continue
            if left["S"] > right["S"]:
                pairs.append({"lower_recall": left["index"],
                              "higher_recall": right["index"],
                              "lower_S": left["S"], "higher_S": right["S"]})
    return {"count": len(pairs), "pairs": pairs,
            "violating_arms": sorted({arm for pair in pairs
                                      for arm in (pair["lower_recall"], pair["higher_recall"])}),
            "tied_recall_pairs": tied_recall,
            "tied_score_pairs": tied_score}


def summary(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    ordered = rank_order(rows)
    recalls = [row["recall"] for row in ordered]
    scores = [row["S"] for row in ordered]
    return {
        "n": len(rows),
        "order_by_recall": [row["index"] for row in ordered],
        "pearson": pearson(recalls, scores),
        "spearman": spearman(recalls, scores),
        "kendall_tau_b": kendall_tau_b(recalls, scores),
        "adjacent_monotonic": adjacent_monotonic(ordered),
        "inversions": inversions(ordered),
    }
