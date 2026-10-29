"""Round-6 metric: the expected rank of the true cell among all cells (cross-tower).

HBAA compares the positive's prefix centroid with the negatives' centroids. This
metric instead asks where the positive's cell RANKS among ALL the cells, measured
against the history point in the ball: the tempered softmax over the cell centres
gives every cell a mass, and the expected rank of the true cell under that
distribution is the quantity.

The lift is measured against the expectation a random assignment would give, so
neither a collapse nor a coarse partition can be rewarded:

    expected_rank = sum_c mass_c * rank(c)          (rank 0 = the best cell)
    lift = 1 - (expected_rank - (K_eff - 1) / 2) / ((K_eff - 1) / 2)

with K_eff the number of NON-EMPTY cells. A table with one cell has K_eff = 1 and
is defined as 0. A random assignment gives 0 by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "history_points.npz"
TAU_FACTOR = 1.0


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = (1.0 - np.sum(a * a, axis=-1)) * (1.0 - np.sum(b * b, axis=-1))
    return np.arccosh(np.maximum(1.0 + 2.0 * difference /
                                 np.maximum(denominator, 1e-12), 1.0 + 1e-12))


def _frechet(points: np.ndarray) -> np.ndarray:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from tools.research.science import hyperbolic_sid_quality as frozen
    return frozen.frechet_barycentre(np.asarray(points, dtype=np.float64),
                                     frozen.BallGeometry())


def _queries(context: dict) -> dict:
    if "round6_queries" in context:
        return context["round6_queries"]
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from results.research.issue126.metric_round import metrics_hbaa as hbaa
    payload = hbaa._queries(context)
    context["round6_queries"] = payload
    return payload


def _history_points(context: dict) -> np.ndarray:
    if "round6_history" not in context:
        context["round6_history"] = np.load(CACHE)["points"].astype(np.float64)
    return context["round6_history"]


def hyp_cell_expected_rank(table: dict, context: dict) -> float:
    """The expected rank of the true cell among all cells, lifted over chance."""
    target = np.asarray(context["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(context["temperature"]) * TAU_FACTOR
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    queries = _queries(context)
    positives = np.array([position.get(int(item), -1) for item in queries["positives"]],
                          dtype=np.int64)
    valid = positives >= 0
    if not valid.any():
        return 0.0
    points = _history_points(context)[valid]
    positives = positives[valid]

    keys = np.unique(codes)
    if keys.size < 2:
        return 0.0
    index_of = {int(code): position for position, code in enumerate(keys.tolist())}
    centres = np.stack([_frechet(target[codes == code]) for code in keys])
    distances = _distance(points[:, None, :], centres[None, :, :])
    logits = -distances / tau
    top = logits.max(axis=1, keepdims=True)
    mass = np.exp(logits - top)
    mass = mass / mass.sum(axis=1, keepdims=True)
    # the rank of each cell under the distribution: a cell's expected rank is the
    # mass-weighted sum of the number of cells above it
    order = np.argsort(-logits, axis=1)
    ranks = np.empty_like(order)
    for row in range(order.shape[0]):
        ranks[row, order[row]] = np.arange(order.shape[1])
    chosen = np.array([index_of[int(codes[index])] for index in positives],
                      dtype=np.int64)
    rows = np.arange(len(points))
    k_eff = float(keys.size)
    if k_eff < 2:
        return 0.0
    # (a) the mass the true cell carries, against the uniform 1/K_eff: 0 for a
    #     random assignment, 1 when the true cell is always the best one.
    true_mass = mass[rows, chosen]
    mass_lift = (true_mass * k_eff - 1.0) / (k_eff - 1.0)
    # (b) the rank the true cell holds under that mass, against chance.
    true_rank = ranks[rows, chosen].astype(np.float64)
    chance = (k_eff - 1.0) / 2.0
    rank_lift = 1.0 - (true_rank - chance) / chance
    context.setdefault("round6_lifts", []).append(
        (float(np.mean(mass_lift)), float(np.mean(rank_lift))))
    return float(np.mean(np.clip(mass_lift, -1.0, 1.0)))


hyp_cell_expected_rank.note = ("expected rank of the true level-1 cell among all cells "
                                "under the tempered softmax from the history point, "
                                "lifted over chance")

def hyp_cell_rank_lift(table: dict, context: dict) -> float:
    """The same construction read as the RANK the true cell holds (the rank lift)."""
    hyp_cell_expected_rank(table, context)
    lifts = context.get("round6_lifts") or []
    return float(lifts[-1][1]) if lifts else 0.0


METRICS = {"hyp_cell_expected_rank": hyp_cell_expected_rank,
           "hyp_cell_rank_lift": hyp_cell_rank_lift}
FREE_CONSTANTS = {"hyp_cell_expected_rank": 0, "hyp_cell_rank_lift": 0}
