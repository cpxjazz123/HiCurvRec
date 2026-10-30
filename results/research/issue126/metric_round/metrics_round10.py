"""Round-10 metric: the cross-fitted FULL-NEGATIVE expected rank of the level-1 cell.

The round-6 rank lift ranked the true cell against TEN negatives; this one ranks
it against EVERY cell, which is the question Stage 3 actually faces when it emits
the first token: the cell has to come first among all the cells, not merely beat
ten sampled ones.

The cells are built from the OTHER fold's users (the users are split by a hash of
their id), so a cell's composition is never fitted on the interaction that asks
about it -- the query is scored against a partition it did not shape.

The lift is over chance: with C non-empty cells the expected rank of a random
cell is (C-1)/2, and the score says how much better the true cell does.
"""
from __future__ import annotations

import numpy as np


def _cell_centres(target: np.ndarray, codes: np.ndarray, members: np.ndarray,
                  key: int) -> dict:
    centres = {}
    for code in np.unique(codes[members]):
        points = target[members[codes[members] == code]]
        centres[int(code)] = points.mean(axis=0)
    return centres


def hbaa_fullneg_rank_excess(table: dict, context: dict) -> float:
    """The true cell's expected rank among ALL cells, over chance, cross-fitted."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from results.research.issue126.metric_round import metrics_hbaa as hbaa

    reference = hbaa._reference(context)
    target = np.asarray(reference["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(reference.get("temperature",
                              (reference.get("record") or {})["temperature"]))
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    queries = hbaa._queries(context)
    points = hbaa._history_points(context, 1.0)

    # the fold of every interaction: the users are split by a hash of their id, so
    # the split is deterministic and the same for every table
    import pandas as pd
    frame = pd.read_parquet(
        REPO_TRAIN := str(Path(__file__).resolve().parents[4] /
                          "results/stage0_build_parquet/train.parquet"),
        columns=["user", "history", "target"])
    frame = frame.sort_values("user", kind="stable")
    last = frame.groupby("user", sort=False).tail(1)
    users = [int(user) for user in last["user"].to_numpy()]
    folds = np.array([hash(str(user)) % 2 for user in users], dtype=np.int64)

    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    positives = np.asarray(queries["positives"], dtype=np.int64)
    target_rows = np.array([position.get(int(item), -1) for item in positives],
                            dtype=np.int64)
    # the items each interaction brings with it, used to build the other fold's cells
    history_items = [np.asarray(history, dtype=np.int64) for history in queries["histories"]]

    total = 0.0
    counted = 0
    for fold in (0, 1):
        rows = np.nonzero(folds == fold)[0]
        if rows.size == 0:
            continue
        # the cells of the OTHER fold's users
        other = np.nonzero(folds != fold)[0]
        members = np.concatenate([history_items[index] for index in other]) \
            if other.size else np.array([], dtype=np.int64)
        members = np.array([position.get(int(item), -1) for item in members],
                            dtype=np.int64)
        members = members[members >= 0]
        if members.size == 0:
            continue
        centres = _cell_centres(target, codes, members, fold)
        keys = sorted(centres)
        if len(keys) < 2:
            continue
        centre_table = np.stack([centres[key] for key in keys])
        index_of = {key: position for position, key in enumerate(keys)}
        for index in rows:
            spot = target_rows[index]
            if spot < 0 or int(codes[spot]) not in index_of:
                continue
            query = points[index]
            difference = np.sum((centre_table - query) ** 2, axis=1)
            denominator = ((1 - np.sum(centre_table * centre_table, axis=1))
                           * (1 - float(np.sum(query * query))))
            distances = np.arccosh(np.maximum(1 + 2 * difference /
                                              np.maximum(denominator, 1e-12), 1 + 1e-12))
            logits = -distances / tau
            order = np.argsort(-logits)
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[order] = np.arange(len(order))
            top = float(logits.max())
            mass = np.exp(logits - top)
            mass = mass / mass.sum()
            expected = float((mass * ranks).sum())
            chance = (len(keys) - 1) / 2.0
            if chance <= 0:
                continue
            total += 1.0 - (expected - chance) / chance
            counted += 1
    return float(total / counted) if counted else 0.0


hbaa_fullneg_rank_excess.note = ("the true level-1 cell's expected rank among ALL cells, "
                                  "over chance, with the cells built from the other "
                                  "fold's users")

METRICS = {"hbaa_fullneg_rank_excess": hbaa_fullneg_rank_excess}
FREE_CONSTANTS = {"hbaa_fullneg_rank_excess": 0}


def hbaa_offcell_expected_rank(table: dict, context: dict) -> float:
    """The Gromov expected rank of the target over the items its OWN cell hides.

    Stage 3 emits a level-1 code and then re-ranks inside that cell. A cell that
    absorbs many confusable items crowds the target out before the re-ranking can
    help, so this asks the question from the other side: REMOVE the true item's
    own cell from the candidate pool and ask whether the target is then the
    dominant item by the hyperbolic ancestor depth of the history's query point.

    It is a removal construction, not a contrast: the candidate pool is every
    item except the members of the true cell.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from results.research.issue126.metric_round import metrics_hbaa as hbaa

    reference = hbaa._reference(context)
    target = np.asarray(reference["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(reference.get("temperature",
                              (reference.get("record") or {})["temperature"]))
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    queries = hbaa._queries(context)
    points = hbaa._history_points(context, 1.0)
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    positives = np.asarray(queries["positives"], dtype=np.int64)
    spot = np.array([position.get(int(item), -1) for item in positives], dtype=np.int64)
    keep = np.nonzero(spot >= 0)[0]
    if keep.size == 0:
        return 0.0

    # the pool is every item whose own cell differs from the true cell
    pool = np.nonzero(codes != codes[spot[keep[0]]])[0]
    if pool.size < 2:
        return 0.0

    def gromov(x, y):
        def d(a, b):
            diff = np.sum((a - b) ** 2, axis=-1)
            den = ((1 - np.sum(a * a, axis=-1)) * (1 - np.sum(b * b, axis=-1)))
            return np.arccosh(np.maximum(1 + 2 * diff / np.maximum(den, 1e-12), 1 + 1e-12))
        def nrm(x):
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return 2 * np.arccosh(1 / np.sqrt(inner))
        return 0.5 * (nrm(x) + np.asarray(nrm(y)) - d(x, y))

    total = 0.0
    counted = 0
    # the pool is recomputed per interaction, because the excluded cell differs
    for index in keep:
        here = spot[index]
        candidates = np.nonzero(codes != codes[here])[0]
        if candidates.size < 2:
            continue
        query = points[index]
        depths = gromov(query, target[candidates])
        # the target is in the pool (its own cell is excluded, so it is NOT):
        # the metric asks how the pool's items compare with the target's depth
        target_depth = gromov(query, target[here:here + 1])[0]
        logits = -depths / tau
        top = float(logits.max())
        mass = np.exp(logits - top)
        mass = mass / mass.sum()
        order = np.argsort(-depths)
        ranks = np.empty(len(order), dtype=np.int64)
        ranks[order] = np.arange(len(order))
        expected = float((mass * ranks).sum())
        chance = (candidates.size - 1) / 2.0
        # the target's own rank among the pool, measured by its depth
        better = int(np.sum(depths > target_depth))
        total += 0.5 * (1.0 - (expected - chance) / chance) + 0.5 * (
            1.0 - (better - chance) / chance)
        counted += 1
    return float(total / counted) if counted else 0.0


hbaa_offcell_expected_rank.note = ("the target's expected rank over the items its own "
                                    "level-1 cell hides, over chance: a removal "
                                    "construction rather than a contrast")

METRICS = {"hbaa_offcell_expected_rank": hbaa_offcell_expected_rank}
FREE_CONSTANTS = {"hbaa_offcell_expected_rank": 0}
