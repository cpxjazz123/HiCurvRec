"""Round-9 metric: the GENERATOR-SIDE view of the level-1 partition.

Every metric measured so far asks whether the table agrees with the behaviour
reference. This one asks the question the generator actually faces: once the
level-1 code is emitted, is the true item among the ten best items of that cell?

For every training interaction the cell of the positive is taken, every item in
it is scored by the hyperbolic distance from the history's query point, and the
rank of the true item inside its own cell is recorded. The score is the share of
interactions where the true item lands in the cell's top ten, measured OVER
CHANCE: the chance level of a cell holding n items is 10/n, so a cell that holds
one item cannot contribute and a cell that holds a thousand is nearly free.

This is the closest a diagnostic can come to Stage 3's own decoding without
training anything: prefix, then re-rank inside the cell.
"""
from __future__ import annotations

import numpy as np

CANDIDATE_CAP = 256      # the cell members scored per interaction, deterministically


def hbaa_emission_hit10(table: dict, context: dict) -> float:
    """The share of interactions whose true item is in its cell's top ten, over chance."""
    reference = context.get("reference") or {}
    source = np.asarray(reference["source"], dtype=np.float64)
    target = np.asarray(reference["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    cells = {}
    for position, code in enumerate(codes.tolist()):
        cells.setdefault(int(code), []).append(position)

    queries = context.get("round9_queries")
    if queries is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
        from results.research.issue126.metric_round import metrics_hbaa as hbaa
        queries = hbaa._queries(context)
        context["round9_queries"] = queries

    positives = np.asarray(queries["positives"], dtype=np.int64)
    histories = queries["histories"]
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}

    total = 0.0
    counted = 0
    for index, item in enumerate(positives.tolist()):
        spot = position.get(int(item))
        if spot is None:
            continue
        members = cells.get(int(codes[spot]), [])
        if len(members) < 2:
            continue
        history = np.asarray(histories[index], dtype=np.int64)
        query = source[history].mean(axis=0, keepdims=True)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        # the cell's members, deterministically capped, with the positive always
        # among them: the generator has it there, so excluding it would measure
        # the cap instead of the partition
        if len(members) <= CANDIDATE_CAP:
            picked = list(members)
        else:
            picked = [member for member in members[:CANDIDATE_CAP - 1] if member != spot]
            picked.append(spot)
        candidates = np.asarray(picked, dtype=np.int64)
        points = target[candidates]
        difference = np.sum((points - query) ** 2, axis=1)
        denominator = ((1 - np.sum(points * points, axis=1))
                       * (1 - float(np.sum(query * query))))
        distances = np.arccosh(np.maximum(1 + 2 * difference /
                                          np.maximum(denominator, 1e-12), 1 + 1e-12))
        order = np.argsort(distances)
        ranks = np.empty(len(order), dtype=np.int64)
        ranks[order] = np.arange(len(order))
        place = int(np.nonzero(candidates == spot)[0][0])
        rank = int(ranks[place]) + 1
        chance = min(1.0, 10.0 / len(members))
        if rank <= 10:
            total += 1.0 - chance
        counted += 1
    return float(total / counted) if counted else 0.0


hbaa_emission_hit10.note = ("the share of training interactions whose true item is in its "
                             "own level-1 cell's hyperbolic top ten, over chance")



def hbaa_cell_retrieval_difficulty(table: dict, context: dict) -> float:
    """How much work the generator has INSIDE the cell, over chance.

    The round-9 measurement showed that the share of interactions whose true item
    is already in its cell's hyperbolic top ten is NEGATIVELY correlated with the
    recall (-0.367): a cell that answers the query by itself leaves the re-ranker
    nothing to choose from, and the top ten fills with near-duplicates. The
    complement asks the useful question -- how much room the re-ranking has -- and
    is the positive statement of the same finding.
    """
    from results.research.issue126.metric_round import metrics_round9 as round9
    return -round9.hbaa_emission_hit10(table, context)


METRICS = {"hbaa_emission_hit10": hbaa_emission_hit10,
           "hbaa_cell_retrieval_difficulty": hbaa_cell_retrieval_difficulty}
FREE_CONSTANTS = {"hbaa_emission_hit10": 0,
                   "hbaa_cell_retrieval_difficulty": 0}
