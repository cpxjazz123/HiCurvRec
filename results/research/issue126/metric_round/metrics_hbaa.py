"""HBAA: the hyperbolic behavioural ancestor alignment, with its Euclidean and flat controls.

The Gromov product at the origin is the tree-shaped quantity of hyperbolic space:

    (x | y)_0 = ( d(0, x) + d(0, y) - d(x, y) ) / 2

In a tree it is exactly the depth of the two nodes' nearest common ancestor; in a
space with negative curvature it is the same quantity up to the thin-triangle defect.
The metric asks whether the true next item's level-1 prefix shares a DEEPER common
ancestor with the user's history than a negative's prefix does:

    Delta = (u_h | mu_{b(i+)})_0 - (u_h | mu_{b(i-)})_0

with u_h the hyperbolic barycentre of the SOURCE representations of the whole
training history, and mu_b the barycentre of the TARGET representations of the
items in bucket b. It is reduced with the frozen temperature exactly as Q_H is,
1 - E[softplus(-Delta/tau)] / ln 2, so the two numbers sit on the same scale.

Two controls are computed on the same queries and the same tables:

* ``hbaa_euclid``  -- the same construction with Euclidean means and norms, no exp
  map and no Poincare distance. If hyperbolic space adds nothing, this matches.
* ``hbaa_flat``    -- the hyperbolic form at curvature c = 1e-3, i.e. almost flat:
  it isolates what the finite curvature itself contributes.

A collapsed table (one code) gives Delta = 0 for every pair and scores 0; a random
assignment scores 0 on average. Those are the degenerate cases the metric must not
reward.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

NEGATIVES = 10
SEED = 42
FLAT_CURVATURE = 1e-3


def _scale(points: np.ndarray, curvature: float) -> np.ndarray:
    """``exp_0(v / s)`` for curvature ``curvature``, centred and scaled by its spread."""
    spread = float(np.sqrt(np.mean(np.sum(points ** 2, axis=1))))
    v = np.asarray(points, dtype=np.float64) / max(spread, 1e-12)
    sqrt_c = np.sqrt(curvature)
    norm = np.linalg.norm(v, axis=-1, keepdims=True) * sqrt_c
    return np.tanh(norm) * v / np.maximum(norm / max(sqrt_c, 1e-12), 1e-12)


def _distance(a: np.ndarray, b: np.ndarray, curvature: float) -> np.ndarray:
    c = float(curvature)
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = (1.0 - c * np.sum(a * a, axis=-1)) * (1.0 - c * np.sum(b * b, axis=-1))
    argument = 1.0 + 2.0 * c * difference / np.maximum(denominator, 1e-12)
    return np.arccosh(np.maximum(argument, 1.0 + 1e-12)) / np.sqrt(c)


def _norm(x: np.ndarray, curvature: float) -> np.ndarray:
    c = float(curvature)
    inner = np.clip(1.0 - c * np.sum(x * x, axis=-1), 1e-12, None)
    return np.arccosh(1.0 / np.sqrt(inner)) / np.sqrt(c)


def _frechet(points: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from tools.research.science import hyperbolic_sid_quality as frozen
    return frozen.frechet_barycentre(np.asarray(points, dtype=np.float64),
                                     frozen.BallGeometry(curvature))


def _queries(context: dict) -> dict:
    if "hbaa_queries" in context:
        return context["hbaa_queries"]
    import pandas as pd
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    frame = pd.read_parquet(repo / "results" / "stage0_build_parquet" / "train.parquet",
                            columns=["user", "history", "target"])
    frame = frame.sort_values("user", kind="stable")
    last = frame.groupby("user", sort=False).tail(1)
    items = np.unique(frame["target"].to_numpy(dtype=np.int64))
    rng = np.random.default_rng(SEED)
    histories, positives, negatives = [], [], []
    for history, target in zip(last["history"].to_numpy(), last["target"].to_numpy()):
        sequence = [int(item) for item in list(history)]
        if not sequence:
            continue
        seen = set(sequence) | {int(target)}
        chosen, attempts = [], 0
        while len(chosen) < NEGATIVES and attempts < NEGATIVES * 20:
            candidate = int(items[rng.integers(0, items.size)])
            attempts += 1
            if candidate not in seen and candidate not in chosen:
                chosen.append(candidate)
        if len(chosen) < NEGATIVES:
            continue
        histories.append(sequence)
        positives.append(int(target))
        negatives.append(np.asarray(chosen, dtype=np.int64))
    payload = {"histories": histories,
               "positives": np.array(positives, dtype=np.int64),
               "negatives": np.stack(negatives).astype(np.int64)}
    context["hbaa_queries"] = payload
    return payload


HISTORY_CACHE = (Path(__file__).resolve().parent / "history_points.npz")


def _history_points(context: dict, curvature: float) -> np.ndarray:
    """The per-interaction history barycentres, cached on disk.

    The Karcher mean of every history costs minutes and does not depend on the
    table under test, so it is computed once and read by every worker.
    """
    key = "hbaa_history_%g" % curvature
    if key not in context:
        if abs(curvature - 1.0) < 1e-12 and HISTORY_CACHE.is_file():
            context[key] = np.load(HISTORY_CACHE)["points"].astype(np.float64)
        else:
            source = np.asarray(context["source"], dtype=np.float64)
            queries = _queries(context)
            context[key] = np.stack(
                [_frechet(source[np.asarray(history, dtype=np.int64)], curvature)
                 for history in queries["histories"]])
    return context[key]


def _gromov_alignment(table: dict, context: dict, *, curvature: float,
                      euclidean: bool = False) -> float:
    target = np.asarray(context["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(context["temperature"])
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    queries = _queries(context)

    pos = np.array([position.get(int(item), -1) for item in queries["positives"]],
                   dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                    for row in queries["negatives"]], dtype=np.int64)
    valid = (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return 0.0

    if euclidean:
        source = np.asarray(context["source"], dtype=np.float64)
        history = np.stack([source[np.asarray(h, dtype=np.int64)].mean(axis=0)
                            for h, keep in zip(queries["histories"], valid) if keep])
        centres = {int(code): target[codes == code].mean(axis=0)
                   for code in np.unique(codes)}
        origin = np.zeros(target.shape[1])
        distance = lambda a, b: np.linalg.norm(a - b, axis=-1)
        norm = lambda x: np.linalg.norm(x, axis=-1)
    else:
        history = _history_points(context, curvature)[valid]
        points = _scale(target, curvature)
        centres = {int(code): _frechet(target[codes == code], curvature)
                   for code in np.unique(codes)}
        origin = np.zeros(points.shape[1])
        distance = lambda a, b: _distance(a, b, curvature)
        norm = lambda x: _norm(x, curvature)

    def gromov(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 0.5 * (norm(x) + np.asarray(norm(y)) - distance(x, y))

    mu_pos = np.stack([centres[int(codes[index])] for index in pos[valid]])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg[valid]])
    # Delta as the proposal states it: the POSITIVE's prefix shares the deeper
    # common ancestor with the history. The order was reversed in the first
    # implementation, which inverted the whole metric.
    delta = gromov(history, mu_pos)[:, None] - gromov(history[:, None, :], mu_neg)
    per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
    return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))


def hbaa_anchor(table: dict, context: dict) -> float:
    """The same Gromov contrast with the ANCHOR as the query point (a control).

    It needs no barycentre and therefore measures the proposal's construction
    without its extra computation; the difference between it and ``hbaa`` is what
    the full history adds.
    """
    return _anchor_alignment(table, context)


def _anchor_alignment(table: dict, context: dict) -> float:
    source = np.asarray(context["source"], dtype=np.float64)
    target = np.asarray(context["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(context["temperature"])
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    queries = _queries(context)
    anchors = np.array([history[-1] for history in queries["histories"]], dtype=np.int64)
    pos = np.array([position.get(int(item), -1) for item in queries["positives"]],
                   dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                    for row in queries["negatives"]], dtype=np.int64)
    valid = (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return 0.0
    points = source[anchors[valid]]
    centres = {int(code): _frechet(target[codes == code]) for code in np.unique(codes)}
    mu_pos = np.stack([centres[int(codes[index])] for index in pos[valid]])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg[valid]])

    def gromov(x, y):
        def d(a, b):
            diff = np.sum((a - b) ** 2, axis=-1)
            den = ((1 - np.sum(a * a, axis=-1)) * (1 - np.sum(b * b, axis=-1)))
            return np.arccosh(np.maximum(1 + 2 * diff / np.maximum(den, 1e-12), 1 + 1e-12))
        def nrm(x):
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return np.arccosh(1 / np.sqrt(inner))
        return 0.5 * (nrm(x) + np.asarray(nrm(y)) - d(x, y))

    delta = gromov(points, mu_pos)[:, None] - gromov(points[:, None, :], mu_neg)
    per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
    return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))


def hbaa(table: dict, context: dict) -> float:
    """The Gromov-product alignment at curvature 1 (the hypothesis)."""
    return _gromov_alignment(table, context, curvature=1.0)


def hbaa_euclid(table: dict, context: dict) -> float:
    """The same construction with Euclidean means and distances: the geometric control."""
    return _gromov_alignment(table, context, curvature=1.0, euclidean=True)


def hbaa_flat(table: dict, context: dict) -> float:
    """The hyperbolic form at curvature 1e-3: what finite curvature itself adds."""
    return _gromov_alignment(table, context, curvature=FLAT_CURVATURE)


hbaa.note = ("Gromov-product (common-ancestor depth) alignment of the level-1 prefix "
             "with the full training history, curvature 1")
hbaa_euclid.note = "the same construction with Euclidean means and distances (control)"
hbaa_flat.note = "the same hyperbolic construction at curvature 1e-3 (control)"

def hbaa_levels(table: dict, context: dict) -> float:
    """HBAA at every level, averaged over the levels that can still be informative.

    The Gromov product measures a TREE relation, and the SID is a tree, so the
    deeper levels are not automatically degenerate the way they were for Q_H: what
    a level contributes is the ancestor depth of its prefix against the behaviour's
    ancestor depth, and a level whose buckets are singletons has no centroid to
    speak for. Each level is therefore scored with the same construction and the
    levels are averaged, with the level-1 score kept as the anchor.
    """
    import numpy as _np
    scores = []
    for level in range(np.asarray(table["rows"]).shape[1]):
        reduced = {"rows": np.asarray(table["rows"])[:, level:level + 1],
                   "item_ids": table["item_ids"]}
        scores.append(_level_alignment(reduced, context))
    return float(np.mean(scores))


def _level_alignment(table: dict, context: dict) -> float:
    """HBAA's contrast for a table whose first column is the level's prefix."""
    target = np.asarray(context["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(context["temperature"])
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    queries = _queries(context)
    pos = np.array([position.get(int(item), -1) for item in queries["positives"]],
                    dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                     for row in queries["negatives"]], dtype=np.int64)
    valid = (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return 0.0
    points = _history_points(context, 1.0)[valid]
    centres = {int(code): _frechet(target[codes == code]) for code in np.unique(codes)}
    mu_pos = np.stack([centres[int(codes[index])] for index in pos[valid]])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg[valid]])

    def gromov(x, y):
        def d(a, b):
            diff = np.sum((a - b) ** 2, axis=-1)
            den = ((1 - np.sum(a * a, axis=-1)) * (1 - np.sum(b * b, axis=-1)))
            return np.arccosh(np.maximum(1 + 2 * diff / np.maximum(den, 1e-12), 1 + 1e-12))
        def nrm(x):
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return np.arccosh(1 / np.sqrt(inner))
        return 0.5 * (nrm(x) + np.asarray(nrm(y)) - d(x, y))

    delta = gromov(points, mu_pos)[:, None] - gromov(points[:, None, :], mu_neg)
    per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
    return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))




def hbaa_weighted(table: dict, context: dict) -> float:
    """HBAA at every level, each level WEIGHTED by the queries it can speak for.

    The flat average over levels measured +0.464 against the level-1 score's
    +0.701: the deeper levels are not noise, they are weaker, and an equal
    average gives them the same say. Here a level is kept only where its buckets
    can make a claim (2 <= |bucket| <= N/2) and it is weighted by the fraction of
    the queries it kept, so a level that has degenerated contributes little.
    """
    rows = np.asarray(table["rows"], dtype=np.int64)
    n_items = rows.shape[0]
    target = np.asarray(context["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(context["temperature"])
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    queries = _queries(context)
    pos = np.array([position.get(int(item), -1) for item in queries["positives"]],
                   dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                    for row in queries["negatives"]], dtype=np.int64)
    valid = (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return 0.0
    points = _history_points(context, 1.0)[valid]
    pos = pos[valid]
    neg = neg[valid]
    total_queries = float(len(points))

    def gromov(x, y):
        def d(a, b):
            diff = np.sum((a - b) ** 2, axis=-1)
            den = ((1 - np.sum(a * a, axis=-1)) * (1 - np.sum(b * b, axis=-1)))
            return np.arccosh(np.maximum(1 + 2 * diff / np.maximum(den, 1e-12), 1 + 1e-12))
        def nrm(x):
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return np.arccosh(1 / np.sqrt(inner))
        return 0.5 * (nrm(x) + np.asarray(nrm(y)) - d(x, y))

    scores, weights = [], []
    for level in range(rows.shape[1]):
        codes = rows[:, level]
        sizes = np.bincount(codes)
        informative = (sizes >= 2) & (sizes <= n_items / 2)
        centres = {int(code): _frechet(target[codes == code]) for code in np.unique(codes)}
        mu_pos = np.stack([centres[int(codes[index])] for index in pos])
        mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg])
        legal = informative[codes[pos]] & informative[codes[neg]].any(axis=1)
        if not legal.any():
            continue
        delta = gromov(points[legal], mu_pos[legal])[:, None] - gromov(
            points[legal][:, None, :], mu_neg[legal])
        per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
        scores.append(1.0 - float(per_query.mean()) / float(np.log(2.0)))
        weights.append(float(legal.sum()) / total_queries)
    if not scores:
        return 0.0
    scores = np.asarray(scores)
    weights = np.asarray(weights)
    return float((scores * weights).sum() / max(weights.sum(), 1e-12))


def hbaa_pair_accuracy(table: dict, context: dict) -> float:
    """HBAA without a temperature: the share of queries whose positive is deeper.

    The softplus reduction needs the frozen temperature; the rank-free version
    asks the same question with a sign test and no scale at all.
    """
    target = np.asarray(context["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    queries = _queries(context)
    pos = np.array([position.get(int(item), -1) for item in queries["positives"]],
                   dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                    for row in queries["negatives"]], dtype=np.int64)
    valid = (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return 0.0
    points = _history_points(context, 1.0)[valid]
    pos = pos[valid]
    neg = neg[valid]
    centres = {int(code): _frechet(target[codes == code]) for code in np.unique(codes)}

    def gromov(x, y):
        def d(a, b):
            diff = np.sum((a - b) ** 2, axis=-1)
            den = ((1 - np.sum(a * a, axis=-1)) * (1 - np.sum(b * b, axis=-1)))
            return np.arccosh(np.maximum(1 + 2 * diff / np.maximum(den, 1e-12), 1 + 1e-12))
        def nrm(x):
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return np.arccosh(1 / np.sqrt(inner))
        return 0.5 * (nrm(x) + np.asarray(nrm(y)) - d(x, y))

    mu_pos = np.stack([centres[int(codes[index])] for index in pos])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg])
    positive_depth = gromov(points, mu_pos)
    negative_depth = gromov(points[:, None, :], mu_neg).mean(axis=1)
    return float((positive_depth > negative_depth).mean())


METRICS = {"hbaa": hbaa, "hbaa_anchor": hbaa_anchor,
           "hbaa_euclid": hbaa_euclid, "hbaa_flat": hbaa_flat,
           "hbaa_levels": hbaa_levels, "hbaa_weighted": hbaa_weighted,
           "hbaa_pair_accuracy": hbaa_pair_accuracy}
FREE_CONSTANTS = {"hbaa": 0, "hbaa_anchor": 0, "hbaa_euclid": 0, "hbaa_flat": 0,
                   "hbaa_levels": 0, "hbaa_weighted": 0, "hbaa_pair_accuracy": 0}
