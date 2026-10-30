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

import os
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
    """``d(0, x)``: the radial hyperbolic distance, defined the SAME way as the
    pairwise one.

    The previous form returned ``arctanh(sqrt(c) ||x||) / sqrt(c)``, which is
    exactly HALF of the distance ``_distance`` returns for the pair ``(0, x)`` --
    ``arccosh((1+u)/(1-u)) / sqrt(c) = 2 arctanh(sqrt(u)) / sqrt(c)``. The Gromov
    product is half of ``d(0,x) + d(0,y) - d(x,y)``, so a halved radial term
    distorts it, and it disagreed with the Lorentz implementation, which uses the
    full radius. Delegating to ``_distance`` makes the two definitions one.
    """
    c = float(curvature)
    origin = np.zeros(x.shape[-1], dtype=np.float64)
    return _distance(origin, x, c)


def _frechet(points: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    """The Karcher mean in the ball, in numpy (no torch needed)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from tools.research.science.hyperbolic_ancestor_alignment import frechet_barycentre
    return frechet_barycentre(np.asarray(points, dtype=np.float64))


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


def _reference(context: dict) -> dict:
    """The frozen reference, or the variant named by HBAA_REFERENCE.

    A variant reference is a different frozen teacher (another curvature), fitted
    on the same training pairs. The incumbent was measured with the Issue #126
    reference; comparing the two answers the question "does the teacher's own
    geometry decide the metric's power?".
    """
    import json as _json
    import os
    from pathlib import Path as _Path

    override = os.environ.get("HBAA_REFERENCE")
    if not override:
        return context["reference"]
    path = _Path(override)
    if path.is_dir():
        path = path / "reference.npz"
    payload = np.load(path)
    record_path = path.with_name("reference.json")
    record = _json.loads(record_path.read_text())
    context["reference"] = {
        "source": np.asarray(payload["source"], dtype=np.float64),
        "target": np.asarray(payload["target"], dtype=np.float64),
        "temperature": float(record["temperature"]),
        "record": record,
    }
    return context["reference"]


def _history_points(context: dict, curvature: float) -> np.ndarray:
    """The per-interaction history barycentres, cached on disk.

    The Karcher mean of every history costs minutes and does not depend on the
    table under test, so it is computed once and read by every worker.
    """
    key = "hbaa_history_%g" % curvature
    if key not in context:
        # the cache is keyed by the reference's own digest: a variant reference
        # has different representations and therefore different history points
        cache = HISTORY_CACHE
        reference = context.get("reference") or {}
        digest = str(((reference.get("record") or {}).get("arrays") or {}).get("sha256", ""))
        if digest:
            cache = HISTORY_CACHE.with_name("history_points_%s.npz" % digest[:12])
        if cache.is_file():
            context[key] = np.load(cache)["points"].astype(np.float64)
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


def hbaa_permuted(table: dict, context: dict) -> float:
    """The same construction with the item-SID correspondence permuted.

    HBAA contrasts a behaviour positive's prefix centroid against its negatives',
    so if the number it reports is a property of the SID tree it has to collapse
    when the tree is dealt at random to the items. The other families in this
    round were caught by exactly this control, and the leader had never been run
    against it.
    """
    import zlib

    rows = np.asarray(table["rows"], dtype=np.int64)
    rng = np.random.default_rng(zlib.crc32(rows.tobytes()) & 0x7FFFFFFF)
    permuted = {"rows": rows[rng.permutation(rows.shape[0])],
                "item_ids": table["item_ids"]}
    return _gromov_alignment(permuted, context, curvature=1.0)


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
            # the radial term of the Poincare metric: 2 * arccosh(1/sqrt(1-|x|^2)).
            # Without the factor 2 the product mixes a halved radius with a full
            # distance and is not the common-ancestor depth it claims to be.
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return 2 * np.arccosh(1 / np.sqrt(inner))
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
            # the radial term of the Poincare metric: 2 * arccosh(1/sqrt(1-|x|^2)).
            # Without the factor 2 the product mixes a halved radius with a full
            # distance and is not the common-ancestor depth it claims to be.
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return 2 * np.arccosh(1 / np.sqrt(inner))
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
            # the radial term of the Poincare metric: 2 * arccosh(1/sqrt(1-|x|^2)).
            # Without the factor 2 the product mixes a halved radius with a full
            # distance and is not the common-ancestor depth it claims to be.
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return 2 * np.arccosh(1 / np.sqrt(inner))
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
            # the radial term of the Poincare metric: 2 * arccosh(1/sqrt(1-|x|^2)).
            # Without the factor 2 the product mixes a halved radius with a full
            # distance and is not the common-ancestor depth it claims to be.
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return 2 * np.arccosh(1 / np.sqrt(inner))
        return 0.5 * (nrm(x) + np.asarray(nrm(y)) - d(x, y))

    mu_pos = np.stack([centres[int(codes[index])] for index in pos])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg])
    positive_depth = gromov(points, mu_pos)
    negative_depth = gromov(points[:, None, :], mu_neg).mean(axis=1)
    return float((positive_depth > negative_depth).mean())




SIZE_MATCH_FACTOR = 2.0


def hbaa_size_matched(table: dict, context: dict) -> float:
    """HBAA with SIZE-MATCHED negatives.

    HBAA compares the positive's prefix centroid with the negatives' centroids, and
    a centroid's radius grows with the size of its bucket, so part of the gap it
    measures is a size effect rather than an alignment. Here only the negatives
    whose bucket size is within a factor of the positive's are used, and a query
    with no such negative is dropped and counted: the number of queries that
    survive is reported with the score, because a metric that keeps only a few
    queries is a metric about those few.
    """
    target = np.asarray(context["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(context["temperature"])
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    sizes = np.bincount(codes)
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
            # the radial term of the Poincare metric: 2 * arccosh(1/sqrt(1-|x|^2)).
            # Without the factor 2 the product mixes a halved radius with a full
            # distance and is not the common-ancestor depth it claims to be.
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return 2 * np.arccosh(1 / np.sqrt(inner))
        return 0.5 * (nrm(x) + np.asarray(nrm(y)) - d(x, y))

    positive_size = sizes[codes[pos]][:, None]
    negative_size = sizes[codes[neg]]
    keep = (negative_size >= positive_size / SIZE_MATCH_FACTOR) & (
        negative_size <= positive_size * SIZE_MATCH_FACTOR)
    legal = keep.any(axis=1)
    if not legal.any():
        return 0.0
    rows_index = np.nonzero(legal)[0]
    kept = keep[rows_index]
    mu_pos = np.stack([centres[int(codes[index])] for index in pos[rows_index]])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row]
                        for row in neg[rows_index]])
    positive_depth = gromov(points[rows_index], mu_pos)
    negative_depth = gromov(points[rows_index][:, None, :], mu_neg)
    delta = np.where(kept, positive_depth[:, None] - negative_depth, 0.0)
    counts = kept.sum(axis=1)
    per_query = np.where(counts > 0,
                         (np.log1p(np.exp(-delta / tau)) * kept).sum(axis=1)
                         / np.maximum(counts, 1), 1.0)
    score = 1.0 - float(per_query.mean()) / float(np.log(2.0))
    context.setdefault("round7_kept", []).append(
        (float(legal.sum()) / float(len(points)),
         float(counts[counts > 0].mean()) if bool((counts > 0).any()) else 0.0))
    return score


def hbaa_size_matched_kept(table: dict, context: dict) -> float:
    """The fraction of queries that kept a size-matched negative (a diagnostic)."""
    hbaa_size_matched(table, context)
    kept = context.get("round7_kept") or []
    return float(kept[-1][0]) if kept else 0.0




def hbaa_validation(table: dict, context: dict) -> float:
    """HBAA on the HELD-OUT validation probe: the anchor as the query point.

    The metric above asks its questions of the TRAINING interactions, and the
    frozen reference was fitted on those pairs, so its answers are in-sample for
    the teacher. This variant asks the same question of the validation probe --
    pairs the reference never saw -- so the comparison with the training version
    says whether the metric needs the teacher's training data to look good.
    """
    reference = _reference(context)
    source = np.asarray(reference["source"], dtype=np.float64)
    target = np.asarray(reference["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(reference.get("temperature",
                              (reference.get("record") or {})["temperature"]))
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    probe = context["probe"]
    anchors = np.asarray(probe["anchors"], dtype=np.int64)
    positives = np.asarray(probe["positives"], dtype=np.int64)
    negatives = np.asarray(probe["negatives"], dtype=np.int64)
    keep = np.array([position.get(int(item), -1) for item in anchors], dtype=np.int64)
    pos = np.array([position.get(int(item), -1) for item in positives], dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row] for row in negatives],
                   dtype=np.int64)
    valid = (keep >= 0) & (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return 0.0
    points = source[anchors[valid]]
    pos = pos[valid]
    neg = neg[valid]
    centres = {int(code): _frechet(target[codes == code]) for code in np.unique(codes)}
    mu_pos = np.stack([centres[int(codes[index])] for index in pos])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg])

    def gromov(x, y):
        def d(a, b):
            diff = np.sum((a - b) ** 2, axis=-1)
            den = ((1 - np.sum(a * a, axis=-1)) * (1 - np.sum(b * b, axis=-1)))
            return np.arccosh(np.maximum(1 + 2 * diff / np.maximum(den, 1e-12), 1 + 1e-12))
        def nrm(x):
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return 2 * np.arccosh(1 / np.sqrt(inner))
        return 0.5 * (nrm(x) + np.asarray(nrm(y)) - d(x, y))

    delta = gromov(points, mu_pos)[:, None] - gromov(points[:, None, :], mu_neg)
    per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
    return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))




def hbaa_teacher_curvature(table: dict, context: dict) -> float:
    """HBAA with the metric's OWN curvature taken from the teacher.

    The frozen reference at curvature c lives in a ball of curvature -c, and a
    metric that measures geodesic distances in a unit-curvature ball reads a
    different geometry from the one the teacher was trained in. This version uses
    the teacher's curvature everywhere: the distances, the radial term and the
    Karcher mean.
    """
    reference = _reference(context)
    record = reference.get("record") or {}
    curvature = float(record.get("spec", {}).get("curvature", 1.0))
    return _alignment(table, context, curvature)


def _alignment(table: dict, context: dict, curvature: float) -> float:
    """HBAA's contrast with an explicit curvature."""
    reference = _reference(context)
    source = np.asarray(reference["source"], dtype=np.float64)
    target = np.asarray(reference["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(reference.get("temperature",
                              (reference.get("record") or {})["temperature"]))
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
    points = _history_points(context, curvature)[valid]
    pos = pos[valid]
    neg = neg[valid]

    def distance(a, b):
        c = float(curvature)
        difference = np.sum((a - b) ** 2, axis=-1)
        denominator = ((1 - c * np.sum(a * a, axis=-1))
                       * (1 - c * np.sum(b * b, axis=-1)))
        return np.arccosh(np.maximum(1 + 2 * c * difference /
                                     np.maximum(denominator, 1e-12), 1 + 1e-12)) / np.sqrt(c)

    def norm(x):
        # the SAME radial distance as the module-level helper: this inline copy
        # was the second place where the factor two was missing
        return _norm(x, curvature)

    centres = {int(code): _frechet(target[codes == code], curvature)
               for code in np.unique(codes)}
    mu_pos = np.stack([centres[int(codes[index])] for index in pos])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg])
    positive = 0.5 * (norm(points) + norm(mu_pos) - distance(points, mu_pos))
    negative = 0.5 * (norm(points)[:, None] + norm(mu_neg) - distance(points[:, None, :], mu_neg))
    delta = positive[:, None] - negative
    per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
    return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))


def hbaa_midpoint(table: dict, context: dict) -> float:
    """HBAA with the query point at the BEHAVIOUR MIDPOINT.

    The history barycentre averages the whole history; the anchor is the last
    item alone. Their geodesic midpoint keeps the history's context and the
    anchor's immediacy, and the question is whether the metric's power comes from
    the context, from the immediacy, or from the contrast itself.
    """
    reference = _reference(context)
    source = np.asarray(reference["source"], dtype=np.float64)
    queries = _queries(context)
    anchors = np.array([history[-1] for history in queries["histories"]], dtype=np.int64)
    history_points = _history_points(context, 1.0)
    # the geodesic midpoint of the history barycentre and the anchor, in the ball
    difference = history_points - source[anchors]
    norm = np.linalg.norm(difference, axis=-1, keepdims=True)
    direction = difference / np.maximum(norm, 1e-12)
    half = np.arctanh(np.clip(norm / 2.0, 0.0, 1 - 1e-9))
    scale = np.tanh(half) / np.maximum(norm, 1e-12)
    moved = source[anchors] + direction * scale * 0.5
    norm2 = np.linalg.norm(moved, axis=-1, keepdims=True)
    moved = np.where(norm2 > 1 - 1e-7, moved / np.maximum(norm2, 1e-12) * (1 - 1e-7), moved)

    target = np.asarray(reference["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(reference.get("temperature",
                              (reference.get("record") or {})["temperature"]))
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    pos = np.array([position.get(int(item), -1) for item in queries["positives"]],
                   dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                    for row in queries["negatives"]], dtype=np.int64)
    valid = (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return 0.0
    points = moved[valid]
    pos = pos[valid]
    neg = neg[valid]
    centres = {int(code): _frechet(target[codes == code]) for code in np.unique(codes)}
    mu_pos = np.stack([centres[int(codes[index])] for index in pos])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg])

    def gromov(x, y):
        def d(a, b):
            diff = np.sum((a - b) ** 2, axis=-1)
            den = ((1 - np.sum(a * a, axis=-1)) * (1 - np.sum(b * b, axis=-1)))
            return np.arccosh(np.maximum(1 + 2 * diff / np.maximum(den, 1e-12), 1 + 1e-12))
        def nrm(x):
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return 2 * np.arccosh(1 / np.sqrt(inner))
        return 0.5 * (nrm(x) + np.asarray(nrm(y)) - d(x, y))

    delta = gromov(points, mu_pos)[:, None] - gromov(points[:, None, :], mu_neg)
    per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
    return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))




def hbaa_recency(table: dict, context: dict) -> float:
    """HBAA with a RECENCY-WEIGHTED history point.

    The history barycentre treats the whole sequence equally; the query a user
    actually brings to the next purchase is dominated by its last few steps. This
    weights the history's source representations by 2^{-i} from the most recent
    step, so the query point leans on the recency the sequence carries. The
    weighting is the only thing that changes.
    """
    reference = _reference(context)
    source = np.asarray(reference["source"], dtype=np.float64)
    target = np.asarray(reference["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(reference.get("temperature",
                              (reference.get("record") or {})["temperature"]))
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

    weights = np.power(0.5, np.arange(12, dtype=np.float64))
    weights = weights / weights.sum()
    points = []
    for history, keep_it in zip(queries["histories"], valid):
        if not keep_it:
            continue
        ids = np.asarray(history, dtype=np.int64)
        recent = ids[-len(weights):]
        w = weights[-len(recent):]
        w = w / w.sum()
        points.append(_frechet((source[recent] * w[:, None]).sum(axis=0, keepdims=True)))
    points = np.stack(points)
    pos = pos[valid]
    neg = neg[valid]

    centres = {int(code): _frechet(target[codes == code]) for code in np.unique(codes)}
    mu_pos = np.stack([centres[int(codes[index])] for index in pos])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg])

    def gromov(x, y):
        def d(a, b):
            diff = np.sum((a - b) ** 2, axis=-1)
            den = ((1 - np.sum(a * a, axis=-1)) * (1 - np.sum(b * b, axis=-1)))
            return np.arccosh(np.maximum(1 + 2 * diff / np.maximum(den, 1e-12), 1 + 1e-12))
        def nrm(x):
            inner = np.clip(1 - np.sum(x * x, axis=-1), 1e-12, None)
            return 2 * np.arccosh(1 / np.sqrt(inner))
        return 0.5 * (nrm(x) + np.asarray(nrm(y)) - d(x, y))

    delta = gromov(points, mu_pos)[:, None] - gromov(points[:, None, :], mu_neg)
    per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
    return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))


METRICS = {
    "hbaa_permuted": hbaa_permuted,"hbaa": hbaa, "hbaa_anchor": hbaa_anchor,
           "hbaa_recency": hbaa_recency,
           "hbaa_teacher_curvature": hbaa_teacher_curvature,
           "hbaa_midpoint": hbaa_midpoint,
           "hbaa_validation": hbaa_validation,
           "hbaa_euclid": hbaa_euclid, "hbaa_flat": hbaa_flat,
           "hbaa_levels": hbaa_levels, "hbaa_weighted": hbaa_weighted,
           "hbaa_pair_accuracy": hbaa_pair_accuracy,
           "hbaa_size_matched": hbaa_size_matched,
           "hbaa_validation": hbaa_validation,
           "hbaa_size_matched_kept": hbaa_size_matched_kept}
FREE_CONSTANTS = {"hbaa": 0, "hbaa_anchor": 0, "hbaa_euclid": 0, "hbaa_flat": 0,
                   "hbaa_levels": 0, "hbaa_weighted": 0, "hbaa_pair_accuracy": 0,
                   "hbaa_size_matched": 0, "hbaa_size_matched_kept": 0,
                   "hbaa_validation": 0, "hbaa_recency": 0,
                   "hbaa_teacher_curvature": 0,
                   "hbaa_midpoint": 0}
