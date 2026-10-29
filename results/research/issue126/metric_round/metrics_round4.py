"""Q_H extended to every level with an informativeness mask (Q_H-H).

What the five measured metrics taught, in one line: a metric of the partition's own
geometry does not track recall, and the contrastive one does. So this keeps Q_H's
contrast exactly -- the difference between the geodesic distance to the negative's
prefix centroid and to the positive's -- and extends it to the deeper levels with
the one control the measurements demand: a level counts only where its buckets can
make a claim.

At level l a bucket is INFORMATIVE when 2 <= |bucket| <= N/2 (a singleton cannot
group anything and a bucket holding half the corpus groups everything), and a query
counts at that level only when the positive's bucket is informative AND at least
one negative's bucket is informative. The level's score is Q_H's own
``1 - E[softplus(-Delta/tau)]/ln2`` over the kept queries, and the levels are
averaged with weight equal to the fraction of queries they kept.

At level 1 the mask almost never binds, so the metric reduces to Q_H; the extension
is what the deeper levels add.
"""
from __future__ import annotations

import numpy as np


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = (1.0 - np.sum(a * a, axis=-1)) * (1.0 - np.sum(b * b, axis=-1))
    denominator = np.maximum(denominator, 1e-12)
    return np.arccosh(np.maximum(1.0 + 2.0 * difference / denominator, 1.0 + 1e-12))


def _frechet(points: np.ndarray, steps: int = 5) -> np.ndarray:
    """The SAME Karcher mean the frozen Issue #126 score uses, imported from it.

    Reimplementing it would make the level-1 case disagree with Q_H by an amount
    that has nothing to do with the extension, so the frozen implementation is
    called directly.
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4]))
    from tools.research.science import hyperbolic_sid_quality as frozen
    return frozen.frechet_barycentre(np.asarray(points, dtype=np.float64),
                                     frozen.BallGeometry())


def q_h_hier(table: dict, context: dict) -> float:
    target = np.asarray(context["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    source = np.asarray(context["source"], dtype=np.float64)
    tau = float(context["temperature"])
    rows = np.asarray(table["rows"], dtype=np.int64)
    n_items, levels = rows.shape

    pairs = context["probe"]
    anchors = np.asarray(pairs["anchors"], dtype=np.int64)
    positives = np.asarray(pairs["positives"], dtype=np.int64)
    negatives = np.asarray(pairs["negatives"], dtype=np.int64)

    position_of = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    queries = np.array([position_of.get(int(item), -1) for item in anchors], dtype=np.int64)
    keep_query = queries >= 0
    pos = np.array([position_of.get(int(item), -1) for item in positives], dtype=np.int64)
    neg = np.array([[position_of.get(int(item), -1) for item in row] for row in negatives],
                   dtype=np.int64)
    keep_query &= (pos >= 0) & (neg >= 0).all(axis=1)
    queries, pos, neg = queries[keep_query], pos[keep_query], neg[keep_query]
    if queries.size == 0:
        return 0.0
    sources = source[anchors[keep_query]]

    scores = []
    weights = []
    for level in range(levels):
        codes = rows[:, level]
        sizes = np.bincount(codes, minlength=int(codes.max()) + 1)
        informative = (sizes >= 2) & (sizes <= n_items / 2)
        centres = {}
        for code in np.unique(codes):
            members = target[codes == code]
            centres[int(code)] = _frechet(members)
        centre_of = lambda index: centres[int(codes[index])]
        pos_ok = informative[codes[pos]]
        neg_ok = informative[codes[neg]]
        legal = pos_ok & neg_ok.any(axis=1)
        if not bool(legal.any()):
            continue
        sub_pos = pos[legal]
        sub_neg = neg[legal]
        sub_query = queries[legal]
        sub_sources = sources[legal]
        mu_pos = np.stack([centre_of(index) for index in sub_pos])
        mu_neg = np.stack([[centre_of(index) for index in row] for row in sub_neg],
                          dtype=np.float64)
        d_pos = _distance(sub_sources, mu_pos) ** 2
        d_neg = _distance(sub_sources[:, None, :], mu_neg) ** 2
        delta = d_neg - d_pos[:, None]
        per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
        score = 1.0 - float(per_query.mean()) / float(np.log(2.0))
        scores.append(score)
        weights.append(float(legal.sum()) / float(len(queries)))
    if not scores:
        return 0.0
    scores = np.asarray(scores)
    weights = np.asarray(weights)
    return float((scores * weights).sum() / max(weights.sum(), 1e-12))


q_h_hier.note = ("Q_H's contrast at every level, kept only where the level's buckets are "
                 "informative (2 <= |bucket| <= N/2), levels weighted by kept queries")

METRICS = {"q_h_hier": q_h_hier}
FREE_CONSTANTS = {"q_h_hier": 0}
