"""HBAA with the NEXT-ITEM reference as the query point.

Issue #126's frozen reference is a pairwise model: its source tower says which
item resembles which. The next-item reference instead reads the user's history
and predicts the behaviour of the item about to be bought, so its prediction is a
statement of INTENT rather than of similarity. This metric asks the HBAA
question with that prediction as the query point.

The predictions are per training interaction (the same interactions the metric
queries), so they are in-sample for the teacher -- the same caveat as the
training-query version of HBAA.
"""
from __future__ import annotations

import numpy as np


def _next_item_points(context: dict) -> np.ndarray:
    if "next_item_points" in context:
        return context["next_item_points"]
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from tools.research.science import next_item_reference
    payload = next_item_reference.load()
    points = payload["predictions"]
    # exp_0 into the unit ball: the prediction lives in the same space as the
    # reference's representations
    norm = np.linalg.norm(points, axis=1, keepdims=True)
    points = np.tanh(norm) * points / np.maximum(norm, 1e-12)
    context["next_item_points"] = points
    return points


def hbaa_nextitem(table: dict, context: dict) -> float:
    """HBAA's contrast with the next-item prediction as the query point."""
    from results.research.issue126.metric_round import metrics_hbaa as hbaa
    reference = hbaa._reference(context)
    target = np.asarray(reference["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(reference.get("temperature",
                              (reference.get("record") or {})["temperature"]))
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    queries = hbaa._queries(context)
    pos = np.array([position.get(int(item), -1) for item in queries["positives"]],
                   dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                    for row in queries["negatives"]], dtype=np.int64)
    valid = (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return 0.0
    points = _next_item_points(context)[valid]
    pos = pos[valid]
    neg = neg[valid]
    centres = {int(code): hbaa._frechet(target[codes == code]) for code in np.unique(codes)}
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


hbaa_nextitem.note = ("HBAA's contrast with the next-item reference's prediction as the "
                      "query point: the teacher's statement of intent, not of similarity")

METRICS = {"hbaa_nextitem": hbaa_nextitem}
FREE_CONSTANTS = {"hbaa_nextitem": 0}
