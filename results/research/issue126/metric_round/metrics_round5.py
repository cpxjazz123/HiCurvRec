"""Q_H with the whole training history as the query point (Q_H-seq).

The measured conclusion of the metric rounds was that no rearrangement of the
single-anchor inputs beats +0.634, and that beating it needs more information.
This is the first of the two information upgrades: the query point is no longer the
last item of the history but the hyperbolic Frechet mean of the SOURCE
representations of every item in the history, and the contrast is Q_H's own -- the
distance to the negative's level-1 prefix centroid minus the distance to the
positive's, reduced with the frozen temperature.

The queries are the TRAINING interactions (user, history, target) with ten negatives
sampled deterministically from the training item set, excluding the user's own
history and the positive. Both the anchor-only and the history-barycentre variants
are computed on that same query set, so the comparison between them is a comparison
of the query representation and not of the sample.
"""
from __future__ import annotations

import numpy as np

CURVATURE = 1.0
NEGATIVES = 10
SEED = 42


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = (1.0 - np.sum(a * a, axis=-1)) * (1.0 - np.sum(b * b, axis=-1))
    return np.arccosh(np.maximum(1.0 + 2.0 * difference /
                                 np.maximum(denominator, 1e-12), 1.0 + 1e-12))


def _frechet(points: np.ndarray) -> np.ndarray:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from tools.research.science import hyperbolic_sid_quality as frozen
    return frozen.frechet_barycentre(np.asarray(points, dtype=np.float64),
                                     frozen.BallGeometry())


def _queries(context: dict) -> dict:
    """The training interactions and their ten deterministic negatives (once)."""
    if "seq_queries" in context:
        return context["seq_queries"]
    import pandas as pd
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    frame = pd.read_parquet(repo / "results" / "stage0_build_parquet" / "train.parquet",
                            columns=["user", "history", "target"])
    frame = frame.sort_values("user", kind="stable")
    last = frame.groupby("user", sort=False).tail(1)
    items = np.unique(frame["target"].to_numpy(dtype=np.int64))
    rng = np.random.default_rng(SEED)
    histories, anchors, positives, negatives = [], [], [], []
    # Rejection sampling instead of building a 24k-item pool per user: the pool
    # form cost minutes per call, this costs microseconds.
    for history, target in zip(last["history"].to_numpy(), last["target"].to_numpy()):
        sequence = [int(item) for item in list(history)]
        if not sequence:
            continue
        seen = set(sequence) | {int(target)}
        chosen = []
        attempts = 0
        while len(chosen) < NEGATIVES and attempts < NEGATIVES * 20:
            candidate = int(items[rng.integers(0, items.size)])
            attempts += 1
            if candidate not in seen and candidate not in chosen:
                chosen.append(candidate)
        if len(chosen) < NEGATIVES:
            continue
        histories.append(sequence)
        anchors.append(sequence[-1])
        positives.append(int(target))
        negatives.append(np.asarray(chosen, dtype=np.int64))
    payload = {"histories": histories, "anchors": np.array(anchors, dtype=np.int64),
               "positives": np.array(positives, dtype=np.int64),
               "negatives": np.stack(negatives).astype(np.int64)}
    context["seq_queries"] = payload
    return payload


def _score(table: dict, context: dict, query_mode: str) -> float:
    source = np.asarray(context["source"], dtype=np.float64)
    target = np.asarray(context["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(context["temperature"])
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}

    queries = _queries(context)
    keep = np.array([position.get(int(item), -1) for item in queries["anchors"]],
                    dtype=np.int64)
    pos = np.array([position.get(int(item), -1) for item in queries["positives"]],
                   dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                    for row in queries["negatives"]], dtype=np.int64)
    valid = (keep >= 0) & (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return 0.0

    if query_mode == "anchor":
        points = source[queries["anchors"][valid]]
    else:
        # the history barycentres do not depend on the table: computed once per
        # context, not once per metric call on 97 tables
        if "seq_query_points" not in context:
            context["seq_query_points"] = np.stack(
                [_frechet(source[np.asarray(history, dtype=np.int64)])
                 for history in queries["histories"]])
        points = context["seq_query_points"][valid]

    centres = {}
    for code in np.unique(codes):
        centres[int(code)] = _frechet(target[codes == code])
    mu_pos = np.stack([centres[int(codes[index])] for index in pos[valid]])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg[valid]])
    d_pos = _distance(points, mu_pos) ** 2
    d_neg = _distance(points[:, None, :], mu_neg) ** 2
    delta = d_neg - d_pos[:, None]
    per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
    return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))


def q_h_seq(table: dict, context: dict) -> float:
    return _score(table, context, "history")


def q_h_anchor_train(table: dict, context: dict) -> float:
    """The same queries with the anchor alone: the control for the upgrade."""
    return _score(table, context, "anchor")


q_h_seq.note = ("Q_H's level-1 contrast with the query point replaced by the hyperbolic "
                "barycentre of the whole training history")
q_h_anchor_train.note = ("the same contrast and the same training queries with the "
                         "single-anchor query point (the control)")

METRICS = {"q_h_seq": q_h_seq, "q_h_anchor_train": q_h_anchor_train}
FREE_CONSTANTS = {"q_h_seq": 0, "q_h_anchor_train": 0}
