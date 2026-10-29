"""Round-2 metric: the hyperbolic tree-conditional softmax path log-likelihood.

For every TRAINING pair (source u, item i) the metric asks what probability the
frozen reference assigns to i's SID path when the tree is walked level by level:
at each level the logit of a child is minus the geodesic distance from the
source representation to that child's hyperbolic prototype, divided by the
reference's own temperature. The score is the mean log-probability of the true
path.

It differs from Q_H in what it measures: Q_H compares the prefix CENTROIDS of a
positive and a negative, while this measures the conditional likelihood of the
true path against all siblings, which is what Stage 3's beam over the trie
maximises. Nothing is fitted; the temperature and the prototypes come from the
frozen reference and the table.
"""
from __future__ import annotations

import numpy as np

CURVATURE = 1.0


def _poincare_from_tangent(tangent: np.ndarray, scale: float) -> np.ndarray:
    v = np.asarray(tangent, dtype=np.float64) / float(scale)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.tanh(norm) * v / np.maximum(norm, 1e-12)


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Poincare distance of curvature 1, broadcasting over the last axis."""
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = (1.0 - np.sum(a * a, axis=-1)) * (1.0 - np.sum(b * b, axis=-1))
    denominator = np.maximum(denominator, 1e-12)
    return np.arccosh(np.maximum(1.0 + 2.0 * difference / denominator, 1.0 + 1e-12))


def _lorentz_barycentre(points: np.ndarray) -> np.ndarray:
    """Closed-form Lorentz barycentre, mapped back to the ball (no iteration)."""
    if points.shape[0] == 1:
        return points[0].copy()
    norm2 = np.sum(points * points, axis=-1, keepdims=True)
    z0 = (1.0 + norm2) / np.maximum(1.0 - norm2, 1e-12)
    z = np.concatenate((z0, 2.0 * points / np.maximum(1.0 - norm2, 1e-12)), axis=-1)
    total = z.sum(axis=0)
    inner = -total[0] * total[0] + np.sum(total[1:] ** 2)
    if inner >= 0:
        return _poincare_from_tangent(points.mean(axis=0), 1.0)
    scale = np.sqrt(-inner)
    bar = total / scale
    denominator = 1.0 + bar[0]
    point = bar[1:] / max(denominator, 1e-12)
    norm = float(np.linalg.norm(point))
    return point / max(1.0, norm / (1.0 - 1e-7))


def hyp_hsoftmax_path(table: dict, context: dict) -> float:
    rows = np.asarray(table["rows"], dtype=np.int64) + 1        # tokens start at 1
    items = np.asarray(table["item_ids"])
    target = np.asarray(context["target"], dtype=np.float64)[items]
    source = np.asarray(context["source"], dtype=np.float64)
    scale = 1.0 / float(context["temperature"])
    tau = float(context["temperature"])
    points = _poincare_from_tangent(target, scale)
    n_items, levels = rows.shape

    # node prototypes: for every prefix, the Lorentz barycentre of its items
    prototypes = {}
    members = {}
    for level in range(levels):
        for position in range(n_items):
            key = tuple(rows[position, :level + 1].tolist())
            members.setdefault(key, []).append(position)
    for key, positions in members.items():
        prototypes[key] = _lorentz_barycentre(points[np.asarray(positions)])

    pairs = context["train_pairs"]
    anchors = np.asarray(pairs["anchors"], dtype=np.int64)
    positives = np.asarray(pairs["positives"], dtype=np.int64)
    # the positive ids index the full catalogue; map them into the table's rows
    position_of = {int(item): index for index, item in enumerate(items.tolist())}
    keep = np.array([position_of.get(int(item), -1) for item in positives], dtype=np.int64)
    valid = keep >= 0
    anchors = anchors[valid]
    keep = keep[valid]
    if anchors.size == 0:
        return 0.0

    sources = _poincare_from_tangent(source[anchors], scale)
    # Grouped by parent so the logits are computed for a whole group at once:
    # the per-pair loop cost 170 s per table, the grouped form a few seconds.
    by_prefix_length = {}
    for node in members:
        by_prefix_length.setdefault(len(node), []).append(node)

    total = 0.0
    counted = 0
    for level in range(levels):
        prefix_length = level + 1
        true_child = rows[keep, level]
        parent_keys = [tuple(row[:level].tolist()) for row in rows[keep]]
        groups = {}
        for index, key in enumerate(parent_keys):
            groups.setdefault(key, []).append(index)
        children_of = {key: [node for node in by_prefix_length.get(prefix_length, [])
                             if node[:level] == key] for key in groups}
        for key, indices in groups.items():
            children = children_of[key]
            if len(children) <= 1:
                continue
            centres = np.stack([prototypes[child] for child in children])
            column = {child[-1]: position for position, child in enumerate(children)}
            row_chunk = 8192
            for start in range(0, len(indices), row_chunk):
                block = np.asarray(indices[start:start + row_chunk], dtype=np.int64)
                logits = -_distance(sources[block][:, None, :],
                                    centres[None, :, :]) / tau
                top = logits.max(axis=1, keepdims=True)
                log_normaliser = top[:, 0] + np.log(np.exp(logits - top).sum(axis=1))
                picks = np.array([column.get(int(child), -1)
                                  for child in true_child[block]], dtype=np.int64)
                rows_ok = picks >= 0
                if not bool(rows_ok.any()):
                    continue
                chosen = logits[np.nonzero(rows_ok)[0], picks[rows_ok]]
                total += float((chosen - log_normaliser[rows_ok]).sum())
                counted += int(rows_ok.sum())
    return float(total / counted) if counted else 0.0


hyp_hsoftmax_path.note = ("mean hyperbolic tree-conditional log-probability of the true "
                          "SID path over the training pairs")

METRICS = {"hyp_hsoftmax_path": hyp_hsoftmax_path}
FREE_CONSTANTS = {"hyp_hsoftmax_path": 0}


def hyp_hsoftmax_path_null(table: dict, context: dict) -> float:
    """The path log-likelihood MINUS its own size-matched null.

    Measured on the 97 pairs, the plain likelihood correlates with recall at
    -0.49, and with the number of level-2 buckets at -0.78: it rewards a trie whose
    branches are few, i.e. a collapsed table (the three highest values in the
    collection are the 12-code, 2-code and 3-code tables). The null removes exactly
    that: the same statistic on tables with the same level sizes but no joint
    structure, averaged over two fixed relabellings. What is left is how much
    better the real assignment predicts the true path than a table that is only as
    granular.
    """
    real = hyp_hsoftmax_path(table, context)
    rows = np.asarray(table["rows"], dtype=np.int64)
    rng = np.random.default_rng(12345)
    nulls = []
    for _ in range(2):
        shuffled = np.empty_like(rows)
        for level in range(rows.shape[1]):
            shuffled[:, level] = rng.permutation(rows[:, level])
        nulls.append(hyp_hsoftmax_path({"rows": shuffled,
                                        "item_ids": table["item_ids"]}, context))
    return float(real - float(np.mean(nulls)))


hyp_hsoftmax_path_null.note = ("path log-likelihood over its size-matched relabelling "
                               "null, two fixed seeds")
METRICS = {"hyp_hsoftmax_path": hyp_hsoftmax_path,
           "hyp_hsoftmax_path_null": hyp_hsoftmax_path_null}
FREE_CONSTANTS = {"hyp_hsoftmax_path": 0, "hyp_hsoftmax_path_null": 0}
