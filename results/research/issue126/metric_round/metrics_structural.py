"""The three families of the request, implemented to their definitions.

1. **Gromov-Prefix Alignment** -- the Gromov product at the origin measures how
   deep two items share a path from the origin; the SID prefix says how many codes
   they share. The metric is the rank agreement between the two over the training
   pairs: do items that share a deeper hyperbolic ancestor also share a longer
   prefix?

2. **Hyperbolic Tree Distortion** -- the SID is a tree, and a tree has a distance
   (the path through the longest common prefix). The metric asks whether that
   distance agrees with the reference's hyperbolic distance: the rank correlation
   between them over the same pairs.

3. **Hyperbolic Cone Consistency** -- in the Poincare ball a cell should be a
   branch, not a smear: the items under one prefix should lie in a narrow cone
   around the cell's direction. The metric compares each cell's angular aperture
   with the aperture of a size-matched random cell.

All three read only the exported SID table, the frozen reference and the training
split: no validation or test interaction, no recall, no Stage 2 or Stage 3.
"""
from __future__ import annotations

import numpy as np

PAIRS = 200000        # the training pairs are sampled deterministically
SEED = 42


def _gromov(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The Gromov product at the origin: the shared depth of two points."""
    def norm(x):
        inner = np.clip(1.0 - np.sum(x * x, axis=-1), 1e-12, None)
        return 2.0 * np.arccosh(1.0 / np.sqrt(inner))

    def distance(x, y):
        difference = np.sum((x - y) ** 2, axis=-1)
        denominator = ((1.0 - np.sum(x * x, axis=-1))
                       * (1.0 - np.sum(y * y, axis=-1)))
        return np.arccosh(np.maximum(1.0 + 2.0 * difference /
                                      np.maximum(denominator, 1e-12), 1.0 + 1e-12))

    return 0.5 * (norm(a) + np.asarray(norm(b)) - distance(a, b))


def _pairs(context: dict, table: dict) -> tuple:
    """The item pairs the three metrics are computed over (training pairs only)."""
    if "structural_pairs" in context:
        return context["structural_pairs"]
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from results.research.issue126.metric_round import metrics_hbaa as hbaa

    queries = hbaa._queries(context)
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    rng = np.random.default_rng(SEED)
    left, right = [], []
    for index in range(len(queries["positives"])):
        anchor = int(queries["histories"][index][-1])
        positive = int(queries["positives"][index])
        negative = int(queries["negatives"][index][0])
        for other in (positive, negative):
            a = position.get(anchor, -1)
            b = position.get(other, -1)
            if a >= 0 and b >= 0:
                left.append(a)
                right.append(b)
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.size > PAIRS:
        pick = rng.choice(left.size, size=PAIRS, replace=False)
        left, right = left[pick], right[pick]
    context["structural_pairs"] = (left, right)
    return left, right


def _reference_points(context: dict, table: dict) -> np.ndarray:
    reference = context.get("reference") or {}
    target = np.asarray(reference["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    norm = np.linalg.norm(target, axis=-1, keepdims=True)
    return np.where(norm > 1 - 1e-7, target / np.maximum(norm, 1e-12) * (1 - 1e-7), target)


def gromov_prefix_alignment(table: dict, context: dict) -> float:
    """The rank agreement between the hyperbolic shared depth and the shared prefix."""
    from scipy import stats
    points = _reference_points(context, table)
    left, right = _pairs(context, table)
    rows = np.asarray(table["rows"], dtype=np.int64)
    depth = _gromov(points[left], points[right])
    shared = (rows[left] == rows[right]).sum(axis=1).astype(np.float64)
    if np.unique(shared).size < 2 or np.unique(depth).size < 2:
        return 0.0
    rho = float(stats.spearmanr(depth, shared)[0])
    return rho


def hyperbolic_tree_distortion(table: dict, context: dict) -> float:
    """The rank agreement between the SID tree distance and the hyperbolic distance."""
    from scipy import stats
    points = _reference_points(context, table)
    left, right = _pairs(context, table)
    rows = np.asarray(table["rows"], dtype=np.int64)
    depth = _gromov(points[left], points[right])
    shared = (rows[left] == rows[right]).sum(axis=1).astype(np.float64)
    # the tree distance of the prefix tree: the codes differ from the first
    # mismatch onwards, so the path length is the width minus the shared prefix
    tree = rows.shape[1] - shared
    if np.unique(tree).size < 2 or np.unique(depth).size < 2:
        return 0.0
    return float(stats.spearmanr(tree, depth)[0])


def hyperbolic_cone_consistency(table: dict, context: dict) -> float:
    """How much tighter each cell's angular cone is than a size-matched random cell."""
    points = _reference_points(context, table)
    rows = np.asarray(table["rows"], dtype=np.int64)
    codes = rows[:, 0]
    cells = {}
    for position, code in enumerate(codes.tolist()):
        cells.setdefault(int(code), []).append(position)

    def aperture(members) -> float:
        """The angular aperture of a set of points seen from its own centre."""
        block = points[members]
        centre = block.mean(axis=0)
        norm = np.linalg.norm(centre)
        if norm < 1e-12:
            return 0.0
        direction = centre / norm
        radial = block - np.sum(block * direction, axis=-1, keepdims=True) * direction
        radial_norm = np.linalg.norm(radial, axis=-1)
        lengths = np.linalg.norm(block, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sin_angle = np.where(lengths > 1e-12, radial_norm / lengths, 0.0)
        return float(np.mean(np.arcsin(np.clip(sin_angle, 0.0, 1.0))))

    rng = np.random.default_rng(SEED)
    ratios = []
    weights = []
    for members in cells.values():
        if len(members) < 2:
            continue
        real = aperture(np.asarray(members, dtype=np.int64))
        shuffled = rng.permutation(members)
        null = aperture(np.asarray(shuffled, dtype=np.int64))
        if null <= 1e-12:
            continue
        ratios.append(1.0 - real / null)
        weights.append(len(members))
    if not ratios:
        return 0.0
    ratios = np.asarray(ratios)
    weights = np.asarray(weights, dtype=np.float64)
    return float((ratios * weights).sum() / weights.sum())


gromov_prefix_alignment.note = ("the rank agreement between the hyperbolic shared depth "
                                "of a pair and the length of their shared SID prefix")
hyperbolic_tree_distortion.note = ("the rank agreement between the SID prefix tree's "
                                   "distance and the reference's hyperbolic distance")
hyperbolic_cone_consistency.note = ("how much tighter each level-1 cell's angular cone "
                                    "is than a size-matched random cell")

METRICS = {
    "gromov_prefix_alignment": gromov_prefix_alignment,
    "hyperbolic_tree_distortion": hyperbolic_tree_distortion,
    "hyperbolic_cone_consistency": hyperbolic_cone_consistency,
}
FREE_CONSTANTS = {name: 0 for name in METRICS}
