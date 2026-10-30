"""The structural families, corrected.

An earlier version of this file had four defects, all of which changed what the
metrics measured:

* the pairs were built from the probe's POSITIVES and NEGATIVES, so the
  "label-free" family was not label-free;
* the shared prefix was counted as "the number of positions that match", which
  is not a prefix: [1,2,3] against [4,2,3] shares no prefix, but that count
  said two;
* the cone metric's null cell was a PERMUTATION of the real cell, so it held the
  same items and the metric was zero to numerical noise;
* the tree distortion was the alignment with the sign flipped, not a distortion.

Everything below is written to the corrected definitions, and the pairs are drawn
from the catalogue without any label.

1. **Gromov-Prefix Alignment** -- over item pairs drawn from the catalogue, how
   well does the hyperbolic shared depth agree with the length of the shared SID
   prefix? The pairs are sampled stratified over the prefix length, so every
   length is represented and the answer does not depend on how rare a long prefix
   is. It names no positive and no negative.

2. **Hyperbolic Tree Distortion** -- the prefix tree's own distance between two
   items (the codes from the first mismatch to the last) against the reference's
   GEODESIC distance: the rank agreement between them says whether the tree
   preserves the geometry.

3. **Hyperbolic Cone Consistency** -- is a cell's angular aperture in the ball
   tighter than that of a DIFFERENT set of the same size drawn from the whole
   catalogue? The null is a different set, not a rearrangement of the same one.

Controls for the alignment: the same construction with the reference's
geodesic distance in Euclidean space, and the same construction at a nearly flat
curvature. If the alignment's power survives neither, it is not the hyperbolic
geometry that carries it.
"""
from __future__ import annotations

import numpy as np

PAIRS_PER_STRATUM = 20000
SEED = 42
FLAT_CURVATURE = 1e-3


# ------------------------------------------------------------------ geometry
def _poincare_distance(a: np.ndarray, b: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    c = float(curvature)
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = ((1.0 - c * np.sum(a * a, axis=-1))
                   * (1.0 - c * np.sum(b * b, axis=-1)))
    return np.arccosh(np.maximum(1.0 + 2.0 * c * difference /
                                 np.maximum(denominator, 1e-12), 1.0 + 1e-12)) / np.sqrt(c)


def _poincare_norm(x: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    c = float(curvature)
    inner = np.clip(1.0 - c * np.sum(x * x, axis=-1), 1e-12, None)
    return 2.0 * np.arccosh(1.0 / np.sqrt(inner)) / np.sqrt(c)


def gromov_depth(a: np.ndarray, b: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    """The Gromov product at the origin: how deep two points share a path."""
    return 0.5 * (_poincare_norm(a, curvature) + _poincare_norm(b, curvature)
                  - _poincare_distance(a, b, curvature))


def euclidean_depth(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The Euclidean analogue of the Gromov product at the origin."""
    return 0.5 * (np.linalg.norm(a, axis=-1) + np.linalg.norm(b, axis=-1)
                  - np.linalg.norm(a - b, axis=-1))


# ------------------------------------------------------------------ prefixes
def shared_prefix_length(rows: np.ndarray, left: np.ndarray,
                         right: np.ndarray) -> np.ndarray:
    """The length of the COMMON PREFIX of two code rows.

    The prefix ends at the FIRST position where the codes differ, so [1,2,3] and
    [4,2,3] share none of it. Counting the positions that happen to match is a
    different quantity and is not a prefix.
    """
    equal = rows[left] == rows[right]
    # the first mismatch, as the position after the last leading match
    mismatch = np.where(equal.all(axis=1), equal.shape[1],
                        equal.argmin(axis=1) if equal.shape[1] else 0)
    return mismatch.astype(np.float64)


def tree_distance(rows: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The prefix tree's own distance: the codes from the first mismatch to the end."""
    return (rows.shape[1] - shared_prefix_length(rows, left, right)).astype(np.float64)


# ------------------------------------------------------------------ sampling
def _stratified_pairs(rows: np.ndarray, rng: np.random.Generator) -> tuple:
    """Item pairs drawn from the catalogue, stratified over the shared prefix length.

    No positive and no negative is read: the pairs are drawn from the items
    themselves, and the stratification only makes sure that every prefix length
    is represented, so the answer does not depend on how rare a long prefix is.
    """
    n_items, width = rows.shape
    levels = width
    left_all, right_all, strata_all = [], [], []
    # strata 1..levels-1: items that share exactly that many codes
    for level in range(1, levels):
        groups = {}
        for item in range(n_items):
            key = tuple(rows[item, :level].tolist())
            groups.setdefault(key, []).append(item)
        usable = [members for members in groups.values() if len(members) > 1]
        if not usable:
            continue
        per_group = max(1, PAIRS_PER_STRATUM // max(1, len(usable)))
        for members in usable:
            picks = rng.choice(len(members), size=(max(2, per_group * 2), 2), replace=True)
            for a, b in picks:
                if a != b:
                    left_all.append(members[a])
                    right_all.append(members[b])
                    strata_all.append(level)
    # stratum 0: items with different first codes
    first = rows[:, 0]
    for _ in range(PAIRS_PER_STRATUM):
        a = int(rng.integers(0, n_items))
        b = int(rng.integers(0, n_items))
        if first[a] != first[b]:
            left_all.append(a)
            right_all.append(b)
            strata_all.append(0)
    return (np.asarray(left_all, dtype=np.int64), np.asarray(right_all, dtype=np.int64),
            np.asarray(strata_all, dtype=np.float64))


def _reference_points(context: dict, table: dict) -> np.ndarray:
    reference = context.get("reference") or {}
    target = np.asarray(reference["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    norm = np.linalg.norm(target, axis=-1, keepdims=True)
    return np.where(norm > 1 - 1e-7, target / np.maximum(norm, 1e-12) * (1 - 1e-7), target)


# ------------------------------------------------------------------ metrics
def _alignment(table: dict, context: dict, *, curvature: float = 1.0,
               euclidean: bool = False) -> float:
    from scipy import stats
    points = _reference_points(context, table)
    rows = np.asarray(table["rows"], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    left, right, strata = _stratified_pairs(rows, rng)
    if left.size < 8:
        return 0.0
    if euclidean:
        depth = euclidean_depth(points[left], points[right])
    else:
        depth = gromov_depth(points[left], points[right], curvature)
    prefix = shared_prefix_length(rows, left, right)
    if np.unique(prefix).size < 2 or np.unique(depth).size < 2:
        return 0.0
    return float(stats.spearmanr(depth, prefix)[0])


def gromov_prefix_alignment(table: dict, context: dict) -> float:
    """Does a deeper hyperbolic shared ancestor come with a longer shared prefix?"""
    return _alignment(table, context, curvature=1.0)


def gromov_prefix_alignment_flat(table: dict, context: dict) -> float:
    """The same construction at a nearly flat curvature (the control)."""
    return _alignment(table, context, curvature=FLAT_CURVATURE)


def gromov_prefix_alignment_euclidean(table: dict, context: dict) -> float:
    """The same construction with a Euclidean shared depth (the control)."""
    return _alignment(table, context, euclidean=True)


def hyperbolic_tree_distortion(table: dict, context: dict) -> float:
    """Does the SID prefix tree preserve the reference's geodesic distances?"""
    from scipy import stats
    points = _reference_points(context, table)
    rows = np.asarray(table["rows"], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    left, right, _ = _stratified_pairs(rows, rng)
    if left.size < 8:
        return 0.0
    tree = tree_distance(rows, left, right)
    geodesic = _poincare_distance(points[left], points[right], 1.0)
    if np.unique(tree).size < 2 or np.unique(geodesic).size < 2:
        return 0.0
    return float(stats.spearmanr(tree, geodesic)[0])


def hyperbolic_cone_consistency(table: dict, context: dict) -> float:
    """Is a cell's angular cone tighter than a DIFFERENT cell of the same size?"""
    points = _reference_points(context, table)
    rows = np.asarray(table["rows"], dtype=np.int64)
    codes = rows[:, 0]
    n_items = rows.shape[0]
    cells = {}
    for position, code in enumerate(codes.tolist()):
        cells.setdefault(int(code), []).append(position)
    rng = np.random.default_rng(SEED)

    def aperture(members: np.ndarray) -> float:
        block = points[members]
        centre = block.mean(axis=0)
        norm = float(np.linalg.norm(centre))
        if norm < 1e-12:
            return 0.0
        direction = centre / norm
        radial = block - np.sum(block * direction, axis=-1, keepdims=True) * direction
        radial_norm = np.linalg.norm(radial, axis=-1)
        lengths = np.linalg.norm(block, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sin_angle = np.where(lengths > 1e-12, radial_norm / lengths, 0.0)
        return float(np.mean(np.arcsin(np.clip(sin_angle, 0.0, 1.0))))

    ratios, weights = [], []
    for members in cells.values():
        if len(members) < 2:
            continue
        real = aperture(np.asarray(members, dtype=np.int64))
        # the null is a DIFFERENT set of the same size, drawn from the whole
        # catalogue: a rearrangement of the same items would make the two the same
        null_members = rng.choice(n_items, size=len(members), replace=False)
        null = aperture(np.asarray(null_members, dtype=np.int64))
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
                                "of a catalogue pair and the length of their shared SID "
                                "prefix, over label-free pairs")
gromov_prefix_alignment_flat.note = "the same at curvature 1e-3 (control)"
gromov_prefix_alignment_euclidean.note = "the same with a Euclidean shared depth (control)"
hyperbolic_tree_distortion.note = ("the rank agreement between the SID prefix tree's "
                                   "distance and the reference's geodesic distance")
hyperbolic_cone_consistency.note = ("how much tighter each level-1 cell's angular cone "
                                    "is than a DIFFERENT set of the same size")

METRICS = {
    "gromov_prefix_alignment": gromov_prefix_alignment,
    "gromov_prefix_alignment_flat": gromov_prefix_alignment_flat,
    "gromov_prefix_alignment_euclidean": gromov_prefix_alignment_euclidean,
    "hyperbolic_tree_distortion": hyperbolic_tree_distortion,
    "hyperbolic_cone_consistency": hyperbolic_cone_consistency,
}
FREE_CONSTANTS = {name: 0 for name in METRICS}
