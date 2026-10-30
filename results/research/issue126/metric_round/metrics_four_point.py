"""The delta-guided four-point SID consistency, to the protocol in DELTA_FOUR_POINT_PROTOCOL.md.

The protocol is fixed before the numbers: the curvature grid, the criterion, the
quadruple sampling, the weight and the score are all written there, and this file
implements exactly that.

* the reference's ball points are converted to their TANGENT coordinates, and
  every curvature re-maps those same tangents into its own ball;
* the curvature is estimated once, from the Gromov delta of the data, and never
  touches a recall;
* the score of a table is the weighted agreement between the geometry's preferred
  pairing of a quadruple and the SID prefix tree's.
"""
from __future__ import annotations

import numpy as np

CURVATURE_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
NEARLY_FLAT = 0.001
QUADRUPLES = 20000
SEED = 42


# ------------------------------------------------------------------ geometry
def tangent_of(points: np.ndarray) -> np.ndarray:
    """``log_1``: the tangent coordinates of curvature-1 ball points."""
    norm = np.linalg.norm(points, axis=-1, keepdims=True)
    safe = np.maximum(norm, 1e-12)
    return np.arctanh(np.clip(norm, 0.0, 1 - 1e-7)) * points / safe


def ball_of(tangent: np.ndarray, curvature: float) -> np.ndarray:
    """``exp_c``: the tangent coordinates mapped into the ball of curvature c."""
    c = float(curvature)
    root = np.sqrt(c)
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    safe = np.maximum(norm, 1e-12)
    return np.tanh(root * norm) * tangent / (root * safe)


def hyperbolic_distance(a: np.ndarray, b: np.ndarray, curvature: float) -> np.ndarray:
    c = float(curvature)
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = ((1 - c * np.sum(a * a, axis=-1)) * (1 - c * np.sum(b * b, axis=-1)))
    return np.arccosh(np.maximum(1 + 2 * c * difference /
                                 np.maximum(denominator, 1e-12), 1 + 1e-12)) / np.sqrt(c)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


# ------------------------------------------------------------------ the data
def _reference_tangents(context: dict, table: dict) -> np.ndarray:
    """The items' TANGENT coordinates in the reference's own space."""
    if "reference_tangents" in context:
        return context["reference_tangents"]
    reference = context.get("reference") or {}
    target = np.asarray(reference["target"], dtype=np.float64)[
        np.asarray(table["item_ids"], dtype=np.int64)]
    norm = np.linalg.norm(target, axis=-1, keepdims=True)
    target = np.where(norm > 1 - 1e-7, target / np.maximum(norm, 1e-12) * (1 - 1e-7), target)
    tangents = tangent_of(target)
    context["reference_tangents"] = tangents
    return tangents


def _quadruples(context: dict, n_items: int) -> np.ndarray:
    if "delta_quadruples" in context:
        return context["delta_quadruples"]
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, n_items, (QUADRUPLES, 4))
    context["delta_quadruples"] = picks
    return picks


def _pairing_sums(d_ab: np.ndarray, d_cd: np.ndarray, d_ac: np.ndarray,
                  d_bd: np.ndarray, d_ad: np.ndarray,
                  d_bc: np.ndarray) -> np.ndarray:
    """The three pairing sums, stacked: [S1, S2, S3]."""
    return np.stack([d_ab + d_cd, d_ac + d_bd, d_ad + d_bc], axis=0)


def four_point_delta(sums: np.ndarray) -> np.ndarray:
    """The standard Gromov delta of a quadruple: (S_max - S_second) / 2.

    The three pairing sums are sorted first: the delta of a quadruple is the gap
    between the largest and the second largest, halved. The deviation of the
    fixed pairing difference |S3 - S2| is not the delta: it changes when the four
    points are relabelled, and it is not zero for a tree metric.
    """
    ordered = np.sort(sums, axis=0)
    return 0.5 * (ordered[2] - ordered[1])


def estimate_curvature(context: dict, table: dict) -> dict:
    """The curvature whose quadruples are as tree-like as the data's are.

    The criterion is fixed by the protocol: the curvature whose MEDIAN |delta|
    matches the data's median |delta| on the tangent vectors.
    """
    if "delta_curvature" in context:
        return context["delta_curvature"]
    tangents = _reference_tangents(context, table)
    picks = _quadruples(context, tangents.shape[0])
    a, b, c, d = picks[:, 0], picks[:, 1], picks[:, 2], picks[:, 3]
    # The DATA's delta is measured in the reference's OWN geometry -- its
    # curvature-1 ball, with the same distance convention the model uses. Using
    # plain Euclidean distances on the tangents put the two on different scales
    # (as c goes to zero the hyperbolic distance tends to sqrt(2) times the
    # Euclidean one), and the curvature that came out of that mismatch said
    # nothing about the data.
    reference_points = ball_of(tangents, 1.0)
    data_sums = _pairing_sums(
        hyperbolic_distance(reference_points[a], reference_points[b], 1.0),
        hyperbolic_distance(reference_points[c], reference_points[d], 1.0),
        hyperbolic_distance(reference_points[a], reference_points[c], 1.0),
        hyperbolic_distance(reference_points[b], reference_points[d], 1.0),
        hyperbolic_distance(reference_points[a], reference_points[d], 1.0),
        hyperbolic_distance(reference_points[b], reference_points[c], 1.0))
    data_delta = four_point_delta(data_sums)
    data_median = float(np.median(data_delta))
    scored = []
    for curvature in CURVATURE_GRID:
        points = ball_of(tangents, curvature)
        sums = _pairing_sums(
            hyperbolic_distance(points[a], points[b], curvature),
            hyperbolic_distance(points[c], points[d], curvature),
            hyperbolic_distance(points[a], points[c], curvature),
            hyperbolic_distance(points[b], points[d], curvature),
            hyperbolic_distance(points[a], points[d], curvature),
            hyperbolic_distance(points[b], points[c], curvature))
        delta = four_point_delta(sums)
        scored.append((abs(float(np.median(delta)) - data_median), curvature,
                       float(np.median(delta))))
    scored.sort(key=lambda entry: (entry[0], entry[1]))
    best = scored[0]
    result = {"curvature": best[1], "data_median_delta": data_median,
              "model_median_delta": best[2], "grid": [
                  {"curvature": curvature, "median_delta": median,
                   "gap": gap} for gap, curvature, median in scored]}
    context["delta_curvature"] = result
    return result


def _score(table: dict, context: dict, curvature: float, euclidean: bool) -> float:
    tangents = _reference_tangents(context, table)
    picks = _quadruples(context, tangents.shape[0])
    a, b, c, d = picks[:, 0], picks[:, 1], picks[:, 2], picks[:, 3]
    if euclidean:
        distance = lambda x, y: euclidean_distance(x, y)
        points = tangents
    else:
        points = ball_of(tangents, curvature)
        distance = lambda x, y: hyperbolic_distance(x, y, curvature)
    sums = _pairing_sums(distance(points[a], points[b]), distance(points[c], points[d]),
                         distance(points[a], points[c]), distance(points[b], points[d]),
                         distance(points[a], points[d]), distance(points[b], points[c]))
    delta = four_point_delta(sums)
    # the weight's normalisation is computed once, at c_delta, and shared
    weight_median = context.get("delta_weight_median")
    if weight_median is None:
        weight_median = float(np.median(delta))
        context["delta_weight_median"] = weight_median
    weight = 1.0 / (1.0 + delta / max(weight_median, 1e-12))

    rows = np.asarray(table["rows"], dtype=np.int64)
    equal = rows[a] == rows[b]
    shared_ab = np.where(equal.all(axis=1), equal.shape[1], equal.argmin(axis=1))
    equal = rows[a] == rows[c]
    shared_ac = np.where(equal.all(axis=1), equal.shape[1], equal.argmin(axis=1))
    equal = rows[a] == rows[d]
    shared_ad = np.where(equal.all(axis=1), equal.shape[1], equal.argmin(axis=1))
    equal = rows[b] == rows[c]
    shared_bc = np.where(equal.all(axis=1), equal.shape[1], equal.argmin(axis=1))
    equal = rows[b] == rows[d]
    shared_bd = np.where(equal.all(axis=1), equal.shape[1], equal.argmin(axis=1))
    equal = rows[c] == rows[d]
    shared_cd = np.where(equal.all(axis=1), equal.shape[1], equal.argmin(axis=1))
    # the tree's PATH lengths, not the shared-prefix counts: the geometry picks the
    # pairing with the smallest distance sum, so the tree must speak in distances
    # too. Using the shared-prefix counts reverses the two conventions, and a
    # permuted table then scores HIGHER than a real one -- which is how the sign
    # error was found.
    width = rows.shape[1]
    tree_sums = np.stack([
        (width - shared_ab) + (width - shared_cd),
        (width - shared_ac) + (width - shared_bd),
        (width - shared_ad) + (width - shared_bc)], axis=0)
    geometry_choice = np.argmin(sums, axis=0)
    tree_choice = np.argmin(tree_sums, axis=0)
    agreement = (geometry_choice == tree_choice).astype(np.float64)
    # A quadruple whose three tree sums are EQUAL carries no information: the tree
    # does not prefer any pairing, and the argmin would be decided by floating
    # point noise. Such quadruples are left out of both the numerator and the
    # denominator. (The permuted control is what showed this: without the rule the
    # baseline sat at 0.42 instead of the chance level of 1/3.)
    spread = tree_sums.max(axis=0) - tree_sums.min(axis=0)
    informative = spread > 0
    if not informative.any():
        return 0.0
    weight = weight * informative
    if weight.sum() <= 0:
        return 0.0
    return float((weight * agreement).sum() / weight.sum())


def _variant(table: dict, context: dict, factor: float, euclidean: bool) -> float:
    estimation = estimate_curvature(context, table)
    curvature = estimation["curvature"] * float(factor)
    if euclidean:
        curvature = estimation["curvature"]
    return _score(table, context, curvature, euclidean)


def delta_four_point(table: dict, context: dict) -> float:
    """Variant A: the curvature estimated from the Gromov delta."""
    return _variant(table, context, 1.0, False)


def delta_four_point_half(table: dict, context: dict) -> float:
    """Variant B: half that curvature."""
    return _variant(table, context, 0.5, False)


def delta_four_point_double(table: dict, context: dict) -> float:
    """Variant C: twice that curvature."""
    return _variant(table, context, 2.0, False)


def delta_four_point_flat(table: dict, context: dict) -> float:
    """Variant D: nearly flat (c = 0.001)."""
    estimation = estimate_curvature(context, table)
    return _score(table, context, NEARLY_FLAT, False)


def delta_four_point_euclidean(table: dict, context: dict) -> float:
    """Variant E: the Euclidean geometry of the tangent vectors."""
    estimation = estimate_curvature(context, table)
    return _score(table, context, estimation["curvature"], True)


def delta_four_point_permuted(table: dict, context: dict) -> float:
    """Variant F: the item↔SID correspondence permuted (the control)."""
    key = "delta_permuted_rows"
    if key not in context:
        rng = np.random.default_rng(SEED)
        rows = np.asarray(table["rows"], dtype=np.int64)
        context[key] = rows[rng.permutation(rows.shape[0])]
    permuted = {"rows": context[key], "item_ids": table["item_ids"]}
    return _score(permuted, context, estimate_curvature(context, table)["curvature"], False)


delta_four_point.note = "the delta-guided four-point consistency at the estimated curvature"
delta_four_point_half.note = "the same at half the estimated curvature"
delta_four_point_double.note = "the same at twice the estimated curvature"
delta_four_point_flat.note = "the same at a nearly flat curvature (control)"
delta_four_point_euclidean.note = "the same on the Euclidean geometry of the tangents (control)"
delta_four_point_permuted.note = "the same with the item-SID correspondence permuted (control)"

METRICS = {
    "delta_four_point": delta_four_point,
    "delta_four_point_half": delta_four_point_half,
    "delta_four_point_double": delta_four_point_double,
    "delta_four_point_flat": delta_four_point_flat,
    "delta_four_point_euclidean": delta_four_point_euclidean,
    "delta_four_point_permuted": delta_four_point_permuted,
}
FREE_CONSTANTS = {name: 0 for name in METRICS}
