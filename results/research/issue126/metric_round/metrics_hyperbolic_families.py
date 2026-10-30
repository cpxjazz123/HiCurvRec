"""Seven hyperbolic properties of the SID table, each with its own controls.

Every family answers one question about the model's geometry and the SID prefix
tree, and every family is measured four times:

* hyperbolic at curvature 1 (the reference's own ball),
* hyperbolic at curvature 8 (the curvature the four-point protocol estimates),
* the Euclidean control, computed on the same points with linear distances,
* a permuted control, where the SID rows are shuffled across the items, so the
  geometry is untouched and only the SID correspondence is destroyed.

The permuted control is the one that decides whether a number means anything: a
property of the SID tree must collapse when the tree is shuffled.

The families are those of the user's brief:

1. ``gromov``        the Gromov product, i.e. how deep two items share an ancestor;
2. ``radial``        the radius from the origin as a hierarchy depth;
3. ``geodesic``      whether a parent's point lies on the root-to-child geodesic;
4. ``cone``          whether a child lies inside its parent's entailment cone;
5. ``angle``         whether branches separate angularly;
6. ``busemann``      whether a prefix shares a horosphere (a boundary direction);
7. ``volume``        whether the item budget grows exponentially with the radius.

None of these is assumed to work. A family that does not beat its Euclidean and
permuted controls is a negative result and is recorded as one.
"""
from __future__ import annotations

import numpy as np

SEED = 42
CURVATURES = (1.0, 8.0)
PAIRS = 20000
CONE_APERTURE = 0.1
SHELLS = 16


# ------------------------------------------------------------------ geometry
def tangent_of(points: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(points, axis=-1, keepdims=True)
    safe = np.maximum(norm, 1e-12)
    return np.arctanh(np.clip(norm, 0.0, 1 - 1e-7)) * points / safe


def ball_of(tangent: np.ndarray, curvature: float) -> np.ndarray:
    root = np.sqrt(float(curvature))
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    safe = np.maximum(norm, 1e-12)
    return np.tanh(root * norm) * tangent / (root * safe)


def origin_of(points: np.ndarray) -> np.ndarray:
    return np.zeros(points.shape[-1], dtype=np.float64)


def distance(a: np.ndarray, b: np.ndarray, curvature: float) -> np.ndarray:
    c = float(curvature)
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = (1 - c * np.sum(a * a, axis=-1)) * (1 - c * np.sum(b * b, axis=-1))
    return np.arccosh(np.maximum(1 + 2 * c * difference /
                                 np.maximum(denominator, 1e-12), 1 + 1e-12)) / np.sqrt(c)


def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def _einstein_midpoint(points: np.ndarray, curvature: float) -> np.ndarray:
    """``exp_0(mean(log_0(.)))``: the cheap, well-defined ball centroid."""
    tangents = tangent_of(np.clip(points, -1 + 1e-7, 1 - 1e-7))
    return ball_of(tangents.mean(axis=0), curvature)


def _euclidean_midpoint(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


# ------------------------------------------------------------------ the data
def _points(context: dict, table: dict, curvature=None) -> np.ndarray:
    """The table's items in the reference ball, or the raw vectors when flat."""
    if "family_points" not in context:
        reference = context.get("reference") or {}
        target = np.asarray(reference["target"], dtype=np.float64)[
            np.asarray(table["item_ids"], dtype=np.int64)]
        norm = np.linalg.norm(target, axis=-1, keepdims=True)
        context["family_points"] = np.where(
            norm > 1 - 1e-7, target / np.maximum(norm, 1e-12) * (1 - 1e-7), target)
    points = context["family_points"]
    if curvature is None:
        return points
    tangent = tangent_of(points)
    return ball_of(tangent, curvature)


def _rows(context: dict, table: dict, permuted: bool) -> np.ndarray:
    """The SID rows, optionally shuffled across the items."""
    if not permuted:
        return np.asarray(table["rows"], dtype=np.int64)
    if "family_permutation" not in context:
        rng = np.random.default_rng(SEED)
        context["family_permutation"] = rng.permutation(len(table["rows"]))
    return np.asarray(table["rows"], dtype=np.int64)[context["family_permutation"]]


def _keys(rows: np.ndarray, level: int) -> np.ndarray:
    """A single integer per prefix at ``level`` (1, 2 or 3)."""
    base = int(rows.max()) + 1
    key = rows[:, 0].astype(np.int64)
    for column in range(1, level):
        key = key * base + rows[:, column]
    return key


def _groups(keys: np.ndarray):
    unique, inverse = np.unique(keys, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(inverse[order])) + 1, [len(order)]))
    return unique, inverse, order, bounds


def _representatives(points: np.ndarray, inverse: np.ndarray, n_groups: int,
                     curvature) -> np.ndarray:
    if curvature is None:
        sums = np.zeros((n_groups, points.shape[1]))
        np.add.at(sums, inverse, points)
        counts = np.bincount(inverse, minlength=n_groups)[:, None]
        return sums / np.maximum(counts, 1)
    tangent = tangent_of(np.clip(points, -1 + 1e-7, 1 - 1e-7))
    sums = np.zeros((n_groups, points.shape[1]))
    np.add.at(sums, inverse, tangent)
    counts = np.bincount(inverse, minlength=n_groups)[:, None]
    return ball_of(sums / np.maximum(counts, 1), curvature)


def _sample_pairs(rows: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Half the pairs uniform, half drawn INSIDE one level-2 prefix.

    Two uniform items share a level-2 prefix about once in a thousand draws, so a
    uniform sample leaves a few dozen positives and the AUC is noise. The
    stratified half puts the positives and the near-miss negatives (same parent,
    different leaf) in the sample at a usable rate.
    """
    rng = np.random.default_rng(seed)
    n_items = len(rows)
    half = PAIRS // 2
    left = rng.integers(0, n_items, half)
    right = rng.integers(0, n_items, half)
    keep = left != right
    uniform = np.stack([left[keep], right[keep]], axis=1)

    keys = _keys(rows, 2)
    unique, inverse, order, bounds = _groups(keys)
    sizes = np.diff(bounds)
    usable = np.flatnonzero(sizes >= 2)
    pick = rng.choice(usable, size=half, replace=True)
    offsets = rng.integers(0, sizes[pick], size=half)
    starts = bounds[pick]
    first = order[starts + offsets]
    second = order[starts + rng.integers(0, sizes[pick], size=half)]
    clash = first == second
    second[clash] = order[starts[clash] + (offsets[clash] + 1) % sizes[pick[clash]]]
    stratified = np.stack([first, second], axis=1)
    return np.concatenate([uniform, stratified], axis=0)


def _shared_level(rows: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    left, right = rows[pairs[:, 0]], rows[pairs[:, 1]]
    shared = np.zeros(len(pairs), dtype=np.int64)
    for column in range(3):
        shared += (left[:, column] == right[:, column]) & (shared == column)
    return shared


def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    if positive.all() or not positive.any():
        return 0.0
    ranks = np.empty(len(scores))
    order = np.argsort(scores, kind="stable")
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(positive.sum())
    n_neg = len(scores) - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _variance_ratio(values: np.ndarray, inverse: np.ndarray, n_groups: int) -> float:
    """1 - E[within-group variance] / total variance (an eta-squared)."""
    total = float(np.var(values))
    if total <= 1e-18:
        return 0.0
    sums = np.bincount(inverse, weights=values, minlength=n_groups)
    counts = np.bincount(inverse, minlength=n_groups)
    means = sums / np.maximum(counts, 1)
    within = float(np.average((values - means[inverse]) ** 2, weights=None))
    return float(1.0 - within / total)


# ------------------------------------------------------------------ families
def gromov(context: dict, table: dict, curvature=None, permuted=False) -> float:
    """Does the Gromov product separate pairs that share a deep prefix?"""
    points = _points(context, table, curvature)
    rows = _rows(context, table, permuted)
    pairs = _sample_pairs(rows)
    if curvature is None:
        origin = origin_of(points)
        left = euclidean(points[pairs[:, 0]], origin)
        right = euclidean(points[pairs[:, 1]], origin)
        direct = euclidean(points[pairs[:, 0]], points[pairs[:, 1]])
    else:
        origin = origin_of(points)
        left = distance(points[pairs[:, 0]], origin, curvature)
        right = distance(points[pairs[:, 1]], origin, curvature)
        direct = distance(points[pairs[:, 0]], points[pairs[:, 1]], curvature)
    product = 0.5 * (left + right - direct)
    shared = _shared_level(rows, pairs)
    return _auc(product, shared >= 2)


def radial_depth(context: dict, table: dict, curvature=None, permuted=False) -> float:
    """Does the radius grow from the level-1 to the level-3 prefix?"""
    points = _points(context, table, curvature)
    rows = _rows(context, table, permuted)
    origin = origin_of(points)
    measure = euclidean if curvature is None else (lambda a, b: distance(a, b, curvature))
    scores = []
    for level in (1, 2, 3):
        keys = _keys(rows, level)
        unique, inverse, _, _ = _groups(keys)
        reps = _representatives(points, inverse, len(unique), curvature)
        scores.append(float(np.median(measure(reps, origin))))
    return float((scores[2] - scores[0]) / max(scores[2] + scores[0], 1e-12))


def radial_purity(context: dict, table: dict, curvature=None, permuted=False) -> float:
    """Do items of one prefix sit at one radius?"""
    points = _points(context, table, curvature)
    rows = _rows(context, table, permuted)
    origin = origin_of(points)
    radii = euclidean(points, origin) if curvature is None else distance(points, origin, curvature)
    unique, inverse, _, _ = _groups(_keys(rows, 3))
    return _variance_ratio(radii, inverse, len(unique))


def geodesic(context: dict, table: dict, curvature=None, permuted=False) -> float:
    """Is the parent on the root-to-child geodesic?  The mean relative slack."""
    points = _points(context, table, curvature)
    rows = _rows(context, table, permuted)
    origin = origin_of(points)
    measure = euclidean if curvature is None else (lambda a, b: distance(a, b, curvature))
    slacks = []
    for level in (1, 2, 3):
        keys = _keys(rows, level)
        unique, inverse, _, _ = _groups(keys)
        parents = _representatives(points, inverse, len(unique), curvature)
        child_root = measure(points, origin)
        parent_root = measure(parents, origin)[inverse]
        child_parent = measure(points, parents[inverse])
        slack = child_root - parent_root - child_parent
        slacks.append(float(np.mean(slack / np.maximum(child_root, 1e-12))))
    return -float(np.mean(slacks))


def cone(context: dict, table: dict, curvature=None, permuted=False) -> float:
    """How far inside its parent's entailment cone does a child sit?

    The 0/1 Ganea test is unusable here and that is a fact about the data: the
    frozen reference's points sit at a nearly constant radius (median 0.9115 by a
    spread of 0.005), so the aperture ``arcsin(K(1-||p||^2)/||p||)`` collapses to
    about one degree for any K that is not tuned to this data, and every child
    falls outside it for every SID (the fraction is identically zero). What is
    measured instead has no knob: the cosine, at the parent, between the outward
    radial direction (origin minus parent) and the direction to the child. It is
    one at full descent and negative for a child that climbs back toward the root.
    """
    points = _points(context, table, curvature)
    rows = _rows(context, table, permuted)
    origin = origin_of(points)
    cosines = []
    for level in (1, 2, 3):
        keys = _keys(rows, level)
        unique, inverse, _, _ = _groups(keys)
        parents = _representatives(points, inverse, len(unique), curvature)
        parent = parents[inverse]
        down = origin - parent
        out = points - parent
        norm = np.maximum(np.linalg.norm(down, axis=1) * np.linalg.norm(out, axis=1), 1e-12)
        cosines.append(float(np.mean(np.sum(down * out, axis=1) / norm)))
    return float(np.mean(cosines))


def angle(context: dict, table: dict, curvature=None, permuted=False) -> float:
    """Do the children of one parent separate angularly more than chance?"""
    points = _points(context, table, curvature)
    rows = _rows(context, table, permuted)
    rng = np.random.default_rng(SEED)
    inside, outside = [], []
    for level in (1, 2):
        child_level = level + 1 if level < 3 else 3
        parent_keys = _keys(rows, level)
        child_keys = _keys(rows, child_level)
        seen = set()
        for key in np.unique(parent_keys):
            members = np.flatnonzero(parent_keys == key)
            if len(members) < 2:
                continue
            _, inverse, _, _ = _groups(child_keys[members])
            if len(np.unique(inverse)) < 2:
                continue
            reps = _representatives(points[members], inverse, len(np.unique(inverse)), curvature)
            centre = _einstein_midpoint(points[members], curvature) if curvature else points[members].mean(axis=0)
            vectors = reps - centre
            norms = np.linalg.norm(vectors, axis=1)
            keep = norms > 1e-9
            vectors, norms = vectors[keep], norms[keep]
            if len(vectors) < 2:
                continue
            cosine = (vectors @ vectors.T) / np.maximum(np.outer(norms, norms), 1e-12)
            upper = np.triu_indices(len(vectors), 1)
            inside.append(np.degrees(np.arccos(np.clip(cosine[upper], -1.0, 1.0))).mean())
            if len(inside) > 60:
                break
        for _ in range(20):
            pick = rng.integers(0, len(points), 2)
            first, second = points[pick[0]], points[pick[1]]
            centre = origin_of(points)
            a, b2 = first - centre, second - centre
            cosine = float(np.dot(a, b2) / max(np.linalg.norm(a) * np.linalg.norm(b2), 1e-12))
            outside.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    if not inside or not outside:
        return 0.0
    return float(np.mean(inside) / max(np.mean(outside), 1e-9))


def busemann(context: dict, table: dict, curvature=None, permuted=False) -> float:
    """Do the items of one prefix share a horosphere (one boundary direction)?"""
    points = _points(context, table, curvature)
    rows = _rows(context, table, permuted)
    norms = np.linalg.norm(points, axis=1)
    direction = points.mean(axis=0)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    if curvature is None:
        values = points @ direction
    else:
        # b_u(x) = log(||x-u||^2 / (1 - ||x||^2)), the curvature-1 horosphere
        squared = np.sum((points - direction[None, :]) ** 2, axis=1)
        values = np.log(np.maximum(squared, 1e-12) / np.maximum(1 - norms ** 2, 1e-12))
    unique, inverse, _, _ = _groups(_keys(rows, 3))
    return _variance_ratio(values, inverse, len(unique))


def volume(context: dict, table: dict, curvature=None, permuted=False) -> float:
    """Does the item count grow exponentially with the radius, as the space allows?"""
    points = _points(context, table, curvature)
    rows = _rows(context, table, permuted)
    origin = origin_of(points)
    radii = euclidean(points, origin) if curvature is None else distance(points, origin, curvature)
    edges = np.quantile(radii, np.linspace(0, 1, SHELLS + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    counts = np.histogram(radii, bins=edges)[0].astype(np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = counts > 0
    if keep.sum() < 3:
        return 0.0
    slope, intercept = np.polyfit(centres[keep], np.log(counts[keep]), 1)
    return float(slope)


def volume_fit(context: dict, table: dict, curvature=None, permuted=False) -> float:
    """How well an exponential growth law describes the item distribution."""
    points = _points(context, table, curvature)
    rows = _rows(context, table, permuted)
    origin = origin_of(points)
    radii = euclidean(points, origin) if curvature is None else distance(points, origin, curvature)
    edges = np.unique(np.quantile(radii, np.linspace(0, 1, SHELLS + 1)))
    if len(edges) < 3:
        return 0.0
    counts = np.histogram(radii, bins=edges)[0].astype(np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = counts > 0
    if keep.sum() < 3:
        return 0.0
    slope, intercept = np.polyfit(centres[keep], np.log(counts[keep]), 1)
    predicted = slope * centres[keep] + intercept
    residual = np.sum((np.log(counts[keep]) - predicted) ** 2)
    total = np.sum((np.log(counts[keep]) - np.log(counts[keep]).mean()) ** 2)
    return float(1.0 - residual / max(total, 1e-18))


# ------------------------------------------------------------------ the metric table
def _wrap(function, curvature=None, permuted=False):
    def metric(table, context):
        return float(function(context, table, curvature=curvature, permuted=permuted))
    return metric


METRICS = {}
FAMILIES = {
    "gromov": gromov,
    "radial_depth": radial_depth,
    "radial_purity": radial_purity,
    "geodesic": geodesic,
    "cone": cone,
    "angle": angle,
    "busemann": busemann,
    "volume": volume,
    "volume_fit": volume_fit,
}
for name, function in FAMILIES.items():
    for curvature in CURVATURES:
        METRICS["%s_c%g" % (name, curvature)] = _wrap(function, curvature=curvature)
    METRICS["%s_euclidean" % name] = _wrap(function, curvature=None)
    METRICS["%s_permuted" % name] = _wrap(function, curvature=1.0, permuted=True)


# ------------------------------------------------------------------ behaviour-aware
def _behaviour_pairs(context: dict, rows: np.ndarray):
    """The training triplets, filtered to the items this table holds."""
    pairs = context.get("train_pairs") or {}
    index = {}
    for place, item in enumerate(np.asarray(context["family_item_ids"], dtype=np.int64)):
        index[int(item)] = place
    anchors, positives, negatives = [], [], []
    negatives_array = np.asarray(pairs.get("negatives", []))
    if negatives_array.ndim == 2:
        negatives_array = negatives_array[:, 0]
    limit = 40000
    for anchor, positive, negative in zip(np.asarray(pairs.get("anchors", []))[:limit],
                                          np.asarray(pairs.get("positives", []))[:limit],
                                          negatives_array[:limit]):
        a, p2, n2 = index.get(int(anchor)), index.get(int(positive)), index.get(int(negative))
        if a is None or p2 is None or n2 is None:
            continue
        anchors.append(a)
        positives.append(p2)
        negatives.append(n2)
    if not anchors:
        return None
    return np.array(anchors), np.array(positives), np.array(negatives)


def _lca_index(rows: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The length of the shared prefix of every pair (0 to 3)."""
    shared = np.zeros(len(left), dtype=np.int64)
    for column in range(3):
        shared += (rows[left, column] == rows[right, column]) & (shared == column)
    return shared


def _prefix_representatives(context: dict, table: dict, rows: np.ndarray,
                            curvature) -> dict:
    points = _points(context, table, curvature)
    store = {"points": points, "group": {}, "representative": {}}
    for level in (1, 2, 3):
        keys = _keys(rows, level)
        unique, inverse, _, _ = _groups(keys)
        store["group"][level] = inverse
        store["representative"][level] = _representatives(points, inverse, len(unique), curvature)
    return store


def _representative_for(context: dict, table: dict, rows: np.ndarray, left: np.ndarray,
                        right: np.ndarray, curvature) -> np.ndarray:
    """The representative of the deepest shared prefix of each pair.

    The group of an item at a level is already known from ``np.unique``, so the
    ancestor of a pair is read directly: the prefix of the left item at the level
    the two share.
    """
    store = context["family_representatives"]
    shared = _lca_index(rows, left, right)
    out = np.zeros((len(left), store["points"].shape[1]))
    for level in (1, 2, 3):
        chosen = shared == level
        if chosen.any():
            out[chosen] = store["representative"][level][store["group"][level][left[chosen]]]
    return out


def behaviour_geodesic(context: dict, table: dict, curvature=None, permuted=False) -> float:
    """Behaviour-aware geodesic alignment (the user's first recommendation).

    For a behaviourally positive pair, the deepest shared SID prefix is the
    candidate lowest common ancestor. If the SID tree were the geometry's own
    tree, that ancestor would sit ON the geodesic between the two items, so its
    slack ``d(root,P) + d(P,child) - d(root,child)`` would be small; for a
    negative pair the ancestor is arbitrary and the slack should be larger. The
    metric is the AUC of that slack separating positives from negatives.
    """
    points = _points(context, table, curvature)
    rows = _rows(context, table, permuted)
    context["family_item_ids"] = table["item_ids"]
    triplets = _behaviour_pairs(context, rows)
    if triplets is None:
        return 0.0
    anchors, positives, negatives = triplets
    if "family_representatives" not in context or permuted:
        context["family_representatives"] = _prefix_representatives(context, table, rows, curvature)
    measure = euclidean if curvature is None else (lambda a, b: distance(a, b, curvature))
    origin = origin_of(points)

    def slack(left, right):
        parent = _representative_for(context, table, rows, left, right, curvature)
        root_left = measure(points[left], origin)
        root_right = measure(points[right], origin)
        to_parent = measure(points[left], parent) + measure(points[right], parent)
        return root_left + root_right - to_parent

    positive_slack = slack(anchors, positives)
    negative_slack = slack(anchors, negatives)
    scores = np.concatenate([-positive_slack, -negative_slack])
    labels = np.concatenate([np.ones(len(positive_slack), dtype=bool),
                             np.zeros(len(negative_slack), dtype=bool)])
    return _auc(scores, labels)


def behaviour_busemann(context: dict, table: dict, curvature=None, permuted=False) -> float:
    """Do a behaviourally related pair share a horosphere more than a negative pair?"""
    points = _points(context, table, curvature)
    rows = _rows(context, table, permuted)
    context["family_item_ids"] = table["item_ids"]
    triplets = _behaviour_pairs(context, rows)
    if triplets is None:
        return 0.0
    anchors, positives, negatives = triplets
    direction = points.mean(axis=0)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    if curvature is None:
        values = points @ direction
    else:
        norms = np.linalg.norm(points, axis=1)
        squared = np.sum((points - direction[None, :]) ** 2, axis=1)
        values = np.log(np.maximum(squared, 1e-12) / np.maximum(1 - norms ** 2, 1e-12))
    gap_positive = np.abs(values[anchors] - values[positives])
    gap_negative = np.abs(values[anchors] - values[negatives])
    scores = np.concatenate([-gap_positive, -gap_negative])
    labels = np.concatenate([np.ones(len(gap_positive), dtype=bool),
                             np.zeros(len(gap_negative), dtype=bool)])
    return _auc(scores, labels)


BEHAVIOUR_FAMILIES = {
    "behaviour_geodesic": behaviour_geodesic,
    "behaviour_busemann": behaviour_busemann,
}
for name, function in BEHAVIOUR_FAMILIES.items():
    for curvature in CURVATURES:
        METRICS["%s_c%g" % (name, curvature)] = _wrap(function, curvature=curvature)
    METRICS["%s_euclidean" % name] = _wrap(function, curvature=None)
    METRICS["%s_permuted" % name] = _wrap(function, curvature=1.0, permuted=True)


def behaviour_geodesic_matched(context: dict, table: dict, curvature=None,
                               permuted=False) -> float:
    """Geodesic alignment, with the ancestor depth held fixed.

    The unmatched version is confounded: a behaviourally positive pair shares a
    deeper prefix, and a deeper ancestor is farther from the root, so its slack is
    larger for a reason that has nothing to do with the geodesic. Here the
    negative for each anchor is an item from the SAME shared prefix that is not a
    behaviour positive, so the ancestor, its depth and the prefix size are
    identical on both sides and only the behaviour differs.
    """
    points = _points(context, table, curvature)
    rows = _rows(context, table, permuted)
    context["family_item_ids"] = table["item_ids"]
    triplets = _behaviour_pairs(context, rows)
    if triplets is None:
        return 0.0
    anchors, positives, _ = triplets
    if "family_representatives" not in context or permuted:
        context["family_representatives"] = _prefix_representatives(context, table, rows, curvature)
    store = context["family_representatives"]
    measure = euclidean if curvature is None else (lambda a, b: distance(a, b, curvature))
    origin = origin_of(points)
    rng = np.random.default_rng(SEED)
    n_items = len(points)

    shared_positive = _lca_index(rows, anchors, positives)
    matched = np.empty_like(positives)
    for place in range(len(anchors)):
        level = int(shared_positive[place])
        if level == 0:
            matched[place] = positives[place]
            continue
        group = store["group"][level][anchors[place]]
        for _ in range(200):
            candidate = int(rng.integers(0, n_items))
            if (store["group"][level][candidate] == group
                    and candidate != int(anchors[place]) and candidate != int(positives[place])):
                matched[place] = candidate
                break
        else:
            matched[place] = positives[place]

    keep = matched != positives
    anchors, positives, matched = anchors[keep], positives[keep], matched[keep]
    if not len(anchors):
        return 0.0

    def slack(right):
        parent = _representative_for(context, table, rows, anchors, right, curvature)
        root_anchor = measure(points[anchors], origin)
        root_right = measure(points[right], origin)
        return root_anchor + root_right - (measure(points[anchors], parent)
                                           + measure(points[right], parent))

    scores = np.concatenate([-slack(positives), -slack(matched)])
    labels = np.concatenate([np.ones(len(anchors), dtype=bool),
                             np.zeros(len(anchors), dtype=bool)])
    return _auc(scores, labels)


for curvature in CURVATURES:
    METRICS["behaviour_geodesic_matched_c%g" % curvature] = _wrap(
        behaviour_geodesic_matched, curvature=curvature)
METRICS["behaviour_geodesic_matched_euclidean"] = _wrap(behaviour_geodesic_matched, curvature=None)
METRICS["behaviour_geodesic_matched_permuted"] = _wrap(
    behaviour_geodesic_matched, curvature=1.0, permuted=True)
