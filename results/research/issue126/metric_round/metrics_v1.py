"""The three metrics the ideation agents proposed, implemented to their formulas.

Each function takes one exported SID table and the frozen Issue #126 reference and
returns a single float; nothing here reads recall, a Stage-3 output, or any
validation/test interaction. The formulas follow the proposals' text, with the
derivations they state:

* ``hyp_voronoi_margin_l1``  -- the mean normalised geodesic margin of the level-1
  Voronoi cells, centres being the Klein/Einstein hyperbolic barycentres.
* ``hyp_adj_r2_l1``          -- the adjusted dispersion R^2 of the level-1
  partition under the hyperbolic Frobenius variance decomposition, both towers.
* ``hsmre``                  -- the size-matched geodesic radius enrichment, the
  ratio of a bucket's geodesic radius to the radius of the same number of
  geodesic nearest neighbours, transformed by sech(ln rho) and masked to buckets
  of a size that can make a claim.

All three use curvature 1 and derive their scale constants from the reference or
the training split, as the proposals state; no constant is fitted to recall.
"""
from __future__ import annotations

import numpy as np

CURVATURE = 1.0


# --------------------------------------------------------------------------- geometry
def _poincare_from_tangent(tangent: np.ndarray, scale: float) -> np.ndarray:
    """``exp_0(v / scale) = tanh(|v/scale|) * unit`` in the unit ball."""
    v = np.asarray(tangent, dtype=np.float64) / float(scale)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    safe = np.maximum(norm, 1e-12)
    return np.tanh(norm) * v / safe


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Poincare distance of curvature 1 between broadcastable arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = (1.0 - np.sum(a * a, axis=-1)) * (1.0 - np.sum(b * b, axis=-1))
    denominator = np.maximum(denominator, 1e-12)
    argument = 1.0 + 2.0 * difference / denominator
    return np.arccosh(np.maximum(argument, 1.0 + 1e-12))


def _klein(points: np.ndarray) -> np.ndarray:
    norm2 = np.sum(points * points, axis=-1, keepdims=True)
    return 2.0 * points / (1.0 + norm2)


def _einstein_centre(points: np.ndarray) -> np.ndarray:
    """The Klein-model Einstein midpoint, mapped back to the Poincare ball."""
    if points.shape[0] == 1:
        return points[0].copy()
    klein = _klein(points)
    norm2 = np.sum(klein * klein, axis=-1, keepdims=True)
    weight = 1.0 / np.sqrt(np.maximum(1.0 - norm2, 1e-12))
    bar = (weight * klein).sum(axis=0) / weight.sum()
    bar_norm = float(np.linalg.norm(bar))
    if bar_norm >= 1.0:
        bar = bar / bar_norm * (1.0 - 1e-6)
        bar_norm = float(np.linalg.norm(bar))
    return bar / (1.0 + np.sqrt(max(0.0, 1.0 - bar_norm ** 2)))


def _barycentre(points: np.ndarray, steps: int = 20) -> np.ndarray:
    """Karcher flow started at the Euclidean mean; used for the geodesic null."""
    if points.shape[0] == 1:
        return points[0].copy()
    centre = _poincare_from_tangent(points.mean(axis=0), 1.0)
    for _ in range(int(steps)):
        # log_p(y) in the ball, then a unit step of size 1/2 towards the mean log
        dots = np.sum(centre * points, axis=-1, keepdims=True)
        centre_norm2 = float(np.sum(centre * centre))
        difference = points - centre
        numerator = difference * (1.0 - centre_norm2)
        denominator = 1.0 - 2.0 * dots + centre_norm2
        tangent = 2.0 * numerator / np.maximum(denominator, 1e-12)
        step = tangent.mean(axis=0)
        norm = float(np.linalg.norm(step))
        if norm < 1e-10:
            break
        moved = _poincare_from_tangent(step * 0.5, 1.0)
        # exp_centre(moved) in the ball, via the Mobius addition of centre and moved
        c2 = centre_norm2
        d2 = float(np.sum(moved * moved))
        cd = float(np.sum(centre * moved))
        add = ((1.0 + 2.0 * cd + d2) * centre + (1.0 - c2) * moved) / max(
            1e-12, 1.0 + 2.0 * cd + c2 * d2)
        shift = float(_distance(centre, add))
        centre = add
        if shift < 1e-9:
            break
    return centre


def _codes(table: dict, level: int) -> np.ndarray:
    return np.asarray(table["rows"], dtype=np.int64)[:, level]


# --------------------------------------------------------------------------- metrics
def hyp_voronoi_margin_l1(table: dict, context: dict) -> float:
    """Mean normalised geodesic margin of the level-1 cells (proposal of agent 3)."""
    source = np.asarray(context["source"], dtype=np.float64)
    scale = float(np.median(np.linalg.norm(source, axis=1)))
    points = _poincare_from_tangent(source, scale)
    codes = _codes(table, 0)
    centres = {}
    for code in np.unique(codes):
        members = points[codes == code]
        centres[int(code)] = _einstein_centre(members) if members.shape[0] > 1 else members[0]
    keys = list(centres)
    if len(keys) < 2:
        return -1.0
    centre_table = np.stack([centres[key] for key in keys])
    position = {key: index for index, key in enumerate(keys)}
    own = np.array([position[int(code)] for code in codes])
    distances = _distance(points[:, None, :], centre_table[None, :, :])
    own_distance = distances[np.arange(len(codes)), own]
    rival = distances.copy()
    rival[np.arange(len(codes)), own] = np.inf
    rival_distance = rival.min(axis=1)
    sizes = np.array([np.sum(codes == key) for key in keys])[own]
    valid = (sizes >= 2) & (rival_distance + own_distance > 0)
    margin = np.where(valid,
                      (rival_distance - own_distance) /
                      np.maximum(rival_distance + own_distance, 1e-12), 0.0)
    return float(margin.mean())


def hyp_adj_r2_l1(table: dict, context: dict) -> float:
    """Adjusted dispersion R^2 of the level-1 partition, both towers (agent 5)."""
    codes = _codes(table, 0)
    values = []
    items = np.asarray(table["item_ids"])
    for tower in ("source", "target"):
        raw = np.asarray(context[tower], dtype=np.float64)[items]
        mean = raw.mean(axis=0, keepdims=True)
        sigma = float(np.sqrt(((raw - mean) ** 2).sum(axis=1).mean()))
        points = _poincare_from_tangent(raw - mean, max(sigma, 1e-12))
        global_centre = _barycentre(points)
        total = float(np.mean(_distance(points, global_centre) ** 2))
        within = 0.0
        keys = np.unique(codes)
        for code in keys:
            members = points[codes == code]
            centre = _einstein_centre(members) if members.shape[0] > 1 else members[0]
            within += float(np.sum(_distance(members, centre) ** 2))
        within /= points.shape[0]
        n = points.shape[0]
        k = len(keys)
        if total <= 0 or n <= k + 1:
            values.append(0.0)
            continue
        values.append(float(np.clip(1.0 - (within / total) * (n - 1) / (n - k), -1.0, 1.0)))
    return float(np.mean(values))


def hsmre(table: dict, context: dict) -> float:
    """Size-matched geodesic radius enrichment, mean over levels (agent 2)."""
    target = np.asarray(context["target"], dtype=np.float64)
    items = np.asarray(table["item_ids"])
    sample = target[items]
    scale = 1.0 / float(context["temperature"])
    points = _poincare_from_tangent(sample, scale)
    n_items = points.shape[0]
    rows = np.asarray(table["rows"], dtype=np.int64)
    levels = rows.shape[1]

    # every level's bucket, and the largest bucket a level can have
    buckets = []
    largest = 1
    for level in range(levels):
        codes = rows[:, level]
        groups = {}
        for position, code in enumerate(codes.tolist()):
            groups.setdefault(int(code), []).append(position)
        buckets.append(groups)
        largest = max(largest, max(len(members) for members in groups.values()))

    # the geodesic nearest-neighbour null: the `largest` nearest points of every item
    wanted = min(largest, n_items - 1)
    neighbours = np.empty((n_items, wanted), dtype=np.int32)
    chunk = 2048
    for start in range(0, n_items, chunk):
        stop = min(start + chunk, n_items)
        block = points[start:stop]
        difference = block[:, None, :] - points[None, :, :]
        numerator = np.sum(difference ** 2, axis=-1)
        denominator = ((1.0 - np.sum(block ** 2, axis=1))[:, None]
                       * (1.0 - np.sum(points ** 2, axis=1))[None, :])
        distances = np.arccosh(np.maximum(1.0 + 2.0 * numerator /
                                          np.maximum(denominator, 1e-12), 1.0 + 1e-12))
        order = np.argpartition(distances, wanted, axis=1)[:, :wanted]
        neighbours[start:stop] = order.astype(np.int32)

    scores = []
    for level, groups in enumerate(buckets):
        numerators = 0.0
        denominators = 0.0
        for code, members in groups.items():
            size = len(members)
            if size < 2 or size > n_items / 2:
                continue
            members = np.asarray(members, dtype=np.int64)
            centre = _einstein_centre(points[members])
            radius = _distance(points[members], centre)
            # The null set of every item in this size class, batched: one row per
            # item, its own position first and then its nearest neighbours. The
            # Karcher flow is the same flow, evaluated for all of them at once,
            # because a Python loop per item is what made the metric cost 156 s.
            # Chunked: the largest level-1 buckets hold a thousand items, and one
            # un-chunked null-set tensor per worker took 12 GB (the run was killed
            # by the scheduler's memory limit).
            chunk = 256
            null_radius = np.empty(members.shape[0], dtype=np.float64)
            for start in range(0, members.shape[0], chunk):
                stop = min(start + chunk, members.shape[0])
                block = members[start:stop]
                null_sets = np.concatenate(
                    (block[:, None], neighbours[block, :size - 1]), axis=1)
                null_centres = _barycentre_batch(points[null_sets])
                null_radius[start:stop] = _distance(points[block], null_centres)
            ratio = radius / np.maximum(null_radius, 1e-12)
            transformed = 2.0 * ratio / (1.0 + ratio ** 2)
            weight = float(size - 1)
            numerators += weight * float(transformed.sum())
            denominators += weight * len(members)
        scores.append(numerators / denominators if denominators > 0 else 0.0)
    return float(np.mean(scores))


def _barycentre_batch(sets: np.ndarray, steps: int = 20) -> np.ndarray:
    """The Karcher flow for many point sets at once (same rule as ``_barycentre``)."""
    sets = np.asarray(sets, dtype=np.float64)
    if sets.shape[1] == 1:
        return sets[:, 0, :].copy()
    centre = _poincare_from_tangent(sets.mean(axis=1), 1.0)
    for _ in range(int(steps)):
        dots = np.sum(centre[:, None, :] * sets, axis=-1, keepdims=True)
        centre_norm2 = np.sum(centre * centre, axis=-1, keepdims=True)[:, None, :]
        numerator = (sets - centre[:, None, :]) * (1.0 - centre_norm2)
        denominator = 1.0 - 2.0 * dots + centre_norm2
        tangent = 2.0 * numerator / np.maximum(denominator, 1e-12)
        step = tangent.mean(axis=1)
        norm = np.linalg.norm(step, axis=-1, keepdims=True)
        if float(np.max(norm)) < 1e-10:
            break
        moved = _poincare_from_tangent(step * 0.5, 1.0)
        c2 = centre_norm2[:, 0, :]
        d2 = np.sum(moved * moved, axis=-1, keepdims=True)
        cd = np.sum(centre * moved, axis=-1, keepdims=True)
        add = ((1.0 + 2.0 * cd + d2) * centre + (1.0 - c2) * moved) / np.maximum(
            1e-12, 1.0 + 2.0 * cd + c2 * d2)
        shift = _distance(centre, add)
        centre = add
        if float(np.max(shift)) < 1e-9:
            break
    return centre


hyp_voronoi_margin_l1.note = "mean geodesic margin of the level-1 Voronoi cells"
hyp_adj_r2_l1.note = "adjusted hyperbolic dispersion R^2 of the level-1 partition"
hsmre.note = "size-matched geodesic radius enrichment, mean over levels"

METRICS = {
    "hyp_voronoi_margin_l1": hyp_voronoi_margin_l1,
    "hyp_adj_r2_l1": hyp_adj_r2_l1,
    "hsmre": hsmre,
}

#: how many derived constants each metric needs, for the tie-break rule
FREE_CONSTANTS = {"hyp_voronoi_margin_l1": 1, "hyp_adj_r2_l1": 2, "hsmre": 2}
