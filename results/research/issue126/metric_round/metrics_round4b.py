"""Round-4 metric: the normalised geodesic covering radius of the level-1 cells.

A worst-case statistic instead of a mean: the typical item's distance to the FARTHEST
member of its own level-1 cell, relative to the covering radius of the whole cloud.
A single straggler inside a cell is invisible to a mean or a variance-explained
score, but it is exactly the item a top-10 cut can never retrieve.
"""
from __future__ import annotations

import numpy as np

CURVATURE = 1.0


def _exp0(tangent: np.ndarray, scale: float) -> np.ndarray:
    v = np.asarray(tangent, dtype=np.float64) / float(scale)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.tanh(norm) * v / np.maximum(norm, 1e-12)


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = (1.0 - np.sum(a * a, axis=-1)) * (1.0 - np.sum(b * b, axis=-1))
    return np.arccosh(np.maximum(1.0 + 2.0 * difference /
                                 np.maximum(denominator, 1e-12), 1.0 + 1e-12))


def _karcher(points: np.ndarray, steps: int = 10) -> np.ndarray:
    """Riemannian gradient descent in the ball (the proposal's own recursion)."""
    if points.shape[0] == 1:
        return points[0].copy()
    centre = points.mean(axis=0)
    norm = float(np.linalg.norm(centre))
    if norm >= 0.99:
        centre = centre / norm * 0.99
    for _ in range(int(steps)):
        dots = np.sum(centre * points, axis=-1, keepdims=True)
        centre_norm2 = float(np.sum(centre * centre))
        numerator = (points - centre) * (1.0 - centre_norm2)
        denominator = 1.0 - 2.0 * dots + centre_norm2
        tangent = 2.0 * numerator / np.maximum(denominator, 1e-12)
        step = tangent.mean(axis=0) * 0.5
        moved_norm = float(np.linalg.norm(step))
        if moved_norm < 1e-10:
            break
        increment = np.tanh(moved_norm) * step / moved_norm
        c2 = centre_norm2
        d2 = float(np.sum(increment * increment))
        cd = float(np.sum(centre * increment))
        centre = ((1.0 + 2.0 * cd + d2) * centre + (1.0 - c2) * increment) / max(
            1e-12, 1.0 + 2.0 * cd + c2 * d2)
    return centre


def hyp_frechet_cover_radius_l1(table: dict, context: dict) -> float:
    target = np.asarray(context["target"], dtype=np.float64)[
        np.asarray(table["item_ids"])]
    tau = float(context["temperature"])
    points = _exp0(target, 2.0 * tau)
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    n_items = points.shape[0]
    cloud_centre = _karcher(points)
    cloud_radius = float(np.max(_distance(points, cloud_centre)))
    if cloud_radius <= 0:
        return 0.0
    total = 0.0
    for code in np.unique(codes):
        members = points[codes == code]
        if members.shape[0] == 1:
            contribution = 0.0
        else:
            centre = _karcher(members)
            contribution = float(np.max(_distance(members, centre))) / cloud_radius
        total += (members.shape[0] / n_items) * contribution
    return float(1.0 - total)


hyp_frechet_cover_radius_l1.note = ("one minus the item-weighted normalised geodesic "
                                    "covering radius of the level-1 cells")

METRICS = {"hyp_frechet_cover_radius_l1": hyp_frechet_cover_radius_l1}
FREE_CONSTANTS = {"hyp_frechet_cover_radius_l1": 0}
