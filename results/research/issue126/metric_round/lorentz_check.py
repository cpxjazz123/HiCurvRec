"""The Lorentz model, on the same points, at the same curvatures.

The Poincare ball and the Lorentz hyperboloid are two charts of one space, so for
a numerically sound implementation the distances must agree and the four-point
agreement must agree. Three questions are answered here:

1. Do the two charts return the same distance?
2. Do they make the same four-point pairing choice?
3. Does the Lorentz chart keep changing as the curvature grows, where the
   Poincare chart's ``tanh`` saturated?

The Lorentz distance is not computed by converting to the hyperboloid and taking
``arccosh(-c <y,y'>)``: for two points on the hyperboloid that is a difference of
two huge products, and it loses every digit exactly when the points are close.
With ``r = sqrt(c) ||t||`` and the unit direction ``u``, the same quantity is

    cosh(sqrt(c) d) = cosh(r1 - r2) + sinh(r1) sinh(r2) (1 - cos(u1, u2))

which adds where the other subtracts, and ``1 - cos(u1,u2) = 0.5 ||u1 - u2||^2``
avoids the cancellation inside the angle. ``arccosh(1 + x)`` is evaluated as
``log1p(x + sqrt(x^2 + 2x))`` so that nearby points keep their precision.

Understanding the result: the two charts must agree wherever the Poincare chart
has not saturated. Where they disagree, the Lorentz number is the trustworthy
one -- unless the two agree and the metric still does not move, in which case the
distance relation itself has stabilised and the curvature has stopped mattering.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from tools.research.science import sid_metric_bench as bench  # noqa: E402

MODULE = Path(__file__).resolve().parent / "metrics_four_point_svd.py"
GRID = (0.001, 0.05, 1.0, 4.0, 16.0, 64.0, 256.0, 4096.0, 65536.0)
OUT = Path(__file__).resolve().parents[1] / "lorentz_check_107.json"


def load_module():
    spec = importlib.util.spec_from_file_location("four_point", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ the charts
def lorentz_parts(tangent: np.ndarray, curvature: float):
    """``(r, u)``: the Lorentz radius and the unit direction of every point.

    The factor two is not a convention: the ball point is
    ``rho = tanh(sqrt(c)||t||)/sqrt(c)``, and the hyperbolic distance from the
    origin to a point at Euclidean ``rho`` is ``(2/sqrt(c)) artanh(sqrt(c) rho)``.
    With ``sqrt(c) rho = tanh(sqrt(c)||t||)`` that is
    ``r = 2 sqrt(c) ||t||`` -- dropping the two makes every Lorentz distance the
    distance of a point at half the radius, which is exactly the factor 2.35 the
    first comparison showed.
    """
    root = np.sqrt(float(curvature))
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    safe = np.maximum(norm, 1e-300)
    return 2.0 * root * norm[..., 0], tangent / safe


def arccosh_stable(value: np.ndarray) -> np.ndarray:
    """``arccosh(1 + x)`` without the cancellation at small ``x``."""
    x = np.maximum(value - 1.0, 0.0)
    return np.log1p(x + np.sqrt(x * x + 2.0 * x))


def _log_cosh(value: np.ndarray) -> np.ndarray:
    magnitude = np.abs(value)
    return magnitude + np.log1p(np.exp(-2.0 * magnitude)) - np.log(2.0)


def _log_sinh(radius: np.ndarray) -> np.ndarray:
    """``log(sinh(r))`` for ``r > 0``, without forming ``sinh``."""
    clipped = np.maximum(radius, 1e-12)
    return clipped + np.log1p(-np.exp(-2.0 * clipped)) - np.log(2.0)


def lorentz_distance(left_r, left_u, right_r, right_u, curvature: float) -> np.ndarray:
    """The Lorentz distance, evaluated in the LOG domain.

    ``cosh(sqrt(c) d) = cosh(r1-r2) + sinh(r1) sinh(r2) (1-cos)`` is the stable
    form, but ``sinh(r1) sinh(r2)`` overflows once ``r`` passes about 350, which
    at this tangent scale is c of order 10^4 -- inside the range this experiment
    has to probe. In logs the two terms are added with ``logaddexp`` and nothing
    overflows; ``arccosh(x) = log(2x)`` to better than 1e-13 once ``x > e^30``.
    """
    root = np.sqrt(float(curvature))
    gap = 0.5 * np.sum((left_u - right_u) ** 2, axis=-1)
    log_first = _log_cosh(left_r - right_r)
    log_second = _log_sinh(left_r) + _log_sinh(right_r) + np.log(np.maximum(gap, 1e-300))
    log_argument = np.logaddexp(log_first, log_second)
    small = log_argument <= 30.0
    distance = np.empty_like(log_argument)
    distance[small] = np.arccosh(np.maximum(np.exp(log_argument[small]), 1.0))
    distance[~small] = log_argument[~small] + np.log(2.0)
    return distance / root


def poincare_distance(points: np.ndarray, curvature: float) -> np.ndarray:
    c = float(curvature)
    difference = np.sum((points[0] - points[1]) ** 2, axis=-1)
    denominator = ((1 - c * np.sum(points[0] * points[0], axis=-1))
                   * (1 - c * np.sum(points[1] * points[1], axis=-1)))
    return np.arccosh(np.maximum(1 + 2 * c * difference
                                 / np.maximum(denominator, 1e-12), 1 + 1e-12)) / np.sqrt(c)


# ------------------------------------------------------------------ the score
def score(table: dict, context: dict, distance, cache: dict, key: str) -> float:
    """The four-point agreement, with the distance law supplied by the caller."""
    embedded = context["lorentz_check_embeddings"].get(key)
    if embedded is None:
        embedded = load_module()._svd_embeddings(context, table)
        context["lorentz_check_embeddings"][key] = embedded
    picks = context["lorentz_check_quadruples"].get(key)
    if picks is None:
        picks = load_module()._graph_local_quadruples(context, table, 20000, 42)
        context["lorentz_check_quadruples"][key] = picks
    a, b, c, d = picks[:, 0], picks[:, 1], picks[:, 2], picks[:, 3]
    sums = np.stack([distance(a, b) + distance(c, d),
                     distance(a, c) + distance(b, d),
                     distance(a, d) + distance(b, c)], axis=0)
    # the SAME delta the metric definition uses, so the two charts are
    # comparable with the numbers already reported
    delta = np.abs(load_module().normalised_delta(sums))
    median = float(np.median(delta)) if cache.get("median") is None else cache["median"]
    cache["median"] = median
    weight = 1.0 / (1.0 + delta / max(median, 1e-12))

    rows = np.asarray(table["rows"], dtype=np.int64)

    def shared(left, right):
        equal = rows[left] == rows[right]
        return np.where(equal.all(axis=1), equal.shape[1], equal.argmin(axis=1)).astype(np.float64)

    width = rows.shape[1]
    tree_sums = np.stack([
        (width - shared(a, b)) + (width - shared(c, d)),
        (width - shared(a, c)) + (width - shared(b, d)),
        (width - shared(a, d)) + (width - shared(b, c))], axis=0)
    geometry_choice = np.argmin(sums, axis=0)
    tree_choice = np.argmin(tree_sums, axis=0)
    agreement = (geometry_choice == tree_choice).astype(np.float64)
    spread = tree_sums.max(axis=0) - tree_sums.min(axis=0)
    informative = spread > 0
    if not informative.any():
        return 0.0, np.zeros(len(delta), dtype=np.int64)
    weight = weight * informative
    if weight.sum() <= 0:
        return 0.0, geometry_choice
    return float((weight * agreement).sum() / weight.sum()), geometry_choice


def main() -> int:
    module = load_module()
    context = bench.load_context()
    context["lorentz_check_embeddings"] = {}
    context["lorentz_check_quadruples"] = {}
    recall = np.array([entry["test_recall@10"] for entry in context["pairs"]])
    report = {"pairs": len(context["pairs"]), "grid": []}

    for curvature in GRID:
        poincare_values, lorentz_values, agreements, distance_errors = [], [], [], []
        for entry in context["pairs"]:
            table = entry["table"]
            key = str(id(table))
            tangent = module._svd_embeddings(context, table)
            ball = module.ball_of(tangent, curvature)
            radius, direction = lorentz_parts(tangent, curvature)

            def to_ball(index):
                return ball[index]

            def to_lorentz(index):
                return radius[index], direction[index]

            def poincare(index_left, index_right):
                return poincare_distance(
                    (ball[index_left], ball[index_right]), curvature)

            def lorentz(index_left, index_right):
                return lorentz_distance(radius[index_left], direction[index_left],
                                        radius[index_right], direction[index_right],
                                        curvature)

            poincare_score, poincare_choice = score(
                table, context, poincare, {"median": None}, key + "_poincare")
            lorentz_score, lorentz_choice = score(
                table, context, lorentz, {"median": None}, key + "_lorentz")
            poincare_values.append(poincare_score)
            lorentz_values.append(lorentz_score)
            agreements.append(float(np.mean(poincare_choice == lorentz_choice)))

            sample = module._graph_local_quadruples(context, table, 500, 7)
            left, right = sample[:, 0], np.roll(sample[:, 1], 1)
            a = poincare(left, right)
            b = lorentz(left, right)
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                relative = np.abs(a[finite] - b[finite]) / np.maximum(np.abs(b[finite]), 1e-12)
                distance_errors.append(float(np.max(relative)))

        rho_p, p_p = stats.spearmanr(poincare_values, recall)
        rho_l, p_l = stats.spearmanr(lorentz_values, recall)
        report["grid"].append({
            "curvature": curvature,
            "poincare": {"spearman": round(float(rho_p), 6), "p_value": float(p_p),
                         "range": [float(np.min(poincare_values)), float(np.max(poincare_values))]},
            "lorentz": {"spearman": round(float(rho_l), 6), "p_value": float(p_l),
                        "range": [float(np.min(lorentz_values)), float(np.max(lorentz_values))]},
            "pairing_agreement": float(np.mean(agreements)),
            "max_relative_distance_error": float(np.max(distance_errors)),
        })
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("%-10s %10s %10s %10s %12s" % ("curvature", "poincare", "lorentz", "pairing", "max rel err"))
    for entry in report["grid"]:
        print("%-10g %10.4f %10.4f %10.3f %12.2e" % (
            entry["curvature"], entry["poincare"]["spearman"],
            entry["lorentz"]["spearman"], entry["pairing_agreement"],
            entry["max_relative_distance_error"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
