"""HBAA with the Lorentz distance: is the leader in the saturated regime too?

HBAA scores the Gromov contrast between a behaviour positive's prefix centroid and
its negatives', with the frozen reference as the ball. The reference sits at radius
0.91 of the curvature-1 ball -- close enough to the boundary that the Poincare
distance there is strongly amplified, and the four-point family lost 0.12 to that
amplification (protocol v8). So the leader has to be re-measured with the same
points, the same centroids and the same history barycentres, and the ONLY thing
changed is the distance law: the Lorentz form

    cosh(sqrt(c) d) = cosh(r1-r2) + sinh(r1) sinh(r2) (1 - cos(u1,u2)),
    r = 2 artanh(sqrt(c) ||x||),

evaluated in the log domain, against the Poincare ``arccosh(1 + ...)``.

If the leader's number survives the swap, its hyperbolicity claim is at least not
an artefact of the chart. If it moves, the same retraction that hit the four-point
family applies to it.
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

MODULE = Path(__file__).resolve().parent / "metrics_hbaa.py"
OUT = Path(__file__).resolve().parents[1] / "lorentz_hbaa_107.json"


def load_module():
    spec = importlib.util.spec_from_file_location("hbaa_module", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lorentz_radius_and_direction(points: np.ndarray, curvature: float):
    norm = np.linalg.norm(points, axis=-1, keepdims=True)
    safe = np.maximum(norm, 1e-300)
    argument = np.clip(np.sqrt(float(curvature)) * norm[..., 0], 0.0, 1.0 - 1e-15)
    return 2.0 * np.arctanh(argument), points / safe


def _log_cosh(value):
    magnitude = np.abs(value)
    return magnitude + np.log1p(np.exp(-2.0 * magnitude)) - np.log(2.0)


def _log_sinh(radius):
    clipped = np.maximum(radius, 1e-12)
    return clipped + np.log1p(-np.exp(-2.0 * clipped)) - np.log(2.0)


def lorentz_distance(left, right, curvature: float):
    """``left`` and ``right`` are ``(radius, direction)`` pairs."""
    root = np.sqrt(float(curvature))
    gap = 0.5 * np.sum((left[1] - right[1]) ** 2, axis=-1)
    log_argument = np.logaddexp(
        _log_cosh(left[0] - right[0]),
        _log_sinh(left[0]) + _log_sinh(right[0]) + np.log(np.maximum(gap, 1e-300)))
    small = log_argument <= 30.0
    distance = np.empty_like(log_argument)
    distance[small] = np.arccosh(np.maximum(np.exp(log_argument[small]), 1.0))
    distance[~small] = log_argument[~small] + np.log(2.0)
    return distance / root


def gromov_alignment(table: dict, context: dict, *, curvature: float, lorentz: bool) -> float:
    module = load_module()
    target = np.asarray(context["target"], dtype=np.float64)[np.asarray(table["item_ids"])]
    tau = float(context["temperature"])
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    position = {int(item): index for index, item in enumerate(table["item_ids"].tolist())}
    queries = module._queries(context)

    pos = np.array([position.get(int(item), -1) for item in queries["positives"]], dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                    for row in queries["negatives"]], dtype=np.int64)
    valid = (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return 0.0

    history = module._history_points(context, curvature)[valid]
    points = module._scale(target, curvature)
    centres = {int(code): module._frechet(target[codes == code], curvature)
               for code in np.unique(codes)}

    if lorentz:
        history_ru = lorentz_radius_and_direction(history, curvature)
        points_ru = lorentz_radius_and_direction(points, curvature)
        centre_ru = {code: lorentz_radius_and_direction(value[None, :], curvature)
                     for code, value in centres.items()}

        def norm(value):
            return value[0]

        def distance(left, right):
            return lorentz_distance(left, right, curvature)

        mu_pos = (np.concatenate([centre_ru[int(codes[index])][0] for index in pos[valid]]),
                  np.vstack([centre_ru[int(codes[index])][1][0] for index in pos[valid]]))
        mu_neg = (np.stack([[centre_ru[int(codes[index])][0][0] for index in row]
                            for row in neg[valid]]),
                  np.stack([[centre_ru[int(codes[index])][1][0] for index in row]
                            for row in neg[valid]]))
        history_ru = (history_ru[0], history_ru[1])
    else:
        origin = np.zeros(points.shape[1])

        def norm(value):
            return module._norm(value, curvature)

        def distance(left, right):
            return module._distance(left, right, curvature)

        mu_pos = np.stack([centres[int(codes[index])] for index in pos[valid]])
        mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg[valid]])
        history_ru = history

    def gromov(x, y):
        return 0.5 * (np.asarray(norm(x)) + np.asarray(norm(y)) - distance(x, y))

    if lorentz:
        positive = gromov(history_ru, mu_pos)[:, None]
        negative = gromov((history_ru[0][:, None], history_ru[1][:, None, :]), mu_neg)
    else:
        positive = gromov(history_ru, mu_pos)[:, None]
        negative = gromov(history_ru[:, None, :], mu_neg)
    delta = positive - negative
    per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
    return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))


def main() -> int:
    context = bench.load_context()
    recall = np.array([entry["test_recall@10"] for entry in context["pairs"]])
    report = {"pairs": len(context["pairs"]), "curvatures": []}
    for curvature in (1e-3, 0.25, 1.0):
        poincare, lorentz = [], []
        for entry in context["pairs"]:
            poincare.append(gromov_alignment(entry["table"], context,
                                             curvature=curvature, lorentz=False))
            lorentz.append(gromov_alignment(entry["table"], context,
                                            curvature=curvature, lorentz=True))
        rho_p, p_p = stats.spearmanr(poincare, recall)
        rho_l, p_l = stats.spearmanr(lorentz, recall)
        report["curvatures"].append({
            "curvature": curvature,
            "poincare": {"spearman": round(float(rho_p), 6), "p_value": float(p_p)},
            "lorentz": {"spearman": round(float(rho_l), 6), "p_value": float(p_l)},
            "max_abs_value_difference": float(np.max(np.abs(np.array(poincare)
                                                            - np.array(lorentz)))),
        })
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("%-10s %10s %10s %14s" % ("curvature", "poincare", "lorentz", "max |diff|"))
    for entry in report["curvatures"]:
        print("%-10g %10.4f %10.4f %14.4f" % (entry["curvature"],
                                              entry["poincare"]["spearman"],
                                              entry["lorentz"]["spearman"],
                                              entry["max_abs_value_difference"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
