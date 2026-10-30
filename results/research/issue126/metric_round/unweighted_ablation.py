"""Weighted or not: which part of the four-point metric predicts Recall?

The metric was a weighted mean of the pairing agreement, the weight coming from
the Gromov delta, ``w = 1/(1 + delta/median)``. Removing it makes every
informative quadruple count once, ``Q = agreements / informative``, and that is
the ablation this script measures:

| law | weighted | unweighted |
| --- | --- | --- |
| Poincare | reproduces the reported number | the plain agreement rate |
| Lorentz | the numerically sound weighted form | **must equal the Poincare one**: the pairing agreement between the charts was 100% from c = 16 on, so the unweighted scores cannot differ |
| Euclidean | -- | the geometry control |

plus the permuted control, where the SID rows are dealt at random to the items.

Two questions: does the geometry still predict Recall without the weight, and does
the hyperbolic geometry beat the Euclidean one at all?
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import zlib

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from tools.research.science import sid_metric_bench as bench  # noqa: E402

MODULE = Path(__file__).resolve().parent / "metrics_four_point_svd.py"
LORENTZ = Path(__file__).resolve().parent / "lorentz_check.py"
OUT = Path(__file__).resolve().parents[1] / "unweighted_ablation_107.json"
CURVATURES = (16.0, 1.0)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shared(rows: np.ndarray, left, right) -> np.ndarray:
    equal = rows[left] == rows[right]
    return np.where(equal.all(axis=1), equal.shape[1], equal.argmin(axis=1)).astype(np.float64)


def score(table: dict, context: dict, distance, weighted: bool) -> float:
    """The agreement rate over the informative quadruples, optionally weighted."""
    module = _load("four_point", MODULE)
    lorentz = _load("lorentz_check", LORENTZ)
    key = str(id(table))
    tangent = context["unweighted_embeddings"].get(key)
    if tangent is None:
        tangent = module._svd_embeddings(context, table)
        context["unweighted_embeddings"][key] = tangent
    picks = context["unweighted_quadruples"].get(key)
    if picks is None:
        picks = module._graph_local_quadruples(context, table, 20000, 42)
        context["unweighted_quadruples"][key] = picks
    a, b, c, d = picks[:, 0], picks[:, 1], picks[:, 2], picks[:, 3]

    sums = np.stack([distance(a, b) + distance(c, d),
                     distance(a, c) + distance(b, d),
                     distance(a, d) + distance(b, c)], axis=0)
    rows = np.asarray(table["rows"], dtype=np.int64)
    width = rows.shape[1]
    tree_sums = np.stack([
        (width - _shared(rows, a, b)) + (width - _shared(rows, c, d)),
        (width - _shared(rows, a, c)) + (width - _shared(rows, b, d)),
        (width - _shared(rows, a, d)) + (width - _shared(rows, b, c))], axis=0)
    informative = (tree_sums.max(axis=0) - tree_sums.min(axis=0)) > 0
    if not informative.any():
        return 0.0
    agreement = (np.argmin(sums, axis=0) == np.argmin(tree_sums, axis=0)).astype(np.float64)
    if not weighted:
        return float(agreement[informative].mean())
    delta = np.abs(module.normalised_delta(sums))
    median = float(np.median(delta))
    weight = 1.0 / (1.0 + delta / max(median, 1e-12))
    weight = weight * informative
    if weight.sum() <= 0:
        return 0.0
    return float((weight * agreement).sum() / weight.sum())


def laws(context: dict, table: dict, curvature: float):
    module = _load("four_point", MODULE)
    lorentz = _load("lorentz_check", LORENTZ)
    tangent = module._svd_embeddings(context, table)
    ball = module.ball_of(tangent, curvature)
    radius, direction = lorentz.lorentz_parts(tangent, curvature)

    poincare = lambda i, j: lorentz.poincare_distance((ball[i], ball[j]), curvature)
    hyperboloid = lambda i, j: lorentz.lorentz_distance(
        radius[i], direction[i], radius[j], direction[j], curvature)
    flat = lambda i, j: np.linalg.norm(tangent[i] - tangent[j], axis=-1)
    return poincare, hyperboloid, flat


def main() -> int:
    context = bench.load_context()
    context["unweighted_embeddings"] = {}
    context["unweighted_quadruples"] = {}
    recall = np.array([entry["test_recall@10"] for entry in context["pairs"]])
    report = {"pairs": len(context["pairs"]), "grid": []}
    for curvature in CURVATURES:
        values = {name: [] for name in ("poincare_weighted", "lorentz_weighted",
                                        "poincare_unweighted", "lorentz_unweighted",
                                        "euclidean_unweighted", "lorentz_unweighted_permuted")}
        for entry in context["pairs"]:
            table = entry["table"]
            poincare, hyperboloid, flat = laws(context, table, curvature)
            values["poincare_weighted"].append(score(table, context, poincare, True))
            values["lorentz_weighted"].append(score(table, context, hyperboloid, True))
            values["poincare_unweighted"].append(score(table, context, poincare, False))
            values["lorentz_unweighted"].append(score(table, context, hyperboloid, False))
            values["euclidean_unweighted"].append(score(table, context, flat, False))
            rows = np.asarray(table["rows"], dtype=np.int64)
            rng = np.random.default_rng(zlib.crc32(rows.tobytes()) & 0x7FFFFFFF)
            shuffled = {"rows": rows[rng.permutation(rows.shape[0])],
                        "item_ids": table["item_ids"]}
            values["lorentz_unweighted_permuted"].append(
                score(shuffled, context, hyperboloid, False))
        entry = {"curvature": curvature, "variants": {}}
        for name, series in values.items():
            array = np.asarray(series, dtype=float)
            rho, p = stats.spearmanr(array, recall)
            entry["variants"][name] = {"spearman": round(float(rho), 6),
                                       "p_value": float(p)}
        entry["poincare_lorentz_unweighted_max_diff"] = float(np.max(np.abs(
            np.asarray(values["poincare_unweighted"]) - np.asarray(values["lorentz_unweighted"]))))
        report["grid"].append(entry)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for entry in report["grid"]:
        print("curvature", entry["curvature"], "| chart agreement of the unweighted score",
              "%.3e" % entry["poincare_lorentz_unweighted_max_diff"])
        for name, value in entry["variants"].items():
            print("   %-32s %+.4f" % (name, value["spearman"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
