"""The curvature sweep, with the checks that decide whether it means anything.

For every curvature in an extended grid this records

* the metric value of every one of the 107 confirmed SIDs,
* the Spearman correlation against Recall@10,
* the largest per-SID change against the previous grid point,
* the share of items that the ``exp_c`` map has pushed onto the boundary.

The last two are the point. A plateau in the correlation means nothing if the
values themselves do not move, and the values do not move once ``tanh(sqrt(c)||t||)``
saturates: the map sends every item to the ball's boundary and the geometry
becomes the same configuration up to floating point.

A second block holds the gauge test: scaling the tangents by ``1/sqrt(c)`` and
using curvature ``c`` is the same ball configuration as curvature 1 with the
unscaled tangents, so the metric must return the same numbers. If it does, the
curvature is a reparameterisation of the radial scale and is not identifiable
from a fixed point set.
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
GRID = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
OUT = Path(__file__).resolve().parents[1] / "curve_sweep_107.json"


def load_module():
    spec = importlib.util.spec_from_file_location("four_point", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boundary_share(module, context, table, curvature):
    embeddings = module._svd_embeddings(context, table)
    radii = np.linalg.norm(module.ball_of(embeddings, curvature), axis=1)
    edge = 0.999 / np.sqrt(float(curvature))
    return float(np.mean(radii > edge))


def main() -> int:
    module = load_module()
    context = bench.load_context()
    recall = np.array([entry["test_recall@10"] for entry in context["pairs"]])
    values = {}
    for curvature in GRID:
        values[curvature] = np.array([
            module._score(entry["table"], context, curvature, False)
            for entry in context["pairs"]
        ])
    shares = np.array([boundary_share(module, context, entry["table"], GRID[len(GRID) // 2])
                       for entry in context["pairs"]])
    report = {"pairs": len(context["pairs"]), "grid": [], "gauge": None}
    previous = None
    for curvature in GRID:
        current = values[curvature]
        rho, p = stats.spearmanr(current, recall)
        entry = {"curvature": curvature,
                 "spearman": round(float(rho), 6),
                 "p_value": float(p),
                 "value_range": [float(current.min()), float(current.max())],
                 "distinct_values": int(len(np.unique(current)))}
        if previous is not None:
            entry["max_abs_change"] = float(np.max(np.abs(current - previous)))
        shares_here = [boundary_share(module, context, pair["table"], curvature)
                       for pair in context["pairs"]]
        entry["boundary_share"] = {"min": float(min(shares_here)),
                                   "median": float(np.median(shares_here)),
                                   "max": float(max(shares_here))}
        report["grid"].append(entry)
        previous = current

    # the gauge test: curvature 1 on the tangents, and curvature c on the tangents
    # divided by sqrt(c), describe the same ball configuration
    reference = values[1.0]
    gauge = []
    for curvature in (4.0, 16.0, 64.0):
        scaled = np.array([
            module._score_scaled(pair["table"], context, curvature, 1.0 / np.sqrt(curvature))
            for pair in context["pairs"]
        ])
        gauge.append({"curvature": curvature,
                      "max_abs_difference_from_curvature_1": float(np.max(np.abs(scaled - reference)))})
    report["gauge"] = gauge
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"grid": report["grid"], "gauge": gauge}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
