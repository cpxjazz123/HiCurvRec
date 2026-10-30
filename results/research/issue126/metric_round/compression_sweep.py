"""Does the four-point correlation come from the hyperbolic geometry, or from
the radial compression that pushes the points toward the ball's boundary?

The cut is made with three distance laws on the SAME quadruples, the SAME items
and the SAME scale:

* ``hyperbolic``  -- ``d_H(x,y)`` of the Poincare ball;
* ``euclidean``   -- the linear distance. It is invariant under a uniform rescale,
  so its correlation does not move with the compression at all: it is the line
  the other two are read against;
* ``weighted``    -- the RADIALLY WEIGHTED EUCLIDEAN: the linear distance divided
  by ``sqrt((1-c||x||^2)(1-c||y||^2))``, i.e. exactly the per-point factor the
  hyperbolic distance carries, with the ``arccosh`` removed.

Why the third one settles the question: the score compares the argmin of three
pairing sums, and an argmin is invariant under a strictly increasing transform of
every distance. When all the points sit at nearly one radius the hyperbolic
distance IS such a transform of the Euclidean one, so the two MUST return the same
value. Any difference between them is caused by the per-point radial weight and by
nothing else. So:

* hyperbolic > euclidean and weighted > euclidean  -> the gain is the radial weight;
* hyperbolic > weighted                            -> the arccosh itself adds something;
* both collapse as the compression is removed      -> the whole effect was the
  compression.

Curvature is held at 8 and the tangent scale sweeps the compression, which is the
one degree of freedom the gauge test identified.
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
CURVATURE = 8.0
SCALES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)
OUT = Path(__file__).resolve().parents[1] / "compression_sweep_107.json"


def load_module():
    spec = importlib.util.spec_from_file_location("four_point", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    context = bench.load_context()
    recall = np.array([entry["test_recall@10"] for entry in context["pairs"]])
    laws = {
        "hyperbolic": lambda table, index, scale, curvature: module._score(
            table, context, curvature, False, tangent_scale=scale),
        "euclidean": lambda table, index, scale, curvature: module._score(
            table, context, curvature, True, tangent_scale=scale),
        "weighted_euclidean": lambda table, index, scale, curvature: module._score(
            table, context, curvature, True, tangent_scale=scale, weighted=True),
    }
    report = {"curvature": CURVATURE, "scales": []}
    previous = {}
    for scale in SCALES:
        entry = {"tangent_scale": scale, "labels": {}}
        for name, law in laws.items():
            values = np.array([law(pair["table"], index, scale, CURVATURE)
                               for index, pair in enumerate(context["pairs"])])
            rho, p = stats.spearmanr(values, recall)
            entry["labels"][name] = {"spearman": round(float(rho), 6),
                                     "p_value": float(p)}
            if name in previous:
                entry["labels"][name]["max_abs_change"] = float(
                    np.max(np.abs(values - previous[name])))
            previous[name] = values
        radii = []
        for pair in context["pairs"]:
            embeddings = module._svd_embeddings(context, pair["table"]) * scale
            points = module.ball_of(embeddings, CURVATURE)
            radii.append(np.linalg.norm(points, axis=1))
        stacked = np.concatenate(radii)
        entry["radius"] = {"median": float(np.median(stacked)),
                           "p10": float(np.percentile(stacked, 10)),
                           "p90": float(np.percentile(stacked, 90))}
        report["scales"].append(entry)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    header = "%-8s %10s %10s %10s %10s" % ("scale", "radius", "hyperbolic", "euclid", "weighted")
    print(header)
    for entry in report["scales"]:
        print("%-8g %10.4f %10.4f %10.4f %10.4f" % (
            entry["tangent_scale"], entry["radius"]["median"],
            entry["labels"]["hyperbolic"]["spearman"],
            entry["labels"]["euclidean"]["spearman"],
            entry["labels"]["weighted_euclidean"]["spearman"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
