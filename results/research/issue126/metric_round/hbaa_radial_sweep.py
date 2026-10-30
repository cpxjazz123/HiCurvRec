"""The radial coefficient of the Gromov product, swept.

HBAA's contrast uses the Gromov product

    (x|y)_0 = ( d(0,x) + d(0,y) - d(x,y) ) / 2

and the radial distance ``d(0,.)`` enters it TWICE, once for the history point and
once for the prefix centroid. The bug that was found multiplied the radial term by
a half, so the scoring rule was not the Gromov product but a family of them:

    g_alpha(x,y) = ( alpha*d(0,x) + alpha*d(0,y) - d(x,y) ) / 2

and the difference between a positive and a negative is

    Delta_alpha = ( d(h,n) - d(h,p) ) / 2 + (alpha/2) * ( r_p - r_n ).

So ``alpha`` is a knob on how much the radial term counts, ``alpha = 1`` is the
Gromov product, and ``alpha = 0.5`` is what the buggy code actually computed.

The sweep holds EVERYTHING else fixed -- the items, the SID trees, the prefix
centroids, the history barycentres, the queries, the distances and the
temperature -- and moves only ``alpha``. The end points are checks on the
arithmetic: ``alpha = 1`` must reproduce the corrected HBAA (+0.6880) and
``alpha = 0.5`` must reproduce the historical, buggy number (+0.7623).

Because the pieces do not depend on alpha, each table is prepared once and every
alpha is then free.

Note on admissibility: alpha is chosen by looking at the recall, so a maximum
over the sweep is a DIAGNOSTIC, not a metric. It says what the radial term does
to the ranking; it does not license a tuned constant.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "hbaa_radial_sweep_107.json"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pieces(table: dict, context: dict, module):
    """Everything the sweep needs, computed once: the two halves of Delta."""
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
        return None
    history = module._history_points(context, 1.0)[valid]
    pos = pos[valid]
    neg = neg[valid]
    centres = {int(code): module._frechet(target[codes == code], 1.0)
               for code in np.unique(codes)}
    mu_pos = np.stack([centres[int(codes[index])] for index in pos])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg])
    radius_history = module._norm(history, 1.0)
    radius_pos = module._norm(mu_pos, 1.0)
    radius_neg = module._norm(mu_neg, 1.0)
    distance_pos = module._distance(history, mu_pos, 1.0)
    distance_neg = module._distance(history[:, None, :], mu_neg, 1.0)
    # the two halves of Delta_alpha
    relative = (distance_neg - distance_pos[:, None]) / 2.0
    radial = (radius_pos[:, None] - radius_neg) / 2.0
    return relative, radial, tau


def score_for(relative: np.ndarray, radial: np.ndarray, tau: float, alpha: float) -> float:
    delta = relative + alpha * radial
    per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
    return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))


def work(index: int):
    from tools.research.science import sid_metric_bench as bench

    module = _load("hbaa_source", HERE / "metrics_hbaa.py")
    context = bench.load_context()
    entry = context["pairs"][index]
    prepared = pieces(entry["table"], context, module)
    if prepared is None:
        return index, None, None
    relative, radial, tau = prepared
    values = {alpha: score_for(relative, radial, tau, alpha) for alpha in ALPHAS}
    return index, values, float(entry["test_recall@10"])


def main() -> int:
    with ProcessPoolExecutor(max_workers=16) as pool:
        rows = [row for row in pool.map(work, range(107)) if row[1] is not None]
    rows.sort()
    recall = np.array([row[2] for row in rows])
    series = {alpha: np.array([row[1][alpha] for row in rows]) for alpha in ALPHAS}
    report = {"pairs": len(rows), "alphas": []}
    for alpha in ALPHAS:
        array = series[alpha]
        rho, p = stats.spearmanr(array, recall)
        report["alphas"].append({"alpha": alpha, "spearman": round(float(rho), 6),
                                 "p_value": float(p),
                                 "range": [float(array.min()), float(array.max())]})
    # paired bootstrap of every alpha against alpha = 1
    rng = np.random.default_rng(42)
    base = series[1.0]
    for entry in report["alphas"]:
        alpha = entry["alpha"]
        other = series[alpha]
        diffs = []
        for _ in range(8000):
            index = rng.integers(0, len(recall), len(recall))
            if len(np.unique(other[index])) < 3 or len(np.unique(base[index])) < 3:
                continue
            diffs.append(stats.spearmanr(other[index], recall[index])[0]
                         - stats.spearmanr(base[index], recall[index])[0])
        diffs = np.array(diffs)
        entry["paired_vs_alpha_1"] = {
            "mean": round(float(diffs.mean()), 6),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 6),
                     round(float(np.percentile(diffs, 97.5)), 6)],
            "p_not_better": round(float((diffs <= 0).mean()), 6)}
    report["values"] = {str(alpha): series[alpha].tolist() for alpha in ALPHAS}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("%-8s %10s %12s %12s" % ("alpha", "rho", "mean diff", "P(not better)"))
    for entry in report["alphas"]:
        print("%-8g %10.4f %+12.4f %12.4f" % (
            entry["alpha"], entry["spearman"],
            entry["paired_vs_alpha_1"]["mean"],
            entry["paired_vs_alpha_1"]["p_not_better"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
