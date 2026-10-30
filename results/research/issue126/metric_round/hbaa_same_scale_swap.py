"""The same weight scale, applied to the radius and to the tightness.

The question: the historical +0.7623 beats the plain distance contrast (+0.7104)
by 0.052. Is that excess the RADIAL information, or the group compactness the
radius stands for?

To put the two terms on one scale, the radius and the tightness are each
standardised over the table's own groups, and the coefficient that the raw
formulation applied is reproduced EXACTLY. With

    Delta = relative + alpha_raw * (r_p - r_n) / 2
    Delta = relative + coef * (z_p - z_n) / 2,   z = (r - mean) / std_r(table)

the two are the same rule iff ``coef = alpha_raw * std_r(table)`` -- a
per-table coefficient, because the raw radius has a scale of its own (median
2.785) that the distance difference does not share. So the equivalent
scale-free coefficient VARIES FROM TABLE TO TABLE, and reproducing the historical
weight means using that per-table coefficient.

Four series are measured on the same 107 pairs:

* ``radial_raw(alpha)``       the historical formulation, for the check;
* ``radial_matched(alpha)``   the standardised radius with the per-table
  equivalent coefficient -- algebraically identical to the first, so it must
  return the same numbers, and that is the implementation check;
* ``tightness_raw(alpha)``    the literal swap: the tightness in its own units
  with the same alpha;
* ``tightness_matched(alpha)`` the tightness standardised, with the same
  per-table coefficient, i.e. the identical weight the radius needed.

If the tightness reaches the same +0.7623 at the same weight, the excess is
compactness. If it stays at the plain contrast's +0.7104, the excess is the
radial information itself.
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
OUT = HERE.parents[0] / "hbaa_same_scale_swap_107.json"
ALPHAS = (0.25, 0.5, 0.75, 1.0)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def work(index: int):
    from tools.research.science import sid_metric_bench as bench

    module = _load("hbaa_source", HERE / "metrics_hbaa.py")
    context = bench.load_context()
    entry = context["pairs"][index]
    table = entry["table"]
    target = np.asarray(context["target"], dtype=np.float64)[np.asarray(table["item_ids"])]
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]

    unique_codes = np.unique(codes)
    centroids, radii, tightness = {}, {}, {}
    for code in unique_codes:
        members = target[codes == code]
        centroid = module._frechet(members, 1.0)
        centroids[int(code)] = centroid
        radii[int(code)] = float(module._norm(centroid[None, :], 1.0)[0])
        distance = module._distance(members, centroid[None, :], 1.0)
        tightness[int(code)] = float(np.mean(distance ** 2))
    radius_values = np.asarray([radii[int(code)] for code in unique_codes])
    tightness_values = np.asarray([tightness[int(code)] for code in unique_codes])
    std_radius = float(max(radius_values.std(), 1e-12))
    std_tightness = float(max(tightness_values.std(), 1e-12))

    queries = module._queries(context)
    position = {int(item): place for place, item in enumerate(table["item_ids"].tolist())}
    pos = np.array([position.get(int(item), -1) for item in queries["positives"]], dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                    for row in queries["negatives"]], dtype=np.int64)
    valid = (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return index, None, None
    history = module._history_points(context, 1.0)[valid]
    pos = pos[valid]
    neg = neg[valid]
    code_pos = codes[pos]
    code_neg = codes[neg]
    mu_pos = np.stack([centroids[int(code)] for code in code_pos])
    mu_neg = np.stack([[centroids[int(code)] for code in row] for row in code_neg])
    relative = (module._distance(history[:, None, :], mu_neg, 1.0)
                - module._distance(history, mu_pos, 1.0)[:, None]) / 2.0

    def raw(values):
        lookup = {int(code): float(value) for code, value in zip(unique_codes, values)}
        positive = np.asarray([lookup[int(code)] for code in code_pos])
        negative = np.asarray([[lookup[int(code)] for code in row] for row in code_neg])
        return (positive[:, None] - negative) / 2.0

    radial_raw = raw(radius_values)
    tightness_raw = raw(tightness_values)
    radial_z = radial_raw / std_radius
    tightness_z = tightness_raw / std_tightness
    tau = float(context["temperature"])

    def score(delta):
        per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
        return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))

    series = {}
    for alpha in ALPHAS:
        series["radial_raw_%g" % alpha] = score(relative + alpha * radial_raw)
        series["radial_matched_%g" % alpha] = score(relative + alpha * std_radius * radial_z)
        series["tightness_raw_%g" % alpha] = score(relative + alpha * tightness_raw)
        series["tightness_matched_%g" % alpha] = score(relative + alpha * std_tightness * tightness_z)
        # THE decisive cell: the SAME dimensionless weight the radius needed
        # (alpha * std_r(table), the coefficient the raw formulation applied),
        # applied to the standardised tightness.
        series["tightness_at_radius_weight_%g" % alpha] = score(
            relative + alpha * std_radius * tightness_z)
        # and the mirror: the radius at the tightness' own scale
        series["radial_at_tightness_weight_%g" % alpha] = score(
            relative + alpha * std_tightness * radial_z)
    series["plain"] = score(relative)
    return index, series, float(std_radius)


def main() -> int:
    with ProcessPoolExecutor(max_workers=16) as pool:
        rows = [row for row in pool.map(work, range(107)) if row[1] is not None]
    rows.sort()
    from tools.research.science import sid_metric_bench as bench

    context = bench.load_context()
    recall = np.asarray([entry["test_recall@10"] for entry in context["pairs"]])[
        np.asarray([row[0] for row in rows], dtype=np.int64)]
    names = sorted({name for row in rows for name in row[1]})
    report = {"pairs": len(rows), "series": {}, "spread": {}}
    for name in names:
        values = np.asarray([row[1][name] for row in rows])
        rho, p = stats.spearmanr(values, recall)
        report["series"][name] = {"spearman": round(float(rho), 6), "p_value": float(p)}
    # the implementation check: the two formulations of the radius must agree
    for alpha in ALPHAS:
        first = np.asarray([row[1]["radial_raw_%g" % alpha] for row in rows])
        second = np.asarray([row[1]["radial_matched_%g" % alpha] for row in rows])
        report["spread"]["radial_raw_vs_matched_%g" % alpha] = float(
            np.max(np.abs(first - second)))
    even = np.arange(0, len(recall), 2)
    odd = np.arange(1, len(recall), 2)
    report["split_half"] = {}
    for family in ("radial_raw", "radial_matched", "tightness_raw", "tightness_matched",
                   "tightness_at_radius_weight", "radial_at_tightness_weight"):
        candidates = [alpha for alpha in ALPHAS]
        best_even = max(candidates, key=lambda a: stats.spearmanr(
            np.asarray([row[1]["%s_%g" % (family, a)] for row in rows])[even], recall[even])[0])
        best_odd = max(candidates, key=lambda a: stats.spearmanr(
            np.asarray([row[1]["%s_%g" % (family, a)] for row in rows])[odd], recall[odd])[0])
        report["split_half"][family] = {"best_on_even": best_even, "best_on_odd": best_odd}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for name in names:
        print("%-26s %+.4f" % (name, report["series"][name]["spearman"]))
    print("implementation check (max |raw - matched|):",
          json.dumps(report["spread"]))
    print("split half:", json.dumps(report["split_half"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
