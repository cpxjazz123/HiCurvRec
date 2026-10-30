"""Why does the saturated chart rank the 107 SIDs better?

Three experiments, all on the same 107 confirmed SID+recall pairs, with the same
quadruples, the same items and the same SID trees.

1. **The weight distribution.** For every quadruple the delta, the weight and the
   pairing agreement are computed in BOTH charts. The question is what saturation
   does to the weights: flatten them, or move the influence onto special
   quadruples?

2. **The weight swap.** The pairing judgement A_k is identical in the two charts
   (the agreement between the charts was 1.000 at c = 16), so the weights can be
   swapped while A_k is held fixed:

   * the Poincare weights with the shared A_k  -> the reported +0.6609,
   * the Lorentz weights with the shared A_k    -> the reported +0.5441,
   * uniform weights with the shared A_k        -> the reported +0.6593.

   If all three come back, the whole gap is the weights. Then the per-quadruple
   weight ratio is related to the properties of the quadruple -- its true delta,
   its tree spread, how far apart its points are, how many prefixes its pairs
   share -- to see WHICH quadruples saturation moves.

3. **The clipping sweep.** The Poincare distance clips its denominator at 1e-12.
   The clip is a numerical parameter, not a geometric one: if the metric, its
   delta or its weights move when only the clip moves, then part of the reported
   correlation is an artefact of the implementation.

The question underneath: why would WRONG weights predict Recall better than the
right ones? The answer has to come from what the down-weighted quadruples have in
common, so the script records, for every quadruple, everything needed to say.
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

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "weight_experiment_107.json"
CURVATURE = 16.0
CLIPS = (1e-6, 1e-9, 1e-12, 1e-15, 1e-18)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def poincare_distance(points, curvature: float, clip: float = 1e-12) -> np.ndarray:
    c = float(curvature)
    difference = np.sum((points[0] - points[1]) ** 2, axis=-1)
    denominator = ((1 - c * np.sum(points[0] * points[0], axis=-1))
                   * (1 - c * np.sum(points[1] * points[1], axis=-1)))
    return np.arccosh(np.maximum(1 + 2 * c * difference / np.maximum(denominator, clip),
                                 1 + 1e-12)) / np.sqrt(c)


def prepare(table: dict, context: dict, module, lorentz):
    tangent = module._svd_embeddings(context, table)
    picks = module._graph_local_quadruples(context, table, module.SCORING_QUADRUPLES, module.SEED)
    a, b, c, d = picks[:, 0], picks[:, 1], picks[:, 2], picks[:, 3]
    rows = np.asarray(table["rows"], dtype=np.int64)

    def shared(left, right):
        equal = rows[left] == rows[right]
        return np.where(equal.all(axis=1), equal.shape[1], equal.argmin(axis=1)).astype(np.float64)

    width = rows.shape[1]
    tree_sums = np.stack([
        (width - shared(a, b)) + (width - shared(c, d)),
        (width - shared(a, c)) + (width - shared(b, d)),
        (width - shared(a, d)) + (width - shared(b, c))], axis=0)
    informative = (tree_sums.max(axis=0) - tree_sums.min(axis=0)) > 0
    tree_choice = np.argmin(tree_sums, axis=0)
    spread = tree_sums.max(axis=0) - tree_sums.min(axis=0)
    shared_pairs = 0.5 * (shared(a, b) + shared(c, d) + shared(a, c)
                          + shared(b, d) + shared(a, d) + shared(b, c))
    return tangent, picks, tree_choice, informative, spread, shared_pairs


def chart_data(tangent, picks, curvature, module, lorentz, clip=None):
    """The delta, the pairing choice and the distances of one chart."""
    a, b, c, d = picks[:, 0], picks[:, 1], picks[:, 2], picks[:, 3]
    ball = module.ball_of(tangent, curvature)
    radius, direction = lorentz.lorentz_parts(tangent, curvature)

    def poincare(left, right):
        return poincare_distance((ball[left], ball[right]), curvature,
                                 1e-12 if clip is None else clip)

    def hyperboloid(left, right):
        return lorentz.lorentz_distance(radius[left], direction[left],
                                        radius[right], direction[right], curvature)

    out = {}
    for name, distance in (("poincare", poincare), ("lorentz", hyperboloid)):
        sums = np.stack([distance(a, b) + distance(c, d),
                         distance(a, c) + distance(b, d),
                         distance(a, d) + distance(b, c)], axis=0)
        delta = np.abs(module.normalised_delta(sums))
        out[name] = {"delta": delta, "choice": np.argmin(sums, axis=0),
                     "scale": np.mean(sums, axis=0)}
    return out


def weights_of(delta: np.ndarray) -> np.ndarray:
    median = float(np.median(delta))
    return 1.0 / (1.0 + delta / max(median, 1e-12)), median


def main() -> int:
    module = _load("four_point", HERE / "metrics_four_point_svd.py")
    lorentz = _load("lorentz_check", HERE / "lorentz_check.py")
    context = bench.load_context()
    recall = np.array([entry["test_recall@10"] for entry in context["pairs"]])

    scores = {name: [] for name in ("poincare_weights", "lorentz_weights", "uniform")}
    kept_indices = []
    weight_ratios, weight_corr = [], []
    property_rows = []
    choice_agree = []
    for index, entry in enumerate(context["pairs"]):
        table = entry["table"]
        tangent, picks, tree_choice, informative, spread, shared_pairs = prepare(
            table, context, module, lorentz)
        data = chart_data(tangent, picks, CURVATURE, module, lorentz)
        delta_p, delta_l = data["poincare"]["delta"], data["lorentz"]["delta"]
        choice_p, choice_l = data["poincare"]["choice"], data["lorentz"]["choice"]
        choice_agree.append(float(np.mean(choice_p == choice_l)))
        agreement = (choice_p == tree_choice).astype(np.float64)
        weight_p, median_p = weights_of(delta_p)
        weight_l, median_l = weights_of(delta_l)
        keep = informative
        if keep.sum() < 16:
            # a table whose quadruples are all uninformative has nothing to say
            continue
        kept_indices.append(index)
        scores["poincare_weights"].append(
            float((weight_p[keep] * agreement[keep]).sum() / max(weight_p[keep].sum(), 1e-12)))
        scores["lorentz_weights"].append(
            float((weight_l[keep] * agreement[keep]).sum() / max(weight_l[keep].sum(), 1e-12)))
        scores["uniform"].append(float(agreement[keep].mean()))

        ratio = np.log(np.maximum(weight_p, 1e-300) / np.maximum(weight_l, 1e-300))
        weight_ratios.append(ratio[keep])
        weight_corr.append(float(np.corrcoef(weight_p[keep], weight_l[keep])[0, 1]))
        property_rows.append({
            "corr_log_ratio_vs_true_delta": float(np.corrcoef(ratio[keep], delta_l[keep])[0, 1]),
            "corr_log_ratio_vs_pairing_scale": float(np.corrcoef(ratio[keep], data["lorentz"]["scale"][keep])[0, 1]),
            "corr_log_ratio_vs_tree_spread": float(np.corrcoef(ratio[keep], spread[keep])[0, 1]),
            "corr_log_ratio_vs_shared_prefixes": float(np.corrcoef(ratio[keep], shared_pairs[keep])[0, 1]),
            "corr_true_delta_vs_pairing_scale": float(np.corrcoef(delta_l[keep], data["lorentz"]["scale"][keep])[0, 1]),
            "corr_poincare_weight_vs_scale": float(np.corrcoef(weight_p[keep], data["lorentz"]["scale"][keep])[0, 1]),
            "corr_lorentz_weight_vs_scale": float(np.corrcoef(weight_l[keep], data["lorentz"]["scale"][keep])[0, 1]),
            "corr_poincare_weight_vs_true_delta": float(np.corrcoef(weight_p[keep], delta_l[keep])[0, 1]),
            "corr_lorentz_weight_vs_true_delta": float(np.corrcoef(weight_l[keep], delta_l[keep])[0, 1]),
            "corr_weight_ratio_vs_agreement": float(np.corrcoef(ratio[keep], agreement[keep])[0, 1]),
            "median_delta_poincare": float(np.median(delta_p[keep])),
            "median_delta_lorentz": float(np.median(delta_l[keep])),
            "weight_ratio_p90_p10": float(np.percentile(ratio[keep], 90) - np.percentile(ratio[keep], 10)),
        })

    recall_kept = recall[np.asarray(kept_indices, dtype=np.int64)]
    report = {"pairs": len(context["pairs"]), "tables_scored": len(kept_indices),
              "curvature": CURVATURE, "experiment_2": {},
              "experiment_1": {}, "experiment_3": []}
    for name, series in scores.items():
        array = np.asarray(series, dtype=float)
        rho, p = stats.spearmanr(array, recall_kept)
        report["experiment_2"][name] = {"spearman": round(float(rho), 6),
                                        "p_value": float(p),
                                        "range": [float(array.min()), float(array.max())]}
    report["experiment_1"] = {
        "pairing_choice_agreement": float(np.mean(choice_agree)),
        "weight_correlation_mean": float(np.mean(weight_corr)),
        "log_weight_ratio_percentiles": {
            "p1": float(np.percentile(np.concatenate(weight_ratios), 1)),
            "p25": float(np.percentile(np.concatenate(weight_ratios), 25)),
            "p50": float(np.percentile(np.concatenate(weight_ratios), 50)),
            "p75": float(np.percentile(np.concatenate(weight_ratios), 75)),
            "p99": float(np.percentile(np.concatenate(weight_ratios), 99))},
        "per_table_properties": {
            key: float(np.mean([row[key] for row in property_rows]))
            for key in property_rows[0]},
        "per_table_properties_std": {
            key: float(np.std([row[key] for row in property_rows]))
            for key in property_rows[0]},
    }

    # experiment 3: the clipping threshold is a numerical parameter only
    for clip in CLIPS:
        series = []
        for entry in [context["pairs"][i] for i in kept_indices]:
            table = entry["table"]
            tangent, picks, tree_choice, informative, spread, shared_pairs = prepare(
                table, context, module, lorentz)
            data = chart_data(tangent, picks, CURVATURE, module, lorentz, clip=clip)
            delta = data["poincare"]["delta"]
            agreement = (data["poincare"]["choice"] == tree_choice).astype(np.float64)
            weight, _ = weights_of(delta)
            keep = informative
            series.append(float((weight[keep] * agreement[keep]).sum()
                                / max(weight[keep].sum(), 1e-12)))
        array = np.asarray(series, dtype=float)
        rho, p = stats.spearmanr(array, recall_kept)
        report["experiment_3"].append({"clip": clip,
                                       "spearman": round(float(rho), 6),
                                       "p_value": float(p),
                                       "median_delta": float(np.median(data["poincare"]["delta"])),
                                       "value_range": [float(array.min()), float(array.max())]})
    report["scores"] = {name: series for name, series in scores.items()}
    report["kept_indices"] = [int(i) for i in kept_indices]
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("experiment 2 (the swap):")
    for name, value in report["experiment_2"].items():
        print("   %-20s %+.4f" % (name, value["spearman"]))
    print("experiment 1:", json.dumps(report["experiment_1"]["per_table_properties"], indent=1))
    print("   pairing choice agreement %.4f | weight correlation %.4f" % (
        report["experiment_1"]["pairing_choice_agreement"],
        report["experiment_1"]["weight_correlation_mean"]))
    print("experiment 3 (the clip):")
    for entry in report["experiment_3"]:
        print("   clip %-8g %+.4f  median delta %.5f" % (
            entry["clip"], entry["spearman"], entry["median_delta"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
