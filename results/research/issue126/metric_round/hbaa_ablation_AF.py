"""The six ablations that split the historical +0.7623 open.

Everything is held fixed except the scoring term: the same items, the same SID
trees, the same prefix centroids, the same history barycentres, the same queries,
the same temperature, the same 107 recalls. No Stage 2 or Stage 3 is touched.

    A  plain               relative only                       (the +0.7104 baseline)
    B  raw radial          relative + 0.5 (r_p - r_n)/2          (must reproduce +0.7623)
    B' reproduction        relative + 0.5 sigma_r,t (z_p - z_n)/2, with
                           sigma_r,t the radius spread of THAT table -- the
                           per-query scores must equal B exactly
    C  compactness         relative + 0.5 (rhat_p - rhat_n)/2, rhat the radius
                           predicted from the compactness, the size and the
                           popularity by a regression INSIDE the table. Same units
                           as the radius, same weight.
    D1 residual            relative + 0.5 (eps_p - eps_n)/2, eps = r - rhat: the
                           radial information the compactness cannot explain, at
                           the same units and the same weight
    D2 residual, matched   the same, with the coefficient rescaled so that the
                           residual carries the radius' own spread (same influence)
    E  random origin       the radius measured from TEN random points of the ball
    F  permuted radius     the radius values shuffled among groups of similar
                           compactness (deciles of D), which keeps the compactness
                           distribution and breaks the group-radius link

The coefficient is never chosen on all 107 pairs and then used to prove itself:
the leave-one-family-out protocol selects the coefficient on the other families
and evaluates it on the held-out one, with the families taken from the run
names.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "hbaa_ablation_AF_107.json"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
RANDOM_ORIGINS = 10


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def family_of(run_dir: str) -> str:
    parts = run_dir.split("/")
    name = "unknown"
    if "stage3_T5Train" in parts:
        name = parts[parts.index("stage3_T5Train") + 1].lower()
    else:
        name = parts[-3].lower()
    if "curvature" in name:
        return "curvature"
    if "resample" in name:
        return "resample"
    if "fork_hyper" in name:
        return "fork_hyper"
    if "ablation" in name:
        return "ablation"
    if "baseline" in name or "behavior" in name:
        return "baseline"
    return "mechanisms"


def work(index: int):
    from tools.research.science import sid_metric_bench as bench

    module = _load("hbaa_source", HERE / "metrics_hbaa.py")
    context = bench.load_context()
    entry = context["pairs"][index]
    table = entry["table"]
    target = np.asarray(context["target"], dtype=np.float64)[np.asarray(table["item_ids"])]
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    items = np.asarray(table["item_ids"], dtype=np.int64)
    popularity = np.zeros(30000, dtype=np.int64)
    try:
        saved = np.load(HERE.parents[0] / "behaviour_hierarchy.npz")
        popularity = np.asarray(saved["popularity"], dtype=np.int64)
    except Exception:
        pass

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
    sizes = np.asarray([int((codes == code).sum()) for code in unique_codes], dtype=float)
    popular = np.asarray([float(np.mean([popularity[int(item)] if int(item) < len(popularity) else 0
                                         for item in items[codes == code]])) for code in unique_codes])

    # the compactness prediction of the radius, fitted INSIDE this table
    design = np.column_stack([np.ones(len(radius_values)), tightness_values,
                              np.log(sizes), np.log1p(popular)])
    beta, *_ = np.linalg.lstsq(design, radius_values, rcond=None)
    radius_predicted = design @ beta
    radius_residual = radius_values - radius_predicted
    std_radius = float(max(radius_values.std(), 1e-12))
    std_residual = float(max(radius_residual.std(), 1e-12))

    # the radius shuffled among groups of similar compactness
    order = np.argsort(tightness_values)
    permuted_radius = radius_values.copy()
    bins = np.array_split(order, 10)
    rng = np.random.default_rng(900 + index)
    for group in bins:
        if len(group) > 1:
            permuted_radius[group] = radius_values[rng.permutation(group)]

    # a term with the radius' own scale and NO information: the control that
    # separates "the radial term carries something" from "adding a term of this
    # size changes how the score is spread across the SIDs"
    noise_values = rng.normal(size=len(radius_values))
    noise_values = (noise_values - noise_values.mean()) / max(noise_values.std(), 1e-12)
    noise_values = noise_values * std_radius + radius_values.mean()

    # the radius measured from random points of the ball
    random_radii = []
    for _ in range(RANDOM_ORIGINS):
        direction = rng.normal(size=target.shape[1])
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        centre = direction * float(np.median(np.linalg.norm(target, axis=1)))
        random_radii.append(np.asarray([module._distance(centroids[int(code)][None, :],
                                                         centre[None, :], 1.0)[0]
                                        for code in unique_codes]))

    queries = module._queries(context)
    position = {int(item): place for place, item in enumerate(table["item_ids"].tolist())}
    pos = np.array([position.get(int(item), -1) for item in queries["positives"]], dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                    for row in queries["negatives"]], dtype=np.int64)
    valid = (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return index, None
    history = module._history_points(context, 1.0)[valid]
    pos = pos[valid]
    neg = neg[valid]
    code_pos = codes[pos]
    code_neg = codes[neg]
    mu_pos = np.stack([centroids[int(code)] for code in code_pos])
    mu_neg = np.stack([[centroids[int(code)] for code in row] for row in code_neg])
    relative = (module._distance(history[:, None, :], mu_neg, 1.0)
                - module._distance(history, mu_pos, 1.0)[:, None]) / 2.0

    code_index = {int(code): place for place, code in enumerate(unique_codes)}

    def term(values):
        # the arrays are indexed by GROUP, not by the SID token
        positive = np.asarray([values[code_index[int(code)]] for code in code_pos])
        negative = np.asarray([[values[code_index[int(code)]] for code in row]
                               for row in code_neg])
        return (positive[:, None] - negative) / 2.0

    radial_raw = term(radius_values)
    radial_predicted = term(radius_predicted)
    radial_residual = term(radius_residual)
    radial_permuted = term(permuted_radius)
    radial_random = [term(values) for values in random_radii]
    radial_noise = term(noise_values)
    # the group-level standardisation: z = (r - mean)/std over the table's groups
    z_radius = (radius_values - radius_values.mean()) / std_radius
    standardised = term(z_radius)
    tau = float(context["temperature"])

    def score(delta):
        per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
        return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))

    series = {}
    for alpha in ALPHAS:
        series["B_raw_radial_%g" % alpha] = score(relative + alpha * radial_raw)
        series["B_prime_reproduction_%g" % alpha] = score(
            relative + alpha * std_radius * standardised)
        series["C_compactness_%g" % alpha] = score(relative + alpha * radial_predicted)
        series["D1_residual_%g" % alpha] = score(relative + alpha * radial_residual)
        series["D2_residual_matched_%g" % alpha] = score(
            relative + alpha * (std_radius / std_residual) * radial_residual)
        series["F_permuted_radius_%g" % alpha] = score(relative + alpha * radial_permuted)
        series["G_random_noise_%g" % alpha] = score(relative + alpha * radial_noise)
    series["E_random_origin_mean"] = float(np.mean([
        score(relative + 0.5 * values) for values in radial_random]))
    series["E_random_origin_per_origin"] = [float(score(relative + 0.5 * values))
                                            for values in radial_random]
    series["A_plain"] = score(relative)
    series["_per_query_B_0.5"] = np.log1p(np.exp(-(relative + 0.5 * radial_raw) / tau))
    series["_per_query_B_prime_0.5"] = np.log1p(np.exp(
        -(relative + 0.5 * std_radius * standardised) / tau))
    return index, series


def main() -> int:
    with ProcessPoolExecutor(max_workers=16) as pool:
        rows = [row for row in pool.map(work, range(107)) if row[1] is not None]
    rows.sort()
    from tools.research.science import sid_metric_bench as bench

    context = bench.load_context()
    keep = np.asarray([row[0] for row in rows], dtype=np.int64)
    recall = np.asarray([entry["test_recall@10"] for entry in context["pairs"]])[keep]
    families = np.asarray([family_of(context["pairs"][index]["row"]["run_dir"]) for index in keep])

    names = sorted({name for row in rows for name in row[1] if not name.startswith("_")})
    series = {name: np.asarray([row[1][name] for row in rows]) for name in names}
    report = {"pairs": len(rows),
              "families": {key: int(value) for key, value in Counter(families).items()},
              "spearman": {}}
    for name in names:
        if name.startswith("E_random_origin_per_origin"):
            continue
        rho, p = stats.spearmanr(series[name], recall)
        report["spearman"][name] = {"spearman": round(float(rho), 6), "p_value": float(p)}

    # the reproduction check, per query
    first = np.stack([row[1]["_per_query_B_0.5"] for row in rows])
    second = np.stack([row[1]["_per_query_B_prime_0.5"] for row in rows])
    report["reproduction_max_abs_per_query_difference"] = float(np.max(np.abs(first - second)))

    # paired bootstrap against the plain baseline
    rng = np.random.default_rng(3)
    plain = series["A_plain"]
    report["paired_vs_plain"] = {}
    for name in ("B_raw_radial_0.5", "C_compactness_0.5", "D1_residual_0.5",
                 "D2_residual_matched_0.5", "F_permuted_radius_0.5",
                 "G_random_noise_0.5"):
        other = series[name]
        differences = []
        for _ in range(8000):
            sample = rng.integers(0, len(recall), len(recall))
            if len(np.unique(other[sample])) < 3 or len(np.unique(plain[sample])) < 3:
                continue
            differences.append(stats.spearmanr(other[sample], recall[sample])[0]
                               - stats.spearmanr(plain[sample], recall[sample])[0])
        differences = np.asarray(differences)
        report["paired_vs_plain"][name] = {
            "mean": round(float(differences.mean()), 6),
            "ci95": [round(float(np.percentile(differences, 2.5)), 6),
                     round(float(np.percentile(differences, 97.5)), 6)],
            "p_not_better": round(float((differences <= 0).mean()), 6)}

    # leave-one-family-out: the coefficient is chosen on the other families
    report["leave_one_family_out"] = {}
    for family in sorted(set(families.tolist())):
        held = families == family
        others = ~held
        chosen = {}
        for prefix in ("B_raw_radial", "C_compactness", "D1_residual",
                       "D2_residual_matched", "F_permuted_radius", "G_random_noise"):
            best = max(ALPHAS, key=lambda a: stats.spearmanr(
                series["%s_%g" % (prefix, a)][others], recall[others])[0])
            chosen[prefix] = best
            report["leave_one_family_out"].setdefault(prefix, {})[family] = {
                "chosen_alpha": best,
                "held_out_spearman": round(float(stats.spearmanr(
                    series["%s_%g" % (prefix, best)][held], recall[held])[0]), 6),
                "held_out_plain": round(float(stats.spearmanr(
                    plain[held], recall[held])[0]), 6)}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for name in names:
        if name.startswith("E_random_origin_per_origin") or name.startswith("_"):
            continue
        print("%-30s %+.4f" % (name, report["spearman"][name]["spearman"]))
    print("reproduction max |per-query difference|:",
          report["reproduction_max_abs_per_query_difference"])
    print("paired vs plain:", json.dumps(report["paired_vs_plain"], indent=1))
    print("leave one family out:", json.dumps(report["leave_one_family_out"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
