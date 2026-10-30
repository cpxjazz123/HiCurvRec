"""Is HBAA's predictive gain the radial term, or is it group tightness?

The radius of a prefix centroid is 91% the group's dispersion (protocol v14), and
a random reference centre reproduces it, so the radius does not measure hierarchy.
That leaves the question the brief asks: does the RADIUS carry HBAA's extra
predictive power, or does the TIGHTNESS it stands for?

This replaces the radial term of the Gromov contrast with the group's tightness
and measures both on the same 107 pairs:

    Delta = ( d(h,n) - d(h,p) ) / 2 + (coef/2) * ( z(term_p) - z(term_n) )

with ``term`` either the radius (standardised over the table's groups) or the
tightness D = mean d(x_i, mu)^2 (standardised the same way), so the two families
share one coefficient scale. Both signs of the tightness term are tried: nothing
in the construction says which direction a tighter positive's prefix should
push the score.

The coefficients are chosen on one half of the 107 pairs and evaluated on the
other half, so the comparison between the families is not made on the sample the
choice was made from.

The per-group records are written out as well (radius, tightness, size,
popularity, and the radius from ten random centres) so the statistics of
protocol v14 can be recomputed against the improved hierarchy reference.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "hbaa_tightness_swap_107.json"
GROUPS_OUT = HERE.parents[0] / "group_records_107.npz"
RADIAL_COEFFICIENTS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
TIGHTNESS_COEFFICIENTS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
RANDOM_CENTRES = 10


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
    items = np.asarray(table["item_ids"], dtype=np.int64)
    popularity = np.zeros(30000, dtype=np.int64)
    try:
        saved = np.load(HERE.parents[0] / "behaviour_hierarchy.npz")
        popularity = np.asarray(saved["popularity"], dtype=np.int64)
    except Exception:
        pass

    unique_codes = np.unique(codes)
    centroids = {}
    radii = {}
    tightness = {}
    for code in unique_codes:
        members = target[codes == code]
        centroid = module._frechet(members, 1.0)
        centroids[int(code)] = centroid
        radii[int(code)] = float(module._norm(centroid[None, :], 1.0)[0])
        distance = module._distance(members, centroid[None, :], 1.0)
        tightness[int(code)] = float(np.mean(distance ** 2))
    radius_values = np.asarray([radii[int(code)] for code in unique_codes])
    tightness_values = np.asarray([tightness[int(code)] for code in unique_codes])
    z_radius = (radius_values - radius_values.mean()) / max(radius_values.std(), 1e-12)
    z_tightness = ((tightness_values - tightness_values.mean())
                   / max(tightness_values.std(), 1e-12))
    z_radius_map = {int(code): float(value) for code, value in zip(unique_codes, z_radius)}
    z_tightness_map = {int(code): float(value) for code, value in zip(unique_codes, z_tightness)}

    queries = module._queries(context)
    position = {int(item): place for place, item in enumerate(table["item_ids"].tolist())}
    pos = np.array([position.get(int(item), -1) for item in queries["positives"]], dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                    for row in queries["negatives"]], dtype=np.int64)
    valid = (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return index, None, None, None
    history = module._history_points(context, 1.0)[valid]
    pos = pos[valid]
    neg = neg[valid]
    code_pos = codes[pos]
    code_neg = codes[neg]
    mu_pos = np.stack([centroids[int(code)] for code in code_pos])
    mu_neg = np.stack([[centroids[int(code)] for code in row] for row in code_neg])
    distance_pos = module._distance(history, mu_pos, 1.0)
    distance_neg = module._distance(history[:, None, :], mu_neg, 1.0)
    relative = (distance_neg - distance_pos[:, None]) / 2.0
    radial = (np.asarray([z_radius_map[int(code)] for code in code_pos])[:, None]
              - np.asarray([[z_radius_map[int(code)] for code in row] for row in code_neg])) / 2.0
    tight = (np.asarray([z_tightness_map[int(code)] for code in code_pos])[:, None]
             - np.asarray([[z_tightness_map[int(code)] for code in row] for row in code_neg])) / 2.0
    tau = float(context["temperature"])

    def score(delta):
        per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
        return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))

    radial_scores = {coefficient: score(relative + coefficient * radial)
                     for coefficient in RADIAL_COEFFICIENTS}
    tightness_scores = {coefficient: score(relative + coefficient * tight)
                        for coefficient in TIGHTNESS_COEFFICIENTS}

    rng = np.random.default_rng(500 + index)
    random_radii = []
    for _ in range(RANDOM_CENTRES):
        direction = rng.normal(size=target.shape[1])
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        centre = direction * float(np.median(np.linalg.norm(target, axis=1)))
        random_radii.append({int(code): float(module._distance(
            centroids[int(code)][None, :], centre[None, :], 1.0)[0])
            for code in unique_codes})

    records = {
        "table": np.full(len(unique_codes), index),
        "code": unique_codes.astype(np.int64),
        "radius": radius_values,
        "tightness": tightness_values,
        "size": np.asarray([int((codes == code).sum()) for code in unique_codes], dtype=np.int64),
        "log_popularity": np.asarray([
            float(np.log1p(np.mean([popularity[int(item)] if int(item) < len(popularity) else 0
                                     for item in items[codes == code]]))) for code in unique_codes]),
        "random_radius": np.asarray([[values[int(code)] for code in unique_codes]
                                     for values in random_radii], dtype=np.float64),
    }
    return index, radial_scores, tightness_scores, records


def main() -> int:
    with ProcessPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(work, range(107)))
    rows.sort()
    usable = [row for row in rows if row[1] is not None]
    recall = np.asarray([row_index for row_index in usable])  # placeholder, fixed below
    from tools.research.science import sid_metric_bench as bench

    context = bench.load_context()
    recall = np.asarray([entry["test_recall@10"] for entry in context["pairs"]])
    keep = [row[0] for row in usable]
    recall = recall[np.asarray(keep, dtype=np.int64)]

    radial = {coefficient: np.asarray([row[1][coefficient] for row in usable])
              for coefficient in RADIAL_COEFFICIENTS}
    tightness = {coefficient: np.asarray([row[2][coefficient] for row in usable])
                 for coefficient in TIGHTNESS_COEFFICIENTS}
    payload = {}
    for key in ("table", "code", "radius", "tightness", "size", "log_popularity"):
        payload[key] = np.concatenate([row[3][key] for row in usable], axis=0)
    for row in usable:
        payload["random_radius_%d" % row[0]] = row[3]["random_radius"]
    np.savez_compressed(GROUPS_OUT, **payload)
    report = {"pairs": len(usable), "radial": {}, "tightness": {}, "split_half": {}}
    for name, family in (("radial", radial), ("tightness", tightness)):
        for coefficient, values in family.items():
            from scipy import stats
            rho, p = stats.spearmanr(values, recall)
            report[name][str(coefficient)] = {"spearman": round(float(rho), 6),
                                               "p_value": float(p)}
    even = np.arange(0, len(recall), 2)
    odd = np.arange(1, len(recall), 2)
    from scipy import stats
    for name, family in (("radial", radial), ("tightness", tightness)):
        best_even = max(family, key=lambda c: stats.spearmanr(family[c][even], recall[even])[0])
        best_odd = max(family, key=lambda c: stats.spearmanr(family[c][odd], recall[odd])[0])
        report["split_half"][name] = {
            "best_on_even": best_even,
            "best_on_odd": best_odd,
            "held_out_even_choice_on_odd": round(float(stats.spearmanr(
                family[best_even][odd], recall[odd])[0]), 6),
            "held_out_odd_choice_on_even": round(float(stats.spearmanr(
                family[best_odd][even], recall[even])[0]), 6),
            "alpha_zero_on_odd": round(float(stats.spearmanr(
                family[0.0][odd], recall[odd])[0]), 6),
            "alpha_zero_on_even": round(float(stats.spearmanr(
                family[0.0][even], recall[even])[0]), 6),
        }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("radial:")
    for coefficient, value in report["radial"].items():
        print("   %-6s %+.4f" % (coefficient, value["spearman"]))
    print("tightness:")
    for coefficient, value in report["tightness"].items():
        print("   %-6s %+.4f" % (coefficient, value["spearman"]))
    print("split half:", json.dumps(report["split_half"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
