"""Which statistical feature of the SID table carries the +0.0522?

The six ablations of protocol v16 showed that the historical gain is reproduced
by an arbitrary origin and by a shuffled radius, so it is not group-specific
radial information. This narrows the question to: WHICH table-level feature?

Five candidates, each fitted INSIDE the table as a prediction of the radius and
then substituted into the same scoring rule at the same units and the same
weight:

    H1 compactness      the mean squared distance of the members to the centroid
    H2 separation       the mean distance from the centroid to the other groups'
                        centroids: how far apart the SID groups sit
    H3 size             the group's item count
    H4 shape            the leading eigenvalue share of the group's own scatter:
                        one blob or a streak
    H5 radial moments   nothing but the table's radius mean and spread, i.e. a
                        rescaling of the score with no group information at all

Plus two permutations that separate "the numbers" from "the correspondence":

    P1 full permutation  every group's radius replaced by another group's radius
                         in the same table, repeated over many draws: the
                         distribution is preserved exactly, the correspondence is
                         destroyed
    P2 noise             a random term of the same scale

The judgement the brief asks for: if the full permutation still gives about
+0.76, which group sits where does not matter and the gain is a table-level
distribution effect; if it collapses to +0.71, the correspondence matters.

Everything else is held fixed: the items, the SID trees, the centroids, the
history barycentres, the queries, the temperature and the recalls. The
coefficient is chosen on the other families and evaluated on the held-out one.
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
OUT = HERE.parents[0] / "hbaa_feature_probe_107.json"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FULL_PERMUTATIONS = 100


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def family_of(run_dir: str) -> str:
    """The mechanism family: the variant name with its iteration suffix removed.

    Every run has its own variant name (``curvature_RQ-VAE_iter53``,
    ``tiger_sample_resample_s43``), so grouping by the full name leaves one or two
    members per family and the held-out split is impossible. The prefix is what
    identifies the mechanism.
    """
    import re

    parts = run_dir.split("/")
    name = (parts[parts.index("stage3_T5Train") + 1].lower()
            if "stage3_T5Train" in parts else parts[-3].lower())
    for marker in ("curvature", "resample", "fork_hyper", "crowd", "bootstrap",
                   "baseline", "behavior", "ablation"):
        if marker in name:
            return marker
    return "other"


def work(index: int):
    from tools.research.science import sid_metric_bench as bench

    module = _load("hbaa_source", HERE / "metrics_hbaa.py")
    context = bench.load_context()
    entry = context["pairs"][index]
    table = entry["table"]
    target = np.asarray(context["target"], dtype=np.float64)[np.asarray(table["item_ids"])]
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]

    unique_codes = np.unique(codes)
    centroids = {}
    members_of = {}
    for code in unique_codes:
        members = target[codes == code]
        members_of[int(code)] = members
        centroids[int(code)] = module._frechet(members, 1.0)
    centroids_matrix = np.stack([centroids[int(code)] for code in unique_codes])
    radius_values = np.asarray([float(module._norm(c[None, :], 1.0)[0])
                                for c in centroids_matrix])
    tightness_values = np.asarray([
        float(np.mean(module._distance(members_of[int(code)],
                                       centroids[int(code)][None, :], 1.0) ** 2))
        for code in unique_codes])
    sizes = np.asarray([len(members_of[int(code)]) for code in unique_codes], dtype=float)

    # H2: the separation of each group from the others, in the ball
    # the mean distance to the OTHER groups: the group itself is excluded from the
    # sum and the mean is over n-1. The previous version set the self-distance to
    # infinity, took the mean (which is then infinite) and replaced the infinity
    # with zero, so the feature was identically zero and the experiment measured
    # nothing.
    separation = np.zeros(len(unique_codes))
    block = 512
    n_groups = len(unique_codes)
    for start in range(0, n_groups, block):
        stop = min(start + block, n_groups)
        distances = module._distance(centroids_matrix[start:stop, None, :],
                                     centroids_matrix[None, :, :], 1.0)
        total = distances.sum(axis=1)
        if n_groups > 1:
            separation[start:stop] = total / (n_groups - 1)
        else:
            separation[start:stop] = 0.0
    separation = np.nan_to_num(separation, nan=0.0, posinf=0.0)

    # H4: the shape of each group, as the leading eigenvalue share of its scatter
    shape = np.zeros(len(unique_codes))
    for place, code in enumerate(unique_codes):
        members = members_of[int(code)]
        if len(members) < 3:
            shape[place] = 1.0
            continue
        centred = members - members.mean(axis=0)
        values, _ = np.linalg.eigh(centred.T @ centred)
        values = np.clip(values, 0.0, None)
        shape[place] = float(values[-1] / max(values.sum(), 1e-12))

    def fit(values):
        design = np.column_stack([np.ones(len(radius_values)), values])
        beta, *_ = np.linalg.lstsq(design, radius_values, rcond=None)
        return design @ beta

    predictions = {
        "H1_compactness": fit(tightness_values),
        "H2_separation": fit(separation),
        "H3_size": fit(np.log(sizes)),
        "H4_shape": fit(shape),
    }
    # H5 is a TABLE-LEVEL offset, and a constant per group cancels in the positive
    # minus negative difference. The offset has to be added to the query difference
    # itself, so the two moments are measured outside this function.
    rng = np.random.default_rng(1700 + index)
    full_permutations = [radius_values[rng.permutation(len(radius_values))]
                         for _ in range(FULL_PERMUTATIONS)]
    noise = rng.normal(size=len(radius_values))
    noise = noise / max(noise.std(), 1e-12) * max(radius_values.std(), 1e-12) + radius_values.mean()

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
        positive = np.asarray([values[code_index[int(code)]] for code in code_pos])
        negative = np.asarray([[values[code_index[int(code)]] for code in row]
                               for row in code_neg])
        return (positive[:, None] - negative) / 2.0

    tau = float(context["temperature"])

    def score(delta):
        per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
        return float(1.0 - float(per_query.mean()) / float(np.log(2.0)))

    # the table-level calibration: an offset on the query difference, standardised
    # against the pooled groups so that its size is comparable to the contrast
    try:
        pooled = np.load(HERE.parents[0] / "group_records_107.npz")["radius"]
        pooled_mean, pooled_std = float(pooled.mean()), float(max(pooled.std(), 1e-12))
    except Exception:
        pooled_mean, pooled_std = float(radius_values.mean()), max(float(radius_values.std()), 1e-12)
    offset_mean = (float(radius_values.mean()) - pooled_mean) / pooled_std
    offset_spread = (float(radius_values.std()) - pooled_std) / pooled_std

    series = {}
    radial_raw = term(radius_values)
    for alpha in ALPHAS:
        series["H5_table_offset_mean_%g" % alpha] = score(relative + alpha * offset_mean)
        series["H5_table_offset_spread_%g" % alpha] = score(relative + alpha * offset_spread)
    for alpha in ALPHAS:
        series["radial_raw_%g" % alpha] = score(relative + alpha * radial_raw)
        for name, values in predictions.items():
            series["%s_%g" % (name, alpha)] = score(relative + alpha * term(values))
        series["noise_%g" % alpha] = score(relative + alpha * term(noise))
    series["plain"] = score(relative)
    full_scores = [score(relative + 0.5 * term(values)) for values in full_permutations]
    return index, series, full_scores


def main() -> int:
    with ProcessPoolExecutor(max_workers=16) as pool:
        rows = [row for row in pool.map(work, range(107)) if row[1] is not None]
    rows.sort()
    from tools.research.science import sid_metric_bench as bench

    context = bench.load_context()
    keep = np.asarray([row[0] for row in rows], dtype=np.int64)
    recall = np.asarray([entry["test_recall@10"] for entry in context["pairs"]])[keep]
    families = np.asarray([family_of(context["pairs"][index]["row"]["run_dir"])
                           for index in keep])
    names = sorted({name for row in rows for name in row[1]})
    series = {name: np.asarray([row[1][name] for row in rows]) for name in names}

    report = {"pairs": len(rows),
              "families": {key: int(value) for key, value in Counter(families).items()},
              "spearman": {}}
    for name in names:
        rho, p = stats.spearmanr(series[name], recall)
        report["spearman"][name] = {"spearman": round(float(rho), 6), "p_value": float(p)}

    # P1: the full permutation, averaged over the draws and per draw
    full = np.stack([np.asarray(row[2]) for row in rows])          # tables x draws
    report["full_permutation"] = {}
    for draw in range(full.shape[1]):
        rho, _ = stats.spearmanr(full[:, draw], recall)
        report["full_permutation"].setdefault("per_draw", []).append(round(float(rho), 6))
    per_draw = np.asarray(report["full_permutation"]["per_draw"])
    report["full_permutation"]["mean"] = round(float(per_draw.mean()), 6)
    report["full_permutation"]["p5"] = round(float(np.percentile(per_draw, 5)), 6)
    report["full_permutation"]["p95"] = round(float(np.percentile(per_draw, 95)), 6)
    report["full_permutation"]["share_above_0.74"] = float(np.mean(per_draw > 0.74))

    # the leave-one-family-out protocol, for every candidate
    report["leave_one_family_out"] = {}
    for prefix in ("radial_raw", "H1_compactness", "H2_separation", "H3_size",
                   "H4_shape", "H5_table_offset_mean", "H5_table_offset_spread", "noise"):
        for family in sorted(set(families.tolist())):
            held = families == family
            others = ~held
            if held.sum() < 5:
                continue
            best = max(ALPHAS, key=lambda a: stats.spearmanr(
                series["%s_%g" % (prefix, a)][others], recall[others])[0])
            report["leave_one_family_out"].setdefault(prefix, {})[family] = {
                "chosen_alpha": best,
                "held_out": round(float(stats.spearmanr(
                    series["%s_%g" % (prefix, best)][held], recall[held])[0]), 6),
                "held_out_plain": round(float(stats.spearmanr(
                    series["plain"][held], recall[held])[0]), 6)}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for name in names:
        print("%-26s %+.4f" % (name, report["spearman"][name]["spearman"]))
    print("full permutation: mean %+.4f  p5 %+.4f  p95 %+.4f  share>0.74 %.2f" % (
        report["full_permutation"]["mean"], report["full_permutation"]["p5"],
        report["full_permutation"]["p95"], report["full_permutation"]["share_above_0.74"]))
    print("leave one family out:", json.dumps(report["leave_one_family_out"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
