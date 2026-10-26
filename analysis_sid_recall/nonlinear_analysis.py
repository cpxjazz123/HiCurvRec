"""Is the metric -> recall relation non-linear, and does a non-linear shape generalise?

Rank correlation only says "monotone on average". With 67 tables a shape can be checked
honestly by out-of-sample prediction: every candidate shape is fitted on four folds and
scored on the fifth, repeated, so a flexible shape only wins if it really generalises.

Shapes compared per metric
--------------------------
``linear``      ``y = a + b x``
``quadratic``   ``y = a + b x + c x^2``
``log``         ``y = a + b ln(x)`` (positive metrics only)
``knn``         the mean recall of the 7 nearest tables in metric value
``isotonic``    monotone piecewise-constant fit, then linear interpolation

For each metric the table reports the out-of-sample R^2 of every shape, the quintile
profile of the mean recall, and — for the strongest metric — a threshold scan with a
permutation test, because "the effect only exists above some level" is the specific
non-linearity worth detecting.

Input: `sid_behaviour_metrics*.csv` (merged on the variant) and `recall@10`.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOLDS = 5
REPEATS = 20
SEED = 42


def load() -> tuple[list, list]:
    """Merge the three metric batteries on the SID's unique identifier.

    The variant name is not a unique identifier: the 67 SIDs carry only 53
    distinct names (``curvature_RQ-VAE`` alone names 15 different SIDs), so
    merging on the name both drops 14 rows and attaches one SID's metrics to
    another SID's recall. The merge key is therefore the SID's sha256, which
    every battery records.
    """
    merged: dict[str, dict] = {}
    coverage: dict[str, set] = {}
    for name in ("sid_behaviour_metrics.csv", "sid_behaviour_metrics2.csv",
                 "sid_behaviour_metrics3.csv"):
        path = HERE / name
        if not path.is_file():
            continue
        seen: set[str] = set()
        for row in csv.DictReader(path.open()):
            key = (row.get("sid_sha256") or "").strip()
            if not key:
                raise ValueError(
                    f"{name} carries no sid_sha256: the merge key has to be the "
                    "identity of the SID, not its name"
                )
            seen.add(key)
            entry = merged.setdefault(
                key, {"sid_sha256": key, "variant": row.get("variant", "")}
            )
            for field, value in row.items():
                if field in ("variant", "sid_sha256"):
                    continue
                try:
                    entry[field] = float(value)
                except (TypeError, ValueError):
                    pass
        coverage[name] = seen
    if coverage:
        reference = max(coverage, key=lambda key: len(coverage[key]))
        for name, seen in coverage.items():
            missing = coverage[reference] - seen
            if missing:
                raise ValueError(
                    f"{name} does not cover {len(missing)} of the "
                    f"{len(coverage[reference])} SIDs in {reference}"
                )
    rows = list(merged.values())
    # recall@5 is a label, not a predictor: it would be the best "metric" by construction
    labels = {"sid_sha256", "variant", "recall@10", "recall@5"}
    metrics = [key for key in rows[0] if key not in labels]
    metrics = [key for key in metrics
               if sum(1 for row in rows if key in row) >= len(rows) * 0.9]
    return rows, metrics


def r2(predicted, actual) -> float:
    predicted = np.asarray(predicted, float)
    actual = np.asarray(actual, float)
    denominator = float(((actual - actual.mean()) ** 2).sum())
    if denominator <= 0:
        return float("nan")
    return 1.0 - float(((actual - predicted) ** 2).sum()) / denominator


def fit_predict(train_x, train_y, test_x, shape: str):
    order = np.argsort(train_x)
    x, y = np.asarray(train_x, float)[order], np.asarray(train_y, float)[order]
    t = np.asarray(test_x, float)
    if shape == "linear":
        coefficients = np.polyfit(x, y, 1)
        return np.polyval(coefficients, t)
    if shape == "quadratic":
        coefficients = np.polyfit(x, y, 2)
        return np.polyval(coefficients, t)
    if shape == "log":
        coefficients = np.polyfit(np.log(np.clip(x, 1e-9, None)), y, 1)
        return np.polyval(coefficients, np.log(np.clip(t, 1e-9, None)))
    if shape == "knn":
        k = min(7, len(x))
        predicted = []
        for value in t:
            distance = np.abs(x - value)
            nearest = np.argsort(distance)[:k]
            predicted.append(float(y[nearest].mean()))
        return np.asarray(predicted)
    if shape == "isotonic":
        # pool adjacent violators, then interpolate the step function
        blocks = [[x[i], y[i], 1] for i in range(len(x))]
        merged = []
        for block in blocks:
            merged.append(block)
            while len(merged) > 1 and merged[-2][1] > merged[-1][1]:
                left, right = merged[-2], merged[-1]
                total = left[2] + right[2]
                mean = (left[1] * left[2] + right[1] * right[2]) / total
                merged[-2:] = [[left[0], mean, total]]
        centres = np.array([block[0] for block in merged])
        levels = np.array([block[1] for block in merged])
        return np.interp(t, centres, levels, left=levels[0], right=levels[-1])
    raise ValueError(shape)


def cross_validate(x, y, shape: str, *, seed: int = SEED) -> float:
    return float(np.nanmean(cross_validate_repeats(x, y, shape, seed=seed)))


def cross_validate_repeats(x, y, shape: str, *, seed: int = SEED) -> list[float]:
    """The out-of-sample R2 of one repeat: the mean over its five folds."""
    rng = np.random.default_rng(seed)
    index = np.arange(len(x))
    scores = []
    for _ in range(REPEATS):
        rng.shuffle(index)
        folds = np.array_split(index, FOLDS)
        fold_scores = []
        for fold in range(FOLDS):
            test = folds[fold]
            train = np.concatenate([folds[other] for other in range(FOLDS) if other != fold])
            predicted = fit_predict([x[i] for i in train], [y[i] for i in train],
                                    [x[i] for i in test], shape)
            fold_scores.append(r2(predicted, [y[i] for i in test]))
        scores.append(float(np.nanmean(fold_scores)))
    return scores


def paired_difference(x, y, better: str, worse: str) -> dict:
    """Paired per-repeat difference of two shapes, with a 95% interval.

    The two shapes see the same folds, so the repeats are paired observations;
    the interval is the bootstrap of the mean difference.
    """
    better_scores = np.asarray(cross_validate_repeats(x, y, better), float)
    worse_scores = np.asarray(cross_validate_repeats(x, y, worse), float)
    difference = better_scores - worse_scores
    rng = np.random.default_rng(SEED)
    draws = [float(np.mean(rng.choice(difference, size=len(difference), replace=True)))
             for _ in range(2000)]
    return {
        "better": better, "worse": worse,
        "mean_difference": float(np.mean(difference)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "repeats": int(len(difference)),
        "wins": int((difference > 0).sum()),
    }


def quintiles(x, y):
    order = np.argsort(x)
    bins = np.array_split(order, 5)
    return [(float(np.mean(np.asarray(x)[bin])), float(np.mean(np.asarray(y)[bin])))
            for bin in bins if len(bin)]


def threshold_scan(x, y, *, rounds: int = 2000, seed: int = SEED):
    """The best split of the metric, and a permutation test of the recall difference."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    best = (None, 0.0)
    for cut in np.quantile(x, np.linspace(0.15, 0.85, 29)):
        above, below = y[x >= cut], y[x < cut]
        if len(above) < 5 or len(below) < 5:
            continue
        difference = float(above.mean() - below.mean())
        if abs(difference) > abs(best[1]):
            best = (float(cut), difference)
    cut, difference = best
    if cut is None:
        return {"cut": None, "difference": None, "p_value": None}
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(rounds):
        shuffled = rng.permutation(y)
        above, below = shuffled[x >= cut], shuffled[x < cut]
        if abs(float(above.mean() - below.mean())) >= abs(difference):
            extreme += 1
    return {"cut": cut, "difference": difference, "p_value": (extreme + 1) / (rounds + 1)}


def main() -> int:
    rows, metrics = load()
    print("merged %d SIDs on sid_sha256 (%d distinct variant names, so the name "
          "would have lost %d rows)" % (
              len(rows), len({row["variant"] for row in rows}),
              len(rows) - len({row["variant"] for row in rows})))
    y = [row["recall@10"] for row in rows]
    shapes = ("linear", "quadratic", "log", "knn", "isotonic")
    scored = []
    for metric in metrics:
        pairs = [(row[metric], row["recall@10"]) for row in rows if metric in row]
        if len(pairs) < 30 or len({value for value, _ in pairs}) < 5:
            continue
        x = [value for value, _ in pairs]
        target = [value for _, value in pairs]
        scores = {shape: cross_validate(x, target, shape) for shape in shapes}
        best = max(scores, key=lambda shape: scores[shape] if scores[shape] == scores[shape] else -9)
        scored.append({"metric": metric, "n": len(pairs), **scores,
                       "best_shape": best, "best_r2": scores[best],
                       "profile": quintiles(x, target)})
    scored.sort(key=lambda row: -(row["best_r2"] if row["best_r2"] == row["best_r2"] else -9))
    with (HERE / "nonlinear_shapes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["metric", "n", *shapes, "best_shape", "best_r2",
                  "quintile_means", "quintile_recall"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in scored:
            writer.writerow({**{key: row[key] for key in ("metric", "n", *shapes,
                                                          "best_shape", "best_r2")},
                             "quintile_means": json_dumps([a for a, _ in row["profile"]]),
                             "quintile_recall": json_dumps([b for _, b in row["profile"]])})

    print("%-28s %6s %7s %7s %7s %6s %9s" % ("metric", "lin", "quad", "log", "knn", "iso", "best"))
    for row in scored[:14]:
        print("%-28s %6.3f %7.3f %7.3f %7.3f %6.3f  %-9s" % (
            row["metric"], row["linear"], row["quadratic"], row["log"], row["knn"],
            row["isotonic"], row["best_shape"]))
    strongest = scored[0]
    pairs = [(row[strongest["metric"]], row["recall@10"]) for row in rows
             if strongest["metric"] in row]
    scan = threshold_scan([value for value, _ in pairs], [value for _, value in pairs])
    print()
    print("strongest metric:", strongest["metric"], "| best shape:", strongest["best_shape"],
          "| out-of-sample R2: %.3f" % strongest["best_r2"])
    print("quintile profile (metric mean -> recall mean):")
    for mean, recall in strongest["profile"]:
        print("   %8.4f -> %.4f" % (mean, recall))
    print("threshold scan:", json_dumps(scan))
    comparison = paired_difference(
        [value for value, _ in pairs], [value for _, value in pairs],
        "isotonic", "linear",
    )
    print("isotonic vs linear (paired over %d repeats): %+.3f  CI95 [%+.3f, %+.3f]  "
          "(isotonic wins %d/%d)" % (
              comparison["repeats"], comparison["mean_difference"],
              comparison["ci95_low"], comparison["ci95_high"],
              comparison["wins"], comparison["repeats"]))
    return 0


def json_dumps(value) -> str:
    import json
    return json.dumps(value)


if __name__ == "__main__":
    sys.exit(main())
