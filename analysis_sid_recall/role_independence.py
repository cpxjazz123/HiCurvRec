"""Does Role 3 carry information the other five roles do not? Regression with 67 arms.

The owner's next question, and the right one before designing any mechanism around a role: the six
rates are not independent measurements, so a strong marginal correlation does not mean a role adds
anything once the other five are in the model.

    recall = b0 + sum_r b_r R_r + g * collision + error

Everything here is out-of-sample, because 67 arms and seven predictors will fit noise in-sample:

    * repeated 5-fold cross-validation, ten repeats, fixed seed
    * ridge with the penalty chosen by an inner 4-fold CV on the training part only
    * out-of-fold R^2 (pooled inside each repeat) and RMSE, reported as mean +- sd over repeats
    * the marginal contribution of a role is the CV R^2 of the full model minus the CV R^2 without it
    * a permutation test shuffles that role's column to say whether the contribution survives noise

Also reported: the collinearity of the six rates (Pearson and Spearman matrices, the share of variance
on the first principal component, the condition number) and the ridge coefficients with their spread
across folds, so no claim rests on a single unstable fit.

Input is `role_pattern_gain.json`: per archived arm, the six anchor-role satisfaction rates.

    python3 analysis_sid_recall/role_independence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "role_pattern_gain.json"
OUT = HERE / "role_independence.json"
RECALL = "test_recall@10"
REPEATS = 10
OUTER_FOLDS = 5
INNER_FOLDS = 4
PERMUTATIONS = 200
SEED = 42
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)
ROLES = tuple(f"Role {index + 1}" for index in range(6))


def standardise(matrix: np.ndarray, mean: np.ndarray | None = None,
                scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = matrix.mean(axis=0)
    if scale is None:
        scale = matrix.std(axis=0)
        scale[scale == 0] = 1.0
    return (matrix - mean) / scale, mean, scale


def fit_ridge(matrix: np.ndarray, target: np.ndarray, penalty: float) -> tuple[float, np.ndarray]:
    """Ridge on already-centred data; the intercept is the mean of the target."""
    intercept = float(target.mean())
    centre = target - intercept
    gram = matrix.T @ matrix + penalty * np.eye(matrix.shape[1])
    coefficients = np.linalg.solve(gram, matrix.T @ centre)
    return intercept, coefficients


def predict(matrix: np.ndarray, intercept: float, coefficients: np.ndarray) -> np.ndarray:
    return intercept + matrix @ coefficients


def folds(size: int, count: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(size)
    return [order[index::count] for index in range(count)]


def inner_penalty(matrix: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> float:
    best, best_error = LAMBDAS[0], float("inf")
    splits = folds(len(target), INNER_FOLDS, rng)
    for penalty in LAMBDAS:
        errors = []
        for test in splits:
            train = np.setdiff1d(np.arange(len(target)), test)
            scaled, mean, scale = standardise(matrix[train])
            intercept, coefficients = fit_ridge(scaled, target[train], penalty)
            held = (matrix[test] - mean) / scale
            errors.append(np.mean((target[test] - predict(held, intercept, coefficients)) ** 2))
        error = float(np.mean(errors))
        if error < best_error:
            best, best_error = penalty, error
    return best


def cross_validate(matrix: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> dict:
    """Out-of-fold R^2 pooled inside each repeat, then averaged over repeats."""
    scores = []
    predictions_all: list[np.ndarray] = []
    chosen = []
    for _ in range(REPEATS):
        splits = folds(len(target), OUTER_FOLDS, rng)
        predicted = np.full(len(target), np.nan)
        for test in splits:
            train = np.setdiff1d(np.arange(len(target)), test)
            if matrix.shape[1] == 0:
                predicted[test] = float(target[train].mean())
                continue
            penalty = inner_penalty(matrix[train], target[train], rng)
            chosen.append(penalty)
            scaled, mean, scale = standardise(matrix[train])
            intercept, coefficients = fit_ridge(scaled, target[train], penalty)
            held = (matrix[test] - mean) / scale
            predicted[test] = predict(held, intercept, coefficients)
        residual = float(np.sum((target - predicted) ** 2))
        total = float(np.sum((target - target.mean()) ** 2))
        scores.append(1.0 - residual / total)
        predictions_all.append(predicted)
    return {"r2_mean": float(np.mean(scores)), "r2_sd": float(np.std(scores)),
            "penalties": chosen, "predictions": np.mean(predictions_all, axis=0)}


def main() -> int:
    payload = json.loads(SOURCE.read_text())
    rows = payload["rows"]
    recall = np.array([row["recall"] for row in rows])
    collision = np.array([row["l3_collision"] for row in rows])
    rates = np.array([[row["role_rate"][str(role)] for role in range(6)] for row in rows])
    gains = np.array([[row["role_gain"][str(role)] for role in range(6)] for row in rows])
    if np.isnan(rates).any():
        raise SystemExit("a role rate is missing; the independence analysis needs all six")

    print(f"{len(rows)} arms, recall range {recall.min():.6f}..{recall.max():.6f}")

    # ---- collinearity of the six rates ----
    def pearson(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.corrcoef(first, second)[0, 1])

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        out = np.empty(len(values))
        out[order] = np.arange(len(values), dtype=np.float64)
        return out

    spearman_matrix = np.array([[float(np.corrcoef(ranks(rates[:, i]), ranks(rates[:, j]))[0, 1])
                                 for j in range(6)] for i in range(6)])
    pearson_matrix = np.corrcoef(rates, rowvar=False)
    standardised, _, _ = standardise(rates)
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(rates, rowvar=False))
    eigenvalue_share = float(eigenvalues[-1] / eigenvalues.sum())
    condition = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
    off_diagonal = [abs(pearson_matrix[i, j]) for i in range(6) for j in range(i + 1, 6)]
    print("\n=== collinearity of the six role rates ===")
    print("       " + "".join(f"{name:>9}" for name in ROLES))
    for index, name in enumerate(ROLES):
        print(f"  {name:<6}" + "".join(f"{pearson_matrix[index, j]:>+9.3f}" for j in range(6)))
    print(f"  max |rho| off-diagonal {max(off_diagonal):.3f}, mean {np.mean(off_diagonal):.3f}")
    print(f"  first principal component holds {eigenvalue_share:.1%} of the variance, "
          f"condition number {condition:.2f}")

    rng = np.random.default_rng(SEED)
    results = {}
    print("\n=== out-of-sample R^2 (repeated 5-fold CV, ridge with an inner penalty) ===")
    print(f"  {'model':<28}{'CV R^2':>10}{'sd':>8}")
    for role in range(6):
        score = cross_validate(rates[:, [role]], recall, rng)
        results[f"only:{ROLES[role]}"] = {key: value for key, value in score.items()
                                          if key != "predictions"}
        print(f"  {'only ' + ROLES[role]:<28}{score['r2_mean']:>+10.4f}{score['r2_sd']:>8.4f}")
    constant = cross_validate(np.zeros((len(rows), 0)), recall, rng)
    print(f"  {'intercept only':<28}{constant['r2_mean']:>+10.4f}{constant['r2_sd']:>8.4f}")
    results["intercept_only"] = {key: value for key, value in constant.items()
                                 if key != "predictions"}

    full = cross_validate(rates, recall, rng)
    print(f"  {'all six':<28}{full['r2_mean']:>+10.4f}{full['r2_sd']:>8.4f}")
    results["all_six"] = {key: value for key, value in full.items() if key != "predictions"}

    with_collision = cross_validate(np.column_stack([rates, collision]), recall, rng)
    print(f"  {'all six + collision':<28}{with_collision['r2_mean']:>+10.4f}"
          f"{with_collision['r2_sd']:>8.4f}")
    results["all_six_plus_collision"] = {key: value for key, value in with_collision.items()
                                         if key != "predictions"}

    full_gains = cross_validate(gains, recall, rng)
    print(f"  {'all six gains':<28}{full_gains['r2_mean']:>+10.4f}{full_gains['r2_sd']:>8.4f}")
    results["all_six_gains"] = {key: value for key, value in full_gains.items()
                                if key != "predictions"}

    # ---- leave one role out: the marginal contribution of each role ----
    print("\n=== marginal contribution: CV R^2(all six) - CV R^2(without that role) ===")
    print(f"  {'role removed':<22}{'without':>10}{'drop':>10}{'p(permutation)':>17}")
    marginal = {}
    for role in range(6):
        keep = [index for index in range(6) if index != role]
        without = cross_validate(rates[:, keep], recall, rng)
        drop = full["r2_mean"] - without["r2_mean"]
        # permutation test: the role's column becomes noise, so a real contribution should vanish
        shuffle_rng = np.random.default_rng(SEED + 101 * (role + 1))
        null_drops = []
        for _ in range(PERMUTATIONS):
            shuffled = rates.copy()
            shuffled[:, role] = shuffle_rng.permutation(rates[:, role])
            permuted_full = cross_validate(shuffled, recall, rng)
            null_drops.append(permuted_full["r2_mean"] - without["r2_mean"])
        p_value = (np.sum(np.array(null_drops) >= drop) + 1) / (PERMUTATIONS + 1)
        marginal[ROLES[role]] = {"without_r2": without["r2_mean"], "drop": drop,
                                 "p_permutation": float(p_value)}
        print(f"  {ROLES[role]:<22}{without['r2_mean']:>+10.4f}{drop:>+10.4f}"
              f"{p_value:>17.3f}")

    # ---- ridge coefficients of the full model, with their spread across folds ----
    scaled, mean, scale = standardise(rates)
    penalty = inner_penalty(scaled, recall, rng)
    intercept, coefficients = fit_ridge(scaled, recall, penalty)
    fold_coefficients = []
    for _ in range(REPEATS):
        for test in folds(len(recall), OUTER_FOLDS, rng):
            train = np.setdiff1d(np.arange(len(recall)), test)
            inner_scaled, inner_mean, inner_scale = standardise(rates[train])
            inner_penalty_value = inner_penalty(inner_scaled, recall[train], rng)
            _, inner_coefficients = fit_ridge(inner_scaled, recall[train], inner_penalty_value)
            fold_coefficients.append(inner_coefficients)
    fold_coefficients = np.array(fold_coefficients)
    print(f"\n=== ridge coefficients on standardised rates (penalty {penalty:g}) ===")
    print(f"  {'role':<10}{'coefficient':>13}{'sd across folds':>17}{'|coef|/sd':>11}")
    coefficient_table = {}
    for index, name in enumerate(ROLES):
        spread = float(fold_coefficients[:, index].std())
        ratio = abs(float(coefficients[index])) / spread if spread else float("inf")
        coefficient_table[name] = {"coefficient": float(coefficients[index]),
                                   "sd_across_folds": spread, "ratio": ratio}
        print(f"  {name:<10}{coefficients[index]:>+13.5f}{spread:>17.5f}{ratio:>11.2f}")

    result = {"n_arms": len(rows), "repeats": REPEATS, "outer_folds": OUTER_FOLDS,
              "inner_folds": INNER_FOLDS, "permutations": PERMUTATIONS,
              "collinearity": {"pearson": pearson_matrix.tolist(),
                               "spearman": spearman_matrix.tolist(),
                               "max_abs_off_diagonal": max(off_diagonal),
                               "mean_abs_off_diagonal": float(np.mean(off_diagonal)),
                               "first_component_share": eigenvalue_share,
                               "condition_number": condition},
              "models": results, "marginal": marginal, "coefficients": coefficient_table}
    OUT.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {OUT.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
