"""Leave-One-Family-Out evaluation of Stage 2 predictors for issue #47.

Linear Ridge only, four pre-registered feature sets, targets are the frozen Stage 3
Recall@10 (primary) and NDCG@10 (secondary).

    M0  log(L1 Hard ER)
    M1  M0 + L1 Gap
    M2  M0 + L1 effective usage ratio + 3-token collision
    M3  M1 + both structural terms

Every one of M0-M3 is fitted inside the SAME outer folds, so their OOF predictions are
directly comparable. Standardisation statistics come from the training folds only, and
the Ridge penalty is chosen by grouped CV over the training families alone. No outer
test family ever influences scaling, the alpha grid, or feature selection.

This module deliberately contains no model-family knowledge: nothing here branches on
which family is being held out, and there is no threshold search.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
CV_SEED = 20240717

FEATURE_SETS: dict[str, list[str]] = {
    "M0": ["log_er_l1"],
    "M1": ["log_er_l1", "gap_l1"],
    "M2": ["log_er_l1", "eff_usage_ratio_l1", "collision_l3"],
    "M3": ["log_er_l1", "gap_l1", "eff_usage_ratio_l1", "collision_l3"],
}


def assemble_matrix(
    rows: Sequence[dict[str, Any]], features: Sequence[str]
) -> np.ndarray:
    """(n_models, n_features) in the caller's feature order."""
    return np.array(
        [[float(row[f]) for f in features] for row in rows], dtype=float
    )


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def _fit_ridge(
    X: np.ndarray, y: np.ndarray, alpha: float
) -> Ridge:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    return model


def _choose_alpha(
    X: np.ndarray, y: np.ndarray, families: Sequence[str], seed: int = CV_SEED
) -> tuple[float, list[dict[str, Any]]]:
    """Grouped CV over training families only.

    With few families the inner split is itself leave-one-family-out; that is the
    strongest grouping available and it never touches an outer test family.
    """
    uniq = sorted(set(families))
    trace: list[dict[str, Any]] = []
    if len(uniq) < 2:
        return ALPHA_GRID[1], trace
    scores = {a: [] for a in ALPHA_GRID}
    for held in uniq:
        mask = np.array([f != held for f in families])
        if mask.sum() < 2 or (~mask).sum() < 1:
            continue
        for a in ALPHA_GRID:
            m = _fit_ridge(X[mask], y[mask], a)
            err = float(np.mean((y[~mask] - m.predict(X[~mask])) ** 2))
            scores[a].append(err)
            trace.append({"alpha": a, "held_family": held, "inner_mse": err})
    means = {a: float(np.mean(v)) for a, v in scores.items() if v}
    best = min(means, key=means.get) if means else ALPHA_GRID[1]
    return best, trace


def lofo_predictions(
    rows: Sequence[dict[str, Any]], target: str, features: Sequence[str], seed: int = CV_SEED
) -> dict[str, Any]:
    X = assemble_matrix(rows, features)
    y = np.array([float(r[target]) for r in rows], dtype=float)
    families = [r["family"] for r in rows]
    uniq = sorted(set(families))

    preds = np.full(len(rows), np.nan)
    alphas: list[dict[str, Any]] = []
    for held in uniq:
        test_mask = np.array([f == held for f in families])
        train_mask = ~test_mask
        X_tr_raw, X_te_raw = X[train_mask], X[test_mask]
        X_tr, X_te = _standardise(X_tr_raw, X_te_raw)
        alpha, trace = _choose_alpha(X_tr, y[train_mask], [f for f, t in zip(families, train_mask) if t], seed)
        model = _fit_ridge(X_tr, y[train_mask], alpha)
        preds[test_mask] = model.predict(X_te)
        alphas.append({
            "held_family": held,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "alpha": alpha,
            "inner_trace": trace,
        })
    return {
        "preds": preds,
        "y": y,
        "families": families,
        "folds": alphas,
        "n_folds": len(uniq),
    }


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    err = p - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rho = spearmanr(p, y).statistic if len(y) > 2 else float("nan")
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "oof_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "oof_spearman": float(rho) if rho == rho else None,
        "n": int(len(y)),
    }


def topk_selection(
    y: np.ndarray, p: np.ndarray, family: Sequence[str], k: int = 10
) -> dict[str, Any]:
    """Model-screening ability: how many of the true top-k models the predictor surfaces.

    This is a property of the Stage 2 proxy used to rank candidate models, NOT the
    recommender's Recall@10.
    """
    true_top = set(np.argsort(-y)[:k].tolist())
    idx = list(np.argsort(-p)[:k])
    hits = len(true_top.intersection(idx))
    per_family: dict[str, int] = {}
    for i in idx:
        f = family[i]
        per_family[f] = per_family.get(f, 0) + (1 if i in true_top else 0)
    return {
        "precision_at_k": hits / k,
        "hits": hits,
        "k": k,
        "true_top_families": {f: int(sum(1 for i in true_top if family[i] == f)) for f in sorted(set(f for i in true_top))},
        "selected_families": per_family,
    }


def family_cluster_bootstrap_delta_mae(
    err_a: np.ndarray,
    err_b: np.ndarray,
    families: Sequence[str],
    delta: np.ndarray,
    replicates: int = 2000,
    seed: int = CV_SEED,
) -> dict[str, Any]:
    """CI for MAE(a) - MAE(b) with whole families resampled as clusters."""
    uniq = sorted(set(families))
    idx_by_family = {f: np.flatnonzero(np.array([g == f for g in families])) for f in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(replicates)
    for b in range(replicates):
        picks = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_family[p] for p in picks])
        stats[b] = np.abs(err_a[idx]).mean() - np.abs(err_b[idx]).mean()
    point = float(np.abs(err_a).mean() - np.abs(err_b).mean())
    lo, hi = float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))
    return {
        "delta_mae": point,
        "ci95": [lo, hi],
        "lower_bound_gt_zero": bool(lo > 0.0),
        "replicates": replicates,
        "n_families": len(uniq),
    }
