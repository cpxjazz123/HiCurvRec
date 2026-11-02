"""Corrected inference for Issue #137.

Fixes the two defects of #136's statistics:
* B1 -- the independence null is a uniform random permutation of all n outcome
  indices drawn WITHOUT replacement (a bijection), not a bootstrap resample; the
  Monte Carlo p is ``(1 + #{|stat_perm| >= |stat_obs|}) / (B + 1)``.
* B2 -- Holm is applied per coefficient column, so ``holm_pearson`` and
  ``holm_spearman`` are separate adjusted p-values over the declared family.

Bootstrap percentile intervals remain a WITH-REPLACEMENT paired resampling and
are reported in their own columns; they are not p-values.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

PERMUTATION_DRAWS = 20000
BOOTSTRAP_DRAWS = 5000
EXHAUSTIVE_MAX_N = 8


# ------------------------------------------------------------------ ranks
def midranks(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks of a 1-D sample."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    for position in range(1, values.size + 1):
        if position == values.size or ordered[position] != ordered[start]:
            ranks[order[start:position]] = (start + position - 1) / 2.0
            start = position
    return ranks


def ranks2(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks along the last axis, vectorised over leading rows."""
    values = np.asarray(values, dtype=float)
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    rows, n = flat.shape
    order = np.argsort(flat, axis=1, kind="stable")
    ordered = np.take_along_axis(flat, order, axis=1)
    new_group = np.ones(flat.shape, dtype=bool)
    new_group[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    positions = np.broadcast_to(np.arange(n), flat.shape)
    starts = np.maximum.accumulate(np.where(new_group, positions, 0), axis=1)
    group_id = np.cumsum(new_group, axis=1) - 1
    counts = np.bincount((group_id + (np.arange(rows) * (n + 1))[:, None]).ravel(),
                         minlength=rows * (n + 1)).reshape(rows, n + 1)
    sizes = np.take_along_axis(counts, group_id, axis=1).astype(float)
    average = starts.astype(float) + (sizes - 1.0) / 2.0
    ranks = np.empty_like(average)
    np.put_along_axis(ranks, order, average, axis=1)
    return ranks.reshape(shape)


# ------------------------------------------------------------ coefficients
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = math.sqrt(float((xc * xc).sum() * (yc * yc).sum()))
    return float((xc * yc).sum() / denominator) if denominator else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan")
    return pearson(midranks(x), midranks(y))


# ------------------------------------------------------------- permutation
def permutation_bijections(n: int, draws: int, seed: int = 0) -> np.ndarray:
    """``draws`` independent uniform permutations of ``0..n-1`` (no replacement)."""
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((draws, n)), axis=1, kind="stable")


def _correlations(chunk: np.ndarray, centred_reference: np.ndarray,
                  denominator_reference: float) -> np.ndarray:
    centre = chunk - chunk.mean(axis=1, keepdims=True)
    denominator = np.sqrt((centre ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (centred_reference @ centre.T) / (denominator_reference * denominator)


def permutation_p(x: np.ndarray, y: np.ndarray, draws: int = PERMUTATION_DRAWS,
                  seed: int = 0, chunk: int = 500) -> dict:
    """Two-sided Monte Carlo p for Pearson and Spearman under index permutation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3:
        raise ValueError("the permutation null needs paired samples of size >= 3")
    observed_r = pearson(x, y)
    observed_rho = spearman(x, y)
    if not (np.isfinite(observed_r) and np.isfinite(observed_rho)):
        return {"draws": 0, "pearson": None, "spearman": None,
                "status": "UNDEFINED_CONSTANT"}
    rank_x = midranks(x)
    rank_y = midranks(y)
    centred_x = x - x.mean()
    denominator_x = float(np.sqrt((centred_x ** 2).sum()))
    centred_rank_x = rank_x - rank_x.mean()
    denominator_rank_x = float(np.sqrt((centred_rank_x ** 2).sum()))
    exceed_r = exceed_rho = 0
    total = 0
    rng = np.random.default_rng(seed)
    while total < draws:
        size = min(chunk, draws - total)
        picks = np.argsort(rng.random((size, x.size)), axis=1, kind="stable")
        values = _correlations(y[picks], centred_x, denominator_x)
        exceed_r += int((np.abs(values) >= abs(observed_r) - 1e-12).sum())
        rank_values = _correlations(rank_y[picks], centred_rank_x, denominator_rank_x)
        exceed_rho += int((np.abs(rank_values) >= abs(observed_rho) - 1e-12).sum())
        total += size
    return {"draws": int(draws), "exhaustive": False,
            "pearson": (exceed_r + 1) / (draws + 1),
            "spearman": (exceed_rho + 1) / (draws + 1),
            "p_resolution": 1.0 / (draws + 1),
            "observed": {"pearson": observed_r, "spearman": observed_rho},
            "status": "DEFINED"}


def exact_permutation_p(x: np.ndarray, y: np.ndarray) -> dict:
    """Exhaustive enumeration of all n! bijections (n <= 8)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    if n > EXHAUSTIVE_MAX_N:
        raise ValueError(f"exhaustive enumeration is limited to n <= {EXHAUSTIVE_MAX_N}")
    observed_r = pearson(x, y)
    observed_rho = spearman(x, y)
    rank_y = midranks(y)
    rank_x = midranks(x)
    exceed_r = exceed_rho = 0
    total = 0
    for permutation in itertools.permutations(range(n)):
        total += 1
        permuted = np.asarray(permutation)
        value = pearson(x, y[permuted])
        if abs(value) >= abs(observed_r) - 1e-12:
            exceed_r += 1
        value = pearson(rank_x, rank_y[permuted])
        if abs(value) >= abs(observed_rho) - 1e-12:
            exceed_rho += 1
    return {"permutations": total, "exhaustive": True,
            "pearson": exceed_r / total, "spearman": exceed_rho / total,
            "observed": {"pearson": observed_r, "spearman": observed_rho},
            "p_resolution": 1.0 / total, "status": "DEFINED"}


# ---------------------------------------------------------------- bootstrap
def bootstrap_ci(x: np.ndarray, y: np.ndarray, draws: int = BOOTSTRAP_DRAWS,
                 seed: int = 0, chunk: int = 500) -> dict:
    """Paired bootstrap percentile CI WITH replacement for both coefficients."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3:
        raise ValueError("the bootstrap needs paired samples of size >= 3")
    rng = np.random.default_rng(seed)
    values_r, values_rho = [], []
    collected = 0
    while collected < draws:
        size = min(chunk, draws - collected)
        picks = rng.integers(0, x.size, (size, x.size))
        sample_x, sample_y = x[picks], y[picks]
        centre_x = sample_x - sample_x.mean(axis=1, keepdims=True)
        centre_y = sample_y - sample_y.mean(axis=1, keepdims=True)
        denominator = np.sqrt((centre_x ** 2).sum(axis=1)) * np.sqrt(
            (centre_y ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            values_r.append((centre_x * centre_y).sum(axis=1) / denominator)
        rank_x = ranks2(sample_x)
        rank_y = ranks2(sample_y)
        rank_x = rank_x - rank_x.mean(axis=1, keepdims=True)
        rank_y = rank_y - rank_y.mean(axis=1, keepdims=True)
        denominator = np.sqrt((rank_x ** 2).sum(axis=1)) * np.sqrt(
            (rank_y ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            values_rho.append((rank_x * rank_y).sum(axis=1) / denominator)
        collected += size
    r_values = np.concatenate(values_r)
    rho_values = np.concatenate(values_rho)
    return {"draws": int(draws), "resampling": "with_replacement",
            "pearson": [float(np.nanquantile(r_values, 0.025)),
                        float(np.nanquantile(r_values, 0.975))],
            "spearman": [float(np.nanquantile(rho_values, 0.025)),
                         float(np.nanquantile(rho_values, 0.975))]}


# --------------------------------------------------------------------- Holm
def holm(pvalues) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(values.size)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (values.size - position) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def holm_columns(p_pearson, p_spearman) -> tuple[list[float], list[float]]:
    """Holm within each coefficient column; no cross-column reuse."""
    if len(p_pearson) != len(p_spearman):
        raise ValueError("the two coefficient columns must cover the same tests")
    return holm(p_pearson), holm(p_spearman)


def family_counts(potential_metric_levels: int, outcomes: int,
                  defined_metric_levels: int = 0) -> dict:
    """66 metric-level entries x 4 outcomes x 2 coefficients = 528 potential tests."""
    potential = potential_metric_levels * outcomes * 2
    if not defined_metric_levels:
        return {"potential": potential, "defined": potential, "undefined": 0}
    defined_entries = defined_metric_levels * outcomes
    defined = defined_entries * 2
    return {"potential": potential, "defined_entries": defined_entries,
            "defined": defined, "undefined": potential - defined}


# -------------------------------------------------------- secondary models
def residualise(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(values.size), controls])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def partial_correlation(x, y, controls, rank: bool = False,
                        draws: int = 5000, seed: int = 0) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]
    if rank:
        x = midranks(x)
        y = midranks(y)
        controls = np.column_stack([midranks(controls[:, index])
                                    for index in range(controls.shape[1])])
    reference = residualise(x, controls)
    residual = residualise(y, controls)
    value = pearson(reference, residual)
    observed = abs(value)
    if not np.isfinite(value):
        return {"n": int(x.size), "value": None, "status": "UNDEFINED_CONSTANT"}
    rng = np.random.default_rng(seed)
    exceed = 0
    total = 0
    while total < draws:
        size = min(500, draws - total)
        picks = rng.random((size, reference.size)).argsort(axis=1)
        sample = residual[picks]
        centre = sample - sample.mean(axis=1, keepdims=True)
        denominator = np.sqrt((centre ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            values = (reference @ centre.T) / (np.sqrt((reference ** 2).sum())
                                               * denominator)
        exceed += int((np.abs(values) >= observed - 1e-12).sum())
        total += size
    return {"n": int(x.size), "value": float(value), "p": (exceed + 1) / (draws + 1),
            "draws": int(draws), "rank": bool(rank), "status": "DEFINED"}


def vif(matrix: np.ndarray) -> list[float]:
    """Variance inflation factors for the columns of a design matrix."""
    matrix = np.asarray(matrix, dtype=float)
    values = []
    for column in range(matrix.shape[1]):
        others = np.delete(matrix, column, axis=1)
        design = np.column_stack([np.ones(others.shape[0]), others])
        beta, *_ = np.linalg.lstsq(design, matrix[:, column], rcond=None)
        fitted = design @ beta
        residual = matrix[:, column] - fitted
        total = float(((matrix[:, column] - matrix[:, column].mean()) ** 2).sum())
        values.append(1.0 / max(1e-12, 1.0 - float((residual ** 2).sum()) / total))
    return values


def cooks_distance(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(design.shape[0]), design])
    beta, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design,
                         np.linalg.pinv(design.T @ design), design)
    p = design.shape[1]
    mse = float((residual ** 2).sum()) / max(1, design.shape[0] - p)
    return (residual ** 2 / (p * mse)) * (leverage / np.maximum(1 - leverage, 1e-12))
