"""Statistics, matched comparison and figures for Issue #129 (WP2).

Pure numpy on purpose: the analysis has to run under the project interpreter
without extra packages, and every number has to be recomputable from the CSVs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ univariate
def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    # average ties
    sorted_values = values[order]
    start = 0
    for position in range(1, values.size + 1):
        if position == values.size or sorted_values[position] != sorted_values[start]:
            if position - start > 1:
                ranks[order[start:position]] = (start + position - 1) / 2.0
            start = position
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx, ry = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
    denominator = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denominator) if denominator else float("nan")


def kendall(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    concordant = discordant = 0
    for i in range(n):
        dx = np.sign(x[i + 1:] - x[i])
        dy = np.sign(y[i + 1:] - y[i])
        concordant += int((dx * dy > 0).sum())
        discordant += int((dx * dy < 0).sum())
    pairs = n * (n - 1) / 2.0
    return float((concordant - discordant) / pairs) if pairs else float("nan")


def permutation_p(x: np.ndarray, y: np.ndarray, observed: float,
                  trials: int = 5000, seed: int = 20261102) -> float:
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(trials):
        shuffled = rng.permutation(y)
        value = spearman(x, shuffled)
        if abs(value) >= abs(observed) - 1e-12:
            extreme += 1
    return (extreme + 1) / (trials + 1)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, trials: int = 2000,
                 seed: int = 20261102) -> tuple:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(trials):
        index = rng.integers(0, x.size, x.size)
        value = spearman(x[index], y[index])
        if value == value:
            values.append(value)
    if not values:
        return (None, None)
    values.sort()
    return (values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1])


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for position, index in enumerate(order):
        value = (len(pvalues) - position) * pvalues[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


# ----------------------------------------------------------------- multivariate
def fit_ols(X: np.ndarray, y: np.ndarray) -> dict:
    design = np.column_stack([np.ones(X.shape[0]), X])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residual ** 2).sum()) / total if total > 0 else float("nan")
    return {"coefficients": coefficients.tolist(), "r2": r2,
            "residual_variance": float((residual ** 2).sum() / max(y.size - design.shape[1], 1))}


def leave_one_series_out(X: np.ndarray, y: np.ndarray, series: np.ndarray) -> dict:
    """Out-of-series predictions; every untested row stays NaN."""
    predictions = np.full(y.size, np.nan)
    for name in np.unique(series):
        test = series == name
        if test.sum() == y.size or test.sum() == 0:
            continue
        train = ~test
        if X[train].shape[0] <= X.shape[1] + 1:
            continue
        model = fit_ols(X[train], y[train])
        design = np.column_stack([np.ones(test.sum()), X[test]])
        predictions[test] = design @ np.asarray(model["coefficients"])
    usable = np.isfinite(predictions)
    if usable.sum() < 3:
        return {"n": int(usable.sum()), "rmse": None, "predictions": predictions.tolist()}
    rmse = float(np.sqrt(((predictions[usable] - y[usable]) ** 2).mean()))
    return {"n": int(usable.sum()), "rmse": rmse, "predictions": predictions.tolist()}


def cluster_bootstrap(X: np.ndarray, y: np.ndarray, series: np.ndarray,
                      trials: int = 1000, seed: int = 20261102) -> dict:
    rng = np.random.default_rng(seed)
    names = np.unique(series)
    draws = {index: [] for index in range(X.shape[1] + 1)}
    for _ in range(trials):
        picked = rng.choice(names, names.size, replace=True)
        index = np.concatenate([np.nonzero(series == name)[0] for name in picked])
        if index.size <= X.shape[1] + 1:
            continue
        model = fit_ols(X[index], y[index])
        for position, value in enumerate(model["coefficients"]):
            draws[position].append(value)
    out = {}
    for position, values in draws.items():
        if not values:
            continue
        values = np.asarray(values)
        out[str(position)] = {"mean": float(values.mean()),
                              "ci95": [float(np.quantile(values, 0.025)),
                                       float(np.quantile(values, 0.975))]}
    return out


# ---------------------------------------------------------------------- figures
PALETTE = ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6"]


def scatter_svg(points: list[dict], x_field: str, path: Path,
                x_label: str, y_label: str = "Recall@10") -> None:
    """A self-contained SVG scatter, one panel per level, coloured by series."""
    width, height = 960, 340
    panels = [("1", 20), ("2", 340), ("3", 660)]
    series = sorted({point["series"] for point in points})
    colour = {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(series)}
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
              f'font-family="inherit">']
    ys = [point["recall"] for point in points]
    y_lo, y_hi = min(ys), max(ys)
    pad = (y_hi - y_lo) * 0.1 or 1e-3
    for level, left in panels:
        values = [point.get(f"{x_field}{level}") for point in points]
        usable = [(value, point["recall"], point["series"], point.get(f"{x_field}_status{level}", "MEASURED"))
                  for value, point in zip(values, points) if value is not None]
        pieces.append(f'<text x="{left + 130}" y="18" font-size="13" text-anchor="middle">'
                      f'level {level}</text>')
        if not usable:
            pieces.append(f'<text x="{left + 130}" y="180" font-size="12" '
                          f'text-anchor="middle" fill="var(--muted)">no defined values</text>')
            continue
        xs = [item[0] for item in usable]
        x_lo, x_hi = min(xs), max(xs)
        span_x = (x_hi - x_lo) or 1e-3
        pieces.append(f'<rect x="{left}" y="30" width="260" height="270" fill="none" '
                      f'stroke="var(--border)"/>')
        for value, recall, name, _status in usable:
            cx = left + 10 + (value - x_lo) / span_x * 240
            cy = 30 + 260 - (recall - (y_lo - pad)) / ((y_hi + pad) - (y_lo - pad)) * 260
            pieces.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.4" '
                          f'fill="var({colour[name]})"/>')
        pieces.append(f'<text x="{left}" y="316" font-size="10" fill="var(--muted)">'
                      f'{x_lo:+.3f}</text>')
        pieces.append(f'<text x="{left + 230}" y="316" font-size="10" fill="var(--muted)">'
                      f'{x_hi:+.3f}</text>')
        pieces.append(f'<text x="{left + 130}" y="334" font-size="10" text-anchor="middle" '
                      f'fill="var(--muted)">{x_label}</text>')
        pieces.append(f'<text x="{left - 6}" y="180" font-size="10" text-anchor="middle" '
                      f'fill="var(--muted)" transform="rotate(-90 {left - 6} 180)">'
                      f'{y_label}</text>')
    pieces.append("</svg>")
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")
