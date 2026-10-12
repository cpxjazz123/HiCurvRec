"""Metric-embedding library for RQ4: same budget, same pairs, two geometries.

A pure function of its arguments so the tests can drive it before the real graph is
touched. Two geometries are compared under an IDENTICAL protocol: same node set,
same embedding dimension, same training pairs with the same ground-truth graph
distances, same evaluation pairs, same optimiser, same learning rate, same number of
steps, same batch size, same seed. The only difference is the distance function.

Each geometry learns ONE global positive scale, and it is fitted on EVALUATION pairs
only, never on training pairs, so a pure rescaling cannot flatter either side.
"""
from __future__ import annotations

import math

import numpy as np
import torch

C_DEFAULT = 1.0
BALL_EPS = 1e-5


# ------------------------------------------------------------------------- distances
def euclidean_pairwise(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(X, Y, p=2).clamp_min(1e-9)


def poincare_pairwise(X: torch.Tensor, Y: torch.Tensor, c: float = C_DEFAULT) -> torch.Tensor:
    """Geodesic distance between every row of X and every row of Y.

    Points are assumed already inside the ball; the boundary rescale is applied here so
    a point that drifts out cannot produce a silent nan. The 1 - c||x||^2 denominator
    is floored, and the arcosh argument is floored just above 1.
    """
    cmax = (1.0 - BALL_EPS) / math.sqrt(c)
    X = _inside(X, c, cmax)
    Y = _inside(Y, c, cmax)
    x2 = (c ** 0.5) ** 2 * X.pow(2).sum(-1, keepdim=True)
    y2 = (c ** 0.5) ** 2 * Y.pow(2).sum(-1, keepdim=True)
    num = 2.0 * c * (X[:, None, :] - Y[None, :, :]).pow(2).sum(-1)
    # The denominator MUST be formed as (n,1) * (1,m). Written as
    # (1-x2) * (1-y2) it broadcasts (n,1) against (m,1) and silently returns an
    # (m, n) matrix whenever n != m; with n == m the square case hides it.
    den = (1.0 - x2).clamp_min(1e-12) * (1.0 - y2).clamp_min(1e-12).t()
    arg = 1.0 + num / den
    return (torch.acosh(arg.clamp_min(1.0 + 1e-7)) / math.sqrt(c)).clamp_min(1e-6)


def _inside(X: torch.Tensor, c: float, cmax: float) -> torch.Tensor:
    n = torch.linalg.vector_norm(X, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(cmax / n, max=1.0)
    return X * scale


def distance(kind: str, X: torch.Tensor, Y: torch.Tensor, c: float = C_DEFAULT) -> torch.Tensor:
    if kind == "euclid":
        return euclidean_pairwise(X, Y)
    if kind == "poincare":
        return poincare_pairwise(X, Y, c)
    raise ValueError(kind)


def ball_radii(X: torch.Tensor, c: float = C_DEFAULT) -> torch.Tensor:
    return (c ** 0.5) * torch.linalg.vector_norm(X, dim=-1)


# ---------------------------------------------------------------------------- training
def train_metric_embedding(kind: str, n_nodes: int, dim: int,
                            train_pairs: np.ndarray, train_dists: np.ndarray,
                            *, steps: int, lr: float, batch: int, seed: int,
                            device: str = "cpu", c: float = C_DEFAULT,
                            log_every: int = 500):
    """Train one embedding under the shared protocol. Returns (emb, scale, history)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "euclid":
        init = torch.randn(n_nodes, dim, generator=g) * (1.0 / math.sqrt(dim))
    else:
        # small tangents, then into the ball, so every point starts strictly inside
        init = torch.randn(n_nodes, dim, generator=g) * (0.1 / math.sqrt(dim))
    X = init.to(device).requires_grad_(True)
    raw_scale = torch.zeros(1, device=device, requires_grad=True)   # softplus -> positive

    opt = torch.optim.AdamW([X], lr=lr, weight_decay=0.0)
    opt_s = torch.optim.AdamW([raw_scale], lr=lr, weight_decay=0.0)
    tp = torch.as_tensor(train_pairs, dtype=torch.long, device=device)
    td = torch.as_tensor(train_dists, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    n_pairs = tp.shape[0]
    history = []

    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.choice(n_pairs, size=min(batch, n_pairs), replace=False),
                             dtype=torch.long, device=device)
        opt.zero_grad(set_to_none=True)
        opt_s.zero_grad(set_to_none=True)
        d = distance(kind, X[tp[idx, 0]], X[tp[idx, 1]], c)
        scale = torch.nn.functional.softplus(raw_scale)
        loss = torch.nn.functional.mse_loss(d, scale * td[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([X], 1.0)
        opt.step()
        opt_s.step()
        if log_every and (step % log_every == 0 or step == 1 or step == steps):
            with torch.no_grad():
                r = ball_radii(X.detach(), c) if kind == "poincare" else None
                history.append({
                    "step": step,
                    "loss": float(loss.detach()),
                    "scale": float(torch.nn.functional.softplus(raw_scale).detach()),
                    "median_ball_radius": float(r.median()) if r is not None else None,
                    "share_radius_gt_0.99": float((r > 0.99).float().mean()) if r is not None else None,
                })
    return X.detach(), float(torch.nn.functional.softplus(raw_scale).detach()), history


# --------------------------------------------------------------------------- evaluation
def fit_scale_eval(true_d: np.ndarray, emb_d: np.ndarray) -> float:
    """The single global positive scale, fitted on EVALUATION pairs."""
    return float((emb_d * true_d).sum() / max((emb_d ** 2).sum(), 1e-12))


def evaluate(kind: str, X: torch.Tensor, pairs: np.ndarray, true_d: np.ndarray,
             k_neighbours: tuple, neighbours_true: dict, n_nodes: int,
             c: float = C_DEFAULT, device: str = "cpu") -> dict:
    tp = torch.as_tensor(pairs, dtype=torch.long, device=device)
    d_raw = distance(kind, X[tp[:, 0]], X[tp[:, 1]], c).cpu().numpy()
    scale = fit_scale_eval(true_d, d_raw)
    d = d_raw * scale
    from scipy import stats
    out = {
        "n_eval_pairs": int(pairs.shape[0]),
        "fitted_scale": scale,
        "normalised_stress": normalised_stress(true_d, d),
        "mean_relative_distortion": mean_relative_distortion(true_d, d),
        "spearman": float(stats.spearmanr(d, true_d).statistic),
        "radius_p99": float(np.quantile(ball_radii(X, c).cpu().numpy(), 0.99)) if kind == "poincare" else None,
        "radius_median": float(np.median(ball_radii(X, c).cpu().numpy())) if kind == "poincare" else None,
        "share_radius_gt_0.99": float((ball_radii(X, c).cpu().numpy() > 0.99).mean()) if kind == "poincare" else None,
        "all_finite": bool(np.isfinite(d).all()),
        "neighbourhood_preservation": {},
    }
    full = distance(kind, X, X, c).cpu().numpy()
    np.fill_diagonal(full, np.inf)
    for k in k_neighbours:
        out["neighbourhood_preservation"][str(k)] = neighbourhood_preservation(
            neighbours_true, full, k, n_nodes)
    return out


def grouped_distortion(kind: str, X: torch.Tensor, pairs: np.ndarray, true_d: np.ndarray,
                       groups: dict, key: str, scale: float, c: float = C_DEFAULT,
                       device: str = "cpu") -> dict:
    """Distortion broken down by a per-pair grouping variable."""
    tp = torch.as_tensor(pairs, dtype=torch.long, device=device)
    d = distance(kind, X[tp[:, 0]], X[tp[:, 1]], c).cpu().numpy() * scale
    out = {}
    for name, mask in groups.items():
        m = np.asarray(mask, dtype=bool)
        if m.sum() < 10:
            continue
        out[name] = {
            "n": int(m.sum()),
            "normalised_stress": normalised_stress(true_d[m], d[m]),
            "mean_relative_distortion": mean_relative_distortion(true_d[m], d[m]),
        }
    return out


def mean_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank statistic, no sklearn dependency."""
    # Rank 1 goes to the SMALLEST score, which is the Mann-Whitney convention
    # behind AUC = (R_pos - n_pos(n_pos+1)/2) / (n_pos * n_neg); with that
    # convention the statistic is 1 exactly when every positive outranks every
    # negative, verified against sklearn in the test.
    order = np.argsort(scores)
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    npos = float(labels.sum())
    nneg = float(labels.size - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))