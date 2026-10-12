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

# The distortion statistics live with the graph side of the study; they are shared
# rather than duplicated so the step-4 and step-5 numbers are computed identically.
import graph_lib as _L

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
    # ||x_i - y_j||^2 by the expansion ||x||^2 + ||y||^2 - 2 x.y, which touches
    # (n, m) and never (n, m, D): the direct (X[:,None,:] - Y[None,:,:]) form
    # materialises a 3-D tensor and dies at 24556 x 24556 x 32. The expansion is the
    # same value up to float cancellation, and it is clamped at 0.
    d2 = (X.pow(2).sum(-1, keepdim=True) + Y.pow(2).sum(-1).unsqueeze(0)
          - 2.0 * (X @ Y.t())).clamp_min(0.0)
    num = 2.0 * c * d2
    # The denominator MUST be formed as (n,1) * (1,m). Written as
    # (1-x2) * (1-y2) it broadcasts (n,1) against (m,1) and silently returns an
    # (m, n) matrix whenever n != m; with n == m the square case hides it.
    den = (1.0 - x2).clamp_min(1e-12) * (1.0 - y2).clamp_min(1e-12).t()
    arg = 1.0 + num / den
    return (torch.acosh(arg.clamp_min(1.0 + 1e-7)) / math.sqrt(c)).clamp_min(1e-6)


def paired_row_distance(kind: str, Xi: torch.Tensor, Yj: torch.Tensor,
                        c: float = C_DEFAULT) -> torch.Tensor:
    """Element-wise distance d(Xi_t, Yj_t): one distance per ROW, not an outer product."""
    if kind == "euclid":
        return (Xi - Yj).pow(2).sum(-1).clamp_min(0.0).sqrt().clamp_min(1e-9)
    xi2 = c * Xi.pow(2).sum(-1)
    yj2 = c * Yj.pow(2).sum(-1)
    num = 2.0 * c * (Xi - Yj).pow(2).sum(-1)
    den = (1.0 - xi2).clamp_min(1e-12) * (1.0 - yj2).clamp_min(1e-12)
    return (torch.acosh((1.0 + num / den).clamp_min(1.0 + 1e-7)) / math.sqrt(c)).clamp_min(1e-6)


def paired_distance(kind: str, X: torch.Tensor, i_idx: np.ndarray, j_idx: np.ndarray,
                    chunk: int = 4096, c: float = C_DEFAULT) -> np.ndarray:
    """Distances between MATCHING index pairs d(X[i_t], X[j_t]), not an outer product.

    The evaluation set holds ~100k pairs. Taking distance(X[i_array], X[j_array])
    there builds a 100k x 100k matrix, 40 GB at float32; only the matching entries
    are wanted, so the pairs are walked in chunks.
    """
    ii = torch.as_tensor(np.asarray(i_idx), dtype=torch.long, device=X.device)
    jj = torch.as_tensor(np.asarray(j_idx), dtype=torch.long, device=X.device)
    out = np.empty(ii.numel(), dtype=np.float64)
    for s0 in range(0, ii.numel(), chunk):
        e0 = min(s0 + chunk, ii.numel())
        Xi = X[ii[s0:e0]]
        Yj = X[jj[s0:e0]]
        # euclid must return the DISTANCE, matching euclidean_pairwise, not its square
        d = (Xi - Yj).pow(2).sum(-1).clamp_min(0.0).sqrt().clamp_min(1e-9) if kind == "euclid" else \
            (torch.acosh((1.0 + 2.0 * c * (Xi - Yj).pow(2).sum(-1)
                          / ((1.0 - c * Xi.pow(2).sum(-1)).clamp_min(1e-12)
                             * (1.0 - c * Yj.pow(2).sum(-1)).clamp_min(1e-12)).t())
             .clamp_min(1.0 + 1e-7)) / math.sqrt(c)).clamp_min(1e-6)
        out[s0:e0] = d.detach().cpu().numpy()
    return out


def topk_neighbours(kind: str, X: torch.Tensor, k: int, chunk: int = 4096,
                    c: float = C_DEFAULT) -> np.ndarray:
    """(N, k) indices of each node's nearest neighbours, computed in row chunks.

    The full (N, N) matrix is never materialised: on a 24556-node graph in float64 it
    is 4.8 GB, and the previous implementation additionally tried to build an
    (N, N, D) intermediate.
    """
    N = X.shape[0]
    best_d = torch.full((N, k), float("inf"), device=X.device, dtype=X.dtype)
    best_i = torch.zeros((N, k), device=X.device, dtype=torch.long)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d = distance(kind, X[s:e], X, c)          # column j is node j
        # A node must not be its own nearest neighbour. The self distance is the
        # floor of the metric, i.e. the SMALLEST possible value, so without this the
        # ranking degenerates to "every node is nearest to itself" and the
        # neighbourhood-preservation metric becomes meaningless.
        rows_local = torch.arange(e - s, device=X.device, dtype=torch.long)
        d[rows_local, rows_local + s] = float("inf")
        col_idx = (torch.arange(N, device=X.device, dtype=torch.long)
                   .unsqueeze(0).expand(e - s, N))
        cat_d = torch.cat([best_d[s:e], d], dim=1)
        cat_i = torch.cat([best_i[s:e], col_idx], dim=1)
        vals, idx = torch.topk(cat_d, k, dim=1, largest=False)
        best_d[s:e] = vals
        best_i[s:e] = torch.gather(cat_i, 1, idx)
    return best_i.cpu().numpy()


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
        # MUST be one distance per training pair. distance(X[a_batch], X[b_batch])
        # is an OUTER PRODUCT here -- (batch, batch) -- and broadcasting it against a
        # (batch,) target silently trains on a different objective; torch only warns.
        d = paired_row_distance(kind, X[tp[idx, 0]], X[tp[idx, 1]], c)
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
    d_raw = paired_distance(kind, X, pairs[:, 0], pairs[:, 1], c=c)
    scale = fit_scale_eval(true_d, d_raw)
    d = d_raw * scale
    from scipy import stats
    out = {
        "n_eval_pairs": int(pairs.shape[0]),
        "fitted_scale": scale,
        "normalised_stress": _L.normalised_stress(true_d, d),
        "mean_relative_distortion": _L.mean_relative_distortion(true_d, d),
        "spearman": float(stats.spearmanr(d, true_d).statistic),
        "radius_p99": float(np.quantile(ball_radii(X, c).cpu().numpy(), 0.99)) if kind == "poincare" else None,
        "radius_median": float(np.median(ball_radii(X, c).cpu().numpy())) if kind == "poincare" else None,
        "share_radius_gt_0.99": float((ball_radii(X, c).cpu().numpy() > 0.99).mean()) if kind == "poincare" else None,
        "all_finite": bool(np.isfinite(d).all()),
        "neighbourhood_preservation": {},
    }
    kmax = max(k_neighbours)
    emb_rank = topk_neighbours(kind, X, kmax, chunk=4096, c=c)
    for k in k_neighbours:
        out["neighbourhood_preservation"][str(k)] = neighbourhood_preservation_from_rank(
            neighbours_true, emb_rank[:, :k], k)
    return out


def grouped_distortion(kind: str, X: torch.Tensor, pairs: np.ndarray, true_d: np.ndarray,
                       groups: dict, key: str, scale: float, c: float = C_DEFAULT,
                       device: str = "cpu") -> dict:
    """Distortion broken down by a per-pair grouping variable."""
    d = paired_distance(kind, X, pairs[:, 0], pairs[:, 1], c=c) * scale
    out = {}
    for name, mask in groups.items():
        m = np.asarray(mask, dtype=bool)
        if m.sum() < 10:
            continue
        out[name] = {
            "n": int(m.sum()),
            "normalised_stress": _L.normalised_stress(true_d[m], d[m]),
            "mean_relative_distortion": _L.mean_relative_distortion(true_d[m], d[m]),
        }
    return out


def neighbourhood_preservation_from_rank(neighbours_true: dict, emb_rank: np.ndarray,
                                         k: int) -> float:
    """Share of each node's k nearest graph neighbours inside its k nearest embedding
    neighbours, given the embedding ranking already computed in chunks."""
    hits = 0
    total = 0
    for i, nb in neighbours_true.items():
        if not nb:
            continue
        total += 1
        hits += len(set(nb[:k]) & set(emb_rank[i].tolist()))
    return float(hits / (total * k)) if total and k else float("nan")


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