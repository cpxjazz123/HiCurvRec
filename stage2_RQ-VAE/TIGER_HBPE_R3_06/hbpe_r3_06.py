"""HBPE-R3-06 (#42): Encoder gradient budget for ER auxiliary objective.

Adaptive ER weight per step:
g_base = norm(grad(L_base + w_b * L_resample, encoder_params))
g_er = norm(grad(L_ER1, encoder_params))
lambda_eff = min(0.010, 0.10 * g_base / (g_er + 1e-12))
Total loss = L_base + w_b * L_resample + lambda_eff * ramp * L_ER1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

LAMBDA_BASE_MAX = 0.010


def _compute_squared_distances(z: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    z_sq = (z ** 2).sum(-1, keepdim=True)
    c_sq = (codebook ** 2).sum(-1)
    d = z_sq + c_sq - 2.0 * torch.matmul(z, codebook.T)
    return d.clamp_min(0.0)


def _compute_reference_gap(z: torch.Tensor, codebook: torch.Tensor) -> float:
    with torch.no_grad():
        d = _compute_squared_distances(z, codebook.detach())
        top2, _ = torch.topk(d, k=2, dim=-1, largest=False)
        gaps = top2[:, 1] - top2[:, 0]
        return float(torch.median(gaps).item())


def _soft_prefix_distribution(
    z: torch.Tensor, codebook: torch.Tensor, temperature: float
) -> torch.Tensor:
    c_det = codebook.detach()
    d = _compute_squared_distances(z, c_det)
    return F.softmax(-d / max(float(temperature), 1e-8), dim=-1)


def hbpe_l1_loss(
    p1_anchor: torch.Tensor,
    p1_pos: torch.Tensor,
    p1_neg: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    s1_pos = (p1_anchor * p1_pos).sum(-1)
    S1_pos = s1_pos.mean()
    s1_neg = (p1_anchor.unsqueeze(1) * p1_neg).sum(-1)
    S1_neg = s1_neg.mean()

    loss = -(torch.log(S1_pos + eps) - torch.log(S1_neg + eps))

    with torch.no_grad():
        stats = {
            "L_A1": float(loss.detach().item()),
            "S1_pos": float(S1_pos.detach().item()),
            "S1_neg": float(S1_neg.detach().item()),
            "Soft_ER1": float((S1_pos / (S1_neg + eps)).item()),
        }
    return loss, stats


def compute_adaptive_er_weight(
    loss_main: torch.Tensor,
    loss_er: torch.Tensor,
    encoder_params: tuple[torch.nn.Parameter, ...],
    max_lambda: float = LAMBDA_BASE_MAX,
    budget_ratio: float = 0.10,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """Compute adaptive lambda_eff = min(max_lambda, budget_ratio * ||g_base|| / ||g_er||)."""
    with torch.enable_grad():
        g_base_list = torch.autograd.grad(
            loss_main, encoder_params, retain_graph=True, allow_unused=True
        )
        g_er_list = torch.autograd.grad(
            loss_er, encoder_params, retain_graph=True, allow_unused=True
        )

    flat_base = [g.reshape(-1) for g in g_base_list if g is not None]
    flat_er = [g.reshape(-1) for g in g_er_list if g is not None]

    if not flat_base or not flat_er:
        return 0.0, 0.0, 0.0

    g_base_norm = float(torch.cat(flat_base).norm().item())
    g_er_norm = float(torch.cat(flat_er).norm().item())

    if not np.isfinite(g_base_norm) or g_base_norm <= 0.0 or not np.isfinite(g_er_norm):
        return 0.0, g_base_norm, g_er_norm

    lambda_eff = min(max_lambda, budget_ratio * g_base_norm / (g_er_norm + eps))
    return float(lambda_eff), g_base_norm, g_er_norm


def _precompute_matched_negatives(
    pairs: np.ndarray,
    n_items: int,
    seed: int = 42,
    k_neg: int = 10,
) -> np.ndarray:
    n_pairs = len(pairs)
    pop = np.bincount(pairs[:, 1], minlength=n_items)
    quantiles = np.quantile(pop[pop > 0], np.linspace(0, 1, 11))
    quantiles[-1] += 1
    buckets = np.clip(np.digitize(pop, quantiles) - 1, 0, 9)

    bucket_lists = [np.where(buckets == b)[0] for b in range(10)]
    bucket_sizes = np.array([len(l) for l in bucket_lists])
    bucket_arr = np.full((10, int(bucket_sizes.max())), -1, dtype=np.int64)
    for b in range(10):
        bucket_arr[b, : bucket_sizes[b]] = bucket_lists[b]

    succ_mat = csr_matrix(
        (np.ones(n_pairs, dtype=bool), (pairs[:, 0], pairs[:, 1])),
        shape=(n_items, n_items),
        dtype=bool,
    )
    succ_mat.sum_duplicates()
    succ_sets = [
        set(succ_mat.indices[succ_mat.indptr[i] : succ_mat.indptr[i + 1]])
        for i in range(n_items)
    ]

    rng = np.random.default_rng(seed)
    anc_buckets = buckets[pairs[:, 0]]
    draws = rng.integers(0, bucket_sizes[anc_buckets][:, None], size=(n_pairs, 18))
    candidates = bucket_arr[anc_buckets[:, None], draws]

    negatives = np.empty((n_pairs, k_neg), dtype=np.int64)
    for i in range(n_pairs):
        anchor = int(pairs[i, 0])
        target = int(pairs[i, 1])
        blocked = succ_sets[anchor]
        chosen: list[int] = []
        for candidate in candidates[i]:
            c = int(candidate)
            if c != anchor and c != target and c not in blocked and c not in chosen:
                chosen.append(c)
                if len(chosen) == k_neg:
                    break
        while len(chosen) < k_neg:
            c = int(rng.integers(0, n_items))
            if c != anchor and c != target and c not in blocked and c not in chosen:
                chosen.append(c)
        negatives[i] = chosen
    return negatives
