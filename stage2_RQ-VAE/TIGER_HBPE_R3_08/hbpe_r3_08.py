"""HBPE-R3-08 (#44): L1-conditional L2 enrichment without unconditional ratio.

Conditioned on anchor and positive sharing real hard L1 id:
Find same-L1 negatives among matched negative items.
Loss: softplus((s2(a, n) - s2(a, p) + 0.1) / 1.0).
Weight 0.0001. Base L1 ER 0.010.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

LAMBDA_ER1 = 0.010
LAMBDA_COND_L2 = 0.0001
COND_MARGIN = 0.1
COND_TEMP = 1.0


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


def hbpe_conditional_l2_loss(
    z_a: torch.Tensor,
    z_p: torch.Tensor,
    z_n: torch.Tensor,
    codebook1: torch.Tensor,
    codebook2: torch.Tensor,
    t2: float,
    margin: float = COND_MARGIN,
    temp: float = COND_TEMP,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute conditional L2 ranking loss on same-L1 transitions."""
    c1 = codebook1.detach()
    c2 = codebook2.detach()

    with torch.no_grad():
        h1_a = _compute_squared_distances(z_a, c1).argmin(dim=-1)  # [B]
        h1_p = _compute_squared_distances(z_p, c1).argmin(dim=-1)  # [B]
        h1_n = _compute_squared_distances(z_n, c1).argmin(dim=-1)  # [B, 10]

        pair_match = (h1_a == h1_p)  # [B]
        neg_match = (h1_n == h1_a.unsqueeze(1))  # [B, 10]
        # Active if positive shares L1 AND at least one negative shares L1
        valid_rows = pair_match & neg_match.any(dim=-1)

    if not valid_rows.any():
        zero = z_a.sum() * 0.0
        return zero, {"L_cond_l2": 0.0, "valid_pairs": 0, "active_rate": 0.0}

    # Residual proxies: r2 = z - C1[h1]
    r2_a = z_a - c1[h1_a]
    r2_p = z_p - c1[h1_p]
    r2_n = z_n - c1[h1_n]

    p2_a = _soft_prefix_distribution(r2_a, c2, t2)  # [B, K]
    p2_p = _soft_prefix_distribution(r2_p, c2, t2)  # [B, K]
    p2_n = _soft_prefix_distribution(r2_n, c2, t2)  # [B, 10, K]

    s2_pos = (p2_a * p2_p).sum(-1)  # [B]
    s2_neg = (p2_a.unsqueeze(1) * p2_n).sum(-1)  # [B, 10]

    # Mask to valid rows and same-L1 negatives
    diff = s2_neg - s2_pos.unsqueeze(1)  # [B, 10]
    rank_loss = F.softplus((diff + margin) / temp)

    # Average only over valid rows and matching negatives
    mask = valid_rows.unsqueeze(1) & neg_match
    total_loss = (rank_loss * mask).sum() / mask.sum().clamp_min(1.0)

    with torch.no_grad():
        stats = {
            "L_cond_l2": float(total_loss.detach().item()),
            "valid_pairs": int(valid_rows.sum().item()),
            "active_rate": float(valid_rows.float().mean().item()),
        }
    return total_loss, stats


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
