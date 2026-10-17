"""HBPE-A3: L1 and L1+L2 Enrichment with Positive-Prefix Coverage Constraints (Issue #33).

Precomputed fixed coverage lower bounds from canonical Resample on train transitions:
RHO1 = 0.11639702
RHO12 = 0.00988457

L_cov1 = ReLU((RHO1 - S1_pos) / RHO1)^2
L_cov12 = ReLU((RHO12 - S12_pos) / RHO12)^2
Total enrichment loss = 0.02 * L_ER1 + 0.02 * L_ER12 + 0.01 * (L_cov1 + L_cov12)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

RHO1 = 0.11639702
RHO12 = 0.00988457


def _compute_squared_distances(z: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean distances between points z and codebook vectors."""
    z_sq = (z ** 2).sum(-1, keepdim=True)
    c_sq = (codebook ** 2).sum(-1)
    d = z_sq + c_sq - 2.0 * torch.matmul(z, codebook.T)
    return d.clamp_min(0.0)


def _compute_reference_gap(z: torch.Tensor, codebook: torch.Tensor) -> float:
    """Compute median squared distance gap between nearest and second-nearest codewords."""
    with torch.no_grad():
        d = _compute_squared_distances(z, codebook.detach())
        top2, _ = torch.topk(d, k=2, dim=-1, largest=False)
        gaps = top2[:, 1] - top2[:, 0]
        return float(torch.median(gaps).item())


def _soft_prefix_distribution(
    z: torch.Tensor, codebook: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Compute soft codeword assignment probability with detached codebook."""
    c_det = codebook.detach()
    d = _compute_squared_distances(z, c_det)
    return F.softmax(-d / max(float(temperature), 1e-8), dim=-1)


def _continuous_l2_residual_proxy(z: torch.Tensor, codebook1: torch.Tensor) -> torch.Tensor:
    """Compute continuous residual proxy r2_proxy = z - C1.detach()[h1]."""
    c1_det = codebook1.detach()
    with torch.no_grad():
        d1 = _compute_squared_distances(z, c1_det)
        h1 = d1.argmin(dim=-1)
    c1_chosen = c1_det[h1]
    return z - c1_chosen


def hbpe_coverage_enrichment_loss(
    p1_a: torch.Tensor,
    p1_p: torch.Tensor,
    p1_n: torch.Tensor,
    p2_a: torch.Tensor,
    p2_p: torch.Tensor,
    p2_n: torch.Tensor,
    rho1: float = RHO1,
    rho12: float = RHO12,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute joint L1 and L1+L2 soft prefix enrichment loss with coverage constraints.
    
    Args:
        p1_a: Tensor of shape [B, K]
        p1_p: Tensor of shape [B, K]
        p1_n: Tensor of shape [B, 10, K]
        p2_a: Tensor of shape [B, K]
        p2_p: Tensor of shape [B, K]
        p2_n: Tensor of shape [B, 10, K]
        rho1: float, positive coverage lower bound for L1
        rho12: float, positive coverage lower bound for L1+L2
        eps: float, numerical stability epsilon
    Returns:
        (total_loss_scalar, stats_dict)
    """
    # L1 similarities
    s1_pos = (p1_a * p1_p).sum(-1)  # shape [B]
    S1_pos = s1_pos.mean()
    s1_neg = (p1_a.unsqueeze(1) * p1_n).sum(-1)  # shape [B, 10]
    S1_neg = s1_neg.mean()

    # L2 similarities
    s2_pos = (p2_a * p2_p).sum(-1)  # shape [B]
    s2_neg = (p2_a.unsqueeze(1) * p2_n).sum(-1)  # shape [B, 10]

    # Two-level joint similarities
    s12_pos = s1_pos * s2_pos  # shape [B]
    S12_pos = s12_pos.mean()
    s12_neg = s1_neg * s2_neg  # shape [B, 10]
    S12_neg = s12_neg.mean()

    l_er1 = -(torch.log(S1_pos + eps) - torch.log(S1_neg + eps))
    l_er12 = -(torch.log(S12_pos + eps) - torch.log(S12_neg + eps))

    # Coverage penalty terms: ReLU((rho - S_pos) / rho)^2
    l_cov1 = torch.relu((float(rho1) - S1_pos) / max(float(rho1), eps)) ** 2
    l_cov12 = torch.relu((float(rho12) - S12_pos) / max(float(rho12), eps)) ** 2

    # Combined loss: 0.02 * L_ER1 + 0.02 * L_ER12 + 0.01 * (L_cov1 + L_cov12)
    total_loss = 0.02 * l_er1 + 0.02 * l_er12 + 0.01 * (l_cov1 + l_cov12)

    with torch.no_grad():
        soft_er1 = float((S1_pos / (S1_neg + eps)).item())
        soft_er12 = float((S12_pos / (S12_neg + eps)).item())
        stats = {
            "L_ER1": float(l_er1.detach().item()),
            "L_ER12": float(l_er12.detach().item()),
            "L_cov1": float(l_cov1.detach().item()),
            "L_cov12": float(l_cov12.detach().item()),
            "S1_pos": float(S1_pos.detach().item()),
            "S1_neg": float(S1_neg.detach().item()),
            "S12_pos": float(S12_pos.detach().item()),
            "S12_neg": float(S12_neg.detach().item()),
            "Soft_ER1": soft_er1,
            "Soft_ER12": soft_er12,
            "rho1": float(rho1),
            "rho12": float(rho12),
        }

    return total_loss, stats


def _precompute_matched_negatives(
    pairs: np.ndarray,
    n_items: int,
    seed: int = 42,
    k_neg: int = 10,
) -> np.ndarray:
    """Precompute k_neg matched popularity negatives for each train transition pair."""
    n_pairs = len(pairs)
    pop = np.bincount(pairs[:, 1], minlength=n_items)
    quantiles = np.quantile(pop[pop > 0], np.linspace(0, 1, 11))
    quantiles[-1] += 1
    buckets = np.clip(np.digitize(pop, quantiles) - 1, 0, 9)

    bucket_lists = [np.where(buckets == b)[0] for b in range(10)]
    bucket_sizes = np.array([len(l) for l in bucket_lists])
    max_sz = max(bucket_sizes)
    bucket_arr = np.full((10, max_sz), -1, dtype=np.int64)
    for b in range(10):
        bucket_arr[b, :bucket_sizes[b]] = bucket_lists[b]

    rows, cols = pairs[:, 0], pairs[:, 1]
    succ_mat = csr_matrix(
        (np.ones(n_pairs, dtype=bool), (rows, cols)),
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
    b_sz = bucket_sizes[anc_buckets]

    draws = rng.integers(0, b_sz[:, None], size=(n_pairs, 18))
    cands = bucket_arr[anc_buckets[:, None], draws]

    neg_out = np.empty((n_pairs, k_neg), dtype=np.int64)
    anchors = pairs[:, 0]
    positives = pairs[:, 1]

    for i in range(n_pairs):
        anc = int(anchors[i])
        pos = int(positives[i])
        blocked = succ_sets[anc]
        chosen: list[int] = []
        for c in cands[i]:
            c_int = int(c)
            if c_int != anc and c_int != pos and c_int not in blocked and c_int not in chosen:
                chosen.append(c_int)
                if len(chosen) == k_neg:
                    break
        while len(chosen) < k_neg:
            c_rand = int(rng.integers(0, n_items))
            if c_rand != anc and c_rand != pos and c_rand not in blocked and c_rand not in chosen:
                chosen.append(c_rand)
        neg_out[i] = chosen

    return neg_out
