"""HBPE-R3-09 (#45): Within-run L1 assignment trust region after behavior warmup.

At epoch 2250, take stop-grad snapshot of encoder and C1.
From epoch 2250 to 3000, compute KL(p_ref || p_now) on uniform catalogue sample.
Weight 0.001 on L_trust, base 0.010 on L1 ER.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

LAMBDA_ER1 = 0.010
LAMBDA_TRUST = 0.001


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


def hbpe_trust_region_loss(
    z_now: torch.Tensor,
    c1_curr: torch.Tensor,
    p_ref: torch.Tensor,
    t1: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute KL(p_ref || p_now) where p_now is differentiable w.r.t z_now."""
    c1_det = c1_curr.detach()
    p_now = _soft_prefix_distribution(z_now, c1_det, t1)

    # KL(p_ref || p_now) = sum p_ref * (log p_ref - log p_now)
    p_r = p_ref.clamp_min(eps)
    p_n = p_now.clamp_min(eps)
    kl = (p_r * (torch.log(p_r) - torch.log(p_n))).sum(dim=-1).mean()

    with torch.no_grad():
        stats = {
            "L_trust": float(kl.detach().item()),
            "trust_kl": float(kl.detach().item()),
        }
    return kl, stats


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
