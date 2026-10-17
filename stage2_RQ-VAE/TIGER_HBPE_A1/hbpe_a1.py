"""HBPE-A1: L1 Soft Prefix Enrichment on top of Exp1a Resample (Issue #31).

1. Computes soft prefix distribution p1_i = softmax(-||z_i - c1||^2 / T1) with C1.detach().
2. S1_pos = mean(p1_a . p1_p) over true train transitions (history[-1] -> target).
3. S1_neg = mean(p1_a . p1_n) over 10 popularity-matched negative items per anchor,
   excluding anchor itself, current target, and all known observed train successors.
4. L_A1 = -[log(S1_pos + 1e-8) - log(S1_neg + 1e-8)].
5. T1 = tau(epoch) * max(D_ref1, 1e-8), tau anneals 1.0 -> 0.2 during behavior ramp.
6. Combined into L_total = L_RQVAE + w_b * L_Resample_3level + 0.02 * ramp(epoch) * L_A1.
7. Crucial: codebooks use C1.detach(), so L_A1 only updates Encoder parameters.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix


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
    """Compute soft codeword assignment probability p1_i with detached codebook."""
    c_det = codebook.detach()
    d = _compute_squared_distances(z, c_det)
    return F.softmax(-d / max(float(temperature), 1e-8), dim=-1)


def hbpe_l1_loss(
    p1_anchor: torch.Tensor,
    p1_pos: torch.Tensor,
    p1_neg: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute L1 soft prefix enrichment ratio loss L_A1.
    
    Args:
        p1_anchor: Tensor of shape [B, K]
        p1_pos: Tensor of shape [B, K]
        p1_neg: Tensor of shape [B, 10, K]
        eps: float, numerical stability epsilon
    Returns:
        (loss_scalar, stats_dict)
    """
    # s1(a, p) = sum_c p1_a[c] * p1_p[c]
    s1_pos = (p1_anchor * p1_pos).sum(-1)  # shape [B]
    S1_pos = s1_pos.mean()

    # s1(a, n) = sum_c p1_a[c] * p1_n[c]
    s1_neg = (p1_anchor.unsqueeze(1) * p1_neg).sum(-1)  # shape [B, 10]
    S1_neg = s1_neg.mean()

    loss = -(torch.log(S1_pos + eps) - torch.log(S1_neg + eps))

    with torch.no_grad():
        soft_er1 = float((S1_pos / (S1_neg + eps)).item())
        stats = {
            "L_A1": float(loss.detach().item()),
            "S1_pos": float(S1_pos.detach().item()),
            "S1_neg": float(S1_neg.detach().item()),
            "Soft_ER1": soft_er1,
        }

    return loss, stats


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
