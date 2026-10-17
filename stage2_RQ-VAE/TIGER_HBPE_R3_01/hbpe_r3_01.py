"""HBPE-R3-01 (#37): Hard-forward / soft-backward prefix pair ranking (Hard-ST).

q_l = onehot(h_l) + p_l - stopgrad(p_l)
s1_ST = q1_a . q1_j, s12_ST = s1_ST * (q2_a . q2_j)
Ranking loss on positive-negative pairs: softplus((s_neg_ST - s_pos_ST + m) / t), m=0.2, t=1.0
L1 weight 0.001, L2 weight 0.0001. Base L1 soft ER 0.010.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

LAMBDA_ER1 = 0.010
LAMBDA_RANK1 = 0.001
LAMBDA_RANK2 = 0.0001
RANK_MARGIN = 0.2
RANK_TEMP = 1.0


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


def hbpe_hard_st_ranking_loss(
    z_a: torch.Tensor,
    z_p: torch.Tensor,
    z_n: torch.Tensor,
    codebook1: torch.Tensor,
    codebook2: torch.Tensor,
    t1: float,
    t2: float,
    m: float = RANK_MARGIN,
    t: float = RANK_TEMP,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute hard-ST prefix ranking loss for L1 and L1+L2."""
    c1 = codebook1.detach()
    c2 = codebook2.detach()
    num_c1 = c1.shape[0]
    num_c2 = c2.shape[0]

    # Level 1
    d1_a = _compute_squared_distances(z_a, c1)
    d1_p = _compute_squared_distances(z_p, c1)
    d1_n = _compute_squared_distances(z_n, c1)

    with torch.no_grad():
        h1_a = d1_a.argmin(dim=-1)
        h1_p = d1_p.argmin(dim=-1)
        h1_n = d1_n.argmin(dim=-1)

    p1_a = F.softmax(-d1_a / max(float(t1), 1e-8), dim=-1)
    p1_p = F.softmax(-d1_p / max(float(t1), 1e-8), dim=-1)
    p1_n = F.softmax(-d1_n / max(float(t1), 1e-8), dim=-1)

    q1_a = F.one_hot(h1_a, num_classes=num_c1).to(p1_a.dtype) + (p1_a - p1_a.detach())
    q1_p = F.one_hot(h1_p, num_classes=num_c1).to(p1_p.dtype) + (p1_p - p1_p.detach())
    q1_n = F.one_hot(h1_n, num_classes=num_c1).to(p1_n.dtype) + (p1_n - p1_n.detach())

    s1_pos_st = (q1_a * q1_p).sum(-1)  # [B]
    s1_neg_st = (q1_a.unsqueeze(1) * q1_n).sum(-1)  # [B, 10]

    # Level 2 residual proxy
    r2_a = z_a - c1[h1_a]
    r2_p = z_p - c1[h1_p]
    r2_n = z_n - c1[h1_n]

    d2_a = _compute_squared_distances(r2_a, c2)
    d2_p = _compute_squared_distances(r2_p, c2)
    d2_n = _compute_squared_distances(r2_n, c2)

    with torch.no_grad():
        h2_a = d2_a.argmin(dim=-1)
        h2_p = d2_p.argmin(dim=-1)
        h2_n = d2_n.argmin(dim=-1)

    p2_a = F.softmax(-d2_a / max(float(t2), 1e-8), dim=-1)
    p2_p = F.softmax(-d2_p / max(float(t2), 1e-8), dim=-1)
    p2_n = F.softmax(-d2_n / max(float(t2), 1e-8), dim=-1)

    q2_a = F.one_hot(h2_a, num_classes=num_c2).to(p2_a.dtype) + (p2_a - p2_a.detach())
    q2_p = F.one_hot(h2_p, num_classes=num_c2).to(p2_p.dtype) + (p2_p - p2_p.detach())
    q2_n = F.one_hot(h2_n, num_classes=num_c2).to(p2_n.dtype) + (p2_n - p2_n.detach())

    s2_pos_st = (q2_a * q2_p).sum(-1)  # [B]
    s2_neg_st = (q2_a.unsqueeze(1) * q2_n).sum(-1)  # [B, 10]

    s12_pos_st = s1_pos_st * s2_pos_st
    s12_neg_st = s1_neg_st * s2_neg_st

    # Ranking loss softplus((s_neg - s_pos + m) / t)
    diff1 = s1_neg_st - s1_pos_st.unsqueeze(1)
    loss_rank1 = F.softplus((diff1 + m) / t).mean()

    diff2 = s12_neg_st - s12_pos_st.unsqueeze(1)
    loss_rank2 = F.softplus((diff2 + m) / t).mean()

    total_rank_loss = LAMBDA_RANK1 * loss_rank1 + LAMBDA_RANK2 * loss_rank2

    with torch.no_grad():
        stats = {
            "loss_rank1": float(loss_rank1.detach().item()),
            "loss_rank2": float(loss_rank2.detach().item()),
            "total_rank_loss": float(total_rank_loss.detach().item()),
            "s1_pos_st_mean": float(s1_pos_st.detach().mean().item()),
            "s1_neg_st_mean": float(s1_neg_st.detach().mean().item()),
            "s12_pos_st_mean": float(s12_pos_st.detach().mean().item()),
            "s12_neg_st_mean": float(s12_neg_st.detach().mean().item()),
        }

    return total_rank_loss, stats


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
