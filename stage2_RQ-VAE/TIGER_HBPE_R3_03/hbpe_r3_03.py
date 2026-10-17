"""HBPE-R3-03 (#39): Multi-successor supervised prefix contrast.

For anchors with >=2 successors, sample up to 4 true positive successors.
L_multi = -mean_a log[(mean_{p in P(a)} s1(a,p) + eps) / (mean_{n in N(a)} s1(a,n) + eps)].
Weight 0.005 on L_multi, base 0.010 on single-pair L1 ER.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

LAMBDA_ER1 = 0.010
LAMBDA_MULTI = 0.005


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


def hbpe_multi_pos_loss(
    p1_anchor: torch.Tensor,
    p1_multi_pos: torch.Tensor,
    pos_mask: torch.Tensor,
    p1_neg: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-successor Soft L1 prefix contrast.

    Args:
        p1_anchor: [B, K]
        p1_multi_pos: [B, 4, K]
        pos_mask: [B, 4] boolean mask
        p1_neg: [B, 10, K]
    """
    # s1(a, p) for each of up to 4 positives
    s1_multi = (p1_anchor.unsqueeze(1) * p1_multi_pos).sum(-1)  # [B, 4]
    s1_pos_mean = (s1_multi * pos_mask).sum(-1) / pos_mask.sum(-1).clamp_min(1.0)  # [B]

    # s1(a, n) for each of 10 negatives
    s1_neg = (p1_anchor.unsqueeze(1) * p1_neg).sum(-1)  # [B, 10]
    s1_neg_mean = s1_neg.mean(-1)  # [B]

    loss = -(torch.log(s1_pos_mean + eps) - torch.log(s1_neg_mean + eps)).mean()

    with torch.no_grad():
        stats = {
            "L_multi": float(loss.detach().item()),
            "multi_pos_mean": float(s1_pos_mean.detach().mean().item()),
            "multi_neg_mean": float(s1_neg_mean.detach().mean().item()),
        }
    return loss, stats


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


def _precompute_multi_positives(
    pairs: np.ndarray,
    n_items: int,
    seed: int = 42,
    max_pos: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute up to max_pos true successors per pair."""
    n_pairs = len(pairs)
    succ_mat = csr_matrix(
        (np.ones(n_pairs, dtype=bool), (pairs[:, 0], pairs[:, 1])),
        shape=(n_items, n_items),
        dtype=bool,
    )
    succ_mat.sum_duplicates()
    succ_lists = [
        succ_mat.indices[succ_mat.indptr[i] : succ_mat.indptr[i + 1]].tolist()
        for i in range(n_items)
    ]

    rng = np.random.default_rng(seed + 999)
    multi_pos = np.zeros((n_pairs, max_pos), dtype=np.int64)
    pos_mask = np.zeros((n_pairs, max_pos), dtype=bool)

    for i in range(n_pairs):
        anc = int(pairs[i, 0])
        tgt = int(pairs[i, 1])
        all_succs = succ_lists[anc]
        
        chosen = [tgt]
        other_succs = [s for s in all_succs if s != tgt]
        if other_succs:
            take = min(len(other_succs), max_pos - 1)
            picks = rng.choice(other_succs, size=take, replace=False).tolist()
            chosen.extend(picks)
            
        multi_pos[i, :len(chosen)] = chosen
        pos_mask[i, :len(chosen)] = True

    return multi_pos, pos_mask
