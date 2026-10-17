"""HBPE-R3-05 (#41): Same-prefix semi-hard negatives dynamically mined for ER.

From 40 candidate negatives per anchor, dynamically at forward step under current model no_grad:
Find candidates sharing same hard L1 prefix with anchor. Pick up to 5 same-prefix negatives + other candidates = 10 negatives for ER branch.
Weight 0.010 on base L1 ER.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

LAMBDA_ER1 = 0.010


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


def _dynamic_mine_semi_hard_negatives(
    cands_40: torch.Tensor,
    z_a: torch.Tensor,
    c1_det: torch.Tensor,
    encoder: torch.nn.Module,
    all_embeddings: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Mine 5 same-prefix + 5 random negatives dynamically from 40 candidates."""
    with torch.no_grad():
        b_sz = z_a.shape[0]
        h1_a = _compute_squared_distances(z_a, c1_det).argmin(dim=-1)  # [B]

        cands_embs = all_embeddings[cands_40.cpu()].to(device, non_blocking=True)  # [B, 40, D]
        z_cands = encoder(cands_embs)  # [B, 40, 32]
        h1_cands = _compute_squared_distances(z_cands, c1_det).argmin(dim=-1)  # [B, 40]

        same = (h1_cands == h1_a.unsqueeze(1)).cpu().numpy()  # [B, 40]
        cands_np = cands_40.cpu().numpy()

        chosen_10 = np.empty((b_sz, 10), dtype=np.int64)
        total_same_mined = 0

        for b in range(b_sz):
            s_items = cands_np[b][same[b]]
            d_items = cands_np[b][~same[b]]
            n_s = min(len(s_items), 5)
            n_d = 10 - n_s
            part_s = s_items[:n_s]
            part_d = d_items[:n_d]
            if len(part_d) < n_d:
                part_d = np.pad(part_d, (0, n_d - len(part_d)), mode="edge")
            chosen_10[b] = np.concatenate([part_s, part_d])
            total_same_mined += n_s

        same_prefix_rate = total_same_mined / (b_sz * 5.0)
        return torch.tensor(chosen_10, device=device, dtype=torch.long), same_prefix_rate


def _precompute_matched_candidates_40(
    pairs: np.ndarray,
    n_items: int,
    seed: int = 42,
    k_cands: int = 40,
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
    draws = rng.integers(0, bucket_sizes[anc_buckets][:, None], size=(n_pairs, 60))
    candidates = bucket_arr[anc_buckets[:, None], draws]

    out_cands = np.empty((n_pairs, k_cands), dtype=np.int64)
    for i in range(n_pairs):
        anchor = int(pairs[i, 0])
        target = int(pairs[i, 1])
        blocked = succ_sets[anchor]
        chosen: list[int] = []
        for candidate in candidates[i]:
            c = int(candidate)
            if c != anchor and c != target and c not in blocked and c not in chosen:
                chosen.append(c)
                if len(chosen) == k_cands:
                    break
        while len(chosen) < k_cands:
            c = int(rng.integers(0, n_items))
            if c != anchor and c != target and c not in blocked and c not in chosen:
                chosen.append(c)
        out_cands[i] = chosen
    return out_cands
