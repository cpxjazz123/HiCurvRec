"""HBPE-R3-07 (#43): Text-semantic neighborhood preservation during prefix enrichment.

Top-8 text semantic neighbors from sentence_t5.npy.
Teacher distribution q = softmax(cos_sim(e_a, e_nb) / 0.1).
Student distribution p = softmax(cos_sim(z_a, z_nb) / 0.1).
L_sem = KL(q || p). Weight 0.001. Base L1 ER 0.010.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

LAMBDA_ER1 = 0.010
LAMBDA_SEM = 0.001
SEM_TEMP = 0.1


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


def hbpe_semantic_neighborhood_loss(
    z_a: torch.Tensor,
    z_nb: torch.Tensor,
    teacher_q: torch.Tensor,
    temp: float = SEM_TEMP,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute KL(q || p) between text embedding similarity q and latent similarity p."""
    # z_a: [B, D], z_nb: [B, 8, D]
    cos_student = F.cosine_similarity(z_a.unsqueeze(1), z_nb, dim=-1)  # [B, 8]
    p = F.softmax(cos_student / max(float(temp), 1e-8), dim=-1)  # [B, 8]

    # KL(q || p) = sum q * (log q - log p)
    q = teacher_q.clamp_min(eps)
    log_q = torch.log(q)
    log_p = torch.log(p.clamp_min(eps))
    kl_per_row = (q * (log_q - log_p)).sum(dim=-1)  # [B]
    loss_sem = kl_per_row.mean()

    with torch.no_grad():
        stats = {
            "L_sem": float(loss_sem.detach().item()),
            "mean_kl": float(loss_sem.detach().item()),
        }
    return loss_sem, stats


def _load_top8_neighbors(tree_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(tree_dir / "text_top8_neighbors.npz")
    indices = torch.from_numpy(data["indices"]).long()
    q = torch.from_numpy(data["q"]).float()
    return indices, q


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
