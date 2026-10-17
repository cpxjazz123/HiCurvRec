"""HBPE-A1-R2 (#34): L1 soft prefix enrichment with an explicit weight, plus an
optional hard-forward/soft-backward L1 occupancy barrier.

The single factor this round ablates is the enrichment weight ``LAMBDA_ER1``; the
original A1 used 0.02. Everything else -- the loss shape, the detached-codebook
soft assignment, the reference distance-gap temperature, the negative sampler, the
behaviour ramp -- is A1's, unchanged, so the weight comparison is clean.

The occupancy barrier exists because A1 bought L1+L2 enrichment by shrinking the L1
codebook (241 -> 172 distinct codes). It penalises a low-entropy L1 occupancy but
must not touch the quantizer: the forward value is the true hard histogram while the
gradient flows only through the soft distribution, so

    q = onehot(h) + (p - p.detach()),   pi = mean(q),   H(pi) = -sum(pi log(pi+eps))

has d(pi)/d(p) = 1 while forward(pi) is exactly the hard histogram. ``argmin`` is
taken under ``no_grad`` and carries no gradient. Nothing here reaches a codebook:
the codebook is detached at every entry point.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

# The single factor this round ablates: A1's L1 enrichment weight.
LAMBDA_ER1 = 0.010

# Occupancy barrier. Only meaningful when USE_OCCUPANCY_BARRIER is True; the
# weight and the reference entropy are read by the trainer at log time.
USE_OCCUPANCY_BARRIER = True
LAMBDA_OCC = 0.01
# Entropy of the hard L1 occupancy histogram of the frozen canonical Resample
# export over the training catalogue. An entropy rather than a per-code target,
# because code identities permute freely between runs.
H_REF_L1 = 5.4243431727412155
# Uniform catalogue sample per step for the occupancy histogram. Items, not
# behaviour pairs: measuring occupancy on pairs would inherit their popularity.
OCC_SAMPLE = 4096


def _compute_squared_distances(z: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between points z and codebook vectors."""
    z_sq = (z ** 2).sum(-1, keepdim=True)
    c_sq = (codebook ** 2).sum(-1)
    d = z_sq + c_sq - 2.0 * torch.matmul(z, codebook.T)
    return d.clamp_min(0.0)


def _compute_reference_gap(z: torch.Tensor, codebook: torch.Tensor) -> float:
    """Median squared distance gap between the nearest and second-nearest codeword."""
    with torch.no_grad():
        d = _compute_squared_distances(z, codebook.detach())
        top2, _ = torch.topk(d, k=2, dim=-1, largest=False)
        gaps = top2[:, 1] - top2[:, 0]
        return float(torch.median(gaps).item())


def _soft_prefix_distribution(
    z: torch.Tensor, codebook: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Soft codeword assignment p1 = softmax(-||z - C.detach()||^2 / T)."""
    c_det = codebook.detach()
    d = _compute_squared_distances(z, c_det)
    return F.softmax(-d / max(float(temperature), 1e-8), dim=-1)


def hbpe_l1_loss(
    p1_anchor: torch.Tensor,
    p1_pos: torch.Tensor,
    p1_neg: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_A1 = -[log(S1_pos + eps) - log(S1_neg + eps)].

    Args:
        p1_anchor: [B, K] soft assignment of the anchor
        p1_pos:    [B, K] soft assignment of the positive target
        p1_neg:    [B, 10, K] soft assignment of the matched negatives
    """
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


def hbpe_l1_occupancy_loss(
    z: torch.Tensor,
    codebook1: torch.Tensor,
    temperature: float,
    h_ref: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hard-forward / soft-backward L1 occupancy entropy barrier.

    The forward value of ``pi`` is the exact hard-assignment histogram; the gradient
    reaches ``z`` only through the soft distribution. ``h_ref`` comes from the frozen
    Resample export, never from valid or test data.

    Args:
        z: [M, D] encoder outputs of uniformly sampled training items
        codebook1: [K, D] L1 codebook, read detached
        temperature: current T1
        h_ref: target entropy in nats
    """
    c_det = codebook1.detach()
    d = _compute_squared_distances(z, c_det)
    with torch.no_grad():
        hard = d.argmin(dim=-1)
    p = F.softmax(-d / max(float(temperature), 1e-8), dim=-1)

    # Straight-through: forward is onehot(hard), backward is the identity on p.
    q = F.one_hot(hard, num_classes=c_det.shape[0]).to(p.dtype) + (p - p.detach())
    pi = q.mean(dim=0).clamp_min(0.0)
    entropy = -(pi * torch.log(pi + eps)).sum()

    h_ref_t = torch.as_tensor(max(float(h_ref), eps), device=z.device, dtype=z.dtype)
    l_occ = torch.relu((h_ref_t - entropy) / h_ref_t) ** 2

    with torch.no_grad():
        hard_counts = torch.bincount(hard, minlength=c_det.shape[0])
        stats = {
            "L_occ": float(l_occ.detach().item()),
            "H_pi": float(entropy.detach().item()),
            "H_ref": float(h_ref_t.item()),
            "effective_codes_exp_H": float(torch.exp(entropy).item()),
            "hard_distinct_l1_codes": int((hard_counts > 0).sum().item()),
            "H_pi_minus_H_ref": float((entropy.detach() - h_ref_t).item()),
        }
    return l_occ, stats


def _precompute_matched_negatives(
    pairs: np.ndarray,
    n_items: int,
    seed: int = 42,
    k_neg: int = 10,
) -> np.ndarray:
    """k_neg popularity-decile-matched negatives per train transition.

    Drawn from the ANCHOR's bucket, excluding the anchor, the current target and
    every observed train successor. Issue #34 step 5 asks for this to stay exactly
    as A1 had it so the weight ablation is single-factor; the audit in
    scripts/hbpe_r2_reference_constants.py measures what it does.
    """
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
