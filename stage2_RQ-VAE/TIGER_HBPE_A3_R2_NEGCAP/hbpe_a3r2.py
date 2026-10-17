"""HBPE-A3-R2 (#36): two-level soft enrichment plus a negative-sharing cap, and an
optional hard-occupancy guard.

A3 (#33) regressed because its coverage floors never fired: the failure was not too
little positive sharing but too much NEGATIVE sharing (L1 codebook collapsed to 55
codes, negative sharing rose roughly 4x). This round points the constraint at the
denominator instead:

    L_neg1  = ReLU(S1_neg  / max(B1_neg(tau),  1e-8) - 1)^2
    L_neg12 = ReLU(S12_neg / max(B12_neg(tau), 1e-8) - 1)^2

``B_neg(tau)`` is measured ONLINE on the same batch from the frozen canonical
Resample export acting purely as a teacher: the same soft formulas are applied to
the teacher's latents with the teacher's own reference distance-gap scale times the
same ``tau(epoch)``. A soft threshold is compared against a soft quantity -- the
hard baseline negative share is never mixed in -- and the comparison depends on
train data only.

The occupancy guard is the #34 barrier: a hard-forward / soft-backward L1 entropy
that penalises a concentrated L1 histogram relative to the frozen Resample's.
Because ``r2_proxy = z - C1.detach()[argmin]`` still carries z, neither safeguard
can guarantee the hard assignment is untouched; both are reported as what they are.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

LAMBDA_ER1 = 0.02
LAMBDA_ER12 = 0.02
LAMBDA_COV = 0.01
LAMBDA_NEG = 0.01

USE_OCCUPANCY_GUARD = False
LAMBDA_OCC = 0.01
H_REF_L1 = 5.4243431727412155
OCC_SAMPLE = 4096

# A3's positive-coverage floors, retained from #33.
RHO1 = 0.11639702
RHO12 = 0.00988457

# Frozen Resample teacher, used only to measure the negative-sharing reference.
TEACHER_CHECKPOINT = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/rqvae_best.pth"
)


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
    """Soft codeword assignment with a detached codebook."""
    c_det = codebook.detach()
    d = _compute_squared_distances(z, c_det)
    return F.softmax(-d / max(float(temperature), 1e-8), dim=-1)


def _continuous_l2_residual_proxy(z: torch.Tensor, codebook1: torch.Tensor) -> torch.Tensor:
    """r2_proxy = z - C1.detach()[h1], argmin taken under no_grad."""
    c1_det = codebook1.detach()
    with torch.no_grad():
        h1 = _compute_squared_distances(z, c1_det).argmin(dim=-1)
    return z - c1_det[h1]


def teacher_negative_sharing(
    z_anchor: torch.Tensor,
    z_neg: torch.Tensor,
    teacher_codebook1: torch.Tensor,
    teacher_codebook2: torch.Tensor,
    teacher_d_ref1: float,
    teacher_d_ref2: float,
    tau: float,
    eps: float = 1e-8,
) -> tuple[float, float]:
    """B1_neg(tau) and B12_neg(tau) under the frozen Resample teacher.

    Evaluated on the same items the current model's own S_neg was measured on, with
    the teacher's own reference distance gaps scaled by the same ``tau(epoch)``, so
    the cap compares like with like. Runs under ``no_grad`` in the caller.
    """
    t1 = tau * max(teacher_d_ref1, eps)
    t2 = tau * max(teacher_d_ref2, eps)
    p1_a = _soft_prefix_distribution(z_anchor, teacher_codebook1, t1)
    p1_n = _soft_prefix_distribution(z_neg, teacher_codebook1, t1)
    s1_neg = (p1_a.unsqueeze(1) * p1_n).sum(-1).mean()

    p2_a = _soft_prefix_distribution(
        _continuous_l2_residual_proxy(z_anchor, teacher_codebook1), teacher_codebook2, t2
    )
    p2_n = _soft_prefix_distribution(
        _continuous_l2_residual_proxy(z_neg, teacher_codebook1), teacher_codebook2, t2
    )
    s2_neg = (p2_a.unsqueeze(1) * p2_n).sum(-1).mean()
    return float(s1_neg), float(s1_neg * s2_neg)


def hbpe_negcap_enrichment_loss(
    p1_a: torch.Tensor,
    p1_p: torch.Tensor,
    p1_n: torch.Tensor,
    p2_a: torch.Tensor,
    p2_p: torch.Tensor,
    p2_n: torch.Tensor,
    b1_neg: float,
    b12_neg: float,
    rho1: float = RHO1,
    rho12: float = RHO12,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """A3's two ER terms, its positive-coverage floors, and the new negative caps.

    Returns the already-weighted total plus the statistics the run has to record:
    whether each cap fired, how large it was, and the soft S_neg it was measured
    against. The caps are on the soft negative sharing, computed in-batch, rather
    than on a sparse hard rate.
    """
    s1_pos = (p1_a * p1_p).sum(-1)
    S1_pos = s1_pos.mean()
    s1_neg = (p1_a.unsqueeze(1) * p1_n).sum(-1)
    S1_neg = s1_neg.mean()
    s2_pos = (p2_a * p2_p).sum(-1)
    S12_pos = (s1_pos * s2_pos).mean()
    s2_neg = (p2_a.unsqueeze(1) * p2_n).sum(-1)
    S12_neg = (s1_neg * s2_neg).mean()

    l_er1 = -(torch.log(S1_pos + eps) - torch.log(S1_neg + eps))
    l_er12 = -(torch.log(S12_pos + eps) - torch.log(S12_neg + eps))
    l_cov1 = torch.relu((float(rho1) - S1_pos) / max(float(rho1), eps)) ** 2
    l_cov12 = torch.relu((float(rho12) - S12_pos) / max(float(rho12), eps)) ** 2

    # ReLU((S_neg / B_neg) - 1)^2 : zero until the current soft negative sharing
    # exceeds the teacher's, then quadratic.
    ratio1 = S1_neg / max(float(b1_neg), eps)
    ratio12 = S12_neg / max(float(b12_neg), eps)
    l_neg1 = torch.relu(ratio1 - 1.0) ** 2
    l_neg12 = torch.relu(ratio12 - 1.0) ** 2

    total = (
        LAMBDA_ER1 * l_er1
        + LAMBDA_ER12 * l_er12
        + LAMBDA_COV * (l_cov1 + l_cov12)
        + LAMBDA_NEG * (l_neg1 + l_neg12)
    )

    with torch.no_grad():
        stats = {
            "L_ER1": float(l_er1.item()),
            "L_ER12": float(l_er12.item()),
            "L_cov1": float(l_cov1.item()),
            "L_cov12": float(l_cov12.item()),
            "L_neg1": float(l_neg1.item()),
            "L_neg12": float(l_neg12.item()),
            "negcap1_active": float(l_neg1.item() > 0.0),
            "negcap12_active": float(l_neg12.item() > 0.0),
            "B1_neg": float(b1_neg),
            "B12_neg": float(b12_neg),
            "S1_pos": float(S1_pos.item()),
            "S1_neg": float(S1_neg.item()),
            "S12_pos": float(S12_pos.item()),
            "S12_neg": float(S12_neg.item()),
            "Soft_ER1": float((S1_pos / (S1_neg + eps)).item()),
            "Soft_ER12": float((S12_pos / (S12_neg + eps)).item()),
        }
    return total, stats


def hbpe_l1_occupancy_loss(
    z: torch.Tensor,
    codebook1: torch.Tensor,
    temperature: float,
    h_ref: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hard-forward / soft-backward L1 occupancy entropy guard."""
    c_det = codebook1.detach()
    d = _compute_squared_distances(z, c_det)
    with torch.no_grad():
        hard = d.argmin(dim=-1)
    p = F.softmax(-d / max(float(temperature), 1e-8), dim=-1)

    q = F.one_hot(hard, num_classes=c_det.shape[0]).to(p.dtype) + (p - p.detach())
    pi = q.mean(dim=0).clamp_min(0.0)
    entropy = -(pi * torch.log(pi + eps)).sum()

    h_ref_t = torch.as_tensor(max(float(h_ref), eps), device=z.device, dtype=z.dtype)
    l_occ = torch.relu((h_ref_t - entropy) / h_ref_t) ** 2

    with torch.no_grad():
        counts = torch.bincount(hard, minlength=c_det.shape[0])
        stats = {
            "L_occ": float(l_occ.item()),
            "H_pi": float(entropy.item()),
            "H_ref": float(h_ref_t.item()),
            "effective_codes_exp_H": float(torch.exp(entropy).item()),
            "hard_distinct_l1_codes": int((counts > 0).sum().item()),
        }
    return l_occ, stats


def hard_prefix_sharing(
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    codebook1: torch.Tensor,
    codebook2: torch.Tensor,
) -> dict[str, float]:
    """Discrete prefix statistics on the training batch, mirroring the quantizer."""
    c1 = codebook1.detach()
    c2 = codebook2.detach()
    with torch.no_grad():
        h1_a = _compute_squared_distances(z_anchor, c1).argmin(-1)
        h1_p = _compute_squared_distances(z_pos, c1).argmin(-1)
        h1_n = _compute_squared_distances(z_neg, c1).argmin(-1)
        h2_a = _compute_squared_distances(z_anchor - c1[h1_a], c2).argmin(-1)
        h2_p = _compute_squared_distances(z_pos - c1[h1_p], c2).argmin(-1)
        h2_n = _compute_squared_distances(z_neg - c1[h1_n], c2).argmin(-1)

        l1_pos = float((h1_a == h1_p).float().mean())
        l1_neg = float((h1_a.unsqueeze(1) == h1_n).float().mean())
        l12_pos = float(((h1_a == h1_p) & (h2_a == h2_p)).float().mean())
        l12_neg = float(
            (((h1_a.unsqueeze(1) == h1_n) & (h2_a.unsqueeze(1) == h2_n))
             .float().mean())
        )
        counts = torch.bincount(h1_a, minlength=c1.shape[0]).float()
        pi = counts / counts.sum().clamp_min(1.0)
        nz = pi[pi > 0]
        entropy = float(-(nz * nz.log()).sum())
    return {
        "train_hard_L1_pos_share": l1_pos,
        "train_hard_L1_neg_share": l1_neg,
        "train_hard_L1_ER": l1_pos / l1_neg if l1_neg > 0 else 1.0,
        "train_hard_L12_pos_share": l12_pos,
        "train_hard_L12_neg_share": l12_neg,
        "train_hard_L12_ER": l12_pos / l12_neg if l12_neg > 0 else 1.0,
        "train_hard_L1_entropy": entropy,
        "train_hard_L1_effective_codes": float(np.exp(entropy)),
        "train_hard_L1_distinct_codes": float(int((counts > 0).sum())),
    }


def _precompute_matched_negatives(
    pairs: np.ndarray,
    n_items: int,
    seed: int = 42,
    k_neg: int = 10,
) -> np.ndarray:
    """A3's sampler, unchanged: anchor-bucket negatives excluding self, target and
    every observed train successor."""
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
