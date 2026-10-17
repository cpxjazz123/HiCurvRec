"""HBPE-A2-R2 (#35): two-level soft prefix enrichment with a tunable L2 weight and
an optional gradient isolation of the L1 branch.

A2 (#32) regressed hard L1 enrichment from 16.39 to 6.50 while its training-side
Soft_ER12 rose to 89.3. Two hypotheses are separated here:

  lambda_er12
      Is the L2 term simply too strong? The sweep runs it at 0.002 and 0.005
      against A2's 0.02, with lambda_er1 pinned at 0.02.

  stopgrad_l1_in_s12
      Is the L2 term dragging the L1 distribution down through ``s12 = s1 * s2``?
      The isolation variant keeps the original L_ER1 (its s1 must NOT be detached)
      but builds the two-level term from ``stopgrad(s1) * s2``, so the L2 objective
      no longer back-propagates through the L1 soft probability directly.

Because ``r2_proxy = z - C1.detach()[argmin]`` still contains z, the isolated variant
is NOT a frozen-L1 method: it can still move the hard L1 assignment indirectly. It is
a gradient-path ablation and is reported as one.

``hard_prefix_sharing`` measures the discrete prefix statistics on the training
batches, so the run can show when Soft_ER12 climbs while the hard ratio falls.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

LAMBDA_ER1 = 0.02
LAMBDA_ER12 = 0.002
# When True, s12 = stopgrad(s1) * s2. L_ER1's own s1 is never detached.
STOPGRAD_L1_IN_S12 = False


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
    """r2_proxy = z - C1.detach()[h1], with the argmin taken under no_grad.

    The index carries no gradient, but z does, so this is the path that keeps the L2
    objective connected to the Encoder.
    """
    c1_det = codebook1.detach()
    with torch.no_grad():
        h1 = _compute_squared_distances(z, c1_det).argmin(dim=-1)
    return z - c1_det[h1]


def hbpe_joint_enrichment_loss(
    p1_a: torch.Tensor,
    p1_p: torch.Tensor,
    p1_n: torch.Tensor,
    p2_a: torch.Tensor,
    p2_p: torch.Tensor,
    p2_n: torch.Tensor,
    lambda_er1: float = LAMBDA_ER1,
    lambda_er12: float = LAMBDA_ER12,
    stopgrad_l1_in_s12: bool = STOPGRAD_L1_IN_S12,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Two-level soft prefix enrichment.

    Returns ``(weighted_total, l_er1, l_er12, stats)``. The two components are
    returned separately so the trainer can measure each one's gradient on the
    Encoder and the angle between them.
    """
    s1_pos = (p1_a * p1_p).sum(-1)
    s1_neg = (p1_a.unsqueeze(1) * p1_n).sum(-1)
    s2_pos = (p2_a * p2_p).sum(-1)
    s2_neg = (p2_a.unsqueeze(1) * p2_n).sum(-1)

    s12_pos = s1_pos.detach() * s2_pos if stopgrad_l1_in_s12 else s1_pos * s2_pos
    s12_neg = s1_neg.detach() * s2_neg if stopgrad_l1_in_s12 else s1_neg * s2_neg

    S1_pos = s1_pos.mean()
    S1_neg = s1_neg.mean()
    S12_pos = s12_pos.mean()
    S12_neg = s12_neg.mean()

    l_er1 = -(torch.log(S1_pos + eps) - torch.log(S1_neg + eps))
    l_er12 = -(torch.log(S12_pos + eps) - torch.log(S12_neg + eps))
    total = float(lambda_er1) * l_er1 + float(lambda_er12) * l_er12

    with torch.no_grad():
        stats = {
            "L_ER1": float(l_er1.item()),
            "L_ER12": float(l_er12.item()),
            "S1_pos": float(S1_pos.item()),
            "S1_neg": float(S1_neg.item()),
            "S12_pos": float(S12_pos.item()),
            "S12_neg": float(S12_neg.item()),
            "Soft_ER1": float((S1_pos / (S1_neg + eps)).item()),
            "Soft_ER12": float((S12_pos / (S12_neg + eps)).item()),
            "lambda_er1": float(lambda_er1),
            "lambda_er12": float(lambda_er12),
            "stopgrad_l1_in_s12": bool(stopgrad_l1_in_s12),
        }
    return total, l_er1, l_er12, stats


def hard_prefix_sharing(
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    codebook1: torch.Tensor,
    codebook2: torch.Tensor,
) -> dict[str, float]:
    """Discrete (argmin) prefix sharing on the current training batch.

    Mirrors the real quantizer's assignments rather than the soft proxy, so the run
    can compare the soft objective against what the exported SIDs actually do.
    Reports pos/neg sharing at L1 and L1+L2 plus the L1 codebook's entropy and
    distinct-code count.
    """
    c1 = codebook1.detach()
    c2 = codebook2.detach()
    with torch.no_grad():
        d1_a = _compute_squared_distances(z_anchor, c1)
        d1_p = _compute_squared_distances(z_pos, c1)
        d1_n = _compute_squared_distances(z_neg, c1)
        h1_a, h1_p, h1_n = d1_a.argmin(-1), d1_p.argmin(-1), d1_n.argmin(-1)

        # The real level-2 input is the hard-quantised residual, not the proxy.
        r2_a = z_anchor - c1[h1_a]
        r2_p = z_pos - c1[h1_p]
        r2_n = z_neg - c1[h1_n]
        h2_a = _compute_squared_distances(r2_a, c2).argmin(-1)
        h2_p = _compute_squared_distances(r2_p, c2).argmin(-1)
        h2_n = _compute_squared_distances(r2_n, c2).argmin(-1)

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
        distinct = int((counts > 0).sum())

    return {
        "train_hard_L1_pos_share": l1_pos,
        "train_hard_L1_neg_share": l1_neg,
        "train_hard_L1_ER": l1_pos / l1_neg if l1_neg > 0 else 1.0,
        "train_hard_L12_pos_share": l12_pos,
        "train_hard_L12_neg_share": l12_neg,
        "train_hard_L12_ER": l12_pos / l12_neg if l12_neg > 0 else 1.0,
        "train_hard_L1_entropy": entropy,
        "train_hard_L1_effective_codes": float(np.exp(entropy)),
        "train_hard_L1_distinct_codes": float(distinct),
    }


def _precompute_matched_negatives(
    pairs: np.ndarray,
    n_items: int,
    seed: int = 42,
    k_neg: int = 10,
) -> np.ndarray:
    """A2's sampler, unchanged: anchor-bucket negatives excluding self, target and
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
