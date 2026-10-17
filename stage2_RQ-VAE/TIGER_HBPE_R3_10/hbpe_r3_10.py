"""HBPE-R3-10 (#46): Degree-corrected behavioral graph modularity for prefixes.

Q = E_{edge}[s1_ST(i,j)] - E_{null}[s1_ST(u,v)]
s1_ST = q_i . q_j, where q = onehot(h1) + p1 - stopgrad(p1)
L_graph = -0.001 * Q. Base L1 ER 0.010.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

LAMBDA_ER1 = 0.010
LAMBDA_GRAPH = 0.001


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


def hbpe_graph_modularity_loss(
    z_edge_src: torch.Tensor,
    z_edge_dst: torch.Tensor,
    z_null_src: torch.Tensor,
    z_null_dst: torch.Tensor,
    codebook1: torch.Tensor,
    t1: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute modularity loss L_graph = -LAMBDA_GRAPH * (E_edge[s1_ST] - E_null[s1_ST])."""
    c1 = codebook1.detach()
    num_codes = c1.shape[0]

    # Edge pairs ST
    d_e_s = _compute_squared_distances(z_edge_src, c1)
    d_e_d = _compute_squared_distances(z_edge_dst, c1)
    with torch.no_grad():
        h_e_s = d_e_s.argmin(dim=-1)
        h_e_d = d_e_d.argmin(dim=-1)
    p_e_s = F.softmax(-d_e_s / max(float(t1), 1e-8), dim=-1)
    p_e_d = F.softmax(-d_e_d / max(float(t1), 1e-8), dim=-1)
    q_e_s = F.one_hot(h_e_s, num_classes=num_codes).to(p_e_s.dtype) + (p_e_s - p_e_s.detach())
    q_e_d = F.one_hot(h_e_d, num_classes=num_codes).to(p_e_d.dtype) + (p_e_d - p_e_d.detach())
    s1_edge = (q_e_s * q_e_d).sum(-1).mean()

    # Null pairs ST
    d_n_s = _compute_squared_distances(z_null_src, c1)
    d_n_d = _compute_squared_distances(z_null_dst, c1)
    with torch.no_grad():
        h_n_s = d_n_s.argmin(dim=-1)
        h_n_d = d_n_d.argmin(dim=-1)
    p_n_s = F.softmax(-d_n_s / max(float(t1), 1e-8), dim=-1)
    p_n_d = F.softmax(-d_n_d / max(float(t1), 1e-8), dim=-1)
    q_n_s = F.one_hot(h_n_s, num_classes=num_codes).to(p_n_s.dtype) + (p_n_s - p_n_s.detach())
    q_n_d = F.one_hot(h_n_d, num_classes=num_codes).to(p_n_d.dtype) + (p_n_d - p_n_d.detach())
    s1_null = (q_n_s * q_n_d).sum(-1).mean()

    q_modularity = s1_edge - s1_null
    loss_graph = -LAMBDA_GRAPH * q_modularity

    with torch.no_grad():
        stats = {
            "L_graph": float(loss_graph.detach().item()),
            "modularity_Q": float(q_modularity.detach().item()),
            "s1_edge": float(s1_edge.detach().item()),
            "s1_null": float(s1_null.detach().item()),
        }
    return loss_graph, stats


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


def _precompute_null_model_samples(
    pairs: np.ndarray,
    n_items: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample null model pairs (u, v) preserving degree distribution."""
    n_pairs = len(pairs)
    d_out = np.bincount(pairs[:, 0], minlength=n_items).astype(np.float64)
    d_in = np.bincount(pairs[:, 1], minlength=n_items).astype(np.float64)

    p_out = d_out / d_out.sum()
    p_in = d_in / d_in.sum()

    rng = np.random.default_rng(seed + 777)
    u = rng.choice(n_items, size=n_pairs, p=p_out)
    v = rng.choice(n_items, size=n_pairs, p=p_in)

    return u.astype(np.int64), v.astype(np.int64)
