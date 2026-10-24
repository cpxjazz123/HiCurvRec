"""Bootstrap assignment consensus and differentiable RQ consensus loss."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def align_assignments(reference: np.ndarray, replica: np.ndarray, codes: int) -> np.ndarray:
    """Map replica labels to reference labels independently at each level."""
    ref, rep = np.asarray(reference), np.asarray(replica)
    if ref.shape != rep.shape or ref.ndim != 2:
        raise ValueError("assignments must have matching [items, levels] shapes")
    result = np.empty_like(rep)
    for level in range(ref.shape[1]):
        overlap = np.zeros((codes, codes), dtype=np.int64)
        np.add.at(overlap, (rep[:, level], ref[:, level]), 1)
        rows, cols = linear_sum_assignment(-overlap)
        mapping = np.arange(codes)
        mapping[rows] = cols
        result[:, level] = mapping[rep[:, level]]
    return result


def consensus_targets(assignments: list[np.ndarray], codes: int) -> torch.Tensor:
    """Return [items, levels, codes] empirical frequencies; assignments include reference."""
    aligned = [np.asarray(assignments[0])]
    aligned.extend(align_assignments(assignments[0], x, codes) for x in assignments[1:])
    ids = np.stack(aligned, axis=0)
    q = np.zeros((*ids.shape[1:], codes), dtype=np.float32)
    for b in range(ids.shape[0]):
        np.put_along_axis(q, ids[b][..., None], np.take_along_axis(q, ids[b][..., None], axis=2) + 1.0 / len(ids), axis=2)
    return torch.from_numpy(q)


def consensus_kl(encoded: torch.Tensor, codebooks: torch.Tensor, targets: torch.Tensor,
                 gamma: float, tau: float = 0.1) -> torch.Tensor:
    """KL(q || softmax(-squared_distance/tau)), averaged over items and levels."""
    # encoded [N,D], codebooks [L,K,D], q [N,L,K]
    residual = encoded
    logits = []
    for level in range(codebooks.shape[0]):
        distances = torch.cdist(residual, codebooks[level]).square()
        logits.append(-distances / tau)
        residual = residual - codebooks[level][distances.argmin(dim=1)]
    logp = F.log_softmax(torch.stack(logits, dim=1), dim=-1)
    q = targets.to(device=encoded.device, dtype=encoded.dtype)
    return gamma * (q * (q.clamp_min(1e-12).log() - logp)).sum(-1).mean()
