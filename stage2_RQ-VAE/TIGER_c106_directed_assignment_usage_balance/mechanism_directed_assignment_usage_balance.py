"""Directed endpoint separation with source-assignment usage balancing."""

import torch
import torch.nn.functional as F


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Compute the proposal loss; source/target assignments are fixed in backward."""
    epsilon = parameters.get("epsilon", 0.001)
    gamma = parameters.get("gamma", 0.1)
    weight = parameters.get("lambda", 0.001)
    rho = parameters.get("rho", 0.2)
    tau = parameters.get("tau", 0.05)

    valid = mask.to(device=codebook.device) > 0
    if not bool(valid.any()):
        return codebook.sum() * 0.0

    # Nearest-neighbour decisions are intentionally constants for differentiation.
    src = source.detach().to(device=codebook.device)
    tgt = target.detach().to(device=codebook.device)
    assignments_src = ((src[:, None, :] - codebook.detach()[None, :, :]) ** 2).sum(-1).argmin(-1)
    assignments_tgt = ((tgt[:, None, :] - codebook.detach()[None, :, :]) ** 2).sum(-1).argmin(-1)
    a, b = assignments_src[valid], assignments_tgt[valid]

    separation = torch.linalg.vector_norm(codebook[a] - codebook[b], dim=-1)
    separation_term = F.softplus((rho - separation) / tau)
    counts = torch.bincount(a, minlength=codebook.shape[0]).to(codebook.dtype)
    n = int(a.numel())
    p = (counts + epsilon) / (n + codebook.shape[0] * epsilon)
    usage_term = gamma * (p - 1.0 / codebook.shape[0]).square().sum()
    return weight * (separation_term.mean() + usage_term)
