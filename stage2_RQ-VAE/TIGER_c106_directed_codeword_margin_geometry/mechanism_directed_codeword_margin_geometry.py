"""Directed midpoint-to-codeword margin loss.

Nearest-code assignments are selected from the endpoints and treated as fixed
indices for differentiation. The distances themselves remain differentiable
with respect to both endpoint representations and the codebook vectors.
"""

import torch
import torch.nn.functional as F


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Compute the masked directed codeword-margin objective.

    Args are source/target ``[B, D]``, codebook ``[C, D]`` and mask ``[B]``.
    """
    delta = float(parameters.get("delta", 0.1))
    weight = float(parameters.get("lambda", 0.001))
    tau = float(parameters.get("tau", 0.5))
    if tau <= 0:
        raise ValueError("tau must be positive")
    if source.ndim != 2 or target.shape != source.shape:
        raise ValueError("source and target must have matching [B, D] shapes")
    if codebook.ndim != 2 or codebook.shape[1] != source.shape[1]:
        raise ValueError("codebook must have shape [C, D] matching representations")
    if mask.ndim != 1 or mask.shape[0] != source.shape[0]:
        raise ValueError("mask must have shape [B]")

    # Indices are hard, endpoint-assigned and deliberately detached from the
    # gradient; the selected codeword distances below are not detached.
    with torch.no_grad():
        source_idx = torch.cdist(source.detach(), codebook.detach()).square().argmin(dim=1)
        target_idx = torch.cdist(target.detach(), codebook.detach()).square().argmin(dim=1)

    midpoint = (source + target) * 0.5
    source_code = codebook[source_idx]
    target_code = codebook[target_idx]
    d_source = (midpoint - source_code).square().sum(dim=-1)
    d_target = (midpoint - target_code).square().sum(dim=-1)
    per_pair = F.softplus((delta + d_source - d_target) / tau)
    valid = mask.to(device=per_pair.device, dtype=per_pair.dtype)
    return weight * (per_pair * valid).sum() / valid.sum().clamp_min(1.0)
