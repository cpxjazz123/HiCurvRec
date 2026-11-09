"""Behavior-conditioned, hard-assignment residual geometry matching."""

import torch


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Match source and corresponding target residuals after nearest-code lookup.

    Nearest-code indices are discrete and consequently fixed for each forward
    pass; gathered codebook rows remain attached to autograd.
    """
    if source.ndim != 2 or target.shape != source.shape:
        raise ValueError("source and target must have the same [B, D] shape")
    if codebook.ndim != 2 or codebook.shape[1] != source.shape[1]:
        raise ValueError("codebook must have shape [C, D]")
    if mask.ndim != 1 or mask.shape[0] != source.shape[0]:
        raise ValueError("mask must have shape [B]")

    weight = float(parameters.get("lambda", parameters.get("weight", 0.001)))
    beta = float(parameters.get("beta", 0.1))
    # Argmin is non-differentiable by definition; gathered rows are not detached.
    source_idx = (source[:, None, :] - codebook[None, :, :]).pow(2).sum(-1).argmin(-1)
    target_idx = (target[:, None, :] - codebook[None, :, :]).pow(2).sum(-1).argmin(-1)
    source_residual = source - codebook.index_select(0, source_idx)
    target_residual = target - codebook.index_select(0, target_idx)
    pair_error = (source_residual - target_residual).pow(2).sum(dim=-1)
    source_anchor = source_residual.pow(2).sum(dim=-1)
    valid = mask.to(device=pair_error.device, dtype=pair_error.dtype)
    denominator = valid.sum().clamp_min(1.0)
    return weight * ((pair_error + beta * source_anchor) * valid).sum() / denominator
