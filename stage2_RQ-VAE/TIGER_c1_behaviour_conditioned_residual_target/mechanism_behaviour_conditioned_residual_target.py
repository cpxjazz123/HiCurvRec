"""Behaviour-conditioned residual target auxiliary objective."""

import torch


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Return the masked directed residual-target loss for one quantization level."""
    alpha = parameters.get("alpha", 0.1)
    beta = parameters.get("beta", 1.0)
    eps = parameters.get("eps", 1e-8)

    # Keep the host's hard nearest-code lookup.  For this auxiliary objective, use
    # the selected codeword directly: putting an identity STE on q here would
    # algebraically cancel the source derivative in both (r - q) and (t - q).
    # The ordinary host quantizer/export path is untouched.
    distances = (source.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(dim=-1)
    indices = distances.argmin(dim=1)
    quantized = codebook[indices]

    source_norm = source.norm(p=2, dim=-1, keepdim=True)
    with torch.no_grad():
        source_direction = source / (source_norm + eps)
        target_direction = target / (target.norm(p=2, dim=-1, keepdim=True) + eps)
        direction_delta = target_direction - source_direction
    target_residual = source + alpha * source_norm.detach() * direction_delta

    pair_loss = (source - quantized).pow(2).sum(dim=-1)
    pair_loss = pair_loss + beta * (target_residual - quantized).pow(2).sum(dim=-1)
    weights = mask.to(device=pair_loss.device, dtype=pair_loss.dtype)
    return (pair_loss * weights).sum() / (weights.sum() + eps)
