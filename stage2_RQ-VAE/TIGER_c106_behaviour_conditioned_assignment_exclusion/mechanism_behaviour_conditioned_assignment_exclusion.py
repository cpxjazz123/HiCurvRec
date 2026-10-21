"""Behaviour-conditioned exclusion of shared level-zero assignments."""

import torch
import torch.nn.functional as F


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Penalize soft assignment overlap for behaviour-similar directed pairs."""
    tau_assign = float(parameters.get("tau_assign", 0.1))
    sigma_beh = float(parameters.get("sigma_beh", 1.0))
    lambda_excl = float(parameters.get("lambda_excl", 1e-5))
    if tau_assign <= 0 or sigma_beh <= 0:
        raise ValueError("tau_assign and sigma_beh must be positive")

    source_prob = F.softmax(-torch.cdist(source, codebook).square() / tau_assign, dim=-1)
    target_prob = F.softmax(-torch.cdist(target, codebook).square() / tau_assign, dim=-1)
    source_beh = parameters.get("source_behaviour", source)
    target_beh = parameters.get("target_behaviour", target)
    behaviour_distance_sq = (source_beh - target_beh).square().sum(dim=-1)
    similarity = torch.exp(-behaviour_distance_sq / (2.0 * sigma_beh**2))
    overlap = (source_prob * target_prob).sum(dim=-1)
    valid = mask.to(device=overlap.device, dtype=overlap.dtype)
    denom = valid.sum().clamp_min(1.0)
    return lambda_excl * (similarity * overlap * valid).sum() / denom
