"""Behavior-conditioned nearest/runner-up margin ranking on L1 residuals."""

import torch
import torch.nn.functional as F


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Compare directed paired L1 assignment margins, without code identity targets."""
    tau = float(parameters.get("tau", 1.0))
    epsilon = float(parameters.get("epsilon", 1e-8))
    if tau <= 0:
        raise ValueError("tau must be positive")
    if codebook.shape[0] < 2:
        raise ValueError("the L1 codebook must contain at least two rows")

    # Indices are discrete constants; selected distance expressions retain gradients.
    source_energy = (source[:, None, :] - codebook[None, :, :]).square().sum(dim=-1)
    target_energy = (target[:, None, :] - codebook[None, :, :]).square().sum(dim=-1)
    source_order = source_energy.argsort(dim=1, stable=True)
    target_order = target_energy.argsort(dim=1, stable=True)
    source_nearest, source_runner = source_order[:, 0], source_order[:, 1]
    target_nearest, target_runner = target_order[:, 0], target_order[:, 1]
    rows = torch.arange(source.shape[0], device=source.device)
    source_margin = source_energy[rows, source_runner] - source_energy[rows, source_nearest]
    target_margin = target_energy[rows, target_runner] - target_energy[rows, target_nearest]
    pair_loss = F.softplus((source_margin - target_margin) / tau)
    weights = mask.to(device=pair_loss.device, dtype=pair_loss.dtype).reshape(-1)
    if weights.numel() != pair_loss.numel():
        raise ValueError("mask must have one entry per directed source-target pair")
    return (pair_loss * weights).sum() / (weights.sum() + epsilon)
