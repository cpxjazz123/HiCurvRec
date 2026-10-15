"""Auxiliary alignment, diversity, and Gumbel quantization losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from anti_collapse_config import ALIGN_WEIGHT, DIVERSITY_WEIGHT, GUMBEL_TAU
from quant_rank_config import LEVEL_WEIGHTS


def align_loss(residuals: torch.Tensor, quantized_sum: torch.Tensor) -> torch.Tensor:
    """Return the mean per-item squared distance from the encoder output."""
    encoder_output = residuals[0]
    return (quantized_sum - encoder_output).pow(2).sum(dim=-1).mean()


def diversity_loss(residuals: torch.Tensor, vq_layers, tau: float) -> torch.Tensor:
    """Penalize non-uniform average soft code assignments at every level."""
    total = residuals.new_zeros(())
    for level, layer in enumerate(vq_layers):
        codebook = layer.get_code_embs()
        distances = (residuals[level, :, None, :] - codebook[None, :, :]).pow(2).sum(dim=-1)
        assignments = F.softmax(-distances / tau, dim=-1)
        mean_assignment = assignments.mean(dim=0)
        uniform = torch.full_like(mean_assignment, 1.0 / codebook.shape[0])
        total = total + (mean_assignment - uniform).pow(2).sum()
    return total


def gumbel_quantized_representation(
    residuals: torch.Tensor,
    vq_layers,
    tau: float,
    hard: bool = False,
) -> torch.Tensor:
    """Build a weighted Euclidean Gumbel representation with an encoder STE."""
    encoder_output = residuals[0]
    quantized_sum = torch.zeros_like(encoder_output)
    for level, (layer, weight) in enumerate(zip(vq_layers, LEVEL_WEIGHTS)):
        codebook = layer.get_code_embs()
        distances = (residuals[level, :, None, :] - codebook[None, :, :]).pow(2).sum(dim=-1)
        probabilities = F.gumbel_softmax(-distances / tau, tau=1.0, hard=hard, dim=-1)
        quantized_level = probabilities @ codebook
        quantized_sum = quantized_sum + float(weight) * quantized_level
    return quantized_sum + (encoder_output - encoder_output.detach())


def compose(behavior_loss: torch.Tensor, term: torch.Tensor) -> torch.Tensor:
    """Add a nonzero auxiliary term while preserving the zero-term identity."""
    if not torch.count_nonzero(term.detach()).item():
        return behavior_loss
    return behavior_loss + term
