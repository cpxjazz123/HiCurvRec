"""The repository's own anti-collapse guard, brought here unchanged.

This is the diversity term of the behaviour-sample tree
(``stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/anti_collapse_losses.py``), the
arm that holds the best recall in this campaign. It penalises a non-uniform
average soft code assignment at every level, which is exactly the failure the
community contrast produces: the community makes the latents of a group
similar, the quantiser reuses codes, and the codebook's usage collapses.

The code is copied verbatim so the guard is the repository's, not a new one;
only the routing constants live in ``anti_collapse_config.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
