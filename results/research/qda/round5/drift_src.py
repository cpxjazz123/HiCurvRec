"""Temporal drift of the encoder output, and the reference it is measured against.

The encoder output moves 10-35 % of its own norm per epoch during the collapse window
while the L1 codebook moves 0.1-0.6 %, so a code that falls behind its items loses them
for good. This module penalises the per-item output displacement between consecutive
epochs and nothing else: the quantiser keeps its argmin rule, and the codebook keeps its
own learning rate.

    L_drift = mean_i || z_i^t - sg(z_i^{t-1}) ||^2 / ( || sg(z_i^{t-1}) ||^2 + eps )

`previous` is a buffer refreshed once per epoch from the whole training set, so every
item is compared against its own last-epoch representation regardless of batch order.
"""
from __future__ import annotations

import torch


def drift_loss(z: torch.Tensor, previous: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Relative squared displacement against a stop-gradient of the previous epoch."""
    previous = previous.detach()
    numerator = (z - previous).pow(2).sum(dim=1)
    denominator = previous.pow(2).sum(dim=1) + eps
    return (numerator / denominator).mean()


@torch.no_grad()
def snapshot(encoder, embeddings: torch.Tensor) -> torch.Tensor:
    """This epoch's reference for every training item, detached."""
    return encoder(embeddings).detach().clone()
