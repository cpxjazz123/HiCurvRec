"""Uncertainty-weighted SID agreement utilities for embedding-view training."""
from __future__ import annotations

import torch


def uncertainty_agreement(encoded_views: torch.Tensor, token_views: torch.Tensor) -> torch.Tensor:
    """Variance-weighted agreement with a differentiable latent surrogate.

    encoded_views: [M,B,D], token_views: [M,B,L]. The discrete term is a
    Hard token disagreement follows the specified indicator. Since hard IDs
    cannot provide gradients, use the corresponding pairwise latent distance
    as a straight-through-style surrogate, retaining the hard value in the
    forward pass and routing gradients through the surrogate.
    """
    if encoded_views.ndim != 3 or token_views.ndim != 3:
        raise ValueError("expected encoded [M,B,D] and tokens [M,B,L]")
    if encoded_views.shape[:2] != token_views.shape[:2] or encoded_views.shape[0] < 2:
        raise ValueError("views must align and include at least two samples")
    variance = (encoded_views - encoded_views.mean(dim=0, keepdim=True)).square().sum(-1).mean(0)
    disagreements = []
    surrogates = []
    for left in range(token_views.shape[0]):
        for right in range(left + 1, token_views.shape[0]):
            disagreements.append((token_views[left] != token_views[right]).float().sum(-1))
            # Smoothly penalize latent instability even when hard assignments
            # happen to match. This is differentiable w.r.t. encoder outputs.
            surrogates.append((encoded_views[left] - encoded_views[right]).square().sum(-1))
    disagreement = torch.stack(disagreements).mean(0)
    surrogate = torch.stack(surrogates).mean(0)
    hard = (variance * disagreement).mean()
    soft = (variance * surrogate).mean()
    return hard.detach() + soft - soft.detach()
