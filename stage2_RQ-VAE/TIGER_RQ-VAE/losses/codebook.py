"""The codebook term: the chosen code must move towards the encoder's output."""

from __future__ import annotations

import torch

from ._pairs import pair_error

NAME = "codebook"
DESCRIPTION = "MSE(sg[latent], quantized) -- pulls the chosen code towards the encoder"


def codebook_loss(latent: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """||sg[x] - q||^2: only the codebook moves, the encoder is the target."""
    return pair_error(latent, quantized.detach(), term=NAME)
