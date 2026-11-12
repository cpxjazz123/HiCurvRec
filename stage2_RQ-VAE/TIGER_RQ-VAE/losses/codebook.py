"""The codebook term: the chosen code must move towards the encoder's output."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ._pairs import validate_pair

NAME = "codebook"
SPEC = {"form": "mse_pair", "symmetric": True, "shift_invariant": True,
        "scale_power": 2, "zero_at_equality": True}
DESCRIPTION = "MSE(sg[latent], quantized) -- pulls the chosen code towards the encoder"


def codebook_loss(latent: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """||sg[x] - q||^2: only the codebook moves, the encoder is the target."""
    validate_pair(latent, quantized.detach(), term=NAME)
    return F.mse_loss(latent, quantized.detach())
