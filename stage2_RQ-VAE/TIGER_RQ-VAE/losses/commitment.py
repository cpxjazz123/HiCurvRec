"""The commitment term: the encoder's output must not drift away from its code."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ._pairs import validate_pair

NAME = "commitment"
SPEC = {"form": "mse_pair", "symmetric": True, "shift_invariant": True,
        "scale_power": 2, "zero_at_equality": True}
DESCRIPTION = "MSE(quantized, sg[latent]) -- pulls the encoder towards the code it chose"


def commitment_loss(latent: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """||q - sg[x]||^2: only the encoder moves, the codebook is the target."""
    validate_pair(quantized, latent.detach(), term=NAME)
    return F.mse_loss(quantized, latent.detach())
