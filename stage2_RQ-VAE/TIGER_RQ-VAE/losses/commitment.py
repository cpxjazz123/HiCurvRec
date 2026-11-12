"""The commitment term: the encoder's output must not drift away from its code."""

from __future__ import annotations

import torch

from ._pairs import pair_error

NAME = "commitment"
DESCRIPTION = "MSE(quantized, sg[latent]) -- pulls the encoder towards the code it chose"


def commitment_loss(latent: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """||q - sg[x]||^2: only the encoder moves, the codebook is the target."""
    return pair_error(quantized, latent.detach(), term=NAME)
