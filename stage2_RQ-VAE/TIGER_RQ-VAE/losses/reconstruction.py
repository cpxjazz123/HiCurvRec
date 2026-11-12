"""The reconstruction term: the decoder must return the embedding it was given."""

from __future__ import annotations

import torch
import torch.nn.functional as F

NAME = "reconstruction"
DESCRIPTION = "mean squared error between the reconstruction and the item embedding"


def reconstruction_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE(reconstruction, embedding) -- the term that keeps the SID decodable."""
    return F.mse_loss(reconstructed, target)
