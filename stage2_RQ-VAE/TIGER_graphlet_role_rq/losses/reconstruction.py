"""The reconstruction term: the decoder must return the embedding it was given."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ._pairs import validate_pair

NAME = "reconstruction"
# What a test may assume about this formula (see tools/research/stage2/tests/test_losses_math.py).
SPEC = {"form": "mse_pair", "symmetric": True, "shift_invariant": True,
        "scale_power": 2, "zero_at_equality": True, "detached": "none"}
DESCRIPTION = "mean squared error between the reconstruction and the item embedding"


def reconstruction_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE(reconstruction, embedding) -- the term that keeps the SID decodable."""
    validate_pair(reconstructed, target, term=NAME)
    return F.mse_loss(reconstructed, target)
