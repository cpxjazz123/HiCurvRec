"""The codebook term: the chosen code must move towards the encoder's output.

    Q(x, q) = MSE(q, sg[x]) = mean((q - sg[x])^2)

`x` is stopped, so the only thing this term can move is `q` -- the codebook entry.
That is the VQ-VAE codebook loss: the entry is pulled towards the vectors assigned to
it. (`SPEC["gradient"]` states this, and the shared suite checks it against the
gradients torch actually produces, so the name cannot drift from the mathematics.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ._pairs import validate_pair

NAME = "codebook"
#: What a test may assume about this formula (see tools/research/stage2/tests/test_losses_math.py).
SPEC = {"form": "mse_pair", "symmetric": True, "shift_invariant": True,
        "scale_power": 2, "zero_at_equality": True, "detached": "first"}
DESCRIPTION = "MSE(quantized, sg[latent]): the gradient reaches the CODEBOOK only"


def codebook_loss(latent: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """||q - sg[x]||^2 -- `latent` is the target, so only the codebook moves."""
    validate_pair(quantized, latent.detach(), term=NAME)
    return F.mse_loss(quantized, latent.detach())
