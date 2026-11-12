"""The commitment term: the encoder must move towards the code it chose.

    C(x, q) = MSE(x, sg[q]) = mean((x - sg[q])^2)

`q` is stopped, so the only thing this term can move is `x` -- the encoder output.
That is the VQ-VAE commitment loss: the encoder is committed to the codebook entry it
picked. (`SPEC["gradient"]` states this, and the shared suite checks it against the
gradients torch actually produces, so the name cannot drift from the mathematics.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ._pairs import validate_pair

NAME = "commitment"
#: What a test may assume about this formula (see tools/research/stage2/tests/test_losses_math.py).
SPEC = {"form": "mse_pair", "symmetric": True, "shift_invariant": True,
        "scale_power": 2, "zero_at_equality": True, "detached": "second"}
DESCRIPTION = "MSE(latent, sg[quantized]): the gradient reaches the ENCODER only"


def commitment_loss(latent: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """||x - sg[q]||^2 -- `quantized` is the target, so only the encoder moves."""
    validate_pair(latent, quantized, term=NAME)
    return F.mse_loss(latent, quantized.detach())
