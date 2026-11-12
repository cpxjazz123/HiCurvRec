"""The quantisation term: the codebook's own objective, per level and summed.

`VQLayer` computes one `(commitment, codebook)` pair per residual level; this module
turns those pairs into the single scalar the training loop sees, so the formula lives
in one place instead of inside the quantiser.

Two shapes exist and both are written here, not in the layer that calls them:
    vq            commitment + beta * codebook
    ema / simvq   beta * codebook        (the codebook is an EMA, there is no
                                          commitment term to add)
"""

from __future__ import annotations

import torch

from .codebook import codebook_loss
from .commitment import commitment_loss

NAME = "quantization"
DESCRIPTION = "the codebook objective: commitment + beta * codebook, averaged over levels"


def quantization_loss(
    latent: torch.Tensor,
    quantized: torch.Tensor,
    beta: float,
    *,
    commitment: bool = True,
) -> torch.Tensor:
    """One level's codebook loss: `commitment + beta * codebook`, or `beta * codebook`."""
    codebook = codebook_loss(latent, quantized)
    if not commitment:
        return beta * codebook
    return commitment_loss(latent, quantized) + beta * codebook


def quantization_total(parts, beta: float):
    """Average the levels' `(commitment, codebook)` pairs into one scalar.

    `commitment` is None for the EMA variant, which has no commitment term; adding a
    zero there would keep the value but claim a term the variant does not have.
    """
    total = 0.0
    levels = 0
    for commitment, codebook in parts:
        levels += 1
        term = beta * codebook if commitment is None else commitment + beta * codebook
        total = total + term
    if levels == 0:
        raise ValueError("the quantiser returned no levels to score")
    return total / levels
