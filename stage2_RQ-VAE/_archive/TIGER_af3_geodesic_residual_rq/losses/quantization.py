"""The quantisation term: the codebook's own objective, per level and averaged.

`VQLayer` reports one pair per residual level, `(commitment, codebook)`, where
commitment is the term whose gradient reaches the encoder and codebook the one whose
gradient reaches the codebook entry. This module turns those pairs into the single
scalar the training loop sees.

The baseline's weighting (RecBole3.0, kept for the frozen numbers) is

    Q + beta * C        C = MSE(x, sg[q])   moves the encoder
                        Q = MSE(q, sg[x])   moves the codebook

Note where beta sits: on the commitment term. That is not the common VQ-VAE ordering
(which weights the codebook term), and it is written down here so nobody re-derives it
from the letters C and Q.

EMA / SimVQ have no gradient term for the codebook -- the entries are an exponential
moving average -- so their pair is `(C, None)` and the level scores `beta * C`.
"""

from __future__ import annotations

import torch

from .codebook import codebook_loss
from .commitment import commitment_loss

NAME = "quantization"
SPEC = {"form": "composite"}
DESCRIPTION = "codebook + beta * commitment, averaged over the residual levels"


def quantization_loss(
    latent: torch.Tensor,
    quantized: torch.Tensor,
    beta: float,
    *,
    codebook_term: bool = True,
) -> torch.Tensor:
    """One level's codebook objective.

    `codebook_term=False` is the EMA/SimVQ shape: the entries move by EMA, so only the
    commitment term is left, weighted by `beta` exactly as the baseline did.
    """
    commitment = commitment_loss(latent, quantized)
    if not codebook_term:
        return beta * commitment
    return codebook_loss(latent, quantized) + beta * commitment


def quantization_total(parts, beta: float):
    """Average the levels' `(commitment, codebook)` pairs into one scalar.

    `codebook` is None for the EMA variant, which has no gradient term for the entries;
    adding a zero there would work numerically but would claim a term the variant does
    not have.
    """
    total = 0.0
    levels = 0
    for commitment, codebook in parts:
        levels += 1
        term = beta * commitment if codebook is None else codebook + beta * commitment
        total = total + term
    if levels == 0:
        raise ValueError("the quantiser returned no levels to score")
    return total / levels
