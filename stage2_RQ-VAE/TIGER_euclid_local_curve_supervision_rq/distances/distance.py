"""The squared Euclidean distance between an encoded vector and every codeword."""

from __future__ import annotations

import torch

NAME = "euclidean"
DESCRIPTION = "squared Euclidean distance |x|^2 - 2 x.c + |c|^2, the expansion RecBole uses"


def distances(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """`(B, D)` against `(K, D)` -> `(B, K)`, squared Euclidean.

    Kept in the expanded form the baseline trained with: it is one matmul instead of
    a broadcasted subtraction, and the numbers must not move.
    """
    return (
        latent.pow(2).sum(1, keepdim=True)
        - 2 * latent @ codebook.t()
        + codebook.pow(2).sum(1, keepdim=True).t()
    )
