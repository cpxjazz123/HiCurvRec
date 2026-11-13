"""The squared Euclidean distance between an encoded vector and every codeword."""

from __future__ import annotations

import torch

NAME = "euclidean"
DESCRIPTION = "squared Euclidean distance |x|^2 - 2 x.c + |c|^2, the expansion RecBole uses"


def distances(latent: torch.Tensor, codebook: torch.Tensor,
              p=None, mode: str | None = None) -> torch.Tensor:
    """`(B, D)` against `(K, D)` -> `(B, K)`, squared Euclidean.

    Kept in the expanded form the baseline trained with: it is one matmul instead of
    a broadcasted subtraction, and the numbers must not move.

    `p` and `mode` exist only so the quantiser can call every registry entry with the
    same positional shape `(latent, codebook, p, mode)`. This entry IGNORES both: the
    Euclidean metric has no per-item share and no geometric switch, which is exactly
    why `geometry_off` can be checked elementwise against the frozen baseline.
    """
    if not torch.is_tensor(latent):
        raise ValueError("the Euclidean distance takes the 32-vector, not the charted "
                         "triple; pass it to the factor_share entry")
    return (
        latent.pow(2).sum(1, keepdim=True)
        - 2 * latent @ codebook.t()
        + codebook.pow(2).sum(1, keepdim=True).t()
    )
