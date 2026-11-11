"""The level-1 assignment cost of issue #180: the contrast's own curvature metric, on the selection side.

``TIGER_r48k2_rq`` measured the behaviour contrast with the pair-curvature Poincare distance while the quantizer still assigned
by squared Euclidean distance, so the space that decided neighbourhoods and the space that decided codewords were not the same.
This module gives :class:`model.layers.VQLayer` that same metric for its cost matrix. The Sinkhorn routine itself is untouched:
only the matrix it centres changes, which is what rule 19h requires.

The item curvature arrives through :func:`set_batch_kappa` because the layer sees residuals, not item ids; the trainer hands it
the batch's kappa before the forward pass and clears it afterwards.
"""
from __future__ import annotations

import torch

from . import r48k_geometry

#: Off: the assignment uses the frozen Euclidean cost even though the contrast is curved (issue's CONTRAST_ONLY).
CONTRAST_ONLY = False

_KAPPA: torch.Tensor | None = None


def set_mode(contrast_only: bool) -> None:
    """``True`` reproduces #179 exactly (curvature in the contrast only); ``False`` is this issue's BOTH_ON."""
    global CONTRAST_ONLY
    CONTRAST_ONLY = bool(contrast_only)


def set_batch_kappa(kappa: torch.Tensor | None) -> None:
    """The current batch's per-item curvature, in row order; ``None`` clears it."""
    global _KAPPA
    _KAPPA = None if kappa is None else torch.as_tensor(kappa).reshape(-1)


def batch_kappa() -> torch.Tensor | None:
    return _KAPPA


def _euclidean(latent: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    return (latent.pow(2).sum(1, keepdim=True) - 2 * latent @ codes.t() + codes.pow(2).sum(1, keepdim=True).t())


def level1_cost(latent: torch.Tensor, codes: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``C[i, k] = d_{kappa_i}(proj(r_i), proj(c_k))``; the Euclidean expression when the geometry is off."""
    if CONTRAST_ONLY or r48k_geometry.GEOMETRY_OFF:
        return _euclidean(latent, codes)
    k = torch.as_tensor(kappa).to(device=latent.device, dtype=torch.float64).reshape(-1, 1)
    if k.shape[0] != latent.shape[0]:
        raise ValueError(f"{k.shape[0]} kappas for {latent.shape[0]} rows")
    src = latent.to(torch.float64)
    books = codes.to(torch.float64)
    rows, cols, dim = src.shape[0], books.shape[0], src.shape[-1]
    p = r48k_geometry.project_to_ball(src.unsqueeze(1).expand(rows, cols, dim), k.unsqueeze(-1))
    q = r48k_geometry.project_to_ball(books.unsqueeze(0).expand(rows, cols, dim), k.unsqueeze(-1))
    cost = r48k_geometry.distance(p, q, k.unsqueeze(-1))
    return cost.to(latent.dtype)
