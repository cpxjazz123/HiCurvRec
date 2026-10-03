"""Standalone RecBole3.0-compatible hyperbolic RQ-VAE wrapper for HG-Rec.

Pure Poincare-ball geometry: codebook assignment, quantization loss and
reconstruction all run in hyperbolic space at a fixed curvature. The
transition-ranking auxiliary objective is applied to origin-tangent encoder
outputs, rescaled onto the same working shell the residual quantizer
quantizes at so both objectives read the same point of the ball.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    MLP,
    RQLayer,
    _curvature_like,
    _poincare_distance_tangent_pairs,
)


class RQVAE(nn.Module):
    """Encode item embeddings into residual-quantized semantic identifiers."""

    def __init__(self, config: Any, *, in_dim: int):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = RQLayer(config)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    def get_curvatures(self) -> torch.Tensor:
        return self.rq.get_curvatures()

    def forward(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(
        self, embeddings: torch.Tensor, *, infer_use_sk: bool = False
    ) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk)
        return tokens

    @torch.no_grad()
    def get_indices_with_stats(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        encoded = self.encoder(embeddings)
        return self.rq.get_indices_with_stats(encoded)

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor) -> None:
        encoded = self.encoder(embeddings)
        self.rq.init_codebook(encoded, embeddings.device)

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        reconstructed: torch.Tensor,
        quant_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(reconstructed, embeddings)
        return recon_loss + quant_loss, recon_loss


def _rescale_to_working_radius(
    x: torch.Tensor, target_radius: float, curvature: torch.Tensor | float
) -> torch.Tensor:
    """Project each tangent vector onto the working shell, direction kept.

    The quantization levels overwrite the magnitude of their input to
    ``target_radius / sqrt(c)`` before the Poincare map, so the encoder latent
    only ever reaches RQ at that radius. Ranking on the raw latent therefore
    measured distances in a region the quantizer discards. Scaling the vectors
    onto the same shell first makes the ranking loss and the residual
    subtraction read the same point of the ball.
    """
    if target_radius == 0.0:
        return x
    sqrt_c = _curvature_like(curvature, x).sqrt()
    target_norm = (target_radius / sqrt_c.clamp_min(
        torch.finfo(x.dtype).tiny
    )).to(dtype=x.dtype)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (target_norm / norm.clamp_min(torch.finfo(x.dtype).tiny))


def behaviour_ranking_loss(
    source: torch.Tensor,
    successor: torch.Tensor,
    negatives: torch.Tensor,
    curvature: float,
    margin: float,
    working_radius: float = 0.0,
) -> torch.Tensor:
    """Hinge ranking over real transition successors and shuffled negatives.

    Each input is an origin-tangent encoder vector. The distance helper maps
    both endpoints to the fixed-curvature Poincare ball before measuring the
    geodesic distance; no quantized-code equality is involved.

    ``working_radius`` is the shell the residual quantizer actually works on.
    A non-zero value measures the ranking on that shell, so the auxiliary loss
    optimizes the same region of the ball that RQ keeps; the encoder's own
    magnitude no longer decides where the loss is evaluated.
    """
    source = _rescale_to_working_radius(source, working_radius, curvature)
    successor = _rescale_to_working_radius(successor, working_radius, curvature)
    negatives = _rescale_to_working_radius(negatives, working_radius, curvature)
    positive = _poincare_distance_tangent_pairs(source, successor, curvature)
    negative = _poincare_distance_tangent_pairs(source, negatives, curvature)
    return F.relu(positive + margin - negative).mean()
