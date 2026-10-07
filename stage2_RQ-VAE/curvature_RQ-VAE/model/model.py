"""Standalone RecBole3.0-compatible hyperbolic RQ-VAE wrapper for HG-Rec.

The encoder may compose tangent residuals with fixed-curvature Mobius addition.
The transition-ranking auxiliary objective acts on origin-tangent encoder
outputs before quantization.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, RQLayer, _poincare_distance_tangent_pairs


class RQVAE(nn.Module):
    """Encode item embeddings into residual-quantized semantic identifiers."""

    def __init__(self, config: Any, *, in_dim: int):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        residual_curvature = (
            float(config.layer_curvatures[0])
            if getattr(config, "curvature_residual_encoder", False)
            else None
        )
        self.encoder = MLP(
            list(self.encoder_sizes),
            dropout=float(config.dropout),
            residual_curvature=residual_curvature,
        )
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


def behaviour_ranking_loss(
    source: torch.Tensor,
    successor: torch.Tensor,
    negatives: torch.Tensor,
    curvature: float,
    margin: float,
) -> torch.Tensor:
    """Hinge ranking over real transition successors and shuffled negatives.

    Each input is an origin-tangent encoder vector. The distance helper maps
    both endpoints to the fixed-curvature Poincare ball before measuring the
    geodesic distance; no quantized-code equality is involved.
    """
    positive = _poincare_distance_tangent_pairs(source, successor, curvature)
    negative = _poincare_distance_tangent_pairs(source, negatives, curvature)
    return F.relu(positive + margin - negative).mean()
