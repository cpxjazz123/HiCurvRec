"""Standalone RecBole3.0-compatible hyperbolic RQ-VAE wrapper for HG-Rec.

Pure Poincare-ball geometry: codebook assignment, quantization loss and
reconstruction all run in hyperbolic space at a fixed curvature. The
behavior-contrastive auxiliary objective is not part of this model.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, RQLayer, TANGENT_RADIUS


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
        self.tangent_radius = float(config.tangent_radius)

    def _to_tangent_space(self, encoded: torch.Tensor) -> torch.Tensor:
        """Place the latent on a fixed-radius shell before expmap0.

        The encoder naturally settles at a small radius (measured ~0.024 for
        the trained checkpoint), where the Poincare expmap is linear to within
        1e-4 and the geometry is a no-op: d_poincare / d_euclidean is a
        constant factor of 2.0, so every curvature gives the same assignment.

        Rescaling to a fixed radius keeps the direction, which is what the
        encoder learns, and fixes the magnitude, which is what the geometry
        needs. A clamp would be wrong here: once saturated it passes no
        gradient back to the encoder.
        """
        radius = torch.linalg.vector_norm(encoded, dim=-1, keepdim=True)
        return encoded * (self.tangent_radius / radius.clamp_min(1e-6))

    def get_curvatures(self) -> torch.Tensor:
        return self.rq.get_curvatures()

    def forward(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        encoded = self._to_tangent_space(self.encoder(embeddings))
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(
        self, embeddings: torch.Tensor, *, infer_use_sk: bool = False
    ) -> torch.Tensor:
        encoded = self._to_tangent_space(self.encoder(embeddings))
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk)
        return tokens

    @torch.no_grad()
    def get_indices_with_stats(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        encoded = self._to_tangent_space(self.encoder(embeddings))
        return self.rq.get_indices_with_stats(encoded)

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor) -> None:
        encoded = self._to_tangent_space(self.encoder(embeddings))
        self.rq.init_codebook(encoded, embeddings.device)

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        reconstructed: torch.Tensor,
        quant_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(reconstructed, embeddings)
        return recon_loss + quant_loss, recon_loss
