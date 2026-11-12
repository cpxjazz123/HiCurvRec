"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from losses import build_loss

from .layers import MLP, RQLayer


class RQVAE(nn.Module):
    """Encode semantic item embeddings into residual-quantized integer SIDs."""

    def __init__(self, config: Any, *, in_dim: int, distance=None):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        # The metric comes from the caller; the model does not choose one.
        self.rq = RQLayer(config, distance=distance)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, list, int, torch.Tensor]:
        """Return the reconstruction, the per-level loss parts, unused codes, tokens.

        The parts are `(commitment, codebook)` pairs, one per residual level; the
        model does not turn them into a scalar. That is `losses/quantization.py`.
        """
        encoded = self.encoder(embeddings)
        quantized, quant_parts, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_parts, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk)
        return tokens

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor) -> None:
        encoded = self.encoder(embeddings)
        self.rq.init_codebook(encoded, embeddings.device)

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        reconstructed: torch.Tensor,
        quant_parts: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Total and reconstruction loss, computed by the registry, not here.

        Kept for callers outside the trainer (a snapshot, a probe); the training
        loop builds the criterion itself from `losses.build_loss`.
        """
        total, components = build_loss(self.config)(reconstructed, embeddings, quant_parts)
        return total, components["reconstruction"]
