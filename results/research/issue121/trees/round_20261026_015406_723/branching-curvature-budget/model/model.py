"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, RQLayer


class RQVAE(nn.Module):
    """Encode semantic item embeddings into residual-quantized integer SIDs."""

    def __init__(self, config: Any, *, in_dim: int):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = RQLayer(config)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    def forward(self, embeddings: torch.Tensor, branching: torch.Tensor | None = None,
                edge_embeddings: torch.Tensor | None = None,
                edge_branching: torch.Tensor | None = None,
                edge_source_branching: torch.Tensor | None = None):
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded, branching=branching)
        reconstructed = self.decoder(quantized)
        if edge_embeddings is None:
            return reconstructed, quant_loss, int(unused_codes), tokens
        edge_encoded = self.encoder(edge_embeddings)
        _, _, _, edge_tokens = self.rq(edge_encoded, branching=edge_branching)
        n = len(edge_embeddings) // 2
        curvature = (1.0 - 0.85 * edge_source_branching).clamp(0.15, 1.0)
        distances = []
        for level, layer in enumerate(self.rq.vq_layers):
            codes = layer.get_code_embs()
            left = codes[edge_tokens[:n, level]]
            right = codes[edge_tokens[n:, level]]
            distances.append(layer._poincare_distance_sq(left, right, curvature).diagonal().mean())
        return reconstructed, quant_loss, int(unused_codes), tokens, torch.stack(distances).mean()

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False, branching: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk, branching=branching)
        return tokens

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
