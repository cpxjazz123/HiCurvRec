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

    def forward(self, embeddings: torch.Tensor, cohort_embeddings: torch.Tensor | None = None, cohort_level: int = 0):
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        cohort_codes = self.cohort_code_vectors(cohort_embeddings, cohort_level) if cohort_embeddings is not None else encoded[:0]
        return reconstructed, quant_loss, int(unused_codes), tokens, cohort_codes

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
        quant_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(reconstructed, embeddings)
        return recon_loss + quant_loss, recon_loss

    def cohort_code_vectors(self, source_embeddings: torch.Tensor, level: int) -> torch.Tensor:
        """Selected RQ-level vectors with gradients to both encoder and codebook."""
        encoded = self.encoder(source_embeddings)
        residual = encoded
        selected = None
        for index, layer in enumerate(self.rq.vq_layers):
            ids = layer._indices(layer._distances(residual), False)
            if index == level:
                code = layer.embed_code(ids)
                # Forward is the hard selected code; backward reaches both the
                # codebook (through `code`) and encoder (straight-through term).
                selected = code + encoded - encoded.detach()
                break
            residual = residual - layer.embed_code(ids)
        if selected is None:
            raise ValueError(f"Invalid RQ level: {level}")
        return selected
