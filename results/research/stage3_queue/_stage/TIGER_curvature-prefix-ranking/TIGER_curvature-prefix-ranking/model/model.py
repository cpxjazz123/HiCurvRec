"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP
from .recording import RecordingRQLayer


class RQVAE(nn.Module):
    """Encode semantic item embeddings into residual-quantized integer SIDs."""

    def __init__(self, config: Any, *, in_dim: int):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = RecordingRQLayer(config)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    def forward(self, embeddings: torch.Tensor,
                item_ids: torch.Tensor = None,
                radius: torch.Tensor = None
                ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """The framework's model contract: a fifth ``reference`` output.

        ``item_ids`` and ``radius`` belong to the framework's probe contract;
        this mechanism ranks whole items with the global curvature, so both are
        accepted and unused.
        ``reference`` is the reconstruction target the loss is measured against,
        which is the input for this arm.
        """
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens, embeddings

    def revert_to_euclidean_assignment(self) -> None:
        """Fall back to the Euclidean assignment rule (see RecordingRQLayer)."""
        self.rq.revert_to_euclidean()

    def level_points(self) -> list[torch.Tensor]:
        """The residual tangent that enters every level of the last forward."""
        return self.rq.level_points()

    def level_codebooks(self) -> list[torch.Tensor]:
        """The codebook of every level."""
        return [layer.get_code_embs() for layer in self.rq.vq_layers]

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False,
                    item_ids: torch.Tensor = None) -> torch.Tensor:
        """``item_ids`` belongs to the framework's probe contract and is unused:
        this mechanism's assignment is the baseline's, item-independent one."""
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
