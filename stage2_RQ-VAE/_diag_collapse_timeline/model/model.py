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

    def forward(
        self,
        embeddings: torch.Tensor,
        item_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Encode, quantise, reconstruct.

        ``item_ids`` are the global ids of the rows.  The mechanism needs them to
        gather each item's frozen behaviour rank; the baseline ignores them, so a
        caller that passes none is the baseline's call exactly.
        """
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded, item_ids=item_ids)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(
        self,
        embeddings: torch.Tensor,
        *,
        infer_use_sk: bool = False,
        item_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk, item_ids=item_ids)
        return tokens

    # The mechanism's two pieces of state, reached through the wrapper: the frozen
    # rank table and the measured scale of the offset.
    def set_stratum_table(self, rank: torch.Tensor, item_ids: torch.Tensor) -> None:
        self.rq.set_stratum_table(rank, item_ids)

    def set_stratum_beta(self, beta: float) -> None:
        self.rq.set_stratum_beta(beta)

    def level1_chosen_distance(self):
        """The level-1 distance of the last forward's chosen codewords."""
        return self.rq.level1_chosen_distance

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
