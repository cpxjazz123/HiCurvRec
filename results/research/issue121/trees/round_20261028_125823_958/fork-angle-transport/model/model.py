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
        # One learned linear map per quantization level: it turns the weighted
        # mean of the successor-neighbourhood representations into the tangent
        # displacement that the curved assignment sees.
        # A ModuleList of Linear modules: ParameterList only accepts
        # Parameters, and a Linear module is not one.
        self.fork_maps = nn.ModuleList([
            nn.Linear(int(config.codebook_dim), int(config.codebook_dim), bias=False)
            for _ in range(int(config.codebook_num))])

    def forward(self, embeddings: torch.Tensor, fork_context: dict | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        encoded = self.encoder(embeddings)
        context = self._fork_context(encoded, fork_context)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded, fork_context=context)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    def _fork_context(self, encoded: torch.Tensor, fork_context: dict | None) -> dict | None:
        """The tangent displacement of every level.

        The displacement is the learned map applied to the weighted mean of the
        successor-neighbourhood representations, scaled by the branching mass
        that the behaviour signal states.
        """
        if fork_context is None:
            return None
        mean = fork_context["neighbour_mean"]
        scale = fork_context["scale"].unsqueeze(-1)
        shifts = [scale * self.fork_maps[level](mean) for level in range(len(self.fork_maps))]
        prepared = dict(fork_context)
        prepared["shifts"] = shifts
        return prepared

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False,
                    fork_context: dict | None = None) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        context = self._fork_context(encoded, fork_context)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk, fork_context=context)
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
