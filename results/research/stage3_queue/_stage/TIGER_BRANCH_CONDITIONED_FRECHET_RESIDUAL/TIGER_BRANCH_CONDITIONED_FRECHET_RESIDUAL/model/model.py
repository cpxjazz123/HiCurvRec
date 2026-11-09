"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec.

The candidate mechanism changes the fine levels only. The coarse level stays
the baseline's Euclidean quantizer, and the decoder, the losses and the export
stay the baseline's.

The fine levels are quantized in the tangent space of a branch anchor: the
training-only transition branches choose the anchor, the anchor is the Frechet
mean of the successor residuals of that branch, and the residual is written as
``u = log_a(r)`` before the codebook sees it. The behaviour signal and the
curved operation therefore meet on the same computation path: removing the
behaviour signal replaces every anchor by one global anchor, and removing the
geometry replaces the curved anchor-relative residual by an ordinary Euclidean
subtraction.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperbolic import (ball_exp_map, ball_log_map, ball_project, frechet_mean,
                         tangent_codes)
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

    def forward(self, embeddings: torch.Tensor, fine_context: dict | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded, fine_context=fine_context)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False,
                    fine_context: dict | None = None) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk,
                                  fine_context=fine_context)
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
