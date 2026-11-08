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

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

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


class MixedRQVAE(nn.Module):
    """The same encoder/decoder skeleton, with the native mixed quantizer inside.

    The reconstruction is decoded from the readout ``phi(q)`` through a
    straight-through path: the decoder sees the quantized value, and the encoder
    receives its gradient. The export takes the deterministic nearest code at
    every level, which is the baseline's inference rule.
    """

    def __init__(self, config: Any, *, in_dim: int, curvature: float = 1.0):
        super().__init__()
        from .learned_curvature import CurvedRQ
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        sizes = ([int(config.codebook_size)] * int(config.codebook_num)
                 if isinstance(config.codebook_size, int)
                 else [int(size) for size in config.codebook_size])
        from .mixed_rq import AlphaTable
        self.rq = CurvedRQ(sizes, AlphaTable(levels=len(sizes)),
                          beta=float(config.beta), sk_epsilon=float(config.sk_epsilon),
                          sk_iters=int(config.sk_iters))
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    @property
    def alpha_table(self):
        """Do not retain a stale alias after DDP broadcasts or checkpoint load."""
        return self.rq.alpha_table

    def forward(self, embeddings: torch.Tensor, item_ids: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        # The item ids are accepted and ignored: the framework's one-epoch smoke
        # probes the trained weights by calling the model the way the baseline's
        # is called, with the item ids of the batch. This mechanism reads no
        # per-item state, so the ids carry nothing.
        encoded = self.encoder(embeddings)
        readout, quant_loss, unused_codes, tokens = self.rq(encoded)
        quantized = encoded + (readout - encoded).detach()
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False) -> torch.Tensor:
        # The export reads each level's curvature, so it must be bound from the
        # learned table first: a restored checkpoint without the binding would
        # export at the constructor's default and disagree with the training run.
        self.rq.bind_curvature()
        from .mixed_rq import encode_tokens
        # The export is the deterministic nearest code at every level; the
        # training-only balanced assignment never runs at inference.
        return encode_tokens(self.encoder(embeddings), self.rq)

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor) -> None:
        self.rq.init_codebooks(self.encoder(embeddings), embeddings.device)

    def compute_loss(self, embeddings: torch.Tensor, reconstructed: torch.Tensor,
                     quant_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(reconstructed, embeddings)
        return recon_loss + quant_loss, recon_loss

