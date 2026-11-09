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
        from .radial_manifold import RadialRQ
        from .mixed_rq import AlphaTable, MixedRQ
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        sizes = ([int(config.codebook_size)] * int(config.codebook_num)
                 if isinstance(config.codebook_size, int)
                 else [int(size) for size in config.codebook_size])
        self.rq = RadialRQ(sizes, AlphaTable(levels=len(sizes)),
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
        quant_loss, tokens = self.rq(encoded)
        # The decoded vector is the sum of the picked joint codes: differentiable in
        # the codebooks, and the same 32-wide object the decoder was built for.
        picked = torch.stack([level.get_code_embs()[tokens[:, position]]
                              for position, level in enumerate(self.rq.levels)], dim=0)
        # Straight-through: the decoder sees the quantized code, and the encoder
        # still receives the reconstruction gradient it is trained by. Without the
        # pass-through the encoder has no gradient at all, which is exactly what
        # the framework's encoder_gradient check exists to catch.
        quantized = encoded + (picked.sum(0) - encoded).detach()
        reconstructed = self.decoder(quantized)
        used = torch.zeros(self.rq.levels[0].n_embed, device=tokens.device)
        used.scatter_add_(0, tokens[:, 0], torch.ones_like(tokens[:, 0], dtype=used.dtype))
        return reconstructed, quant_loss, int((used == 0).sum().item()), tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False) -> torch.Tensor:
        # The export is this quantizer's own nearest code at every level: the
        # frozen helper splits the code 16/16 and this mechanism splits it 15/16/1,
        # so it cannot be reused. The training-only balanced assignment never runs
        # at inference.
        _, tokens = self.rq(self.encoder(embeddings), infer_use_sk=False)
        return tokens

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor) -> None:
        self.rq.init_codebooks(self.encoder(embeddings), embeddings.device)

    def compute_loss(self, embeddings: torch.Tensor, reconstructed: torch.Tensor,
                     quant_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(reconstructed, embeddings)
        return recon_loss + quant_loss, recon_loss

