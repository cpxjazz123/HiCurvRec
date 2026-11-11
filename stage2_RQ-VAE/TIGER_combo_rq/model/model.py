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

    def forward_with_codes(self, embeddings: torch.Tensor):
        """Like :meth:`forward`, but also returns the residual stack's readout as ``levels[-1]``.

        Issue #192 needs item positions for the structure term, and they must be the quantiser's own readout rather than a second codebook path: the
        straight-through value is differentiable in the encoder, which is what the structure term's gradient is supposed to reach.
        """
        encoded = self.encoder(embeddings)
        # walk the stack so the raw codeword readout is available beside the straight-through one: the raw sum is differentiable in the
        # codebook as well as the encoder, which is what the structure term's budget needs (issue #192)
        residual = encoded
        straight_total = 0.0
        raw_total = 0.0
        indices = []
        for layer in self.rq.vq_layers:
            straight, raw, index = layer.quantize_with_raw(residual)
            residual = residual - straight
            straight_total = straight_total + straight
            raw_total = raw_total + raw
            indices.append(index)
        tokens = torch.stack(indices, dim=-1)
        quantized = straight_total
        # this entry point exists for the structure term's positions; its loss outputs are placeholders and must not be used for training
        quant_loss = torch.zeros((), device=encoded.device)
        unused_codes = 0
        reconstructed = self.decoder(quantized)
        # both paths at once: the raw codeword sum carries gradient into the codebook, and the straight-through term carries it into the encoder, so
        # neither supervised block sees a zero structure norm (the budget rule refuses when either does)
        positions = raw_total + (encoded - encoded.detach())
        return reconstructed, quant_loss, int(unused_codes), tokens, [positions]

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
