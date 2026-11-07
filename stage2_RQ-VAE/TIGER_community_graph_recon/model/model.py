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

    def forward(self, embeddings: torch.Tensor, item_ids: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Reconstruct every row.

        ``item_ids`` is the identity of each row, which every Stage-2 arm's
        forward receives and this mechanism's loss does not need here: the
        community graph reconstruction reads the representations, not the ids,
        and it is computed in its own forward. The argument is kept so the
        forward's contract is the same one the framework's smokes and the other
        arms use.
        """
        del item_ids
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    def quantised_views(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The quantised representation of every row, in its two gradient forms.

        The first is the straight-through sum the frozen baseline's decoder also
        consumes: its value is the sum of the three codeword vectors and its
        gradient reaches the encoder. The second is the plain sum of those same
        three codeword vectors without any detach, so its gradient reaches the
        codebook and never the encoder.

        The two are provided so that a loss can be written once and routed to
        both parameter groups: a loss built on the straight-through tensor alone
        moves the encoder and leaves the codebook untouched, which is the
        mistake this method exists to make impossible.
        """
        encoded = self.encoder(embeddings)
        quantized, _, _, tokens = self.rq(encoded)
        codewords = torch.zeros_like(quantized)
        for level, layer in enumerate(self.rq.vq_layers):
            codewords = codewords + F.embedding(tokens[:, level], layer.get_code_embs())
        return quantized, codewords, tokens

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
