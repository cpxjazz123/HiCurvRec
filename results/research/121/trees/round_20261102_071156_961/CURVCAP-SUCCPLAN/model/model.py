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
        # the mechanism's state: the Euclidean code table the behaviour field is
        # read from (recomputed as the codebook moves) and the per-level
        # curvature the profile carries
        self.register_buffer("euclid_codes", torch.zeros(0, dtype=torch.long),
                             persistent=False)
        self.register_buffer("level_curvature", torch.ones(int(config.codebook_num)),
                             persistent=True)
        self._successors = {}
        self._mechanism_on = False

    def forward(self, embeddings: torch.Tensor,
                item_ids: torch.Tensor = None) -> tuple:
        """The baseline's forward, plus the item ids the mechanism's field needs."""
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded, item_ids=item_ids)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False,
                    item_ids: torch.Tensor = None) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk, item_ids=item_ids)
        return tokens

    def set_mechanism_state(self, successors: dict, curvature, euclid_codes) -> None:
        """Bind the successor lists, the per-level curvature and the code table."""
        self._successors = successors
        self._mechanism_on = True
        self.level_curvature = torch.as_tensor(curvature, dtype=torch.float32)
        if euclid_codes is not None:
            self.euclid_codes = torch.as_tensor(euclid_codes, dtype=torch.long)
        for level, layer in enumerate(self.rq.vq_layers):
            layer.set_mechanism_state(float(self.level_curvature[level]), successors,
                                      self.euclid_codes if self.euclid_codes.numel() else None)

    @torch.no_grad()
    def euclidean_codes(self, embeddings: torch.Tensor) -> torch.Tensor:
        """The plain Euclidean nearest-code table, WITHOUT the mechanism.

        It is the table the behaviour field is read from and the table the
        capacity freeze is measured against, so it must not go through the
        mechanism itself. Every rank computes it from the replicated codebook,
        so no collective is involved and the ranks cannot deadlock.
        """
        encoded = self.encoder(embeddings)
        residual = encoded
        out = torch.empty((residual.shape[0], len(self.rq.vq_layers)),
                          dtype=torch.long, device=residual.device)
        for level, layer in enumerate(self.rq.vq_layers):
            indices = layer._distances(residual).argmin(dim=-1)
            out[:, level] = indices
            residual = residual - layer.embed_code(indices)
        return out

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
