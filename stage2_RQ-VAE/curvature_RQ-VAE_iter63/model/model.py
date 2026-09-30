from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CURVATURE_MAX, CURVATURE_MIN, MLP, RQLayer


class RQVAE(nn.Module):
    def __init__(self, config: Any, *, in_dim: int, item_curvatures: torch.Tensor):
        super().__init__()
        sizes = (int(in_dim), *tuple(int(size) for size in config.hidden_sizes), int(config.codebook_dim))
        self.encoder = MLP(list(sizes), dropout=float(config.dropout))
        self.rq = RQLayer(
            config.codebook_size,
            config.codebook_dim,
            beta=float(config.beta),
            sk_epsilon=float(config.sk_epsilon),
            sk_iters=int(config.sk_iters),
        )
        self.decoder = MLP(list(sizes[::-1]), dropout=float(config.dropout))
        curvatures = torch.as_tensor(item_curvatures, dtype=torch.float32).detach().contiguous()
        if (
            curvatures.ndim != 2
            or curvatures.shape[1] != len(config.codebook_size)
            or not torch.isfinite(curvatures).all()
        ):
            raise ValueError("Item curvatures must be a finite [items, residual_levels] tensor")
        if float(curvatures.min()) < CURVATURE_MIN or float(curvatures.max()) > CURVATURE_MAX:
            raise ValueError("Item curvatures are outside the supported positive range")
        self.register_buffer("item_curvatures", curvatures)

    def get_item_curvatures(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = item_ids.to(device=self.item_curvatures.device, dtype=torch.long)
        return self.item_curvatures.index_select(0, item_ids.reshape(-1)).reshape(
            *item_ids.shape, self.item_curvatures.shape[1]
        )

    def forward(self, embeddings: torch.Tensor, item_ids: torch.Tensor):
        curvatures = self.get_item_curvatures(item_ids)
        quantized, quant_loss, unused, tokens = self.rq(self.encoder(embeddings), curvatures)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused), tokens, curvatures

    @torch.no_grad()
    def init_codebook(self, embeddings, item_ids):
        encoded = self.encoder(embeddings)
        curvature = self.get_item_curvatures(item_ids)
        self.rq.init_codebook(encoded, curvature, embeddings.device)

    @torch.no_grad()
    def get_indices_with_stats(self, embeddings, item_ids):
        encoded = self.encoder(embeddings)
        curvature = self.get_item_curvatures(item_ids)
        tokens, stats = self.rq.get_indices_with_stats(encoded, curvature)
        return tokens, stats, curvature

    def compute_loss(self, embeddings, reconstructed, quant_loss):
        recon_loss = F.mse_loss(reconstructed, embeddings)
        return recon_loss + quant_loss, recon_loss


__all__ = ["RQVAE"]
