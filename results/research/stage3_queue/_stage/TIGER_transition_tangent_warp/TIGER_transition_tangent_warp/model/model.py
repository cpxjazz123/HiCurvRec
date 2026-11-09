"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _exp0(v: torch.Tensor, c: float) -> torch.Tensor:
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.tanh((c ** 0.5) * norm) * v / ((c ** 0.5) * norm)


def _log0(x: torch.Tensor, c: float) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8, max=(1.0 - 1e-5) / (c ** 0.5))
    return torch.atanh((c ** 0.5) * norm) * x / ((c ** 0.5) * norm)


def _mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    xx, yy = (x * x).sum(-1, keepdim=True), (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    numerator = (1 + 2*c*xy + c*yy)*x + (1-c*xx)*y
    denominator = (1 + 2*c*xy + c*c*xx*yy).clamp_min(1e-8)
    out = numerator / denominator
    max_norm = (1.0 - 1e-5) / (c ** 0.5)
    return out * torch.clamp(max_norm / out.norm(dim=-1, keepdim=True).clamp_min(1e-8), max=1.0)

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
        self.behavior_map = nn.Linear(64, int(config.codebook_dim), bias=False)
        nn.init.normal_(self.behavior_map.weight, mean=0.0, std=0.01)
        self.curvature = 1.0
        self.tangent_radius = 0.1

    def _warp(self, encoded: torch.Tensor, profile: torch.Tensor, behavior_on: bool = True, geometry_on: bool = True) -> torch.Tensor:
        if not behavior_on:
            return encoded
        u = self.tangent_radius * torch.tanh(self.behavior_map(profile))
        if not geometry_on:
            return encoded + u
        c = self.curvature
        x = _exp0(encoded, c)
        y = _mobius_add(x, _exp0(u, c), c)
        return _log0(y, c)

    def forward(self, embeddings: torch.Tensor, profile: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        encoded = self.encoder(embeddings)
        if profile is not None:
            encoded = self._warp(encoded, profile)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, profile: Optional[torch.Tensor] = None, infer_use_sk: bool = False) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        if profile is not None:
            encoded = self._warp(encoded, profile)
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
