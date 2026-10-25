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

    def _behavior_transform(self, encoded, neighbor_embeddings, behavior_gate, transition_probs, behavior_on, geometry_on):
        if not behavior_on or neighbor_embeddings is None or behavior_gate is None:
            return encoded
        from .utils import expmap0, expmap_at, logmap0, logmap_at, mobius_add, proj_to_ball
        # c=0.5 is the positive magnitude corresponding to sectional curvature -0.5.
        c, beta = 0.5, 0.2
        weights = transition_probs
        if weights is None:
            weights = torch.ones(neighbor_embeddings.shape[:2], device=encoded.device, dtype=encoded.dtype)
        weights = weights.to(device=encoded.device, dtype=encoded.dtype).clamp_min(0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        z = expmap0(encoded, c)
        points = expmap0(self.encoder(neighbor_embeddings), c)
        # Three weighted Frechet/Karcher updates in the Poincare ball.
        prototype = proj_to_ball((points * weights[..., None]).sum(dim=1), c)
        for _ in range(3):
            tangent = logmap_at(prototype[:, None, :], points, c)
            update = (tangent * weights[..., None]).sum(dim=1)
            prototype = proj_to_ball(mobius_add(prototype, expmap_at(prototype, update, c), c), c)
        if geometry_on:
            tangent = logmap0(mobius_add(-z, prototype, c), c)
            target_ball = mobius_add(z, expmap0(beta * behavior_gate[:, None] * tangent, c), c)
            return proj_to_ball(target_ball, c)
        neighbor_mean = (self.encoder(neighbor_embeddings) * weights[..., None]).sum(dim=1)
        return encoded + beta * behavior_gate[:, None] * (neighbor_mean - encoded)

    def forward(self, embeddings: torch.Tensor, neighbor_embeddings: torch.Tensor | None = None, behavior_gate: torch.Tensor | None = None, transition_probs: torch.Tensor | None = None, behavior_on: bool = True, geometry_on: bool = True) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        encoded = self.encoder(embeddings)
        encoded = self._behavior_transform(encoded, neighbor_embeddings, behavior_gate, transition_probs, behavior_on, geometry_on)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded, curved=bool(behavior_on and geometry_on and neighbor_embeddings is not None))
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, neighbor_embeddings=None, behavior_gate=None, transition_probs=None, *, behavior_on=True, geometry_on=True, infer_use_sk: bool = False) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        encoded = self._behavior_transform(encoded, neighbor_embeddings, behavior_gate, transition_probs, behavior_on, geometry_on)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk, curved=bool(behavior_on and geometry_on and neighbor_embeddings is not None))
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
