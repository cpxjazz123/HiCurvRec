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
        self.context_w = nn.Linear(int(config.codebook_dim), int(config.codebook_dim), bias=False)
        self.context_u = nn.Linear(int(in_dim), int(config.codebook_dim), bias=False)
        self.branch_alpha = 0.15
        self.curvature = 0.5
        self.behavior_enabled = True
        self.geometry_enabled = True
        self.rq = RQLayer(config)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    @staticmethod
    def _project_ball(x: torch.Tensor, c: float) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return x * (torch.tanh((c ** 0.5) * norm) / ((c ** 0.5) * norm))

    @staticmethod
    def _exp0(x: torch.Tensor, c: float) -> torch.Tensor:
        return RQVAE._project_ball(x, c)

    @staticmethod
    def _log0(x: torch.Tensor, c: float) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scaled = ((c ** 0.5) * norm).clamp(max=1.0 - 1e-5)
        return x * (torch.atanh(scaled) / ((c ** 0.5) * norm))

    @staticmethod
    def _log_at(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
        # Poincare log map at x, using Mobius subtraction (-x) (+_c) y.
        cxy = c * (x * y).sum(-1, keepdim=True)
        x2, y2 = x.square().sum(-1, keepdim=True), y.square().sum(-1, keepdim=True)
        denom = (1 - 2 * cxy + c * c * x2 * y2).clamp_min(1e-8)
        # Mobius subtraction (-x) (+_c) y:
        #   ((1 - 2c<x,y> + c|y|^2)(-x) + (1 - c|x|^2) y) / (1 - 2c<x,y> + c^2|x|^2|y|^2)
        # The two coefficients belong to those two vectors, in that order.
        delta = (-(1 - 2 * cxy + c * y2) * x + (1 - c * x2) * y) / denom
        norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scaled = (c ** .5 * norm).clamp(max=1 - 1e-5)
        lambda_x = (2 / (1 - c * x2).clamp_min(1e-8))
        return delta * (2 / ((c ** .5) * lambda_x)) * (torch.atanh(scaled) / norm)

    @staticmethod
    def _mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
        x2 = x.square().sum(-1, keepdim=True)
        y2 = y.square().sum(-1, keepdim=True)
        xy = (x * y).sum(-1, keepdim=True)
        numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denominator = (1 + 2 * c * xy + c * c * x2 * y2).clamp_min(1e-8)
        point = numerator / denominator
        # Keep finite-precision results strictly inside the Poincare ball.
        max_norm = (1.0 - 1e-5) / (c ** 0.5)
        norm = point.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return point * torch.clamp(max_norm / norm, max=1.0)

    @staticmethod
    def _exp_at(x: torch.Tensor, tangent: torch.Tensor, c: float) -> torch.Tensor:
        x2 = x.square().sum(-1, keepdim=True)
        lambda_x = 2.0 / (1.0 - c * x2).clamp_min(1e-8)
        return RQVAE._mobius_add(x, RQVAE._exp0((lambda_x / 2.0) * tangent, c), c)

    def _branch_target(self, encoded: torch.Tensor, contexts: torch.Tensor | None) -> torch.Tensor:
        if contexts is None or not self.behavior_enabled:
            return encoded
        active = (contexts.norm(dim=-1, keepdim=True) > 0).to(encoded.dtype)
        if not self.geometry_enabled:
            return encoded + self.branch_alpha * (self.context_u(contexts) - self.context_w(encoded)) * active
        c = self.curvature
        z = self._exp0(self.context_w(encoded), c)
        b = self._exp0(self.context_u(contexts), c)
        tangent = self._log_at(z, b, c)
        return self._log0(self._exp_at(z, self.branch_alpha * tangent * active, c), c)

    def forward(self, embeddings: torch.Tensor, contexts: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        encoded = self.encoder(embeddings)
        quant_target = self._branch_target(encoded, contexts)
        quantized, quant_loss, unused_codes, tokens = self.rq(
            quant_target, geometry=self.geometry_enabled, curvature=self.curvature
        )
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, contexts: torch.Tensor | None = None, *, infer_use_sk: bool = False) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(
            self._branch_target(encoded, contexts), infer_use_sk=infer_use_sk,
            geometry=self.geometry_enabled, curvature=self.curvature,
        )
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
