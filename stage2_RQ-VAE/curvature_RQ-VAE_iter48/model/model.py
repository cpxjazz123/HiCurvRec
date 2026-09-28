"""Standalone RecBole3.0-compatible RQ-VAE wrapper for HG-Rec."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, RQLayer, _pairwise_poincare_distance_tangents


class RQVAE(nn.Module):
    """Encode item embeddings into residual-quantized semantic identifiers."""

    def __init__(self, config: Any, *, in_dim: int):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = RQLayer(config)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    def set_curriculum_step(self, step: int) -> None:
        self.rq.set_curriculum_step(step)

    def get_curvatures(self) -> torch.Tensor:
        return self.rq.get_curvatures()

    def curvature_regularization(self) -> torch.Tensor:
        return self.rq.curvature_regularization()

    def _behavior_contrastive_loss(
        self,
        residuals: torch.Tensor,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(source_ids.shape[0])
        if batch_size == 0:
            return residuals.sum() * 0.0
        losses = []
        for level, layer in enumerate(self.rq.vq_layers):
            source = residuals[level, :batch_size]
            candidates = residuals[level, batch_size:]
            distances = _pairwise_poincare_distance_tangents(
                source, candidates, layer.get_curvature()
            )
            logits = -distances / self.rq.behavior_temperature
            duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
            duplicate_targets.fill_diagonal_(False)
            logits = logits.masked_fill(duplicate_targets, -torch.inf)
            valid = source_ids.ne(target_ids)
            if bool(valid.any()):
                labels = torch.arange(batch_size, device=source_ids.device)
                losses.append(F.cross_entropy(logits[valid], labels[valid]))
        if not losses:
            return residuals.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(
        self,
        embeddings: torch.Tensor,
        behavior_ids: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(embeddings)
        behavior_active = (
            behavior_ids is not None and self.rq.get_behavior_weight() > 0.0
        )
        if behavior_active:
            quantized, quant_loss, unused_codes, tokens, residuals = self.rq(
                encoded, return_residuals=True
            )
            behavior_loss = self._behavior_contrastive_loss(
                residuals, behavior_ids[0], behavior_ids[1]
            )
        else:
            quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
            behavior_loss = quant_loss.new_zeros(())
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens, behavior_loss

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, *, infer_use_sk: bool = False) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded, infer_use_sk=infer_use_sk)
        return tokens

    @torch.no_grad()
    def get_indices_with_stats(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        encoded = self.encoder(embeddings)
        return self.rq.get_indices_with_stats(encoded)

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor) -> None:
        encoded = self.encoder(embeddings)
        self.rq.init_codebook(encoded, embeddings.device)

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        reconstructed: torch.Tensor,
        quant_loss: torch.Tensor,
        behavior_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(reconstructed, embeddings)
        curvature_reg = self.rq.curvature_regularization()
        curvature_alpha = self.rq.get_curriculum_alpha()
        behavior_weight = self.rq.get_behavior_weight()
        total_loss = (
            recon_loss
            + quant_loss
            + behavior_weight * behavior_loss
            + self.rq.curvature_reg_weight * curvature_alpha * curvature_reg
        )
        return total_loss, recon_loss
