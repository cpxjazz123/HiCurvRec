from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CURVATURE_MAX, CURVATURE_MIN, MLP, RQLayer, _poincare_pairwise_distances


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

    def set_curriculum_step(self, step: int) -> None:
        self.rq.set_curriculum_step(step)

    def get_item_curvatures(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = item_ids.to(device=self.item_curvatures.device, dtype=torch.long)
        return self.item_curvatures.index_select(0, item_ids.reshape(-1)).reshape(
            *item_ids.shape, self.item_curvatures.shape[1]
        )

    def _behavior_contrastive_loss(self, residuals, curvatures, source_ids, target_ids):
        batch_size = int(source_ids.shape[0])
        if batch_size == 0:
            return residuals.sum() * 0.0
        losses = []
        duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
        duplicate_targets.fill_diagonal_(False)
        valid = source_ids.ne(target_ids)
        labels = torch.arange(batch_size, device=source_ids.device)
        for level in range(residuals.shape[0]):
            source_curvature = curvatures[:batch_size, level]
            target_curvature = curvatures[batch_size:, level]
            pair_curvature = torch.sqrt(source_curvature[:, None] * target_curvature[None, :])
            source = residuals[level, :batch_size]
            candidates = residuals[level, batch_size:]
            distances = _poincare_pairwise_distances(source, candidates, pair_curvature)
            logits = -distances / self.rq.behavior_temperature
            logits = logits.masked_fill(duplicate_targets, -torch.inf)
            if bool(valid.any()):
                losses.append(F.cross_entropy(logits[valid], labels[valid]))
        if not losses:
            return residuals.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(
        self,
        embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        behavior_ids: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        curvatures = self.get_item_curvatures(item_ids)
        behavior_active = behavior_ids is not None and self.rq.get_behavior_weight() > 0.0
        if behavior_active:
            quantized, quant_loss, unused, tokens, residuals = self.rq(
                self.encoder(embeddings), curvatures, return_residuals=True
            )
            behavior_loss = self._behavior_contrastive_loss(
                residuals, curvatures, behavior_ids[0], behavior_ids[1]
            )
        else:
            quantized, quant_loss, unused, tokens = self.rq(self.encoder(embeddings), curvatures)
            behavior_loss = quant_loss.new_zeros(())
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused), tokens, behavior_loss, curvatures

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

    def compute_loss(self, embeddings, reconstructed, quant_loss, behavior_loss):
        recon_loss = F.mse_loss(reconstructed, embeddings)
        loss = recon_loss + quant_loss + self.rq.get_behavior_weight() * behavior_loss
        return loss, recon_loss


__all__ = ["RQVAE"]
