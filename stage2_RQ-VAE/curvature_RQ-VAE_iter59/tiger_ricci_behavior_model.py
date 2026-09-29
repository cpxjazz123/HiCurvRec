"""Euclidean TIGER RQ-VAE with Ricci behavior supervision on quantized codes.

Quantization stays on the original Euclidean TIGER path, including hard vector
assignments, residual subtraction, and the unchanged Sinkhorn/EMA contracts.
Fixed item curvature affects only per-level behavior distances over the
straight-through quantized codeword outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TIGER_MODEL_DIR = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_RQ-VAE/model"
)
sys.path.insert(0, str(TIGER_MODEL_DIR))
from layers import MLP, RQLayer


_BALL_EPS = 1e-6


class QuantizedLevelTraceRQLayer(RQLayer):
    """Original TIGER residual stack with optional quantized-level capture."""

    def forward(
        self,
        x: torch.Tensor,
        infer_use_sk: bool = False,
        return_level_quantized: bool = False,
    ):
        if not return_level_quantized:
            return super().forward(x, infer_use_sk)
        quantized_x = torch.zeros(
            x.shape[0], self.codebook_dim, device=x.device, dtype=x.dtype
        )
        sum_quant_loss: torch.Tensor | float = 0.0
        num_unused_codes = 0.0
        output = torch.empty(
            x.shape[0], self.codebook_num, dtype=torch.long, device=x.device
        )
        residual = x
        layer_quantized = []
        for level, vq_layer in enumerate(self.vq_layers):
            quant, quant_loss, unused, indices = vq_layer(residual, infer_use_sk)
            layer_quantized.append(quant)
            residual = residual - quant
            quantized_x = quantized_x + quant
            sum_quant_loss = sum_quant_loss + quant_loss
            num_unused_codes += unused
            output[:, level] = indices
        result = (
            quantized_x,
            sum_quant_loss / self.codebook_num,
            num_unused_codes,
            output,
        )
        return (*result, torch.stack(layer_quantized, dim=0))


def _poincare_pairwise_distances(
    x: torch.Tensor, y: torch.Tensor, pair_curvature: torch.Tensor
) -> torch.Tensor:
    """Poincare distance for every row pair under one curvature per pair."""
    c = pair_curvature.clamp_min(torch.finfo(x.dtype).tiny)
    sqrt_c = c.sqrt()
    x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    y_norm = torch.linalg.vector_norm(y, dim=-1).unsqueeze(0)
    x_scaled = sqrt_c * x_norm
    y_scaled = sqrt_c * y_norm
    x_factor = torch.tanh(x_scaled) / x_scaled.clamp_min(torch.finfo(x.dtype).eps)
    y_factor = torch.tanh(y_scaled) / y_scaled.clamp_min(torch.finfo(y.dtype).eps)
    max_norm = (1.0 - _BALL_EPS) / sqrt_c
    x_factor = x_factor * torch.clamp(
        max_norm / (x_norm * x_factor).clamp_min(torch.finfo(x.dtype).eps), max=1.0
    )
    y_factor = y_factor * torch.clamp(
        max_norm / (y_norm * y_factor).clamp_min(torch.finfo(y.dtype).eps), max=1.0
    )
    x2 = (x_norm * x_factor).square()
    y2 = (y_norm * y_factor).square()
    dot = (x @ y.t()) * x_factor * y_factor
    difference2 = (x2 + y2 - 2.0 * dot).clamp_min(1e-12)
    denominator = (1.0 - 2.0 * c * dot + c.square() * x2 * y2).clamp_min(
        torch.finfo(x.dtype).tiny
    )
    scaled_norm = (sqrt_c * torch.sqrt(difference2 / denominator)).clamp(
        max=1.0 - _BALL_EPS
    )
    return (2.0 / sqrt_c) * torch.atanh(scaled_norm)


class TIGERRicciBehaviorRQVAE(nn.Module):
    """Euclidean TIGER with fixed curvature supervising hard codeword vectors."""

    def __init__(self, config: Any, *, in_dim: int, item_curvatures: torch.Tensor):
        super().__init__()
        self.config = config
        self.encoder_sizes = (
            int(in_dim),
            *tuple(int(size) for size in config.hidden_sizes),
            int(config.codebook_dim),
        )
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = QuantizedLevelTraceRQLayer(config)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))
        curvature_table = torch.as_tensor(item_curvatures, dtype=torch.float32)
        if curvature_table.ndim != 1 or not torch.isfinite(curvature_table).all():
            raise ValueError("Expected a finite scalar curvature per item")
        if bool((curvature_table <= 0).any()):
            raise ValueError("Poincare curvature magnitudes must be positive")
        self.register_buffer("item_curvatures", curvature_table.clone(), persistent=True)
        self.behavior_temperature = float(config.behavior_temperature)
        self.behavior_weight_max = float(config.behavior_weight_max)
        self._global_step = 0

    def set_global_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("Global step must be nonnegative")
        self._global_step = int(step)

    def get_behavior_weight(self) -> float:
        alpha = min(max((self._global_step - 20_000) / 20_000.0, 0.0), 1.0)
        return self.behavior_weight_max * alpha

    def _curvature_for(self, item_ids: torch.Tensor) -> torch.Tensor:
        ids = item_ids.to(device=self.item_curvatures.device, dtype=torch.long)
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("item_ids must identify every embedding row")
        if bool((ids < 0).any()) or bool((ids >= self.item_curvatures.numel()).any()):
            raise IndexError("item_ids exceed the fixed curvature table")
        return self.item_curvatures[ids]

    def _behavior_contrastive_loss(
        self, quantized_levels, source_ids, target_ids, curvatures
    ):
        batch_size = int(source_ids.shape[0])
        if batch_size == 0:
            return quantized_levels.sum() * 0.0
        source_curvature = curvatures[:batch_size]
        target_curvature = curvatures[batch_size:]
        pair_curvature = torch.sqrt(source_curvature[:, None] * target_curvature[None, :])
        duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
        duplicate_targets.fill_diagonal_(False)
        valid = source_ids.ne(target_ids)
        if not bool(valid.any()):
            return quantized_levels.sum() * 0.0
        labels = torch.arange(batch_size, device=source_ids.device)
        losses = []
        for level in range(self.rq.codebook_num):
            source = quantized_levels[level, :batch_size]
            candidates = quantized_levels[level, batch_size:]
            distances = _poincare_pairwise_distances(source, candidates, pair_curvature)
            logits = -distances / self.behavior_temperature
            logits = logits.masked_fill(duplicate_targets, -torch.inf)
            losses.append(F.cross_entropy(logits[valid], labels[valid]))
        return torch.stack(losses).mean()

    def forward(self, embeddings, *, item_ids, behavior_ids: Optional[tuple[torch.Tensor, torch.Tensor]] = None):
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.encoder_sizes[0]:
            raise ValueError("embeddings must be a batch of source-dimension item vectors")
        curvatures = self._curvature_for(item_ids)
        if curvatures.shape[0] != embeddings.shape[0]:
            raise ValueError("one item ID is required for each embedding row")
        behavior_weight = self.get_behavior_weight()
        encoded = self.encoder(embeddings)
        if behavior_ids is None or behavior_weight == 0.0:
            quantized, quant_loss, unused, tokens = self.rq(encoded)
            return self.decoder(quantized), quant_loss, int(unused), tokens, quant_loss.new_zeros(())
        quantized, quant_loss, unused, tokens, quantized_levels = self.rq(
            encoded, return_level_quantized=True
        )
        behavior_loss = self._behavior_contrastive_loss(
            quantized_levels, behavior_ids[0], behavior_ids[1], curvatures
        )
        return self.decoder(quantized), quant_loss, int(unused), tokens, behavior_loss

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Codes are assigned by the Euclidean quantizer alone; c_i is not consulted."""
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded)
        return tokens
