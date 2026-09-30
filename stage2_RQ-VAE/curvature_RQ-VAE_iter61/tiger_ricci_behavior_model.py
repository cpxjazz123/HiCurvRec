"""Euclidean TIGER RQ-VAE with topology-gated mixed behavior geometry.

Quantization, residual subtraction, and codebooks remain the Euclidean TIGER
implementation. Fixed Iter53 item curvature and topology scores are consumed
only by the behavior contrastive loss. Its pair distance is the specified
geometric interpolation between Euclidean distance and Poincare distance.
"""
from __future__ import annotations

import math
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


class ResidualTraceRQLayer(RQLayer):
    """Original TIGER RQLayer with optional pre-quantization residual capture."""

    def forward(
        self,
        x: torch.Tensor,
        infer_use_sk: bool = False,
        return_residuals: bool = False,
    ):
        if not return_residuals:
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
        layer_residuals = []
        for level, vq_layer in enumerate(self.vq_layers):
            layer_residuals.append(residual)
            quant, quant_loss, unused, indices = vq_layer(residual, infer_use_sk)
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
        return (*result, torch.stack(layer_residuals, dim=0))


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
    """Euclidean TIGER with fixed curvature and a trainable topology gate."""

    def __init__(
        self,
        config: Any,
        *,
        in_dim: int,
        item_curvatures: torch.Tensor,
        item_topology_scores: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        self.encoder_sizes = (
            int(in_dim),
            *tuple(int(size) for size in config.hidden_sizes),
            int(config.codebook_dim),
        )
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = ResidualTraceRQLayer(config)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))
        curvature_table = torch.as_tensor(item_curvatures, dtype=torch.float32)
        topology_table = torch.as_tensor(item_topology_scores, dtype=torch.float32)
        if curvature_table.ndim != 1 or not torch.isfinite(curvature_table).all():
            raise ValueError("Expected a finite scalar curvature per item")
        if bool((curvature_table <= 0).any()):
            raise ValueError("Poincare curvature magnitudes must be positive")
        if topology_table.ndim != 1 or topology_table.shape != curvature_table.shape:
            raise ValueError("Expected one finite topology score per item")
        if not torch.isfinite(topology_table).all():
            raise ValueError("Topology scores must be finite")
        self.register_buffer("item_curvatures", curvature_table.clone(), persistent=True)
        self.register_buffer("item_topology_scores", topology_table.clone(), persistent=True)
        # softplus keeps alpha positive, preserving the specified low-q -> Euclidean
        # and high-q -> hyperbolic orientation. Initial alpha=1, tau=0.
        self.geometry_alpha_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.geometry_tau = nn.Parameter(torch.tensor(0.0))
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

    def _item_values(self, item_ids: torch.Tensor, table: torch.Tensor, name: str):
        ids = item_ids.to(device=table.device, dtype=torch.long)
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("item_ids must identify every embedding row")
        if bool((ids < 0).any()) or bool((ids >= table.numel()).any()):
            raise IndexError(f"item_ids exceed the fixed {name} table")
        return table[ids]

    def _geometry_gate(self, topology_scores: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.geometry_alpha_raw)
        return torch.sigmoid(alpha * (topology_scores - self.geometry_tau))

    def gate_diagnostics(self) -> dict[str, float]:
        with torch.no_grad():
            gates = self._geometry_gate(self.item_topology_scores)
            quantiles = torch.quantile(
                gates, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=gates.device)
            )
            return {
                "geometry_alpha": float(F.softplus(self.geometry_alpha_raw).cpu()),
                "geometry_tau": float(self.geometry_tau.cpu()),
                "gate_mean": float(gates.mean().cpu()),
                "gate_min": float(quantiles[0].cpu()),
                "gate_p10": float(quantiles[1].cpu()),
                "gate_median": float(quantiles[2].cpu()),
                "gate_p90": float(quantiles[3].cpu()),
                "gate_max": float(quantiles[4].cpu()),
            }

    def _behavior_contrastive_loss(
        self, residuals, source_ids, target_ids, curvatures, topology_scores
    ):
        batch_size = int(source_ids.shape[0])
        alpha_gate = self._geometry_gate(topology_scores)
        gate_zero = (alpha_gate.sum() + self.geometry_tau) * 0.0
        if batch_size == 0:
            return residuals.sum() * 0.0 + gate_zero
        source_curvature = curvatures[:batch_size]
        target_curvature = curvatures[batch_size:]
        pair_curvature = torch.sqrt(source_curvature[:, None] * target_curvature[None, :])
        source_gate = alpha_gate[:batch_size]
        target_gate = alpha_gate[batch_size:]
        pair_gate = torch.sqrt(source_gate[:, None] * target_gate[None, :])
        duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
        duplicate_targets.fill_diagonal_(False)
        valid = source_ids.ne(target_ids)
        if not bool(valid.any()):
            return residuals.sum() * 0.0 + gate_zero
        labels = torch.arange(batch_size, device=source_ids.device)
        losses = []
        for level in range(self.rq.codebook_num):
            source = residuals[level, :batch_size]
            candidates = residuals[level, batch_size:]
            euclidean = torch.cdist(source, candidates, p=2)
            hyperbolic = _poincare_pairwise_distances(source, candidates, pair_curvature)
            distances = (1.0 - pair_gate) * euclidean + pair_gate * hyperbolic
            logits = -distances / self.behavior_temperature
            logits = logits.masked_fill(duplicate_targets, -torch.inf)
            losses.append(F.cross_entropy(logits[valid], labels[valid]))
        return torch.stack(losses).mean()

    def forward(
        self,
        embeddings,
        *,
        item_ids,
        behavior_ids: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.encoder_sizes[0]:
            raise ValueError("embeddings must be a batch of source-dimension item vectors")
        curvatures = self._item_values(item_ids, self.item_curvatures, "curvature")
        topology_scores = self._item_values(
            item_ids, self.item_topology_scores, "topology score"
        )
        if curvatures.shape[0] != embeddings.shape[0]:
            raise ValueError("one item ID is required for each embedding row")
        behavior_weight = self.get_behavior_weight()
        encoded = self.encoder(embeddings)
        if behavior_ids is None or behavior_weight == 0.0:
            quantized, quant_loss, unused, tokens = self.rq(encoded)
            gate_zero = (self.geometry_alpha_raw + self.geometry_tau) * 0.0
            return (
                self.decoder(quantized), quant_loss, int(unused), tokens,
                quant_loss.new_zeros(()) + gate_zero,
            )
        quantized, quant_loss, unused, tokens, residuals = self.rq(
            encoded, return_residuals=True
        )
        behavior_loss = self._behavior_contrastive_loss(
            residuals,
            behavior_ids[0],
            behavior_ids[1],
            curvatures,
            topology_scores,
        )
        return self.decoder(quantized), quant_loss, int(unused), tokens, behavior_loss

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Codes depend on Euclidean TIGER quantization alone."""
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded)
        return tokens
