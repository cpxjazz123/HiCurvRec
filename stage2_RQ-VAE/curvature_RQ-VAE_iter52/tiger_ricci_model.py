"""TIGER residual quantization with graph-derived, fixed item curvature."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

TIGER_MODEL_DIR = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_RQ-VAE/model"
)
sys.path.insert(0, str(TIGER_MODEL_DIR))
from layers import MLP, VQLayer


_BALL_EPS = 1e-6


def _expmap0_tangent(x: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    c = curvature.reshape(-1, 1).clamp_min(torch.finfo(x.dtype).tiny)
    sqrt_c = c.sqrt()
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    scaled = sqrt_c * norm
    factor = torch.tanh(scaled) / scaled.clamp_min(torch.finfo(x.dtype).eps)
    point = factor * x
    point_norm = torch.linalg.vector_norm(point, dim=-1, keepdim=True)
    max_norm = (1.0 - _BALL_EPS) / sqrt_c
    return point * torch.clamp(max_norm / point_norm.clamp_min(torch.finfo(x.dtype).eps), max=1.0)


def _logmap0_point(x: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    c = curvature.reshape(-1, 1).clamp_min(torch.finfo(x.dtype).tiny)
    sqrt_c = c.sqrt()
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    norm_safe = norm.clamp_min(torch.finfo(x.dtype).eps)
    scaled = (sqrt_c * norm).clamp(max=1.0 - _BALL_EPS)
    return (torch.atanh(scaled) / (sqrt_c * norm_safe)) * x


def _mobius_add(x: torch.Tensor, y: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    c = curvature.reshape(-1, 1).clamp_min(torch.finfo(x.dtype).tiny)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    denominator = 1.0 + 2.0 * c * xy + c.square() * x2 * y2
    return numerator / denominator.clamp_min(torch.finfo(x.dtype).tiny)


def _hyperbolic_residual(
    residual_tangent: torch.Tensor,
    code_tangent: torch.Tensor,
    curvature: torch.Tensor,
) -> torch.Tensor:
    residual_point = _expmap0_tangent(residual_tangent, curvature)
    code_point = _expmap0_tangent(code_tangent, curvature)
    difference = _mobius_add(-code_point, residual_point, curvature)
    return _logmap0_point(difference, curvature)


def _poincare_item_code_distances(
    x: torch.Tensor, codebook: torch.Tensor, curvature: torch.Tensor
) -> torch.Tensor:
    c = curvature.reshape(-1, 1).clamp_min(torch.finfo(x.dtype).tiny)
    sqrt_c = c.sqrt()
    x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    code_norm = torch.linalg.vector_norm(codebook, dim=-1).unsqueeze(0)
    x_scaled = sqrt_c * x_norm
    code_scaled = sqrt_c * code_norm
    x_factor = torch.tanh(x_scaled) / x_scaled.clamp_min(torch.finfo(x.dtype).eps)
    code_factor = torch.tanh(code_scaled) / code_scaled.clamp_min(torch.finfo(x.dtype).eps)
    max_norm = (1.0 - _BALL_EPS) / sqrt_c
    x_factor = x_factor * torch.clamp(
        max_norm / (x_norm * x_factor).clamp_min(torch.finfo(x.dtype).eps), max=1.0
    )
    code_factor = code_factor * torch.clamp(
        max_norm / (code_norm * code_factor).clamp_min(torch.finfo(x.dtype).eps), max=1.0
    )
    x2 = (x_norm * x_factor).square()
    code2 = (code_norm * code_factor).square()
    dot = (x @ codebook.t()) * x_factor * code_factor
    difference2 = (x2 + code2 - 2.0 * dot).clamp_min(1e-12)
    denominator = (1.0 - 2.0 * c * dot + c.square() * x2 * code2).clamp_min(
        torch.finfo(x.dtype).tiny
    )
    scaled_norm = (sqrt_c * torch.sqrt(difference2 / denominator)).clamp(
        max=1.0 - _BALL_EPS
    )
    return (2.0 / sqrt_c) * torch.atanh(scaled_norm)


def _poincare_row_distances(
    x: torch.Tensor, y: torch.Tensor, curvature: torch.Tensor
) -> torch.Tensor:
    c = curvature.reshape(-1, 1).clamp_min(torch.finfo(x.dtype).tiny)
    x_point = _expmap0_tangent(x, c)
    y_point = _expmap0_tangent(y, c)
    difference = _mobius_add(-x_point, y_point, c)
    norm = torch.linalg.vector_norm(difference, dim=-1, keepdim=True).clamp_min(1e-12)
    sqrt_c = c.sqrt()
    return (2.0 / sqrt_c) * torch.atanh((sqrt_c * norm).clamp(max=1.0 - _BALL_EPS))


def _poincare_pairwise_distances(
    x: torch.Tensor, y: torch.Tensor, pair_curvature: torch.Tensor
) -> torch.Tensor:
    c = pair_curvature.clamp_min(torch.finfo(x.dtype).tiny)
    sqrt_c = c.sqrt()
    x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    y_norm = torch.linalg.vector_norm(y, dim=-1).unsqueeze(0)
    x_scaled = sqrt_c * x_norm
    y_scaled = sqrt_c * y_norm
    x_factor = torch.tanh(x_scaled) / x_scaled.clamp_min(torch.finfo(x.dtype).eps)
    y_factor = torch.tanh(y_scaled) / y_scaled.clamp_min(torch.finfo(x.dtype).eps)
    max_norm = (1.0 - _BALL_EPS) / sqrt_c
    x_factor = x_factor * torch.clamp(
        max_norm / (x_norm * x_factor).clamp_min(torch.finfo(x.dtype).eps), max=1.0
    )
    y_factor = y_factor * torch.clamp(
        max_norm / (y_norm * y_factor).clamp_min(torch.finfo(x.dtype).eps), max=1.0
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


class RicciVQLayer(VQLayer):
    """TIGER codebook parameters with Poincare assignment and losses."""

    @staticmethod
    def center_distance(distances: torch.Tensor) -> torch.Tensor:
        high, low = distances.max(), distances.min()
        middle = (high + low) / 2.0
        return (distances - middle) / (high - middle + 1e-5)

    @torch.no_grad()
    def sinkhorn(self, distances: torch.Tensor, epsilon: float, iterations: int):
        q = torch.exp(-distances / epsilon)
        batch_size, codebook_size = q.shape
        q /= q.sum(-1, keepdim=True).sum(-2, keepdim=True)
        for _ in range(iterations):
            q /= q.sum(0, keepdim=True)
            q /= codebook_size
            q /= q.sum(1, keepdim=True)
            q /= batch_size
        q *= batch_size
        return q

    def forward(self, x, curvature, infer_use_sk=False):
        latent = x.reshape(-1, self.dim)
        c = curvature.reshape(-1)
        distances = _poincare_item_code_distances(latent, self.get_code_embs(), c)
        if (self.training and self.use_sk) or (self.use_sk and infer_use_sk):
            centered = self.center_distance(distances).double()
            assignments = self.sinkhorn(centered, self.sk_epsilon, self.sk_iters)
            if not torch.isfinite(assignments).all():
                raise RuntimeError("Poincare Sinkhorn assignment returned NaN or infinity")
            indices = torch.argmax(assignments, dim=-1)
        else:
            indices = torch.argmin(distances, dim=-1)
        used = F.one_hot(indices, self.n_embed).sum(0)
        if dist.is_initialized() and infer_use_sk is False and not getattr(
            self, "_skip_ddp_reduce", False
        ):
            dist.all_reduce(used, op=dist.ReduceOp.SUM)
        unused = int((used == 0).sum().item())
        quantized = F.embedding(indices, self.get_code_embs()).view_as(x)
        codebook_loss = _poincare_row_distances(
            latent.detach(), quantized.reshape(-1, self.dim), c
        ).square().mean()
        commitment_loss = _poincare_row_distances(
            latent, quantized.detach().reshape(-1, self.dim), c
        ).square().mean()
        quant_loss = codebook_loss + self.beta * commitment_loss
        quantized = x + (quantized - x).detach()
        return quantized, quant_loss, unused, indices.view(*x.shape[:-1])


class RicciRQLayer(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.codebook_num = int(config.codebook_num)
        self.codebook_dim = int(config.codebook_dim)
        if isinstance(config.codebook_size, int):
            sizes = [int(config.codebook_size)] * self.codebook_num
        else:
            sizes = [int(size) for size in config.codebook_size]
        if len(sizes) != self.codebook_num:
            raise ValueError("codebook_size must have one entry per quantization level")
        self.vq_layers = nn.ModuleList(
            RicciVQLayer(
                size,
                self.codebook_dim,
                float(config.beta),
                float(config.sk_epsilon) if level == self.codebook_num - 1 else -1.0,
                int(config.sk_iters) if level == self.codebook_num - 1 else -1,
            )
            for level, size in enumerate(sizes)
        )

    def forward(self, x, curvature, infer_use_sk=False, return_residuals=False):
        curvature = curvature.reshape(-1)
        quantized_sum = torch.zeros_like(x)
        residual = x
        quant_losses = []
        tokens = torch.empty(x.shape[0], self.codebook_num, dtype=torch.long, device=x.device)
        residuals = []
        unused_codes = 0
        for level, layer in enumerate(self.vq_layers):
            if return_residuals:
                residuals.append(residual)
            quantized, loss, unused, indices = layer(
                residual, curvature, infer_use_sk=infer_use_sk
            )
            residual = _hyperbolic_residual(residual, quantized, curvature)
            quantized_sum = quantized_sum + quantized
            quant_losses.append(loss)
            unused_codes += unused
            tokens[:, level] = indices
        result = (quantized_sum, torch.stack(quant_losses).mean(), unused_codes, tokens)
        if return_residuals:
            return (*result, torch.stack(residuals, dim=0))
        return result


class TIGERRicciBehaviorRQVAE(nn.Module):
    """TIGER architecture with fixed per-item curvature from train-only graph signals."""

    def __init__(self, config: Any, *, in_dim: int, item_curvatures: torch.Tensor):
        super().__init__()
        self.config = config
        self.encoder_sizes = (int(in_dim), *tuple(int(size) for size in config.hidden_sizes), int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = RicciRQLayer(config)
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

    def _behavior_contrastive_loss(self, residuals, source_ids, target_ids, curvatures):
        batch_size = int(source_ids.shape[0])
        if batch_size == 0:
            return residuals.sum() * 0.0
        source_curvature = curvatures[:batch_size]
        target_curvature = curvatures[batch_size:]
        pair_curvature = torch.sqrt(source_curvature[:, None] * target_curvature[None, :])
        duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
        duplicate_targets.fill_diagonal_(False)
        valid = source_ids.ne(target_ids)
        if not bool(valid.any()):
            return residuals.sum() * 0.0
        labels = torch.arange(batch_size, device=source_ids.device)
        losses = []
        for level in range(self.rq.codebook_num):
            source = residuals[level, :batch_size]
            candidates = residuals[level, batch_size:]
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
            quantized, quant_loss, unused, tokens = self.rq(encoded, curvatures)
            return self.decoder(quantized), quant_loss, int(unused), tokens, quant_loss.new_zeros(())
        quantized, quant_loss, unused, tokens, residuals = self.rq(
            encoded, curvatures, return_residuals=True
        )
        behavior_loss = self._behavior_contrastive_loss(
            residuals, behavior_ids[0], behavior_ids[1], curvatures
        )
        return self.decoder(quantized), quant_loss, int(unused), tokens, behavior_loss

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        curvatures = self._curvature_for(item_ids)
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded, curvatures)
        return tokens
