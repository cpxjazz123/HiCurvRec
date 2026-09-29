"""Standard Euclidean TIGER RQ-VAE plus its Euclidean behavior variant."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

TIGER_MODEL_DIR = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_RQ-VAE/model"
)
sys.path.insert(0, str(TIGER_MODEL_DIR))
from layers import MLP, RQLayer


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


class TIGERRQVAE(nn.Module):
    """The repository's standard Euclidean TIGER model and quantizer."""

    def __init__(
        self,
        config: Any,
        *,
        in_dim: int,
        rq_layer_type: Type[RQLayer] = RQLayer,
    ):
        super().__init__()
        self.config = config
        hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.encoder_sizes = (int(in_dim), *hidden_sizes, int(config.codebook_dim))
        self.encoder = MLP(list(self.encoder_sizes), dropout=float(config.dropout))
        self.rq = rq_layer_type(config)
        self.decoder = MLP(list(self.encoder_sizes[::-1]), dropout=float(config.dropout))

    def forward(self, embeddings: torch.Tensor):
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens = self.rq(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded)
        return tokens

    @torch.no_grad()
    def init_codebook(self, embeddings: torch.Tensor) -> None:
        encoded = self.encoder(embeddings)
        self.rq.init_codebook(encoded, embeddings.device)


class TIGERBehaviorRQVAE(TIGERRQVAE):
    """Euclidean TIGER with a ramped contrastive loss on residual vectors."""

    def __init__(self, config: Any, *, in_dim: int):
        super().__init__(config, in_dim=in_dim, rq_layer_type=ResidualTraceRQLayer)
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
        for level in range(self.rq.codebook_num):
            source = residuals[level, :batch_size]
            candidates = residuals[level, batch_size:]
            distances = torch.cdist(source, candidates, p=2)
            logits = -distances / self.behavior_temperature
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
        *,
        behavior_ids: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        behavior_weight = self.get_behavior_weight()
        if behavior_ids is None or behavior_weight == 0.0:
            reconstructed, quant_loss, unused_codes, tokens = super().forward(embeddings)
            return reconstructed, quant_loss, unused_codes, tokens, quant_loss.new_zeros(())
        encoded = self.encoder(embeddings)
        quantized, quant_loss, unused_codes, tokens, residuals = self.rq(
            encoded, return_residuals=True
        )
        behavior_loss = self._behavior_contrastive_loss(
            residuals, behavior_ids[0], behavior_ids[1]
        )
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, int(unused_codes), tokens, behavior_loss
