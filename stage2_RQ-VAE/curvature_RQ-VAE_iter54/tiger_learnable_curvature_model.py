"""Euclidean TIGER with a learnable nonlinear structure-to-curvature controller.

The RQ-VAE is the original Euclidean TIGER path.  Curvature is produced by a
small `6 -> 8 -> 1` controller over train-graph structure features and is used
only for the Behavior contrastive loss, exactly as in Iter53.

The controller is initialized to reproduce the validated Iter53 linear map:

    q_i = (R_i + G_i + H_i) / 3      c_i = c_min + (c_max - c_min) * sigmoid(q_i)

so Iter54 begins at the Iter53 operating point and any later difference is
attributable to what the controller learned, not to a different starting map.
Primary-feature weights are positive by construction, so a structure feature
rising does not invert its structural meaning.  Zero-sum slope offsets break
hidden-unit symmetry without changing the initial Iter53 linear map.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TIGER_MODEL_DIR = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_RQ-VAE/model"
)
sys.path.insert(0, str(TIGER_MODEL_DIR))
from layers import MLP, RQLayer


_BALL_EPS = 1e-6
PRIMARY_FEATURES = 3


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


class CurvatureController(nn.Module):
    """Learnable nonlinear structure-to-curvature map with monotone primaries."""

    def __init__(self, config: Any, features: torch.Tensor):
        super().__init__()
        self.hidden = int(config.controller_hidden)
        self.curvature_min = float(config.curvature_min)
        self.curvature_max = float(config.curvature_max)
        self.reference_curvature_mean = float(config.reference_curvature_mean)
        self.reference_curvature_std = float(config.reference_curvature_std)
        primary = features[:, :PRIMARY_FEATURES]
        self.register_buffer("primary_abs_max", primary.abs().max(dim=0).values)
        effective = float(config.controller_weight_init)
        diversity = float(config.controller_diversity)
        if self.hidden != 8:
            raise ValueError("controller initialization is defined for eight hidden units")
        pattern = primary.new_tensor(
            (
                (1.0, 0.0, -1.0),
                (-1.0, 0.0, 1.0),
                (0.0, 1.0, -1.0),
                (0.0, -1.0, 1.0),
                (1.0, -1.0, 0.0),
                (-1.0, 1.0, 0.0),
                (1.0, 1.0, -2.0),
                (-1.0, -1.0, 2.0),
            )
        )
        effective_weights = effective + diversity * pattern
        if torch.any(effective_weights <= 0):
            raise ValueError("controller diversity must preserve positive primary weights")
        self.primary_weight = nn.Parameter(torch.log(torch.expm1(effective_weights)))
        self.primary_bias = nn.Parameter(
            torch.full((self.hidden,), float(config.controller_bias_init))
        )
        self.interaction_weight = nn.Parameter(
            torch.zeros(self.hidden, PRIMARY_FEATURES * (PRIMARY_FEATURES - 1) // 2)
        )
        self.interaction_bias = nn.Parameter(torch.zeros(self.hidden))
        output_init = 1.0 / self.hidden
        self.output_weight = nn.Parameter(
            torch.full((1, self.hidden), float(np.log(np.expm1(output_init))))
        )
        self.output_bias = nn.Parameter(
            torch.full((1,), -float(config.controller_bias_init))
        )

    def effective_primary_weight(self) -> torch.Tensor:
        """Keep d(preactivation)/d(R,G,H) positive over the observed feature box."""
        interaction = self.interaction_weight
        r_max, g_max, h_max = self.primary_abs_max
        compensation = torch.stack(
            (
                interaction[:, 0].abs() * g_max + interaction[:, 1].abs() * h_max,
                interaction[:, 0].abs() * r_max + interaction[:, 2].abs() * h_max,
                interaction[:, 1].abs() * r_max + interaction[:, 2].abs() * g_max,
            ),
            dim=1,
        )
        return F.softplus(self.primary_weight) + compensation

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        primary = features[..., :PRIMARY_FEATURES]
        interaction = features[..., PRIMARY_FEATURES:]
        if primary.shape[-1] != PRIMARY_FEATURES or interaction.shape[-1] != 3:
            raise ValueError("controller expects (R,G,H,RG,RH,GH) features")
        hidden = primary @ self.effective_primary_weight().t()
        hidden = hidden + interaction @ self.interaction_weight.t()
        hidden = F.relu(hidden + self.primary_bias + self.interaction_bias)
        q = hidden @ F.softplus(self.output_weight).t() + self.output_bias
        return self.curvature_min + (self.curvature_max - self.curvature_min) * torch.sigmoid(
            q.squeeze(-1)
        )

    def regularizer(self, curvature: torch.Tensor, config: Any) -> torch.Tensor:
        """Anchor mean and spread to the validated Iter53 curvature distribution."""
        mean_term = (curvature.mean() - self.reference_curvature_mean).square()
        std_term = (curvature.std(unbiased=False) - self.reference_curvature_std).square()
        return float(config.curvature_mean_reg) * mean_term + float(
            config.curvature_std_reg
        ) * std_term

    def architecture(self) -> dict:
        return {
            "hidden": self.hidden,
            "input_features": 6,
            "primary_features": ["negative_orc_need", "two_hop_expansion", "transition_entropy"],
            "interaction_features": ["R*G", "R*H", "G*H"],
            "primary_monotone_over_observed_feature_box": True,
            "output_weight_positive": True,
            "mean_target": self.reference_curvature_mean,
            "std_target": self.reference_curvature_std,
            "spread_penalty": "(std(c) - iter53_std)^2",
            "curvature_min": self.curvature_min,
            "curvature_max": self.curvature_max,
        }



class TIGERLearnableCurvatureRQVAE(nn.Module):
    """Euclidean TIGER with learned curvature used only in the behavior loss."""

    def __init__(self, config: Any, *, in_dim: int, curvature_features: torch.Tensor):
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
        features = torch.as_tensor(curvature_features, dtype=torch.float32)
        if features.ndim != 2 or features.shape[-1] != 6 or not torch.isfinite(features).all():
            raise ValueError("Expected six finite structure features per item")
        self.register_buffer("curvature_features", features.clone(), persistent=True)
        self.controller = CurvatureController(config, features)
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
        item_indices = item_ids.to(
            device=self.curvature_features.device, dtype=torch.long
        )
        if item_indices.ndim != 1 or item_indices.shape[0] != embeddings.shape[0]:
            raise ValueError("one item ID is required for each embedding row")
        if bool((item_indices < 0).any()) or bool(
            (item_indices >= self.curvature_features.shape[0]).any()
        ):
            raise IndexError("item_ids exceed the fixed curvature feature table")
        behavior_weight = self.get_behavior_weight()
        encoded = self.encoder(embeddings)
        if behavior_ids is None or behavior_weight == 0.0:
            quantized, quant_loss, unused, tokens = self.rq(encoded)
            all_curvature = self.controller(self.curvature_features)
            return (
                self.decoder(quantized),
                quant_loss,
                int(unused),
                tokens,
                quant_loss.new_zeros(()),
                all_curvature[item_indices],
                self.controller.regularizer(all_curvature, self.config),
            )
        quantized, quant_loss, unused, tokens, residuals = self.rq(
            encoded, return_residuals=True
        )
        all_curvature = self.controller(self.curvature_features)
        batch_curvature = all_curvature[item_indices]
        behavior_loss = self._behavior_contrastive_loss(
            residuals, behavior_ids[0], behavior_ids[1], batch_curvature
        )
        return (
            self.decoder(quantized),
            quant_loss,
            int(unused),
            tokens,
            behavior_loss,
            batch_curvature,
            self.controller.regularizer(all_curvature, self.config),
        )

    @torch.no_grad()
    def get_indices(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Codes are assigned by the Euclidean quantizer alone; c_i is not consulted."""
        encoded = self.encoder(embeddings)
        _, _, _, tokens = self.rq(encoded)
        return tokens
