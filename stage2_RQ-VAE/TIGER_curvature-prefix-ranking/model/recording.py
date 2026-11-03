"""The residual recorder of the curvature-prefix-ranking mechanism.

The baseline's ``RQLayer`` is left byte-identical to its source tree. This
subclass repeats its loop and additionally keeps the residual that enters every
level, so the mechanism can read the same tangents the SID was assigned from.
The assignment itself is the frozen baseline's: Euclidean nearest code, with the
baseline's Sinkhorn only where it already was (the last level), so the collision
stays at the baseline's measured level while the curved/behaviour terms train.
"""
from __future__ import annotations

import torch

from .layers import RQLayer


class RecordingRQLayer(RQLayer):
    """``RQLayer`` plus ``level_points()`` and per-level balanced assignment."""

    def forward(self, x: torch.Tensor, infer_use_sk: bool = False):
        quantized_x = torch.zeros(x.shape[0], self.codebook_dim, device=x.device)
        sum_quant_loss: torch.Tensor | float = 0.0
        num_unused_codes = 0.0
        output = torch.empty(x.shape[0], self.codebook_num, dtype=torch.long,
                             device=x.device)
        residual = x
        level_points = []
        for level, vq_layer in enumerate(self.vq_layers):
            level_points.append(residual)
            quant, quant_loss, unused, indices = vq_layer(residual, infer_use_sk)
            residual = residual - quant
            quantized_x = quantized_x + quant
            sum_quant_loss = sum_quant_loss + quant_loss
            num_unused_codes += unused
            output[:, level] = indices
        self._level_points = level_points
        return quantized_x, sum_quant_loss / self.codebook_num, num_unused_codes, output

    def level_points(self) -> list[torch.Tensor]:
        points = getattr(self, "_level_points", None)
        if not points:
            raise RuntimeError("level_points() is only available after a forward pass")
        return list(points)
