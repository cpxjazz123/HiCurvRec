"""The residual recorder of the curvature-prefix-ranking mechanism.

The baseline's ``RQLayer`` is left byte-identical to its source tree.  This
subclass repeats its ten-line loop and additionally keeps the residual that
enters every level, so the mechanism can read the same tangents the SID was
assigned from.  A test pins the equivalence of the two implementations on the
same input and state.
"""
from __future__ import annotations

import torch

from .layers import RQLayer


class RecordingRQLayer(RQLayer):
    """``RQLayer`` plus ``level_points()`` for the mechanism's curve side.

    The collision gate (three levels <= 20%) needs a spread code usage, so the
    balanced Sinkhorn assignment is enabled on EVERY level during training; at
    inference the assignment stays the deterministic argmin, so the exported SID
    of an item does not depend on its batch.
    """

    def __init__(self, config):
        super().__init__(config)
        for layer in self.vq_layers:
            layer.use_sk = bool(self.sk_epsilon > 0 and self.sk_iters > 0)
            layer.sk_epsilon = float(self.sk_epsilon)
            layer.sk_iters = int(self.sk_iters)

    def forward(self, x: torch.Tensor, infer_use_sk: bool = False,
                item_ids: torch.Tensor = None):
        quantized_x = torch.zeros(x.shape[0], self.codebook_dim, device=x.device)
        sum_quant_loss: torch.Tensor | float = 0.0
        num_unused_codes = 0.0
        output = torch.empty(x.shape[0], self.codebook_num, dtype=torch.long,
                             device=x.device)
        residual = x
        level_points = []
        for level, vq_layer in enumerate(self.vq_layers):
            level_points.append(residual)
            quant, quant_loss, unused, indices = vq_layer(residual, infer_use_sk, item_ids)
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
