"""The residual recorder / product quantizer of the curvature-prefix mechanism.

The baseline's ``RQLayer`` is left byte-identical to its source tree. This
subclass replaces the *assignment rule*: every level quantises the same encoder
latent with its own codebook and the levels are summed for the decoder. Residual
quantisation makes the third-level residual nearly constant inside a coarse
bucket, which is why items in the same (k1,k2) bucket kept colliding; three
independent codebooks break that correlation and are trained end to end.

The first codebook is initialised by KMeans on the latent, the other two keep
their random initialisation (a different KMeans on the same latent would give
identical centres and identical tokens).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .layers import RQLayer


class RecordingRQLayer(RQLayer):
    """``RQLayer`` with product assignment and ``level_points()``."""

    def __init__(self, config):
        super().__init__(config)
        for layer in self.vq_layers:
            layer.use_sk = bool(self.sk_epsilon > 0 and self.sk_iters > 0)
            layer.sk_epsilon = float(self.sk_epsilon)
            layer.sk_iters = int(self.sk_iters)

    def forward(self, x: torch.Tensor, infer_use_sk: bool = False,
                item_ids: torch.Tensor = None):
        quantized = torch.zeros_like(x)
        total_loss = x.new_zeros(())
        unused_codes = 0.0
        tokens = []
        for layer in self.vq_layers:
            quant, quant_loss, unused, indices = layer(x, infer_use_sk, item_ids)
            # The level's raw code vector, not its straight-through image: three
            # straight-through images would add the latent twice into the sum.
            quantized = quantized + layer.embed_code(indices).view(x.shape)
            total_loss = total_loss + quant_loss
            unused_codes += unused
            tokens.append(indices.view(x.shape[0]))
        self._level_points = [x, x, x]
        # Straight-through on the summed code, so the encoder receives the
        # reconstruction gradient once instead of three times.
        quantized = x + (quantized - x).detach()
        return quantized, total_loss / self.codebook_num, unused_codes, torch.stack(tokens, dim=1)

    def level_points(self) -> list[torch.Tensor]:
        points = getattr(self, "_level_points", None)
        if not points:
            raise RuntimeError("level_points() is only available after a forward pass")
        return list(points)

    @torch.no_grad()
    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        first, *others = self.vq_layers
        first.init_codebook(x, device)
        # The remaining levels keep their random, independently drawn codebooks:
        # a second KMeans on the same latent would reproduce the first one.
        return x
