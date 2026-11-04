"""The residual recorder of the curvature-prefix-ranking mechanism.

The baseline's ``RQLayer`` is left byte-identical to its source tree. This
subclass repeats its loop and additionally keeps the residual that enters every
level, so the mechanism can read the same tangents the SID was assigned from.
The assignment itself is the frozen baseline's: Euclidean nearest code, with the
baseline's Sinkhorn only where it already was (the last level), so the collision
stays at the baseline's measured level while the curved/behaviour terms train.
"""
from __future__ import annotations

import types

import torch

import curvature_prefix

from .layers import RQLayer


def _curved_distances(layer, latent: torch.Tensor) -> torch.Tensor:
    """The Poincare cost, so the curve selects the code instead of scoring it.

    ``layer.curvature`` is measured by the trainer from the level-1 residual
    norms before training (Issue #138 C); until it is set the unit curvature is
    used, which is the framework geometry contract's default.
    """
    curvature = float(getattr(layer, "curvature", None) or 1.0)
    return curvature_prefix.curved_codes_cost(latent, layer.get_code_embs(), curvature)


class RecordingRQLayer(RQLayer):
    """``RQLayer`` plus ``level_points()`` and a balanced second level."""

    def __init__(self, config):
        super().__init__(config)
        # The assignment rule is the candidate's (CLAUDE.md 66). The exported L2
        # prefix count is the quantity the historical SID record ties to recall
        # (rho=+0.45 against R@10; the 0.0636 arm exported 16,145 prefixes where
        # this recipe exports ~7,200), and an unbalanced second level spends its
        # 256 codes on a few items. The baseline balances the LAST level only; the
        # candidate balances the second level with the same frozen Sinkhorn
        # constants. The coarse level keeps the plain nearest-code rule, because
        # it carries the behaviour cohesion the mechanism ranks.
        if len(self.vq_layers) > 1 and self.sk_epsilon > 0 and self.sk_iters > 0:
            second = self.vq_layers[1]
            second.sk_epsilon = self.sk_epsilon
            second.sk_iters = self.sk_iters
            second.use_sk = second.sk_epsilon > 0 and second.sk_iters > 0
        # With the curve on, every level selects its code by the Poincare cost, in
        # training and in the export pass alike (get_indices uses these layers).
        self._curved = bool(getattr(config, "use_curvature", True))
        if self._curved:
            for layer in self.vq_layers:
                layer.curvature = None
                layer._euclidean_distances = layer._distances
                layer._distances = types.MethodType(_curved_distances, layer)

    def revert_to_euclidean(self) -> None:
        """Abandon the curved rule when it cannot keep the collision budget.

        The trainer calls this when the first evaluation is already outside the
        budget and no state was ever accepted: a run that cannot produce an
        in-budget partition under the curved assignment must fall back rather
        than export a collapsed SID.
        """
        for layer in self.vq_layers:
            original = getattr(layer, "_euclidean_distances", None)
            if original is not None:
                layer._distances = original
        self._curved = False

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
