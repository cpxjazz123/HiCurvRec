from __future__ import annotations

import torch
from torch import nn, Tensor
from typing import Optional


class ReconstructionLoss(nn.Module):
    """在当前 assignment 使用的动态曲率下计算 reconstruction loss。"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat: Tensor, x: Tensor, c: Tensor) -> Tensor:
        from modules.hyperbolic import _expmap0_t, _poincare_distance_t

        if c.shape != (1, 1):
            raise ValueError(
                f"ReconstructionLoss curvature shape 必须为 (1, 1)，实际为 {tuple(c.shape)}"
            )
        if not torch.isfinite(c).all().item() or c.item() <= 0:
            raise ValueError("ReconstructionLoss curvature 必须为有限正数")
        c = c.to(device=x.device, dtype=x.dtype)
        x_hat_h = _expmap0_t(x_hat, c)
        x_h = _expmap0_t(x, c)
        distance = _poincare_distance_t(x_hat_h, x_h, c).squeeze(-1)
        return distance.square()


class QuantizeLoss(nn.Module):
    """在 assignment 使用的当前曲率下计算 commitment/codebook loss。"""

    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = float(commitment_weight)

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        c: Tensor,
        log_tau_l: Optional[Tensor] = None,
        c_min_reference: float = 0.05,
        c_max_reference: float = 1.5,
    ) -> Tensor:
        from modules.hyperbolic import _expmap0_t, _poincare_distance_t

        if c.shape != (1, 1):
            raise ValueError(
                f"QuantizeLoss curvature shape 必须为 (1, 1)，实际为 {tuple(c.shape)}"
            )
        if not torch.isfinite(c).all().item() or c.item() <= 0:
            raise ValueError("QuantizeLoss curvature 必须为有限正数")
        c = c.to(device=query.device, dtype=query.dtype)
        query_h = _expmap0_t(query, c)
        value_h = _expmap0_t(value, c)
        codebook_loss = _poincare_distance_t(
            query_h.detach(), value_h, c
        ).squeeze(-1).square()
        commitment_loss = _poincare_distance_t(
            query_h, value_h.detach(), c
        ).squeeze(-1).square()
        # iter13: per-layer learnable commitment multiplier.
        # `tau_l = exp(log_tau_l)` multiplies the commitment_weight (gradient flows
        # through log_tau_l since it's a leaf parameter). The c-modulated term
        # from iter11 is also kept so per-layer commitment is jointly modulated
        # by (c) and (tau_l). When log_tau_l = 0 -> tau_l = 1 (identity).
        c_modulation = 1.0
        if log_tau_l is not None:
            tau_l = torch.exp(log_tau_l).clamp(0.05, 20.0)
            c_value = c.item()
            if c_max_reference > c_min_reference and c_value > 0:
                normalized = max(min(c_value / c_max_reference, 1.0), 1e-6)
                c_modulation = float(normalized ** 0.25)
            effective_commitment = self.commitment_weight * c_modulation * tau_l
        else:
            c_value = c.item()
            if c_max_reference > c_min_reference and c_value > 0:
                normalized = max(min(c_value / c_max_reference, 1.0), 1e-6)
                c_modulation = float(normalized ** 0.25)
            effective_commitment = self.commitment_weight * c_modulation
        return codebook_loss + effective_commitment * commitment_loss
