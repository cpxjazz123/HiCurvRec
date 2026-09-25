import torch
from torch import nn, Tensor


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
        # iter11: mild c-modulated commitment weight. We multiply the base
        # commitment_weight by a power of normalized curvature, so high-c phases
        # get slightly stronger commitment pull (the encoder pushes residuals
        # toward codebook faster at high c) and low-c phases get a softer pull.
        # The exponent 0.25 is gentle: ratio (c_min/c_max)**0.25 ≈ 0.43, so the
        # low-c commitment weight is ~0.43x the high-c weight. Per-layer
        # utilization should remain balanced because the assignment step is
        # unchanged (Sinkhorn ε linear, 3 iters, iter8's mechanism).
        c_value = c.item()
        if c_value <= 0 or c_max_reference <= c_min_reference:
            c_modulation = 1.0
        else:
            normalized = max(min(c_value / c_max_reference, 1.0), 1e-6)
            c_modulation = float(normalized ** 0.25)
        effective_commitment = self.commitment_weight * c_modulation
        return codebook_loss + effective_commitment * commitment_loss
