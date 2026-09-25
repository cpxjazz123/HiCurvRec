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

    def forward(self, query: Tensor, value: Tensor, c: Tensor) -> Tensor:
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
        return codebook_loss + self.commitment_weight * commitment_loss
