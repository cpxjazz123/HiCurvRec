import torch
from torch import nn
from torch import Tensor


class ReconstructionLoss(nn.Module):
    """HG-Rec 风格: hyperbolic reconstruction loss = poincare_distance² (projected to Poincaré ball)."""
    def __init__(self, c: float = 1.0) -> None:
        super().__init__()
        self.c = float(c)

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        from modules.hyperbolic import (
            _expmap0_t, _poincare_distance_t, _logmap0_t,
        )
        # HG-Rec 风格: 重建损失在 Poincaré 球 (c=self.c, HG-Rec 用 1.0)
        # expmap0 → Poincaré 球 → poincare_distance² → loss
        c_t = torch.tensor(self.c, device=x.device, dtype=x.dtype)
        c_exp = c_t.view(1, 1)
        x_hat_h = _expmap0_t(x_hat, c_exp)
        x_h = _expmap0_t(x, c_exp)
        d = _poincare_distance_t(x_hat_h, x_h, c_exp).squeeze(-1) ** 2
        return d


class CategoricalReconstuctionLoss(nn.Module):
    def __init__(self, n_cat_feats: int) -> None:
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.n_cat_feats = n_cat_feats

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        reconstr = self.reconstruction_loss(
            x_hat[:, : -self.n_cat_feats], x[:, : -self.n_cat_feats]
        )
        if self.n_cat_feats > 0:
            cat_reconstr = nn.functional.binary_cross_entropy_with_logits(
                x_hat[:, -self.n_cat_feats :],
                x[:, -self.n_cat_feats :],
                reduction="none",
            ).sum(axis=-1)
            reconstr += cat_reconstr
        return reconstr


class QuantizeLoss(nn.Module):
    """HG-Rec 风格: commitment/codebook loss = poincare_distance² (projected to Poincaré 球 c=1)."""
    def __init__(self, commitment_weight: float = 1.0, c: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight
        self.c = float(c)

    def forward(self, query: Tensor, value: Tensor) -> Tensor:
        from modules.hyperbolic import (
            _expmap0_t, _poincare_distance_t,
        )
        c_t = torch.tensor(self.c, device=query.device, dtype=query.dtype)
        c_exp = c_t.view(1, 1)
        # HG-Rec 风格 commitment/codebook loss: 在 Poincaré 球 (c=1) 上计算距离²
        # expmap0 → ball → poincare_distance² (per-sample, B only — HG-Rec 同口径)
        query_h = _expmap0_t(query, c_exp)
        value_h = _expmap0_t(value, c_exp)
        # commitment: distance² between detached_query and value (gradient → codebook)
        # codebook: distance² between query and detached_value (gradient → encoder)
        emb_loss = _poincare_distance_t(query_h.detach(), value_h, c_exp).squeeze(-1) ** 2
        query_loss = _poincare_distance_t(query_h, value_h.detach(), c_exp).squeeze(-1) ** 2
        return emb_loss + self.commitment_weight * query_loss
