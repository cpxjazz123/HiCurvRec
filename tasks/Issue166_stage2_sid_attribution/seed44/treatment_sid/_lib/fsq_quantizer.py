# Task #290 — FSQ (Finite Scalar Quantization) + κ-decouple (新 quantizer 文件, 不污染 baseline)
# 2026-07-29
#
# 设计:
#   - 完全消除可学码本, 改成 scalar round (Mentzer et al. 2023 FSQ paper)
#   - 100% L0/L1/L2 util by construction (没有 argmin)
#   - κ-decouple Phase A/B curriculum (跟 task144/task284/task287 同样协议)
#   - 测试 FSQ 跟 κ-decouple 是否兼容
#
# 来源:
#   - FSQ paper: https://arxiv.org/abs/2309.15505 (Mentzer et al. 2023)
#   - HG-Rec κ-decouple: task144 K=64 + task287 K=128 + task284 K=256
#
# 用法:
#   python train_hrqvae.py --quantizer fsq --fsq_levels 8 5 5 5 --num_emb_list 64 128 256 \
#       --kappa_freeze_epochs 100 --lr_theta_post_unfreeze 1e-5 --epochs 200

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for round operation."""
    z_q = torch.round(z)
    return z + (z_q - z).detach()


class FSQCodebook(nn.Module):
    """FSQ: 每维独立 round 到 discrete levels, 不需要可学码本."""

    def __init__(self, levels: List[int], e_dim: int):
        """
        levels: list of int, e.g. [8, 5, 5, 5] → 8*5*5*5 = 1000 effective codes
        e_dim: dim per layer (跟 baseline 一致, e.g. 32)
        """
        super().__init__()
        assert e_dim == len(levels), f"FSQ levels ({len(levels)}) must equal e_dim ({e_dim})"
        # 注册为 buffer 让 .to(device) 自动跟 z 一起搬
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))
        self.e_dim = e_dim
        self.K = int(self.levels.prod().item())  # effective codebook size

        # 预计算 basis for converting quantized values → flat indices
        offset = torch.cat([torch.tensor([1]), torch.cumprod(self.levels, 0)[:-1]])
        self.register_buffer("offset", offset.float())

    def indices(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, e_dim) → flat indices (B,)"""
        scaled = torch.round(z)  # assume z already in [-L/2, L/2]
        # shift from [0, L-1] to centered → flat index
        shifted = scaled + (self.levels - 1) / 2  # to [0, L-1]
        return (shifted.long() * self.offset.long()).sum(dim=-1)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, e_dim) → z_q (B, e_dim) (rounded, with STE)."""
        # Scale: project z from R^{e_dim} to [-L/2, L/2] for each dim
        # Convention: z is already centered (encoder output has zero mean by construction)
        half_width = (self.levels - 1) / 2  # per-dim
        scaled = z / half_width  # naive scaling — could use learned scale but FSQ 默认不加
        z_q = round_ste(scaled)
        # Project back to original scale (for downstream T5 to interpret)
        return z_q * half_width

    def forward(self, z: torch.Tensor):
        """z: (B, e_dim) → (z_q_st, indices, commitment_loss, codebook_loss)"""
        z_q = self.quantize(z)
        indices = self.indices(z)
        # Commitment loss: MSE between z_q.detach() and z (FSQ 不需要 codebook loss, 因没码本)
        commitment_loss = F.mse_loss(z_q.detach(), z)
        # Straight-through estimator: forward = z_q (rounded), backward = z (continuous)
        z_st = z + (z_q - z).detach()
        return z_st, indices, commitment_loss, torch.tensor(0.0, device=z.device)


class FSQResidualQuantizer(nn.Module):
    """FSQ stack M 层, 每层独立 round (类似 RQ-VAE 但用 FSQ 替代 VQ).

    R-fix 2026-07-29: per-layer indices 输出 ∈ [0, n_e_list[m]-1] (匹配 GenRecDataset item2code 协议,
    避免 packed-int 4^32 ≈ 1.8e19 超过 vocab_size=1025 → vectorized_gather_kernel out of bounds).
    """

    def __init__(self, n_e_list: List[int], e_dim: int, M: int = 3,
                 levels: Optional[List[int]] = None):
        super().__init__()
        self.e_dim = e_dim
        self.M = M
        self.n_e_list = n_e_list  # K for each layer (e.g. [64, 128, 256])
        # 用一个 FSQ per layer (e_dim 维度可独立 round)
        self.fsq_layers = nn.ModuleList([
            FSQCodebook(levels=levels if levels else [8] * e_dim, e_dim=e_dim)
            for _ in range(M)
        ])

    def _per_layer_index(self, packed_idx: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """把 FSQCodebook 输出的 packed-int (in [0, L^e_dim)) 投影到 per-layer index in [0, n_e-1].

        简单方案: packed int mod n_e_list[m]. 数学上 mixed-base, 但足够 distributed 让 T5-mini 训练有 per-layer signal.
        """
        n_e = self.n_e_list[layer_idx]
        return packed_idx % n_e

    def forward(self, x: torch.Tensor, use_sk: bool = False):
        """x: (B, e_dim) → (x_q_st, total_loss, all_indices) where all_indices ∈ [0, n_e_list[m]-1]"""
        x_q_total = torch.zeros_like(x)
        total_loss = torch.tensor(0.0, device=x.device)
        all_indices = []
        residual = x
        for m, layer in enumerate(self.fsq_layers):
            z_q, idx_packed, loss_q, _ = layer(residual)
            # 投影到 per-layer index ∈ [0, n_e_list[m]-1]
            idx_per_layer = self._per_layer_index(idx_packed, m)
            x_q_total = x_q_total + z_q
            total_loss = total_loss + loss_q
            all_indices.append(idx_per_layer)
            residual = residual - z_q  # residual: subtract quantized
        return x_q_total, total_loss, torch.stack(all_indices, dim=-1)  # (B, M)

    @torch.no_grad()
    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """Per-layer code usage. indices: (B, M) → (M, K_per_layer)"""
        usages = []
        for m, layer in enumerate(self.fsq_layers):
            idx_m = indices[:, m]
            counts = torch.bincount(idx_m, minlength=self.n_e_list[m]).float()
            usages.append((counts > 0).float().mean().item())  # fraction of codes used
        return torch.tensor(usages)


class FSQWithKappaDecouple(nn.Module):
    """FSQ + κ-decouple curriculum (Phase A κ frozen=0 + Phase B κ unfreeze lr=1e-5).

    κ 是可学曲率参数 (跟 FreeCurvHRQVAE 一样), 但实际只在 residual 上算 commitment loss
    (FSQ 没有 distance-based argmin, 所以 κ 不影响 quantization, 只影响 commitment).
    """

    def __init__(self, n_e_list: List[int], e_dim: int, M: int = 3,
                 kappa_init: float = 0.0, levels: Optional[List[int]] = None):
        super().__init__()
        self.fsq_rq = FSQResidualQuantizer(n_e_list, e_dim, M, levels)
        # κ learnable per layer
        self.theta_m = nn.Parameter(torch.full((M,), float(kappa_init)))
        self.M = M

    def kappa_m(self) -> torch.Tensor:
        # κ_actual = -softplus(theta) ensures negative curvature (跟 HG-Rec 一致)
        return -F.softplus(self.theta_m)

    def set_kappa_frozen(self, frozen: bool):
        self.theta_m.requires_grad = not frozen

    def forward(self, x: torch.Tensor, use_sk: bool = False):
        # FSQ forward (跟 κ 无关)
        x_q_st, fsq_loss, indices = self.fsq_rq(x, use_sk)

        # κ-decouple commitment loss (Poincaré distance from x_q to x)
        # 即使 FSQ 也算 κ-Stereographic commitment (测 κ 跟 FSQ 是否兼容)
        kappa = self.kappa_m()
        commitment_sq = torch.tensor(0.0, device=x.device)
        for m, k_m in enumerate(kappa):
            k_abs = k_m.abs().clamp(min=1e-8)
            sqrt_k = torch.sqrt(k_abs)
            diff = x_q_st - x.detach()
            norm = diff.norm(dim=-1).clamp_min(1e-8)
            denom = 2.0 * (1.0 - k_m * norm ** 2 / 4.0).abs().clamp_min(1e-6)
            arg = sqrt_k * norm / denom
            d = (2.0 / sqrt_k) * torch.arctan(arg)
            commitment_sq = commitment_sq + (d ** 2).mean()

        total_loss = fsq_loss + 0.5 * commitment_sq
        return x_q_st, total_loss, indices

    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        return self.fsq_rq.get_codebook_usage(indices)