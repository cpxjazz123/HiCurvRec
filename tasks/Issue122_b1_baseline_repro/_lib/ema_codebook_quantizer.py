# Task #291 — EMA codebook update wrapper for FreeCurvHRQVAE (Task #292 复用)
# 2026-07-29
#
# 设计:
#   - 包装现有 FreeCurvResidualVectorQuantization (不污染 src/)
#   - 在 forward 后用 EMA 更新 codebook embeddings (跟 VQ-VAE-2 Razavi et al. 2019 一致)
#   - decay=0.99 (跟 VQ-VAE-2 默认一致)
#   - 切断 κ+codebook 反馈循环: codebook 不通过 gradient 更新, 只通过 EMA 平滑
#
# 来源:
#   - VQ-VAE-2: https://arxiv.org/abs/1906.00446 (Razavi et al. 2019)
#   - κ learnable (跟 baseline 一样, 不冻结)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .hrqvae_free_curv import FreeCurvResidualVectorQuantization


class EMAQuantizerWrapper(nn.Module):
    """EMA codebook update wrapper for FreeCurvResidualVectorQuantization.

    Args:
        base_rq: FreeCurvResidualVectorQuantization 实例
        decay: EMA decay (0.99 跟 VQ-VAE-2 默认一致)
        eps: 防止除零
    """

    def __init__(self, base_rq: FreeCurvResidualVectorQuantization,
                 decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.base_rq = base_rq
        self.decay = decay
        self.eps = eps
        # 初始化 EMA cluster_size + embed_avg per layer
        self.M = len(base_rq.vq_layers)
        # FreeCurvVectorQuantization 用 self.n_e (不是 K)
        n_e_list = [vq.n_e for vq in base_rq.vq_layers]
        e_dim = base_rq.vq_layers[0].e_dim
        # EMA buffers per layer
        self.register_buffer("ema_cluster_size", torch.zeros(self.M, max(n_e_list)))
        self.register_buffer("ema_embed_avg", torch.zeros(self.M, max(n_e_list), e_dim))

    @torch.no_grad()
    def _ema_update(self, layer_idx: int, x_flat: torch.Tensor, indices: torch.Tensor):
        """One-hot encode indices, EMA update cluster_size and embed_avg, write back to codebook."""
        vq = self.base_rq.vq_layers[layer_idx]
        K = vq.n_e  # FreeCurvVQ 用 n_e
        e_dim = vq.e_dim

        # Clamp indices to [0, K-1] (R12 fix): FreeCurvHRQVAE indices may include padding=4
        # which is outside [0, K-1]. Without clamp, F.one_hot triggers CUDA device-side assert.
        indices_clamped = indices.clamp(min=0, max=K - 1)

        # One-hot encoding
        one_hot = F.one_hot(indices_clamped, K).float()  # (B, K)

        # EMA cluster size (codebook usage frequency)
        cluster_size = one_hot.sum(dim=0)  # (K,)
        self.ema_cluster_size[layer_idx, :K] = (
            self.decay * self.ema_cluster_size[layer_idx, :K] + (1 - self.decay) * cluster_size
        )

        # EMA embed average (sum of x assigned to each codeword)
        embed_sum = one_hot.T @ x_flat  # (K, e_dim)
        self.ema_embed_avg[layer_idx, :K] = (
            self.decay * self.ema_embed_avg[layer_idx, :K] + (1 - self.decay) * embed_sum
        )

        # Normalize: new embed = embed_avg / cluster_size (Laplace smoothing)
        n = self.ema_cluster_size[layer_idx, :K].sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size[layer_idx, :K] + self.eps) / (n + K * self.eps) * n
        )
        new_embed = self.ema_embed_avg[layer_idx, :K] / cluster_size_smoothed.unsqueeze(-1)
        # Write back to codebook (in-place, no gradient)
        vq.embeddings.weight.data[:K] = new_embed

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        """Forward passes through base FreeCurvHRQVAE, then EMA updates codebooks."""
        # Base forward
        x_q_st, total_loss, all_indices = self.base_rq(x, use_sk=use_sk)
        # all_indices shape: (B, M) (stack dim=-1 of M x (B,) tensors)

        # EMA update per layer (only during training)
        if self.training:
            x_flat = x.view(-1, self.base_rq.vq_layers[0].e_dim)
            for m, vq in enumerate(self.base_rq.vq_layers):
                # R12 fix: all_indices is (B, M), index along dim 1 (not dim 0)
                # Previously all_indices[m] gave (M,) shape, causing shape mismatch in matmul.
                idx_m = all_indices[:, m].flatten()  # (B,)
                self._ema_update(m, x_flat, idx_m)

        return x_q_st, total_loss, all_indices

    def get_codebook_usage(self, indices: torch.Tensor):
        return self.base_rq.get_codebook_usage(indices)