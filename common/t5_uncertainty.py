"""T5 logits uncertainty decay head (Issue #140 v76) — 借鉴 DIGER AutoSigmaGumbel 思想.

DIGER 论文: 给 RQ-VAE codebook 选择加 Gumbel noise, σ 参数 learnable,
compute_uncertainty_loss = task_loss * exp(-k*σ) + c*σ. 训练初期 σ 大 → high noise
→ 探索码字空间; 训练后期 σ→0 → 无 noise → 利用学到的码字.

本模块: 给 T5 logits 加 Gumbel noise (Stage3 训练时). 注意 SID 已经固定
(be9be8f8 SHA256), 这里不确定性是给 T5 decoder 输出, 不是给 SID codebook.
期望效果: 训练初期高 noise → T5 学到 robust 模式; 后期 σ→0 → 精确预测.

参考 DIGER AutoSigmaGumbel (vq.py:86-178):
- sigma = nn.Parameter(log2 形式)
- s = 2^sigma (std)
- noise = gumbels * s
- noisy_logits = (logits + noise) / tau
- compute_uncertainty_loss(task_loss, sigma) = task_loss * exp(-k*sigma) + c*sigma

用法:
    head = T5UncertaintyHead(initial_sigma=0.0, k=0.458145, c=1.361442)
    loss_total = head.compute_total_loss(task_loss)
    noisy_logits = head(logits)  # 仅在 training 时有 noise, eval 时直通
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class T5UncertaintyHead(nn.Module):
    """T5 logits uncertainty decay head (DIGER 思想改造版).

    Args:
        initial_sigma: 初始 σ (log2 形式). DIGER 默认 -20 (≈ 0 std) 或 log2(initial_std).
                       v76 推荐 0.0 (= std 1.0, 中等 noise).
        initial_std: 直接给 std (覆盖 initial_sigma).
        k: uncertainty decay rate (DIGER Gumbel 默认 0.458145 / Fast 0.018127).
        c: uncertainty reg coefficient (DIGER Gumbel 默认 1.361442 / Fast 0.036916).
        reg_weight: uncertainty loss 缩放 (DIGER 默认 1.0).
        clamp_min: s 下界 (DIGER 默认 1e-5).
        clamp_max: s 上界 (DIGER 默认 100.0).
    """

    def __init__(self,
                 initial_sigma: float = 0.0,
                 initial_std: float = None,
                 k: float = 0.458145,
                 c: float = 1.361442,
                 reg_weight: float = 1.0,
                 clamp_min: float = 1e-5,
                 clamp_max: float = 100.0):
        super().__init__()
        if initial_std is not None:
            if initial_std <= 1e-5:
                _init_sigma = -20.0
            else:
                _init_sigma = math.log2(initial_std)
        else:
            _init_sigma = initial_sigma

        self.sigma = nn.Parameter(torch.tensor(_init_sigma))
        self.k = k
        self.c = c
        self.reg_weight = reg_weight
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    @property
    def sigma_value(self):
        return self.sigma

    @property
    def std_value(self):
        """实际 std = 2^σ."""
        return torch.pow(2.0, self.sigma).clamp(min=self.clamp_min, max=self.clamp_max)

    def forward(self, logits: torch.Tensor, tau: float = 1.0, dim: int = -1) -> torch.Tensor:
        """注入 Gumbel noise 到 logits. Training 时有 noise, eval 时直通.

        Args:
            logits: T5 decoder 输出 logits (B, L, V).
            tau: softmax 温度, 默认 1.0.
            dim: softmax dim, 默认 -1.
        Returns:
            noisy_logits: 加 noise 后的 logits (B, L, V).
        """
        if self.training:
            sigma_value = self.sigma
            s = torch.pow(2.0, sigma_value)
            s = s.clamp(min=self.clamp_min, max=self.clamp_max)
            # Gumbel noise: g = -log(-log(U)) = -log(exponential().log())  (DIGER vq.py:119 简化版)
            gumbels = -torch.empty_like(logits).exponential_().log()
            if s.item() < 1e-4:
                noise = torch.zeros_like(logits)
            else:
                noise = gumbels * s
            noisy_logits = (logits + noise) / tau
            return noisy_logits
        else:
            return logits

    def compute_uncertainty_loss(self, task_loss: torch.Tensor) -> torch.Tensor:
        """DIGER uncertainty loss = task_loss * exp(-k*σ) + c*σ.

        Returns:
            scalar loss tensor.
        """
        sigma_value = self.sigma
        exp_term = task_loss * torch.exp(-self.k * sigma_value)
        reg_term = self.c * self.reg_weight * sigma_value
        return exp_term + reg_term