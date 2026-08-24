"""PoincaréCodebook: RQ-VAE codebook 在 Poincaré ball 内 (Stage 2 κ 学习备胎).

设计 (vs 标准 Euclidean RQ-VAE codebook):
1. codebook 向量归一化 ||e_i|| < 1/√c (在 Poincaré ball B_c^d 内)
2. 量化距离 = poincare_dist_sq(z, e) 替代欧氏距离 ||z - e||^2
3. learnable per-codebook curvature κ ∈ R^1 (单一全局曲率, 让 codebook 自适应)
4. loss 加 poincare_reg: codebook 向量趋近原点 (鼓励 tree 化结构)

文献支撑 (3 篇 2025 hyperbolic codebook):
- UCQ: Uncertain-aware Contrastive Quantization (arXiv 2507.02722)
- Quadratic Quantization (arXiv 2507.08616)
- SOF: Scalable One-stage Binarization (arXiv 2507.08622)

数学公式:
- poincare_dist_sq(u, v, c) = (1/c) * arcosh(1 + 2c * ||u-v||^2 / ((1-c||u||^2)(1-c||v||^2)))^2
- poincare_norm_sq(x, c) = ||x||^2 / (1 - c*||x||^2)
- 量化: argmin_i poincare_dist_sq(z, e_i)

实施路径 (Stage 2 κ 备胎, 待用户明确"启动"后执行):
1. 在 train_rqvae.py 实例化 PoincaréCodebook 替代 EuclideanCodebook
2. 跑 Stage 2 训练 (curve-rqvae checkpoint)
3. 跑 infer_sids_instruments.py 生成新 SID
4. 重跑 Stage 3 HALC v2 (用新 SID)
5. R37 对照 baseline (HALC v2 + Euclidean codebook) test_R@10=0.1072

注意: 此模块只准备代码骨架, 实际启动训练需用户明确指令 (R50).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoincareCodebook(nn.Module):
    """RQ-VAE 第一层 codebook 在 Poincaré ball 内, 量化距离用 poincare_dist_sq."""

    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 32,
        init_curvature: float = 1.0,
        learnable_curvature: bool = True,
        reg_weight: float = 0.01,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # codebook: 标准欧氏空间的 weight, 量化时投影到 Poincaré ball
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        # learnable curvature
        if learnable_curvature:
            self.log_curvature = nn.Parameter(torch.tensor(math.log(math.expm1(init_curvature))))
        else:
            self.register_buffer("log_curvature", torch.tensor(math.log(math.expm1(init_curvature))))
        self.reg_weight = reg_weight

    @property
    def curvature(self) -> torch.Tensor:
        return F.softplus(self.log_curvature) + 1e-5

    def poincare_project(self, x: torch.Tensor) -> torch.Tensor:
        """Project x to Poincaré ball B_c^d: ||x|| < 1/√c."""
        c = self.curvature
        max_norm = (1.0 / torch.sqrt(c) - 1e-5)
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        scale = (max_norm / norm).clamp_max(1.0)
        return x * scale

    def poincare_dist_sq(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Squared Poincaré distance (u, v) ∈ B_c^d. Args: (..., d), (..., d) -> (...,)."""
        c = self.curvature
        sqrt_c = torch.sqrt(c)
        u_norm_sq = (u ** 2).sum(dim=-1) * c
        v_norm_sq = (v ** 2).sum(dim=-1) * c
        diff_norm_sq = ((u - v) ** 2).sum(dim=-1) * c
        # poincare_dist = arcosh(1 + 2 * ||u-v||^2 * c / ((1-c||u||^2)(1-c||v||^2)))
        denominator = (1 - u_norm_sq) * (1 - v_norm_sq)
        arg = 1 + 2 * diff_norm_sq / denominator.clamp_min(1e-10)
        # arcosh(x) = log(x + sqrt(x^2 - 1)), 数值稳定: clamp 到 [1+ε, ∞)
        arg = arg.clamp_min(1 + 1e-7)
        dist = torch.acosh(arg)
        return dist ** 2  # squared distance

    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        """Quantize z_e ∈ R^{d} → codebook index. Args: (..., d). Returns: (...)."""
        # project z_e to ball
        z_proj = self.poincare_project(z_e)
        # project codebook
        e = self.poincare_project(self.embedding.weight)
        # poincare distance
        # z_proj: (B, d), e: (K, d) → (B, K)
        dist = self.poincare_dist_sq(
            z_proj.unsqueeze(1).expand(-1, self.num_embeddings, -1),  # (B, K, d)
            e.unsqueeze(0).expand(z_proj.shape[0], -1, -1),            # (B, K, d)
        )  # (B, K)
        indices = dist.argmin(dim=-1)
        return indices

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up quantized vector by indices."""
        e = self.poincare_project(self.embedding.weight)
        return e[indices]

    def poincare_reg_loss(self) -> torch.Tensor:
        """Poincaré regularization: encourage codebook vectors to be well inside ball."""
        e = self.poincare_project(self.embedding.weight)
        c = self.curvature
        # poincare_norm_sq: 让 codebook 不太靠近 ball 边界 (提高量化稳定性)
        norm_sq = (e ** 2).sum(dim=-1) * c
        # target: norm_sq < 0.5 (远离边界 1.0)
        return torch.clamp(norm_sq - 0.5, min=0.0).mean()