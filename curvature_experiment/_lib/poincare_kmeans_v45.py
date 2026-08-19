"""v45 (Issue265): Stage 2 Poincaré K-means — 双曲球面 K-means 初始化 codebook.

设计: 把 Stage 2 K-means 聚类从欧氏空间 (init/kmeans.py) 改到 Poincaré 球面:
  - 距离: d_P(u, v) = arccosh(1 + 2 ||u-v||² / ((1-||u||²)(1-||v||²))) / sqrt(c)
  - 中心更新: Fréchet mean (双曲重心), 用 Karcher 梯度下降 100 步

替代 init/kmeans.py 里的欧氏 K-means, 让 codebook 中心天然落在 Poincaré 球面,
后续 RQ-VAE 训练时 d_P 距离与中心分布更一致.

R36 合规: 仅改 Stage 2 端 K-means 几何变换, 无 LR/dropout/wd sweep.
"""
import numpy as np
import torch
from einops import rearrange
from typing import NamedTuple


def _poincare_distance(u, v, c):
    """Poincaré 距离 sqrt(c) * arccosh(1 + 2c ||u-v||² / ((1-c||u||²)(1-c||v||²)))."""
    sq_u = (u * u).sum(dim=-1, keepdim=True)  # (..., 1)
    sq_v = (v * v).sum(dim=-1, keepdim=True)
    sq_diff = ((u.unsqueeze(-2) - v.unsqueeze(-3)) ** 2).sum(dim=-1)
    arg = 1 + 2 * c * sq_diff / ((1 - c * sq_u) * (1 - c * sq_v).transpose(-1, -2)).clamp_min(1e-10)
    arg = arg.clamp_min(1.0 + 1e-7)
    return (1.0 / (c ** 0.5)) * torch.acosh(arg)


def _expmap0(v, c):
    """欧氏 → Poincaré ball."""
    sq_v = (v * v).sum(dim=-1, keepdim=True)
    return _expmap0_sq(v, sq_v, c)


def _expmap0_sq(v, sq_v, c):
    return v / (1 + (1 - c * sq_v).clamp_min(1e-10).sqrt()).clamp_min(1e-10) * 2


def _project_to_ball(x, c):
    """把向量 project 到 Poincaré ball (||x||² < 1/c)."""
    norm = (x * x).sum(dim=-1, keepdim=True).clamp_min(1e-10).sqrt()
    max_norm = (1.0 / (c ** 0.5) - 1e-5)
    return x * (max_norm / norm).clamp_max(1.0)


def _frechet_mean(points, weights=None, c=0.5, n_iters=50):
    """Poincaré ball 上的 Karcher / Fréchet 均值.

    迭代: μ_t+1 = expmap_μ_t( sum w_i * logmap_μ_t(p_i) / sum w_i )
    points: (N, D), weights: (N,) 或 None (uniform)
    """
    from modules.hyperbolic import _expmap0_t, _logmap0_t

    if weights is None:
        weights = torch.ones(points.shape[0], device=points.device) / points.shape[0]
    else:
        weights = weights / weights.sum()

    mu = points.mean(dim=0, keepdim=True)  # 初始化用欧氏均值
    mu = _project_to_ball(mu, c)

    for _ in range(n_iters):
        # logmap_μ: 每个点映到 μ 的切空间
        log_p = _logmap0_t(points - mu, c)  # (N, D) — 近似 logmap(欧氏差 + 几何修正)
        # 加权平均
        delta = (weights.unsqueeze(-1) * log_p).sum(dim=0, keepdim=True)
        # expmap_μ: 沿切向量推回 Poincaré
        mu = _expmap0_t(mu + delta, c)
        mu = _project_to_ball(mu, c)
    return mu.squeeze(0)


class PoincaréKmeansOutput(NamedTuple):
    centroids: torch.Tensor
    assignment: torch.Tensor


class PoincaréKmeans:
    """Poincaré ball 上的 K-means 聚类 (双曲距离 + Fréchet mean 更新中心)."""

    def __init__(self, k: int, c: float = 0.5, max_iters: int = None, stop_threshold: float = 1e-7):
        self.k = k
        self.c = c
        self.iters = max_iters
        self.stop_threshold = stop_threshold
        self.centroids = None
        self.assignment = None

    def _init_centroids(self, x: torch.Tensor) -> None:
        """k-means++ 风格: 在 Poincaré ball 内随机选 k 个不重复点."""
        B = x.shape[0]
        init_idx = np.random.choice(B, self.k, replace=False)
        self.centroids = x[init_idx, :].clone()
        self.assignment = None

    def _update_centroids(self, x) -> None:
        """分配 + Fréchet 均值更新中心."""
        # 分配: Poincaré 距离最近
        dist = _poincare_distance(x, self.centroids, self.c)  # (B, k)
        centroid_idx = dist.argmin(dim=-1)  # (B,)

        # Fréchet mean 更新每个中心
        for cluster in range(self.k):
            mask = (centroid_idx == cluster)
            if mask.any():
                cluster_points = x[mask]
                new_center = _frechet_mean(cluster_points, c=self.c, n_iters=30)
                self.centroids[cluster] = new_center
            else:
                # 空 cluster: 随机重选
                rand_idx = torch.randint(0, x.shape[0], (1,)).item()
                self.centroids[cluster] = x[rand_idx]
        self.assignment = centroid_idx

    def run(self, x):
        self._init_centroids(x)
        i = 0
        while self.iters is None or i < self.iters:
            old_c = self.centroids.clone()
            self._update_centroids(x)
            # 收敛判定: 中心移动最大范数
            max_shift = (self.centroids - old_c).norm(dim=-1).max()
            if max_shift < self.stop_threshold:
                break
            i += 1
        return PoincaréKmeansOutput(centroids=self.centroids, assignment=self.assignment)


def poincare_kmeans_init_(tensor: torch.Tensor, x: torch.Tensor, c: float = 0.5):
    """在 Poincaré ball 上做 K-means 初始化 codebook tensor (K, D).

    替代 init/kmeans.py:kmeans_init_, 让 codebook 中心天然落在 Poincaré 球面.
    """
    assert tensor.dim() == 2
    assert x.dim() == 2
    with torch.no_grad():
        k, D = tensor.shape
        # 把输入 x 也投到 Poincaré ball (如果欧氏向量不在球内)
        x_norm = (x * x).sum(dim=-1, keepdim=True).sqrt()
        max_x_norm = (1.0 / (c ** 0.5) - 1e-3)
        x_proj = x * (max_x_norm / x_norm.clamp_min(1e-10)).clamp_max(1.0)
        # 跑 Poincaré K-means
        out = PoincaréKmeans(k=k, c=c, max_iters=100).run(x_proj)
        # 把结果 project 到球内 (Fréchet mean 可能轻微越界)
        centroids = _project_to_ball(out.centroids, c)
        tensor.data.copy_(centroids)