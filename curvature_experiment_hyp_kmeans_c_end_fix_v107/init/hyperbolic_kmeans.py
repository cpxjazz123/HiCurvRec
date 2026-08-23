"""Hyperbolic K-Means (Poincaré distance + Fréchet mean centroid).
Used for codebook initialization in Riemannian RQ-VAE (Chami et al. 2019 / Sala et al. 2018).
签名与 kmeans.kmeans_init_ 兼容, 可直接替换.
"""
import os, sys
import numpy as np
import torch
from einops import rearrange
from typing import NamedTuple

# Make hyperbolic ops importable (init/hyperbolic_kmeans.py lives one level above modules/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from modules.hyperbolic import _expmap0_t, _logmap0_t, _poincare_distance_t


def _proj_to_ball(x, eps=1e-5):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = ((1 - eps) / norm).clamp(max=1.0)
    return x * scale


class HyperbolicKmeansOutput(NamedTuple):
    centroids: torch.Tensor
    assignment: torch.Tensor


class HyperbolicKmeans:
    """Poincaré K-Means: Poincaré distance assignment + Fréchet mean (Riemannian centroid)."""

    def __init__(self, k: int, max_iters: int = 30, stop_threshold: float = 1e-6, c: float = 0.5):
        self.k = k
        self.iters = max_iters
        self.stop_threshold = stop_threshold
        self.c = c
        self.centroids = None
        self.assignment = None

    def _init_centroids(self, x):
        B, D = x.shape
        init_idx = np.random.choice(B, self.k, replace=False)
        self.centroids = x[init_idx].clone()

    def _frechet_mean(self, pts, c_t, n_iter=15, tol=1e-7):
        """Riemannian Fréchet mean via alternating logmap/expmap."""
        if pts.shape[0] == 0:
            return None
        # init with Euclidean mean projected to ball
        mu = pts.mean(dim=0)
        mu = _proj_to_ball(mu.unsqueeze(0)).squeeze(0)
        for _ in range(n_iter):
            u_tan = torch.zeros_like(mu)
            valid = 0
            for p in pts:
                p_norm = p.norm()
                if p_norm >= 1 - 1e-5:
                    continue
                try:
                    u = _logmap0_t(p.unsqueeze(0), c_t).squeeze(0)
                    u_tan = u_tan + u
                    valid += 1
                except Exception:
                    continue
            if valid == 0:
                break
            u_tan = u_tan / valid
            mu_new = _expmap0_t(u_tan.unsqueeze(0), c_t).squeeze(0)
            mu_new = _proj_to_ball(mu_new.unsqueeze(0)).squeeze(0)
            if (mu_new - mu).norm() < tol:
                mu = mu_new
                break
            mu = mu_new
        return mu

    def _update_centroids(self, x, assignment):
        c_t = torch.tensor(self.c, dtype=x.dtype, device=x.device)
        for cluster in range(self.k):
            mask = (assignment == cluster)
            if not mask.any():
                idx = torch.randint(0, x.shape[0], (1,))
                self.centroids[cluster] = x[idx].squeeze(0)
                continue
            pts = x[mask]
            new_c = self._frechet_mean(pts, c_t)
            if new_c is None:
                self.centroids[cluster] = pts.mean(dim=0)
            else:
                self.centroids[cluster] = new_c

    def _assign(self, x):
        B, D = x.shape
        c_t = torch.tensor(self.c, dtype=x.dtype, device=x.device).view(1, 1, 1)
        x_h = _expmap0_t(x.unsqueeze(1), c_t).squeeze(1)
        c_h = _expmap0_t(self.centroids.unsqueeze(1), c_t).squeeze(1)
        x_h_exp = x_h.unsqueeze(1).expand(B, self.k, D)
        c_h_exp = c_h.unsqueeze(0).expand(B, self.k, D)
        d = _poincare_distance_t(x_h_exp, c_h_exp, c_t).squeeze(-1)
        return d.argmin(dim=-1)

    def run(self, x):
        x = _proj_to_ball(x)
        self._init_centroids(x)
        for i in range(self.iters):
            old_c = self.centroids.clone()
            self.assignment = self._assign(x)
            self._update_centroids(x, self.assignment)
            if (self.centroids - old_c).norm(dim=-1).max() < self.stop_threshold:
                break
        return HyperbolicKmeansOutput(centroids=self.centroids, assignment=self.assignment)


def hyperbolic_kmeans_init_(tensor: torch.Tensor, x: torch.Tensor, c: float = 0.5):
    """Drop-in replacement for kmeans_init_ using Poincaré K-Means + logmap0 back to Euclidean.

    R36c v100 阶段 C 微调: 原版直接拷贝 ball 中心导致 forward 时球内距离压缩 → collapse.
    修正: 在 ball 上做 hyperbolic K-means, 然后 logmap0 把 centroids 映射回 Euclidean
    tangent space, 保留 hyperbolic-aware cluster 结构但避免球内 distance 压缩.
    """
    assert tensor.dim() == 2
    assert x.dim() == 2
    with torch.no_grad():
        k, _ = tensor.shape
        out = HyperbolicKmeans(k=k, c=c).run(x)
        # 把球内 centroids 通过 logmap0 映射回 Euclidean tangent space (保留几何)
        c_t = torch.tensor(c, dtype=out.centroids.dtype, device=out.centroids.device)
        centroids_eucl = _logmap0_t(out.centroids, c_t)
        tensor.data.copy_(centroids_eucl)