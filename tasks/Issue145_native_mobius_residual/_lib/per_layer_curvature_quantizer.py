"""Issue #145 — Per-layer shared learnable curvature RQ-VAE.

Spec 摘录 (R18 D1):
  c_l = c_min + (c_max - c_min) * sigmoid(theta_l), l = 0, 1, 2
  三层共享 c_l, 每层所有 item/codeword 使用同一个 c_l.
  A: 切空间 residual (普通减法/加法)
  B: Möbius residual (减法 = ⊕ with −y) + 跨曲率 T_{a→b} 切空间传输

唯一变量: residual/reconstruction algebra. 距离公式、归一化、Sinkhorn/argmin、anchor、optimizer 全相同.

实施要点:
  - 与 Issue #140 共享 HVectorQuantization + per-layer curvature 框架 (c_l = C_MIN+(C_MAX-C_MIN)·sigmoid(theta_l))
  - 通过 mobius_residual=False/True 切换 A/B
  - A 路径: tangent 减法/加法, 与基线一致 (Issue #139/#140)
  - B 路径: expmap0 → Möbius subtract → logmap0 产生下一层 residual; 重构时 transport + Möbius add
  - B 路径 c_l→0 极限必须数值收敛到 A (Gate 1 check 5)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_

# 从 utils 复用: proj_to_ball, mobius_add, expmap0, logmap0, poincare_distance, MLP
from utils import (
    proj_to_ball, mobius_add, expmap0, logmap0, poincare_distance,
    MLP, sinkhorn_algorithm, kmeans, _eps
)

# ──────────────────────────────────────────────────────────────────
# Issue #145 spec: c_l = C_MIN + (C_MAX - C_MIN) * sigmoid(theta_l) 有界可学习
# ──────────────────────────────────────────────────────────────────
C_MIN = 0.5
C_MAX = 2.0
THETA_INIT = 0.0  # sigmoid(0) = 0.5 → c_init = 1.0 (与基线 c=1 严格对齐, 起点等价)
RADIUS_SCALE_EPS = 1e-6


def _logit(p: float) -> float:
    """sigmoid 反函数: θ = logit(p) ⟹ sigmoid(θ) = p."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def init_theta_for_c(c_target: float) -> float:
    """反解 θ 使 c_l = c_target: θ = logit((c - C_MIN) / (C_MAX - C_MIN))."""
    p = (c_target - C_MIN) / (C_MAX - C_MIN)
    return _logit(p)


# ──────────────────────────────────────────────────────────────────
# Per-layer vector quantizer with shared learnable curvature c_l
# ──────────────────────────────────────────────────────────────────
class PerLayerCurvatureVQ(nn.Module):
    """Issue #145: 每层共享一个有界可学习曲率 c_l = C_MIN + (C_MAX-C_MIN)·sigmoid(θ_l).

    与 Issue #140 (per-codeword θ_b) 不同: θ_l 是层标量 (3 个参数总计).
    所有 item/codeword 在同一层用同一个 c_l.
    """

    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=True, kmeans_iters=10,
                 sk_eps=0.003, sk_iters=3, fix_c=False, layer_idx=0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        self.fix_c = fix_c
        self.layer_idx = layer_idx

        self.embeddings = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            self.initted = True
            with torch.no_grad():
                self.embeddings.weight.data.uniform_(-0.01, 0.01)
        else:
            self.initted = False
            self.embeddings.weight.data.zero_()

        # 单层共享 θ_l
        self.theta = nn.Parameter(torch.tensor(THETA_INIT, dtype=torch.float32))
        self.mix_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def get_c(self) -> torch.Tensor:
        """Issue #145: c_l = C_MIN + (C_MAX-C_MIN)·sigmoid(θ_l) ∈ (C_MIN, C_MAX) 有界"""
        return C_MIN + (C_MAX - C_MIN) * torch.sigmoid(self.theta)

    def get_codebook(self) -> torch.Tensor:
        return self.embeddings.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embeddings(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embeddings.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-10
        assert amplitude > 0, "Amplitude must be positive"
        return (distances - middle) / amplitude

    def forward(self, x, use_sk=True):
        """Input x: tangent-space latent (B, e_dim).
        Output x_q: tangent-space quantized (B, e_dim)."""
        latent = x.view(-1, self.e_dim)
        codebook_e = self.embeddings.weight

        if not self.initted and self.training:
            self.init_emb(latent)

        c = self.get_c()  # 标量, 带梯度
        # 与基线一致: 距离对 c stop-grad 防"距离随 c 减"尺度作弊; κ 梯度由结构目标/先验提供
        c_geom = c.detach() if not self.fix_c else c

        latent_h = proj_to_ball(expmap0(latent, c_geom), c_geom)
        codebook_h = proj_to_ball(expmap0(codebook_e, c_geom), c_geom)

        B = latent_h.shape[0]
        K = codebook_h.shape[0]

        latent_exp = latent_h.unsqueeze(1).expand(B, K, -1)
        cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)
        d = poincare_distance(latent_exp, cb_exp, c_geom).squeeze(-1)

        if not use_sk or self.sk_eps <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_centered = self.center_distance_for_constraint(d).double()
            Q = sinkhorn_algorithm(d_centered, self.sk_eps, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise ValueError("Sinkhorn produced NaN/Inf")
            indices = torch.argmax(Q, dim=-1)

        x_q = codebook_e.index_select(0, indices)
        # 量化损失: 与基线 HVectorQuantization 同公式
        commitment_loss = torch.mean(poincare_distance(x_q.detach(), latent, c_geom) ** 2)
        codebook_loss = torch.mean(poincare_distance(x_q, latent.detach(), c_geom) ** 2)
        loss = commitment_loss + self.beta * codebook_loss

        # logmap0 输入先 proj_to_ball 兜底
        x_q_safe = proj_to_ball(x_q, c_geom)
        latent_safe = proj_to_ball(latent, c_geom)
        x_q = logmap0(x_q_safe, c_geom)
        latent = logmap0(latent_safe, c_geom)
        x_q = x + (x_q - x).detach()  # straight-through estimator
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices


# ──────────────────────────────────────────────────────────────────
# RQ with optional Möbius residual algebra
# ──────────────────────────────────────────────────────────────────
class PerLayerCurvatureHRQ(nn.Module):
    """Issue #145: 3-layer RQ with optional Möbius residual.

    A 路径 (mobius_residual=False):
      residual_t = x_t (encoder 输出, 切空间)
      x_q_t = 0
      for l in layers:
          x_res_t, loss, idx = quantizer_l(residual_t)   # 切空间操作
          residual_t = residual_t - x_res_t              # 切空间减法
          x_q_t = x_q_t + x_res_t                        # 切空间加法

    B 路径 (mobius_residual=True):
      residual_t = x_t
      for l in layers:
          x_res_t, loss, idx = quantizer_l(residual_t)   # 同 A
          c_l = cs[l]
          h_l = expmap0(residual_t, c_l)
          q_l = expmap0(x_res_t, c_l)
          u_l = mobius_add(-h_l, q_l, c_l)               # Möbius subtract
          residual_t = logmap0(u_l, c_l)                  # 下一层 residual (切空间)

      # 重构 (B): hyperbolic addition with cross-curvature transport
      x_hat_H = zeros_like(x_q[0])
      for l in layers:
          c_l = cs[l]
          q_l_h = expmap0(x_q[l], c_l)
          if l == 0:
              x_hat_H = q_l_h  # origin ⊕ q = q
          else:
              c_prev = cs[l-1]
              x_prev = expmap0(logmap0(x_hat_H, c_prev), c_l)  # T_{c_{l-1}→c_l}
              x_hat_H = mobius_add(x_prev, q_l_h, c_l)
      x_q_t = logmap0(x_hat_H, cs[-1])
    """

    def __init__(self, n_e_list, e_dim, sk_eps, beta=0.25, kmeans_init=True,
                 kmeans_iters=10, sk_iters=3, fix_c=False, mobius_residual=False):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.beta = beta
        self.fix_c = fix_c
        self.mobius_residual = mobius_residual
        self.vq_layers = nn.ModuleList([
            PerLayerCurvatureVQ(n_e, e_dim, beta=beta, kmeans_init=kmeans_init,
                                kmeans_iters=kmeans_iters, sk_eps=eps, sk_iters=sk_iters,
                                fix_c=fix_c, layer_idx=i)
            for i, (n_e, eps) in enumerate(zip(n_e_list, sk_eps))
        ])
        # 审计用缓存
        self._last_residual_norms = []  # 每层 residual norm
        self._last_recon_errors = []    # 每层 reconstruction error
        self._last_roundtrip_errors = []  # Exp/Log round-trip error per layer
        self._last_transport_errors = []  # T_{a→b}+T_{b→a} round-trip
        self._last_boundary_hit = []     # 每层 boundary hit rate

    def get_c_list(self):
        return [q.get_c() for q in self.vq_layers]

    def forward(self, x, use_sk=True):
        """Input x: tangent-space (B, e_dim).
        Output x_q: tangent-space quantized."""
        cs = self.get_c_list() if self.mobius_residual else None

        all_losses = []
        all_indices = []
        all_xq_t = []  # 切空间累积
        residual_t = x

        # 清空审计缓存
        self._last_residual_norms = []
        self._last_recon_errors = []
        self._last_roundtrip_errors = []
        self._last_transport_errors = []
        self._last_boundary_hit = []

        for l, quantizer in enumerate(self.vq_layers):
            x_res_t, loss, idx = quantizer(residual_t, use_sk=use_sk)
            all_losses.append(loss)
            all_indices.append(idx)
            all_xq_t.append(x_res_t)

            # 审计: residual norm (训练信息)
            with torch.no_grad():
                self._last_residual_norms.append(float(residual_t.norm(dim=-1).mean().item()))

            if self.mobius_residual:
                # B 路径: Möbius residual
                c_l = cs[l]
                # 数值安全: 强制 proj 后再 exp/log
                residual_h = proj_to_ball(expmap0(residual_t, c_l), c_l)
                q_h = proj_to_ball(expmap0(x_res_t, c_l), c_l)
                # round-trip audit
                rt_back = logmap0(residual_h, c_l)
                rt_err = (rt_back - residual_t).norm(dim=-1).mean()
                self._last_roundtrip_errors.append(float(rt_err.item()))

                # Möbius subtract: h_l ⊖ q_l = h_l ⊕ (-q_l)
                u_l = mobius_add(-residual_h, q_h, c_l)
                residual_t_new = logmap0(u_l, c_l)

                # boundary hit audit
                r_norm = residual_h.norm(dim=-1)
                r_max = (1.0 / c_l.sqrt()).item() if c_l.item() > 0 else 1e10
                hit_rate = float((r_norm > 0.999 * r_max).float().mean().item())
                self._last_boundary_hit.append(hit_rate)

                residual_t = residual_t_new
            else:
                # A 路径: tangent residual (基线)
                residual_t = residual_t - x_res_t

        # 重构
        if self.mobius_residual:
            # B: hyperbolic addition with cross-curvature transport
            x_hat_H = torch.zeros_like(all_xq_t[0])
            for l, x_q_t in enumerate(all_xq_t):
                c_l = cs[l]
                q_l_h = proj_to_ball(expmap0(x_q_t, c_l), c_l)
                if l == 0:
                    x_hat_H = q_l_h
                else:
                    c_prev = cs[l-1]
                    x_prev = proj_to_ball(expmap0(logmap0(x_hat_H, c_prev), c_l), c_l)
                    # transport audit: T_{a→b}+T_{b→a} round-trip
                    with torch.no_grad():
                        rt_back = proj_to_ball(expmap0(logmap0(x_prev, c_l), c_prev), c_prev)
                        t_err = (rt_back - x_hat_H).norm(dim=-1).mean()
                        self._last_transport_errors.append(float(t_err.item()))
                    x_hat_H = mobius_add(x_prev, q_l_h, c_l)
            x_q_total = logmap0(x_hat_H, cs[-1])
        else:
            # A: tangent accumulation (基线)
            x_q_total = sum(all_xq_t)

        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        return x_q_total, mean_loss, all_indices


# ──────────────────────────────────────────────────────────────────
# Full HRQVAE wrapper
# ──────────────────────────────────────────────────────────────────
class PerLayerCurvatureHRQVAE(nn.Module):
    def __init__(self, in_dim=768, num_emb_list=None, e_dim=64, layers=None,
                 dropout_prob=0.0, bn=False, loss_type='mse',
                 quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
                 kmeans_iters=10, sk_eps=None, sk_iters=100,
                 fix_c=False, mobius_residual=False):
        super().__init__()
        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.fix_c = fix_c

        self.encode_layer_dims = [in_dim] + list(layers) + [e_dim]
        self.encoder = MLP(layers=self.encode_layer_dims, dropout=dropout_prob, use_bn=bn)
        self.hrq = PerLayerCurvatureHRQ(
            n_e_list=num_emb_list, e_dim=e_dim, sk_eps=sk_eps,
            beta=beta, kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
            sk_iters=sk_iters, fix_c=fix_c, mobius_residual=mobius_residual
        )
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLP(layers=self.decode_layer_dims, dropout=dropout_prob, use_bn=bn)

    def forward(self, x, use_sk=True):
        z = self.encoder(x)
        z_q, rq_loss, indices = self.hrq(z, use_sk=use_sk)
        out = self.decoder(z_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, x, use_sk=True):
        z = self.encoder(x)
        _, _, indices = self.hrq(z, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quent_loss, xs=None):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'poincare':
            o = expmap0(out, c=1)
            t = expmap0(xs, c=1)
            o = proj_to_ball(o, c=1)
            t = proj_to_ball(t, c=1)
            loss_recon = torch.mean(poincare_distance(o, t, c=1) ** 2)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        loss_total = loss_recon + self.quant_loss_weight * quent_loss
        return loss_total, loss_recon
