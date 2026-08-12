"""Issue #148 — Factorized Product-Manifold Quantizer.

A (control): 每层 32 维位于共享曲率 c_l 的双曲空间, d_A(i,b) = d_H(z_l,i, e_l,b; c_l)
B (treatment): 每层 32 维严格拆分 16 维 Euclidean + 16 维 Hyperbolic 因子, 乘积距离:
  d_E(i,b) = ||z_l,i^E - e_l,b^E||_2
  d_H(i,b) = d_Poincare(z_l,i^H, e_l,b^H; c_l^H)
  d_product(i,b) = sqrt(d_E^2 + d_H^2)
  c_l^H = C_MIN + (C_MAX - C_MIN) * sigmoid(theta_l^H)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import proj_to_ball, expmap0, logmap0, poincare_distance, kmeans, MLP, _eps

C_MIN = 0.5
C_MAX = 2.0
E_DIM_EUCLIDEAN = 16
E_DIM_HYPERBOLIC = 16
E_DIM_TOTAL = 32  # = E_DIM_EUCLIDEAN + E_DIM_HYPERBOLIC
SCALE_COEFF_INIT = 1.0  # 固定单位换算系数 (不允许更新)


def _proj_to_ball_t(x, c, eps=1e-6):
    r = (1.0 / c) ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1 - eps) * r
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def _expmap0_t(u, c):
    sqrt_c = c ** 0.5
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(_eps(u))
    factor = torch.tanh(sqrt_c * norm_u) / (sqrt_c * norm_u)
    return _proj_to_ball_t(factor * u, c)


def _logmap0_t(x, c):
    sqrt_c = c ** 0.5
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
    factor = torch.atanh((sqrt_c * norm_x).clamp(max=1 - 1e-5)) / (sqrt_c * norm_x)
    return factor * x


# ── 单层量化器 ──
class FactorizedProductVQ(nn.Module):
    """A: 单一双曲因子 (per-layer shared c_l)
       B: 16 维 Euclidean + 16 维 Hyperbolic 乘积 (per-layer shared c_l^H)"""

    def __init__(self, n_e, e_dim_total=E_DIM_TOTAL, e_dim_euc=E_DIM_EUCLIDEAN,
                 e_dim_hyp=E_DIM_HYPERBOLIC, beta=0.25, kmeans_init=True, kmeans_iters=10,
                 sk_eps=0.0, sk_iters=3, layer_idx=0, factorized=False):
        super().__init__()
        assert e_dim_total == e_dim_euc + e_dim_hyp, f"e_dim_total={e_dim_total} != euc+hyp"
        self.n_e = n_e
        self.e_dim_total = e_dim_total
        self.e_dim_euc = e_dim_euc
        self.e_dim_hyp = e_dim_hyp
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        self.layer_idx = layer_idx
        self.factorized = factorized
        # A: 全局共享 c_l (单参数)
        # B: 单一曲率 c_l^H (仅作用于双曲 16 维; 欧氏 16 维无曲率)
        # 两者都用单个 theta (B 是 c_l^H, A 是 c_l)
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.embeddings = nn.Embedding(n_e, e_dim_total)
        if not kmeans_init:
            self.initted = True
            with torch.no_grad():
                self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            self.initted = False
            self.embeddings.weight.data.zero_()

    def get_c(self) -> torch.Tensor:
        return C_MIN + (C_MAX - C_MIN) * torch.sigmoid(self.theta)

    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embeddings.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim_total)  # (B, 32)
        if not self.initted and self.training:
            self.init_emb(latent)
        B = latent.shape[0]
        K = self.n_e
        if not self.factorized:
            # A 路径: 32 维全部位于共享 c_l 的双曲空间
            c = self.get_c()  # scalar
            latent_h = proj_to_ball(expmap0(latent, c), c)  # (B, 32)
            codebook_h = proj_to_ball(expmap0(self.embeddings.weight, c), c)  # (K, 32)
            x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
            cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)
            d = poincare_distance(x_exp, cb_exp, c).squeeze(-1)  # (B, K)
        else:
            # B 路径: 拆分为 16 维 Euclidean + 16 维 Hyperbolic, 乘积距离
            c = self.get_c()  # c_l^H (scalar, 用于双曲部分)
            latent_euc = latent[:, :self.e_dim_euc]  # (B, 16)
            latent_hyp = latent[:, self.e_dim_euc:]  # (B, 16)
            codebook_euc = self.embeddings.weight[:, :self.e_dim_euc]  # (K, 16)
            codebook_hyp = self.embeddings.weight[:, self.e_dim_euc:]  # (K, 16)
            # 欧氏距离: ||z^E - e^E||_2
            d_euc = torch.cdist(latent_euc.unsqueeze(1), codebook_euc.unsqueeze(0)).squeeze(1)  # (B, K)
            # 双曲距离: 把 16 维投影到 c_l^H 双曲空间, 再算 Poincaré 距离
            latent_hyp_h = proj_to_ball(expmap0(latent_hyp, c), c)  # (B, 16)
            codebook_hyp_h = proj_to_ball(expmap0(codebook_hyp, c), c)  # (K, 16)
            x_exp = latent_hyp_h.unsqueeze(1).expand(B, K, -1)
            cb_exp = codebook_hyp_h.unsqueeze(0).expand(B, K, -1)
            d_hyp = poincare_distance(x_exp, cb_exp, c).squeeze(-1)  # (B, K)
            # 乘积距离: sqrt(d_E^2 + d_H^2)
            d = torch.sqrt(d_euc ** 2 + d_hyp ** 2 + 1e-12)
        if not use_sk or self.sk_eps <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_centered = (d - d.mean(dim=-1, keepdim=True)) / (d.std(dim=-1, keepdim=True) + 1e-8)
            d_centered = d_centered.double()
            from utils import sinkhorn_algorithm
            Q = sinkhorn_algorithm(d_centered, self.sk_eps, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise ValueError("Sinkhorn NaN/Inf")
            indices = torch.argmax(Q, dim=-1)
        # 量化 loss (用 selected codeword 的距离)
        x_q = self.embeddings.weight.index_select(0, indices)  # (B, 32)
        if not self.factorized:
            c_sel = c
            # 投影 x_q 和 latent 到 ball 避免 poincare_distance 数值溢出
            x_q_proj = proj_to_ball(x_q, c_sel)
            latent_proj = proj_to_ball(latent, c_sel)
            commitment_loss = torch.mean(poincare_distance(x_q_proj.detach(), latent_proj, c_sel) ** 2)
            codebook_loss = torch.mean(poincare_distance(x_q_proj, latent_proj.detach(), c_sel) ** 2)
            x_q_safe = x_q_proj
            latent_safe = latent_proj
            x_q = _logmap0_t(x_q_safe, c_sel)
            latent = _logmap0_t(latent_safe, c_sel)
        else:
            # B 路径: commitment + codebook loss 用乘积距离
            x_q_euc = x_q[:, :self.e_dim_euc]
            x_q_hyp = x_q[:, self.e_dim_euc:]
            latent_euc = latent[:, :self.e_dim_euc]
            latent_hyp = latent[:, self.e_dim_euc:]
            d_euc_q = torch.sum((x_q_euc - latent_euc) ** 2, dim=-1)  # (B,)
            latent_hyp_h = proj_to_ball(expmap0(latent_hyp, c), c)
            x_q_hyp_h = proj_to_ball(expmap0(x_q_hyp, c), c)
            d_hyp_q = poincare_distance(x_q_hyp_h, latent_hyp_h, c).squeeze(-1)  # (B,)
            d_q = torch.sqrt(d_euc_q + d_hyp_q + 1e-12)  # (B,)
            commitment_loss = torch.mean(d_q.detach() ** 2)
            codebook_loss = torch.mean(d_q ** 2)
            x_q_safe = proj_to_ball(x_q_hyp, c)
            latent_safe = proj_to_ball(latent_hyp, c)
            x_q_proj = _logmap0_t(x_q_safe, c)
            latent_proj = _logmap0_t(latent_safe, c)
            x_q_new = x_q.clone()
            x_q_new[:, self.e_dim_euc:] = x_q_proj
            latent_new = latent.clone()
            latent_new[:, self.e_dim_euc:] = latent_proj
            x_q = x_q_new
            latent = latent_new
        loss = commitment_loss + self.beta * codebook_loss
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices


# ── 完整 HRQVAE ──
class FactorizedProductHRQVAE(nn.Module):
    def __init__(self, in_dim=768, num_emb_list=(64, 128, 256), e_dim=E_DIM_TOTAL,
                 layers=(512, 256, 128, 64), beta=0.25, kmeans_init=True,
                 kmeans_iters=10, factorized=False):
        super().__init__()
        self.in_dim = in_dim
        self.num_emb_list = list(num_emb_list)
        self.e_dim = e_dim
        self.layers = layers
        self.factorized = factorized
        encode_layer_dims = [in_dim] + list(layers) + [e_dim]
        self.encoder = MLP(layers=encode_layer_dims, dropout=0.0, use_bn=False)
        decode_layer_dims = encode_layer_dims[::-1]
        self.decoder = MLP(layers=decode_layer_dims, dropout=0.0, use_bn=False)
        self.vq_layers = nn.ModuleList([
            FactorizedProductVQ(n_e, e_dim_total=e_dim, beta=beta,
                                kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                layer_idx=i, factorized=factorized)
            for i, n_e in enumerate(num_emb_list)
        ])

    def forward(self, x, use_sk=True):
        z = self.encoder(x)
        z_q, rq_loss, indices = self._rq_forward(z, use_sk=use_sk)
        out = self.decoder(z_q)
        return out, rq_loss, indices, z_q, z

    def _rq_forward(self, x, use_sk=True):
        all_losses, all_indices = [], []
        x_q = 0
        residual = x
        for q in self.vq_layers:
            x_res, loss, idx = q(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(idx)
        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        return x_q, mean_loss, all_indices

    def get_indices(self, x, use_sk=True):
        z = self.encoder(x)
        _, _, indices = self._rq_forward(z, use_sk=use_sk)
        return indices