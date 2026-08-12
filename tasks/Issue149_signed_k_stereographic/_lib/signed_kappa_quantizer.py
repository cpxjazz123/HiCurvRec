"""Issue #149 — Signed κ-Stereographic Quantizer.

A (control): 三层只能学习负曲率, c_l = C_MIN + (C_MAX-C_MIN)·sigmoid(θ_l), sectional_curvature_l = -c_l.
  Distance: Poincaré ball d_H(x, y; c_l).

B (treatment): 三层各学习 signed sectional curvature κ_l = κ_max · tanh(θ_l),
  κ_l ∈ (-κ_max, +κ_max). κ<0 双曲, κ=0 欧氏, κ>0 球面. 统一 κ-stereographic 形式.

B 初始化: 与 A 同 (负曲率), 训练起点 forward/assignment/loss/SID 严格一致.
训练后允许跨越 0.

关键设计 (R18): B 路径在 |κ|>K_THRESH 时使用与 A 完全相同的 c=|κ| Poincaré 公式
(保证 init κ=-1.25 时 B distance/assignment 与 A c=1.25 完全相同), 在 κ≈0 时切换到
Euclidean (||x-y||) 保证可微, 在 κ>K_THRESH 时切换到 spherical ((2/√κ)·arctan(√κ·d)).
所有切换用 sigmoid 平滑, 保证 θ 梯度连续.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import proj_to_ball, expmap0, logmap0, poincare_distance, kmeans, MLP, _eps, mobius_add

C_MIN = 0.5
C_MAX = 2.0
KAPPA_MAX = 2.0  # |κ| 上限 (与 c_max 对齐, 保证 A/B 起点一致: 初始 c=1.25 → κ=-1.25)
C_INITIAL = 1.25  # sigmoid(0)·(C_MAX-C_MIN)+C_MIN = 1.25, A 路径初始 c
E_DIM = 32
K_THRESH = 0.05  # |κ| 切换阈值: |κ|>K_THRESH → κ-stereographic, 否则 Euclidean
SWITCH_SCALE = 20.0  # sigmoid 切换的锐度 (大=接近硬切换)


def _signed_kappa_mobius_diff(x, y, kappa):
    """Compute ||(-x) ⊕ y||_c with c = |κ| (always positive).

    Rationale (R18): κ 的符号只用于分支选择 (hyperbolic/spherical/Euclidean), 不改变 Möbius add.
    Möbius add 在 hyperbolic/spherical 分支都用 c=|κ|>0, 保证 init κ=-1.25 时
    c_eff=1.25, 与 A 路径的 c=1.25 严格等价.
    """
    if not isinstance(kappa, torch.Tensor):
        c_abs = abs(float(kappa))
    else:
        c_abs = abs(kappa).item()
    diff = mobius_add(-x, y, c_abs)
    d = diff.norm(dim=-1, keepdim=True).clamp_min(_eps(diff))
    return d


def signed_kappa_distance(x, y, kappa):
    """Signed κ distance with smooth blending.

    - |κ|>K_THRESH, κ<0: D = (2/√|κ|) · artanh(√|κ|·d_mobius)  [hyperbolic, 与 A 同 c=|κ|]
    - |κ|>K_THRESH, κ>0: D = (2/√κ) · arctan(√κ·d_mobius)      [spherical]
    - |κ|<K_THRESH: D = ||x-y||                                 [Euclidean limit]

    三段用 sigmoid 平滑, 保证可微.
    """
    if not isinstance(kappa, torch.Tensor):
        kappa = torch.tensor(float(kappa), dtype=x.dtype, device=x.device)
    abs_k = abs(kappa).clamp_min(1e-7)
    sqrt_abs_k = abs_k.sqrt()
    d = _signed_kappa_mobius_diff(x, y, kappa)
    d_euc = torch.norm(x - y, dim=-1, keepdim=True)
    u_hyp = (sqrt_abs_k * d).clamp(max=1 - 1e-7)
    d_hyp = (2.0 / sqrt_abs_k) * torch.arctanh(u_hyp)
    u_sph = (sqrt_abs_k * d).clamp(max=math.pi / 2 - 1e-5)
    d_sph = (2.0 / sqrt_abs_k) * torch.arctan(u_sph)
    w_geo = torch.sigmoid(SWITCH_SCALE * (abs_k - K_THRESH))
    w_hyp = (1.0 - torch.sigmoid(SWITCH_SCALE * kappa)) * w_geo
    w_sph = torch.sigmoid(SWITCH_SCALE * kappa) * w_geo
    w_euc = 1.0 - w_geo
    D = w_hyp * d_hyp + w_sph * d_sph + w_euc * d_euc
    return D


def signed_kappa_expmap0(u, kappa):
    """Signed κ expmap at origin with smooth blending.

    - |κ|>K_THRESH, κ<0: exp_0(v) = tanh(√|κ|·||v||) · v/(√|κ|·||v||)  [与 A 同 c=|κ|]
    - |κ|>K_THRESH, κ>0: exp_0(v) = tan(√κ·||v||) · v/(√κ·||v||)
    - |κ|<K_THRESH: exp_0(v) = v  [Euclidean limit]

    投影到 ball/sphere: R = 1/√|κ|.
    """
    if not isinstance(kappa, torch.Tensor):
        kappa = torch.tensor(float(kappa), dtype=u.dtype, device=u.device)
    abs_k = abs(kappa).clamp_min(1e-7)
    sqrt_k = abs_k.sqrt()
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(_eps(u))
    sqrt_k_norm = sqrt_k * norm_u  # 不要 clamp, tanh 永远有界
    # tan 仅在球面分支需要 clamp 防溢出
    sqrt_k_norm_sph = sqrt_k_norm.clamp(max=math.pi / 2 - 1e-5)
    tanh_part = torch.tanh(sqrt_k_norm)
    tan_part = torch.tan(sqrt_k_norm_sph)
    w_geo = torch.sigmoid(SWITCH_SCALE * (abs_k - K_THRESH))
    w_hyp = (1.0 - torch.sigmoid(SWITCH_SCALE * kappa)) * w_geo
    w_sph = torch.sigmoid(SWITCH_SCALE * kappa) * w_geo
    w_euc = 1.0 - w_geo
    hyp_factor = tanh_part / sqrt_k_norm
    sph_factor = tan_part / sqrt_k_norm_sph
    euc_factor = 1.0
    factor = w_hyp * hyp_factor + w_sph * sph_factor + w_euc * euc_factor
    out = factor * u
    R = 1.0 / sqrt_k
    out_norm = out.norm(dim=-1, keepdim=True).clamp_min(_eps(out))
    max_norm = (1 - 1e-5) * R
    scale = torch.where(out_norm > max_norm, max_norm / out_norm, torch.ones_like(out_norm))
    return out * scale


def signed_kappa_logmap0(x, kappa):
    """Signed κ logmap at origin with smooth blending.

    - |κ|>K_THRESH, κ<0: log_0(y) = artanh(√|κ|·||y||) · y/(√|κ|·||y||)
    - |κ|>K_THRESH, κ>0: log_0(y) = arctan(√κ·||y||) · y/(√κ·||y||)
    - |κ|<K_THRESH: log_0(y) = y
    """
    if not isinstance(kappa, torch.Tensor):
        kappa = torch.tensor(float(kappa), dtype=x.dtype, device=x.device)
    abs_k = abs(kappa).clamp_min(1e-7)
    sqrt_k = abs_k.sqrt()
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
    sqrt_k_norm = sqrt_k * norm_x
    # artanh 仅在双曲分支需要 clamp 防溢出 (√|κ|·||y|| < 1)
    sqrt_k_norm_hyp = sqrt_k_norm.clamp(max=1 - 1e-7)
    atanh_part = torch.arctanh(sqrt_k_norm_hyp)
    atan_part = torch.arctan(sqrt_k_norm.clamp(max=math.pi / 2 - 1e-5))
    w_geo = torch.sigmoid(SWITCH_SCALE * (abs_k - K_THRESH))
    w_hyp = (1.0 - torch.sigmoid(SWITCH_SCALE * kappa)) * w_geo
    w_sph = torch.sigmoid(SWITCH_SCALE * kappa) * w_geo
    w_euc = 1.0 - w_geo
    hyp_factor = atanh_part / sqrt_k_norm_hyp
    sph_factor = atan_part / sqrt_k_norm.clamp(min=_eps(x))
    euc_factor = 1.0
    factor = w_hyp * hyp_factor + w_sph * sph_factor + w_euc * euc_factor
    return factor * x


def signed_kappa_proj(x, kappa):
    """Project x onto valid domain: ball of radius 1/√|κ| (hyperbolic) or sphere of radius 1/√κ (spherical)."""
    if not isinstance(kappa, torch.Tensor):
        kappa = torch.tensor(float(kappa), dtype=x.dtype, device=x.device)
    abs_k = abs(kappa).clamp_min(1e-7)
    R = 1.0 / abs_k.sqrt()
    norm = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
    max_norm = (1 - 1e-5) * R
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


# ── 单层量化器 ──
class SignedKappaVQ(nn.Module):
    """A: standard hyperbolic (c_l > 0) with full Poincaré distance.
       B: signed κ-stereographic (κ ∈ [-κ_max, κ_max]) with smooth distance/expmap/logmap.
    """

    def __init__(self, n_e, e_dim=E_DIM, beta=0.25, kmeans_init=True,
                 kmeans_iters=10, sk_eps=0.0, sk_iters=3, layer_idx=0,
                 signed_kappa=False, kappa_max=KAPPA_MAX):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        self.layer_idx = layer_idx
        self.signed_kappa = signed_kappa
        self.kappa_max = kappa_max
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        # B 路径: 初始化 theta 使 kappa_init = -1.25 (与 A 初始 c=1.25 对应, 严格等价)
        if signed_kappa:
            with torch.no_grad():
                self.theta.data.fill_(math.atanh(-C_INITIAL / kappa_max))
        self.embeddings = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            self.initted = True
            with torch.no_grad():
                self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            self.initted = False
            self.embeddings.weight.data.zero_()

    def get_c_or_kappa(self):
        """Return (c_or_kappa, kind) where kind ∈ {'c_pos', 'kappa_signed'}."""
        if self.signed_kappa:
            return self.kappa_max * torch.tanh(self.theta), 'kappa_signed'
        return C_MIN + (C_MAX - C_MIN) * torch.sigmoid(self.theta), 'c_pos'

    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embeddings.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)
        if not self.initted and self.training:
            self.init_emb(latent)
        B = latent.shape[0]
        K = self.n_e
        curv, kind = self.get_c_or_kappa()
        if kind == 'c_pos':
            # A 路径: 标准 Poincaré 球 (c > 0)
            c = curv
            latent_h = proj_to_ball(expmap0(latent, c), c)
            codebook_h = proj_to_ball(expmap0(self.embeddings.weight, c), c)
            x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
            cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)
            d = poincare_distance(x_exp, cb_exp, c).squeeze(-1)  # (B, K)
        else:
            # B 路径: signed κ-stereographic
            kappa = curv
            latent_h = signed_kappa_proj(signed_kappa_expmap0(latent, kappa), kappa)
            codebook_h = signed_kappa_proj(signed_kappa_expmap0(self.embeddings.weight, kappa), kappa)
            x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
            cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)
            d = signed_kappa_distance(x_exp, cb_exp, kappa).squeeze(-1)  # (B, K)
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
        x_q = self.embeddings.weight.index_select(0, indices)
        if kind == 'c_pos':
            c_sel = c
            x_q_proj = proj_to_ball(x_q, c_sel)
            latent_proj = proj_to_ball(latent, c_sel)
            commitment_loss = torch.mean(poincare_distance(x_q_proj.detach(), latent_proj, c_sel) ** 2)
            codebook_loss = torch.mean(poincare_distance(x_q_proj, latent_proj.detach(), c_sel) ** 2)
            x_q_safe = x_q_proj
            latent_safe = latent_proj
            x_q = logmap0(x_q_safe, c_sel)
            latent = logmap0(latent_safe, c_sel)
        else:
            kappa_sel = kappa
            x_q_proj = signed_kappa_proj(x_q, kappa_sel)
            latent_proj = signed_kappa_proj(latent, kappa_sel)
            commitment_loss = torch.mean(signed_kappa_distance(x_q_proj.detach(), latent_proj, kappa_sel) ** 2)
            codebook_loss = torch.mean(signed_kappa_distance(x_q_proj, latent_proj.detach(), kappa_sel) ** 2)
            x_q_safe = x_q_proj
            latent_safe = latent_proj
            x_q = signed_kappa_logmap0(x_q_safe, kappa_sel)
            latent = signed_kappa_logmap0(latent_safe, kappa_sel)
        loss = commitment_loss + self.beta * codebook_loss
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices


# ── 完整 HRQVAE ──
class SignedKappaHRQVAE(nn.Module):
    def __init__(self, in_dim=768, num_emb_list=(64, 128, 256), e_dim=E_DIM,
                 layers=(512, 256, 128, 64), beta=0.25, kmeans_init=True,
                 kmeans_iters=10, signed_kappa=False, kappa_max=KAPPA_MAX):
        super().__init__()
        self.in_dim = in_dim
        self.num_emb_list = list(num_emb_list)
        self.e_dim = e_dim
        self.layers = layers
        self.signed_kappa = signed_kappa
        self.kappa_max = kappa_max
        encode_layer_dims = [in_dim] + list(layers) + [e_dim]
        self.encoder = MLP(layers=encode_layer_dims, dropout=0.0, use_bn=False)
        decode_layer_dims = encode_layer_dims[::-1]
        self.decoder = MLP(layers=decode_layer_dims, dropout=0.0, use_bn=False)
        self.vq_layers = nn.ModuleList([
            SignedKappaVQ(n_e, e_dim=e_dim, beta=beta,
                          kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                          layer_idx=i, signed_kappa=signed_kappa, kappa_max=kappa_max)
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
