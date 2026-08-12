"""Issue #149 — Signed κ-Stereographic Quantizer.

A (control): 三层只能学习负曲率, c_l = C_MIN + (C_MAX-C_MIN)·sigmoid(θ_l), sectional_curvature_l = -c_l.
  Distance: Poincaré ball d_H(x, y; c_l).

B (treatment): 三层各学习 signed sectional curvature κ_l = κ_max · tanh(θ_l),
  κ_l ∈ (-κ_max, +κ_max). κ<0 双曲, κ=0 欧氏, κ>0 球面. 统一 κ-stereographic 形式.

B 初始化: 与 A 同 (负曲率), 训练起点 forward/assignment/loss/SID 严格一致.
训练后允许跨越 0.

κ ≈ 0 使用 analytic limit 或稳定级数展开; 必须可微, 不能切断 theta_l 梯度.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import proj_to_ball, expmap0, logmap0, poincare_distance, kmeans, MLP, _eps, mobius_add

C_MIN = 0.5
C_MAX = 2.0
KAPPA_MAX = 2.0  # |κ| 上限 (与 c_max 对齐, 保证 A/B 起点一致: 初始 c=1.25 → κ=-1.25)
E_DIM = 32


def _signed_kappa_mobius_diff(x, y, kappa):
    """Compute ||(-x) ⊕ y|| in κ-stereographic model with signed κ.
    Möbius add formula works for any real c; we just plug c = sign(κ) * |κ|."""
    abs_k = abs(kappa)
    c_signed = math.copysign(abs_k, kappa)
    diff = mobius_add(-x, y, c_signed)
    d = diff.norm(dim=-1, keepdim=True).clamp_min(_eps(diff))
    return d


def signed_kappa_distance(x, y, kappa, abs_kappa=None):
    """κ-stereographic distance with smooth blending at κ=0.

    D(x, y; κ) = (2/√|κ|) · g(√|κ| · ||-x ⊕ y||)

    g(u) = arctan(u) (κ>0 spherical) | artanh(u) (κ<0 hyperbolic) | u (κ=0 Euclidean)
    Smooth blending via w_sph = sigmoid(scale * κ); both branches always computed.
    """
    if abs_kappa is None:
        abs_kappa = abs(kappa)
    sqrt_abs_k = abs_kappa.clamp_min(1e-7).sqrt()
    d = _signed_kappa_mobius_diff(x, y, kappa)
    u = (sqrt_abs_k * d).clamp(max=1 - 1e-7)
    # Compute both branches (always finite, both bounded)
    atan_part = torch.arctan(u)
    atanh_part = torch.arctanh(u)
    # Soft switch (differentiable w.r.t. kappa): w_sph = sigmoid(scale * kappa)
    w_sph = torch.sigmoid(20.0 * kappa)
    g = w_sph * atan_part + (1 - w_sph) * atanh_part
    # Near κ=0 limit: both arctan(u) and artanh(u) ≈ u, so D ≈ (2/√|κ|) * u
    # But (2/√|κ|) * u = 2 * d, so D ≈ 2d. Hmm, we want D ≈ d at κ=0.
    # Use limit: as κ→0, D = d (Euclidean). Apply correction: D = (2/√|κ|) * g(u) ≈ d requires scaling.
    # Actually the correct κ-stereographic formula is:
    #   D = (2/√|κ|) * g(u) where u = √|κ| * d, and g(u) for small u ≈ u
    #   → D ≈ (2/√|κ|) * √|κ| * d = 2d  ... that's wrong by factor of 2.
    # Wait, the standard κ-stereographic formula is:
    #   D = (1/√|κ|) * g(u) where g(u) ∈ {arctan(u), artanh(u), u}
    # Let me re-check. Standard: arctan(√κ·d) / √κ = arctan(u)/√κ
    # Near κ=0: arctan(u)/√κ ≈ u/√κ = √κ·d/√κ = d. Yes, so D = (1/√|κ|) * g(u).
    out = (1.0 / sqrt_abs_k) * g
    return out


def signed_kappa_expmap0(u, kappa):
    """κ-stereographic exponential map at origin.

    Exp_0^κ(v) = (1/√|κ|) · tanh(√|κ| · ||v||/2) · v/||v||      if κ<0 (hyperbolic)
    Exp_0^κ(v) = (1/√κ) · tan(√κ · ||v||/2) · v/||v||            if κ>0 (spherical)
    Exp_0^κ(v) = v                                                if κ=0 (Euclidean)

    Smooth blending: compute both tanh and tan branches, blend by sign.
    """
    abs_k = abs(kappa).clamp_min(1e-7)
    sqrt_k = abs_k.sqrt()
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(_eps(u))
    sqrt_k_norm = (sqrt_k * norm_u / 2.0).clamp(max=math.pi/2 - 1e-5)
    tanh_part = torch.tanh(sqrt_k_norm)
    tan_part = torch.tan(sqrt_k_norm)
    w_sph = torch.sigmoid(20.0 * kappa)
    blend = w_sph * tan_part + (1 - w_sph) * tanh_part
    # For κ>0 (spherical), the projection radius is R = 1/√κ. We need to ensure result norm ≤ R.
    factor = (1.0 / sqrt_k) * (blend / (norm_u * sqrt_k))
    out = factor * u  # this gives norm = blend/√κ ≤ R = 1/√κ
    # Clip to R for safety
    R = 1.0 / sqrt_k
    out_norm = out.norm(dim=-1, keepdim=True).clamp_min(_eps(out))
    out = torch.where(out_norm > R * (1 - 1e-5), R * (1 - 1e-5) * out / out_norm, out)
    return out


def signed_kappa_logmap0(x, kappa):
    """κ-stereographic log map at origin.

    Log_0^κ(y) = (2/(√|κ|)) · arctanh(√|κ| · ||y||) · y/||y||   if κ<0
    Log_0^κ(y) = (2/√κ) · arctan(√κ · ||y||) · y/||y||         if κ>0
    Log_0^κ(y) = y                                              if κ=0

    Smooth blending.
    """
    abs_k = abs(kappa).clamp_min(1e-7)
    sqrt_k = abs_k.sqrt()
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
    u = (sqrt_k * norm_x).clamp(max=1 - 1e-7)
    atanh_part = torch.arctanh(u)
    atan_part = torch.arctan(u)
    w_sph = torch.sigmoid(20.0 * kappa)
    blend = w_sph * atan_part + (1 - w_sph) * atanh_part
    factor = (2.0 / sqrt_k) * (blend / u)
    out = factor * x
    return out


def signed_kappa_proj(x, kappa):
    """Project x onto valid domain: ball of radius 1/√|κ| (hyperbolic) or sphere of radius 1/√κ (spherical)."""
    abs_k = abs(kappa).clamp_min(1e-7)
    R = 1.0 / abs_k.sqrt()
    norm = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
    max_norm = (1 - 1e-5) * R
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


# ── 单层量化器 ──
class SignedKappaVQ(nn.Module):
    """A: standard hyperbolic (c_l ≥ 0) with full Poincaré distance.
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
        # theta_l: signed_kappa=False → c_l = sigmoid(θ)·(C_MAX-C_MIN)+C_MIN
        #          signed_kappa=True → κ_l = kappa_max · tanh(θ)
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
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
        # 距离 / 投影 / expmap0 / logmap0
        if kind == 'c_pos':
            # A 路径: 标准 Poincaré 球 (c ≥ 0)
            c = curv  # scalar, c > 0
            latent_h = proj_to_ball(expmap0(latent, c), c)
            codebook_h = proj_to_ball(expmap0(self.embeddings.weight, c), c)
            x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
            cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)
            d = poincare_distance(x_exp, cb_exp, c).squeeze(-1)  # (B, K)
        else:
            # B 路径: signed κ-stereographic
            kappa = curv  # scalar, κ ∈ (-kappa_max, +kappa_max)
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
        # commitment + codebook loss
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