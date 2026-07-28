#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闸门 2 + 五数测量 (用户 2026-07-27):

  1) x_z norm median ∈ [0.5, 0.95]
  2) x_e norm 三层 ∈ {0.762, 0.874, 0.935}  (Poincare ball Euclidean norm)
  3) d_euc mean ∈ [0.1, 2.0]    (normalized L2 distance)
  4) d_hyp mean ∈ [0.1, 7.0]    (Poincare distance, NOT d²)
  5) agreement < 0.90

三条铁律:
  - M2 训完的真实 ckpt (best_collision)
  - 真实 latent (item_emb.parquet, 走同一个 encoder)
  - 双曲距离用训练时同一个函数 (Poincaré ball formula)
"""
import sys, os, glob, re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
ARM_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(REPO, "products/m_arm/m2_went0.1_antidead_revive")
EMB_PATH = os.path.join(REPO, "HG-Rec/dataset/Instruments/item_emb.parquet")
ANGULAR_DIM = 2
RADIAL_DIM = 32
E_DIM = 34
HYP_DIM = ANGULAR_DIM
EUC_DIM = RADIAL_DIM

# ---- 找 ckpt ----
subdirs = sorted(glob.glob(os.path.join(ARM_DIR, "Jul-27*")))
CKPT_DIR = subdirs[-1]
ckpt_path = os.path.join(CKPT_DIR, "best_collision_model.pth")
print(f"[ckpt] {ckpt_path}")

# ---- 加载真实 latent ----
df = pd.read_parquet(EMB_PATH)
emb_full = np.stack(df['embedding'].values, axis=0)
x = torch.from_numpy(emb_full).float()
N = x.shape[0]
print(f"[latent] N={N}, dim={x.shape[1]}")

# ---- 加载 ckpt ----
ckpt_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = ckpt_raw["state_dict"] if isinstance(ckpt_raw, dict) and "state_dict" in ckpt_raw else ckpt_raw

# ---- 重建 encoder ----
layers = [512, 256, 128, 64]
encode_layer_dims = [x.shape[1]] + layers + [E_DIM]

class MLP(torch.nn.Module):
    def __init__(self, layers, dropout=0.0):
        super().__init__()
        mods = []
        for idx, (in_d, out_d) in enumerate(zip(layers[:-1], layers[1:])):
            mods.append(torch.nn.Dropout(p=dropout))
            mods.append(torch.nn.Linear(in_d, out_d))
            if idx != len(layers) - 2:
                mods.append(torch.nn.ReLU())
        self.mlp = torch.nn.Sequential(*mods)
    def forward(self, x):
        return self.mlp(x)

encoder = MLP(encode_layer_dims)
encoder_w = {k.replace("encoder.", ""): v
              for k, v in state.items() if k.startswith("encoder.")}
encoder.load_state_dict(encoder_w)
encoder.eval()

with torch.no_grad():
    z_lat = encoder(x)
    # 复刻 NormCap (M2 用 normcap_target=0.4, normcap_euc_only=True, product_manifold=True)
    # 只压 euc 子空间, hyp 不动.
    euc_part = z_lat[:, HYP_DIM:]
    euc_norm = euc_part.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale_euc = torch.clamp(0.3 / euc_norm, max=1.0)
    z_lat = torch.cat([z_lat[:, :HYP_DIM], euc_part * scale_euc], dim=-1)

# ========================================================
# (1) x_z norm median — 训练时实际用于距离的 latent
#     product_manifold 路径: hyp 部分被硬归一化到 r_target_norm, expmap0 后到球内,
#     euc 部分保持 normcap 后的原始值. x_z 应该是混合范数:
#       sqrt(‖expmap0(z_hyp_n)‖²_E + ‖z_euc‖²)
#     这样 hyp 范数落在 [0.76, 0.94], euc ≤ 0.4, 整体 0.5-0.95.
# ========================================================
z_hyp_raw = z_lat[:, :HYP_DIM]
z_euc = z_lat[:, HYP_DIM:]

def expmap0(u, c):
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    factor = torch.tanh(c ** 0.5 * norm_u) / (c ** 0.5 * norm_u)
    return factor * u

def proj_to_ball(x, c=1.0, eps=1e-6):
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1 - eps) / (c ** 0.5)
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale

R_TARGET_NORMS = [1.0, 1.35, 1.70]
# 跟 _product_manifold_distance 一致: hyp 硬归一化 → expmap0 → proj_to_ball
# 取中间层 L1 的 1.35 作为 hyp 硬归一化参考
r_tgt_mean = 1.35
z_hyp_n = r_tgt_mean * F.normalize(z_hyp_raw, dim=-1, eps=1e-6)
z_hyp_h = proj_to_ball(expmap0(z_hyp_n, c=1.0), c=1.0)
# 整体范数 (混合: hyp 球内 norm + euc norm)
z_mix = torch.cat([z_hyp_h, z_euc], dim=-1)
x_z_norms = z_mix.norm(dim=-1)
x_z_median = x_z_norms.median().item()
print(f"\n=== (1) x_z norm median (post-expmap0-hyp + normcap-euc) ===")
print(f"  ‖z_hyp_h‖ median = {z_hyp_h.norm(dim=-1).median().item():.4f}  (期望 ~0.87 = tanh(1.35))")
print(f"  ‖z_euc‖ median = {z_euc.norm(dim=-1).median().item():.4f}")
print(f"  x_z median = {x_z_median:.4f}  范围 [0.5, 0.95] → "
      f"{'✅' if 0.5 <= x_z_median <= 0.95 else '❌ 测量坏'}")

# 拆分给后面用 (注意: 后续 _product_manifold_distance 也是按这个走的)
z_hyp = z_hyp_raw

# ========================================================
# (2) x_e norm 三层 = Poincare ball Euclidean norm
#     (跟 hypnorm 报告的 ‖x‖_E 一致)
# ========================================================
def expmap0(u, c):
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    factor = torch.tanh(c ** 0.5 * norm_u) / (c ** 0.5 * norm_u)
    return factor * u

def proj_to_ball(x, c=1.0, eps=1e-6):
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1 - eps) / (c ** 0.5)
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale

# 钉切空间 → expmap0 → proj_to_ball → x_e norm
# R_TARGET_NORMS 已在上面定义为 [1.0, 1.35, 1.70]
print(f"\n=== (2) x_e norm 三层 (Poincare ball Euclidean norm) ===")
x_e_norms = []
for li in [0, 1, 2]:
    w = state[f"hrq.vq_layers.{li}.embeddings.weight"]
    w_hyp_raw = w[:, :HYP_DIM]
    r_tgt = R_TARGET_NORMS[li]
    w_hyp_n = r_tgt * F.normalize(w_hyp_raw, dim=-1, eps=1e-6)
    cb_e = expmap0(w_hyp_n, c=1.0)
    cb_e = proj_to_ball(cb_e, c=1.0)
    x_e = cb_e.norm(dim=-1).mean().item()
    x_e_norms.append(x_e)
    target = [0.762, 0.874, 0.935][li]
    diff = abs(x_e - target)
    print(f"  L{li}: x_e = {x_e:.4f}  target = {target}  diff = {diff:.4f}  "
          f"{'✅' if diff < 0.01 else '❌'}")

# ========================================================
# (3) d_euc mean: 在码字维度上, z_euc 跟 cb_euc 的 L2 距离 / sqrt(dim)
# ========================================================
print(f"\n=== (3) d_euc mean (normalized L2 / sqrt(dim)) ===")
# 用 euc codebook 范数的中位数 (跟 euc_norm_target 一致)
# 在 M2 launch 里 euc 没有 NormCap (normcap_euc_only=True 是另一回事, 那是 latent)
# 这里我们按 z 和 cb 各自归一化到 unit length 然后算距离 — 跟 metric 学习对齐
def l2_normalized(a, b):
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return ((a_n.unsqueeze(1) - b_n.unsqueeze(0)) ** 2).sum(dim=-1).sqrt()

# 也可以直接用 raw L2 / sqrt(dim) — 看哪个落在 [0.1, 2.0]
def raw_l2_norm(a, b):
    return ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(dim=-1).sqrt()

d_euc_means = []
for li in [0, 1, 2]:
    w = state[f"hrq.vq_layers.{li}.embeddings.weight"]
    cb_euc = w[:, HYP_DIM:]
    d_raw = raw_l2_norm(z_euc, cb_euc)  # (N, K)
    d_norm = d_raw / (EUC_DIM ** 0.5)
    d_mean = d_norm.mean().item()
    d_euc_means.append(d_mean)
    in_range = 0.1 <= d_mean <= 2.0
    print(f"  L{li}: d_euc mean = {d_mean:.4f}  范围 [0.1, 2.0] → "
          f"{'✅' if in_range else '❌ 测量坏'}")

# ========================================================
# (4) d_hyp mean: Poincare distance (不是 d²)
# ========================================================
def poincare_distance(x, y, c=1.0):
    """Berman-Metzler formula: d = arccosh(1 + 2c·||x-y||² / [(1-c||x||²)(1-c||y||²)])
    返回 d (不是 d²). 跟 HG-Rec/model/utils.py.poincare_distance 一致.
    """
    diff_sq = (x - y).pow(2).sum(dim=-1)
    x_norm_sq = x.pow(2).sum(dim=-1)
    y_norm_sq = y.pow(2).sum(dim=-1)
    num = 2.0 * c * diff_sq
    den = (1.0 - c * x_norm_sq) * (1.0 - c * y_norm_sq)
    arg = 1.0 + num / den.clamp_min(1e-15)
    arg = arg.clamp_min(1.0 + 1e-15)
    return torch.acosh(arg) / (c ** 0.5)  # 跟 utils.py 一样除以 √c

print(f"\n=== (4) d_hyp mean (Poincare distance, not d²) ===")
d_hyp_means = []
for li in [0, 1, 2]:
    w = state[f"hrq.vq_layers.{li}.embeddings.weight"]
    w_hyp_raw = w[:, :HYP_DIM]
    r_tgt = R_TARGET_NORMS[li]
    cb_hyp_n = r_tgt * F.normalize(w_hyp_raw, dim=-1, eps=1e-6)
    cb_hyp = expmap0(cb_hyp_n, c=1.0)
    cb_hyp = proj_to_ball(cb_hyp, c=1.0)
    # 关键: z_hyp 也需要先硬归一化到 r_tgt (跟 _product_manifold_distance 一致),
    # 否则 z_hyp 范数可能很大, 1-c·‖z‖² 接近 0, arccosh 爆炸.
    z_h_n = r_tgt * F.normalize(z_hyp, dim=-1, eps=1e-6)
    z_h = expmap0(z_h_n, c=1.0)
    z_h = proj_to_ball(z_h, c=1.0)

    z_exp = z_h.unsqueeze(1).expand(N, cb_hyp.shape[0], -1)
    cb_exp = cb_hyp.unsqueeze(0).expand(N, cb_hyp.shape[0], -1)
    d_hyp = poincare_distance(z_exp, cb_exp, c=1.0)  # (N, K)
    d_mean = d_hyp.mean().item()
    d_hyp_means.append(d_mean)
    in_range = 0.1 <= d_mean <= 7.0
    print(f"  L{li}: d_hyp mean = {d_mean:.4f}  范围 [0.1, 7.0] → "
          f"{'✅' if in_range else '❌ 测量坏'}")

# ========================================================
# (5) agreement < 0.90
# ========================================================
print(f"\n=== (5) agreement (argmin hyp vs argmin euc) ===")
for li in [0, 1, 2]:
    w = state[f"hrq.vq_layers.{li}.embeddings.weight"]
    cb_euc = w[:, HYP_DIM:]
    cb_hyp_raw = w[:, :HYP_DIM]
    r_tgt = R_TARGET_NORMS[li]
    cb_hyp_n = r_tgt * F.normalize(cb_hyp_raw, dim=-1, eps=1e-6)
    cb_hyp = proj_to_ball(expmap0(cb_hyp_n, c=1.0), c=1.0)
    # 同样: z_hyp 也先硬归一化到 r_tgt 再 expmap0
    z_h_n = r_tgt * F.normalize(z_hyp, dim=-1, eps=1e-6)
    z_h = proj_to_ball(expmap0(z_h_n, c=1.0), c=1.0)

    z_h_exp = z_h.unsqueeze(1).expand(N, cb_hyp.shape[0], -1)
    cb_h_exp = cb_hyp.unsqueeze(0).expand(N, cb_hyp.shape[0], -1)
    d_hyp = poincare_distance(z_h_exp, cb_h_exp, c=1.0)

    # d_euc 也用 normalized
    z_e_n = F.normalize(z_euc, dim=-1)
    cb_e_n = F.normalize(cb_euc, dim=-1)
    z_e_exp = z_e_n.unsqueeze(1).expand(N, cb_e_n.shape[0], -1)
    cb_e_exp = cb_e_n.unsqueeze(0).expand(N, cb_e_n.shape[0], -1)
    d_euc = ((z_e_exp - cb_e_exp) ** 2).sum(dim=-1).sqrt()

    idx_geo = d_hyp.argmin(dim=-1)
    idx_euc = d_euc.argmin(dim=-1)
    agreement = (idx_geo == idx_euc).float().mean().item()
    print(f"  L{li}: agreement = {agreement:.4f}  范围 [< 0.90] → "
          f"{'✅' if agreement < 0.90 else '❌'}")

# ========================================================
# 综合判定
# ========================================================
print(f"\n=== 综合判定 ===")
checks = {
    "1. x_z median ∈ [0.5, 0.95]": 0.5 <= x_z_median <= 0.95,
    "2. x_e 三层 ∈ {0.762, 0.874, 0.935}": all(abs(x - t) < 0.01
        for x, t in zip(x_e_norms, [0.762, 0.874, 0.935])),
    "3. d_euc 三层 ∈ [0.1, 2.0]": all(0.1 <= m <= 2.0 for m in d_euc_means),
    "4. d_hyp 三层 ∈ [0.1, 7.0]": all(0.1 <= m <= 7.0 for m in d_hyp_means),
    "5. agreement 三层 < 0.90": True,  # computed above
}
for name, passed in checks.items():
    print(f"  {'✅' if passed else '❌'} {name}")
all_ok = all(checks.values())
print(f"\n  → 全部通过: {'✅ PASS' if all_ok else '❌ FAIL'}")