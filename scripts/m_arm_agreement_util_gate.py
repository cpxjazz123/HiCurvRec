#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用户 2026-07-27 第三步: 两个核心闸门.

约束 (必须严格):
  - 两边: 同一批点, 同一个 2 维子空间, 只换度量.
  - x_z = proj_to_ball(expmap0(z_hyp2))    # latent 的双曲部分 (B, 2)
  - x_e = proj_to_ball(expmap0(cb_hyp2))   # 码字的双曲部分 (K, 2), 半径已分化
  - d_hyp = poincare_distance(x_z[:, None, :], x_e[None, :, :])   # (B, K)
  - d_euc = torch.cdist(x_z, x_e)                                  # (B, K)
  - agreement = (d_hyp.argmin(-1) == d_euc.argmin(-1)).float().mean()
  - util = d_hyp.argmin(-1).unique().numel() / K

合理性检查 (先贴这三个, 防止又测错):
  - x_e 的范数 min/max: L0≈[0.60, 0.86] / L1≈[0.78, 0.93] / L2≈[0.89, 0.96]
  - d_euc 均值: 0.05 – 2.0
  - d_hyp 均值: 0.1 – 8.0

目标 (两个必须同时达到):
  - agreement < 0.90 (三层)
  - util ≥ 0.80 (三层)
"""
import sys, os, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
CKPT_DIR = sys.argv[1] if len(sys.argv) > 1 else None
if CKPT_DIR is None:
    # 默认找 best_collision
    candidates = sorted(glob.glob(os.path.join(
        REPO, "products/m_arm/m_radius_spread_step2_50ep", "Jul-27-*"
    )))
    if not candidates:
        print(f"❌ FAIL: no ckpt dir found under products/m_arm/m_radius_spread_step2_50ep/")
        sys.exit(1)
    CKPT_DIR = candidates[-1]
print(f"[ckpt_dir] {CKPT_DIR}")

# ---- 找 best_collision ckpt ----
# 注意: 目录名含 [64,128,256], glob 不支持字面方括号; 用 os.listdir 过滤.
best_collision = [f for f in os.listdir(CKPT_DIR) if f.startswith("best_collision") and f.endswith(".pth")]
if not best_collision:
    print(f"❌ FAIL: no best_collision_model.pth in {CKPT_DIR}")
    sys.exit(1)
ckpt_path = os.path.join(CKPT_DIR, best_collision[0])
print(f"[ckpt] {ckpt_path}")

# ---- 加载真实 latent ----
EMB_PATH = os.path.join(REPO, "HG-Rec/dataset/Instruments/item_emb.parquet")
df = pd.read_parquet(EMB_PATH)
emb_full = np.stack(df['embedding'].values, axis=0)
x_full = torch.from_numpy(emb_full).float()
N = x_full.shape[0]
print(f"[latent] N={N}, dim={x_full.shape[1]}")

# ---- 加载 ckpt ----
ckpt_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = ckpt_raw["state_dict"] if isinstance(ckpt_raw, dict) and "state_dict" in ckpt_raw else ckpt_raw

# ---- 重建 encoder ----
HYP_DIM = 2
EUC_DIM = 32
E_DIM = HYP_DIM + EUC_DIM
layers = [512, 256, 128, 64]
encode_layer_dims = [x_full.shape[1]] + layers + [E_DIM]

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
    z_lat = encoder(x_full)
    # 复刻 NormCap (normcap_euc_only=True, normcap_target=0.3, product_manifold=True)
    # 只压 euc 子空间, hyp 不动.
    euc_part = z_lat[:, HYP_DIM:]
    euc_norm = euc_part.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale_euc = torch.clamp(0.3 / euc_norm, max=1.0)
    z_lat = torch.cat([z_lat[:, :HYP_DIM], euc_part * scale_euc], dim=-1)

z_hyp_raw = z_lat[:, :HYP_DIM]  # (N, 2)

# ========================================================
# 双曲工具函数
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

def poincare_distance(x, y, c=1.0):
    """复刻 utils.poincare_distance (mobius add + artanh 公式)."""
    sqrt_c = c ** 0.5
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    neg_x = -x
    nxy = -(xy)
    num = (1 + 2 * c * nxy + c * y2) * neg_x + (1 - c * x2) * y
    den = 1 + 2 * c * nxy + (c ** 2) * x2 * y2
    den = den.clamp_min(1e-15)
    diff = num / den
    norm = diff.norm(dim=-1).clamp_min(1e-15)
    norm = norm.clamp(max=(1.0 - 1e-5) / sqrt_c)
    x_in = sqrt_c * norm
    x_in = x_in.clamp(max=1.0 - 1e-7)
    artanh = 0.5 * torch.log((1.0 + x_in) / (1.0 - x_in))
    d = (2.0 / sqrt_c) * artanh
    return d

# ---- 码字: 应用 per-codeword radius ----
R_TARGET_NORMS = [1.0, 1.35, 1.70]
R_SPREAD = 0.3

print("\n=== 合理性检查 (sanity check) ===\n")

all_pass_sanity = True
all_pass_targets = True
results = []

for li in [0, 1, 2]:
    print(f"--- L{li} ---")
    # 加载 vq layer 权重 + log_r
    w = state[f"hrq.vq_layers.{li}.embeddings.weight"]
    log_r = state[f"hrq.vq_layers.{li}.log_r"]
    w_hyp_raw = w[:, :HYP_DIM]
    K = w.shape[0]
    r_target = R_TARGET_NORMS[li]
    # per-codeword radii
    r_per_k = r_target + R_SPREAD * (2 * torch.sigmoid(log_r) - 1)  # (K,)

    # x_z: latent 侧硬归一化到 r_target (跟训练时 _product_manifold_distance 一致)
    z_h_n = r_target * F.normalize(z_hyp_raw, dim=-1, eps=1e-6)
    x_z = proj_to_ball(expmap0(z_h_n, c=1.0), c=1.0)  # (N, 2)

    # x_e: 码字侧用 per-codeword 半径
    w_h_n = r_per_k.unsqueeze(-1) * F.normalize(w_hyp_raw, dim=-1, eps=1e-6)
    x_e = proj_to_ball(expmap0(w_h_n, c=1.0), c=1.0)  # (K, 2)

    # ---- 合理性检查 ----
    x_e_norms = x_e.norm(dim=-1)
    x_e_min, x_e_max = x_e_norms.min().item(), x_e_norms.max().item()

    # d_hyp
    z_exp = x_z.unsqueeze(1).expand(N, K, -1)
    e_exp = x_e.unsqueeze(0).expand(N, K, -1)
    d_hyp = poincare_distance(z_exp, e_exp, c=1.0)  # (N, K)
    d_hyp_mean = d_hyp.mean().item()

    # d_euc
    d_euc = torch.cdist(x_z, x_e)  # (N, K)
    d_euc_mean = d_euc.mean().item()

    # 合理性预期范围
    expected_range = [[0.60, 0.86], [0.78, 0.93], [0.89, 0.96]][li]
    x_e_ok = expected_range[0] - 0.02 <= x_e_min and x_e_max <= expected_range[1] + 0.02
    d_euc_ok = 0.05 <= d_euc_mean <= 2.0
    d_hyp_ok = 0.1 <= d_hyp_mean <= 8.0
    sanity_pass = x_e_ok and d_euc_ok and d_hyp_ok
    print(f"  x_e norm: min={x_e_min:.4f} max={x_e_max:.4f}  expected≈[{expected_range[0]},{expected_range[1]}]  {'✅' if x_e_ok else '❌'}")
    print(f"  d_euc mean: {d_euc_mean:.4f}  range [0.05, 2.0]  {'✅' if d_euc_ok else '❌'}")
    print(f"  d_hyp mean: {d_hyp_mean:.4f}  range [0.1, 8.0]  {'✅' if d_hyp_ok else '❌'}")

    if not sanity_pass:
        all_pass_sanity = False
        print(f"  ⚠️ 合理性检查失败 — 测量可能错, 不算数")
        continue

    # ---- 核心指标 ----
    idx_hyp = d_hyp.argmin(dim=-1)
    idx_euc = d_euc.argmin(dim=-1)
    agreement = (idx_hyp == idx_euc).float().mean().item()
    unique_hyp = int(idx_hyp.unique().numel())
    util = unique_hyp / K
    util_pct = 100 * util

    print(f"  agreement = {agreement:.4f}  target < 0.90  {'✅' if agreement < 0.90 else '❌'}")
    print(f"  util = {unique_hyp}/{K} = {util_pct:.1f}%  target ≥ 80%  {'✅' if util_pct >= 80.0 else '❌'}")

    results.append({"li": li, "agreement": agreement, "util": util_pct,
                    "x_e_min": x_e_min, "x_e_max": x_e_max,
                    "d_euc_mean": d_euc_mean, "d_hyp_mean": d_hyp_mean})

# ---- 综合判定 ----
print("\n=== 综合判定 ===")
print(f"  合理性检查: {'✅' if all_pass_sanity else '❌'}")

if not all_pass_sanity:
    print("  ❌ FAIL: 合理性不通过, 测量可能错, 不进入目标判定")
    sys.exit(1)

# 检查两个核心目标
agreement_pass = all(r["agreement"] < 0.90 for r in results)
util_pass = all(r["util"] >= 80.0 for r in results)
print(f"\n  agreement < 0.90 (三层): {'✅' if agreement_pass else '❌'}")
for r in results:
    print(f"    L{r['li']}: agreement={r['agreement']:.4f}")
print(f"\n  util ≥ 80% (三层): {'✅' if util_pass else '❌'}")
for r in results:
    print(f"    L{r['li']}: util={r['util']:.1f}%")

if agreement_pass and util_pass:
    print("\n  ✅✅✅ 进第四步 — 十七次以来第一次'几何参与 + 码本健康'共存")
elif agreement_pass and not util_pass:
    # 检查 L2 是否卡在 73.8%
    l2_util = next(r["util"] for r in results if r["li"] == 2)
    if 70 <= l2_util < 80:
        print(f"\n  🟡 agreement 达标, L2 util {l2_util:.1f}% 差一点 → 加大熵正则或多跑 20 epoch 只补 L2")
    else:
        print(f"\n  🟡 agreement 达标, util 部分不达标 (L2={l2_util:.1f}%) → 加大熵正则或多跑 epoch")
elif not agreement_pass:
    max_agreement = max(r["agreement"] for r in results)
    if max_agreement > 0.95:
        print(f"\n  ❌ agreement > 0.95 ({max_agreement:.4f}) → 半径分化也压不出几何 → 收线")
    else:
        print(f"\n  ❌ agreement 不达标 (max={max_agreement:.4f}), util 状态: {'达标' if util_pass else '不达标'}")