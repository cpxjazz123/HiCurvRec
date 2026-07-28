#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""① 纯 hyp 球内坐标 agreement 检查.

双方都用 proj_to_ball(expmap0(r * F.normalize(...))) 把 2D hyp 投到球内坐标.
然后 d_hyp = poincare_distance, d_euc = torch.cdist (在 2 维球内坐标上算 L2).
预期: agreement = 100% (因为半径钉死, 双方缩放相同, 单调变换).

如果不是 100% → 实现里还有别的东西在动.
"""
import sys, os, glob
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
    # normcap_euc_only: 只压 euc
    euc_part = z_lat[:, HYP_DIM:]
    euc_norm = euc_part.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale_euc = torch.clamp(0.3 / euc_norm, max=1.0)
    z_lat = torch.cat([z_lat[:, :HYP_DIM], euc_part * scale_euc], dim=-1)

z_hyp_raw = z_lat[:, :HYP_DIM]

# ========================================================
# 双方用 2 维球内坐标
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
    """复刻 utils.poincare_distance (mobius add + artanh 公式):
      d = (2/√c) · artanh(√c · ‖(-x) ⊕ y‖)
    用户 2026-07-27 闸门 2 复现要求: 用训练时同一个公式 (utils 版), 不是 Berman-Metzler 简化版.
    """
    sqrt_c = c ** 0.5
    # mobius_add(-x, y, c)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    neg_x = -x
    nx2 = x2  # (-x)^2 = x^2
    nxy = -(xy)  # (-x) · y = -(x · y)
    num = (1 + 2 * c * nxy + c * y2) * neg_x + (1 - c * nx2) * y
    den = 1 + 2 * c * nxy + (c ** 2) * nx2 * y2
    den = den.clamp_min(1e-15)
    diff = num / den
    norm = diff.norm(dim=-1).clamp_min(1e-15)
    # Phase 0.4 (Task #178) 数值稳定: clamp 防止 sqrt_c * norm > 1
    norm = norm.clamp(max=(1.0 - 1e-5) / sqrt_c)
    # artanh(x) = 0.5 * log((1+x)/(1-x))
    x_in = sqrt_c * norm
    x_in = x_in.clamp(max=1.0 - 1e-7)
    artanh = 0.5 * torch.log((1.0 + x_in) / (1.0 - x_in))
    d = (2.0 / sqrt_c) * artanh
    return d

R_TARGET_NORMS = [1.0, 1.35, 1.70]

print("\n=== ① 纯 hyp 球内坐标 agreement (预期 100%) ===\n")
all_pass = True
for li in [0, 1, 2]:
    w = state[f"hrq.vq_layers.{li}.embeddings.weight"]
    w_hyp_raw = w[:, :HYP_DIM]
    r = R_TARGET_NORMS[li]

    # 双方: 钉半径 + expmap0 + proj_to_ball
    x_z2 = proj_to_ball(expmap0(r * F.normalize(z_hyp_raw, dim=-1, eps=1e-6), c=1.0), c=1.0)
    x_e2 = proj_to_ball(expmap0(r * F.normalize(w_hyp_raw, dim=-1, eps=1e-6), c=1.0), c=1.0)

    # d_hyp: poincare distance (N, K)
    x_z_exp = x_z2.unsqueeze(1).expand(N, x_e2.shape[0], -1)
    x_e_exp = x_e2.unsqueeze(0).expand(N, x_e2.shape[0], -1)
    d_hyp = poincare_distance(x_z_exp, x_e_exp, c=1.0)

    # d_euc: torch.cdist 在 2 维球内坐标上 (Euclidean L2)
    d_euc = torch.cdist(x_z2, x_e2)  # (N, K)

    idx_hyp = d_hyp.argmin(dim=-1)
    idx_euc = d_euc.argmin(dim=-1)
    agreement = (idx_hyp == idx_euc).float().mean().item()

    print(f"  L{li}: r={r}, 球内 ‖x_z2‖={x_z2.norm(dim=-1).mean().item():.4f}, "
          f"‖x_e2‖={x_e2.norm(dim=-1).mean().item():.4f}, "
          f"agreement = {agreement:.6f}  "
          f"{'✅ 100%' if agreement == 1.0 else '❌ NOT 100%'}")

    if agreement != 1.0:
        all_pass = False
        # 诊断: 不一致的有多少
        disagree = (idx_hyp != idx_euc).sum().item()
        print(f"        disagree_count = {disagree}/{N} = {disagree/N:.6f}")
        # 看 d_hyp / d_euc 是否单调相关 (理论上应该是单调变换)
        # 取一个 disagree 样本看 d 分布
        if disagree > 0:
            sample_idx = (idx_hyp != idx_euc).nonzero()[0].item()
            d_h = d_hyp[sample_idx]
            d_e = d_euc[sample_idx]
            print(f"        sample {sample_idx}: hyp_argmin={idx_hyp[sample_idx].item()} "
                  f"d_h={d_h[idx_hyp[sample_idx]].item():.6f}, "
                  f"euc_argmin={idx_euc[sample_idx].item()} d_e={d_e[idx_euc[sample_idx]].item():.6f}")
            # 看 hyp argmin 的 d_euc 在该样本里排第几 (理论应该也是 min)
            d_e_rank = (d_e < d_e[idx_euc[sample_idx]]).sum().item()
            d_h_rank = (d_h < d_h[idx_hyp[sample_idx]]).sum().item()
            print(f"        d_euc at hyp_argmin: {d_e[idx_hyp[sample_idx]].item():.6f} (rank {d_e_rank})")
            print(f"        d_hyp at euc_argmin: {d_h[idx_euc[sample_idx]].item():.6f} (rank {d_h_rank})")
            # 看 (hyp_argmin, euc_argmin) 这两点的真实 d
            print(f"        d_hyp matrix[hyp_argmin, euc_argmin]: {d_hyp[sample_idx, idx_euc[sample_idx]].item():.6f}")
            print(f"        d_euc matrix[euc_argmin, hyp_argmin]: {d_euc[sample_idx, idx_hyp[sample_idx]].item():.6f}")
            # 检查: 双曲距离跟欧氏距离在所有 K 上是否单调?
            # spearman corr
            from scipy.stats import spearmanr
            rho, _ = spearmanr(d_h.cpu().numpy(), d_e.cpu().numpy())
            print(f"        spearman(d_h, d_e) = {rho:.6f} (1.0 = 完全单调)")

print(f"\n=== 综合 ===")
print(f"  {'✅ PASS: 所有层 100% agreement (半径钉死后单调变换)' if all_pass else '❌ FAIL: 至少一层 < 100%, 实现里还有别的东西在动'}")