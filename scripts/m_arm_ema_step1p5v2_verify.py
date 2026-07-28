#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用户 2026-07-27 第 1.5 步 (修正版) 验证: batch EMA 中心化 + 尺度控制.

跑 1 epoch (LR_LOG_R=1.0), 报告 5 个数:
  1. z_c 归一化前的范数 ∈ [0.5, 5]
  2. 1000 样本两两余弦 std > 0.3
  3. 1000 样本两两余弦 MEAN < 0.3 (现在是 0.9952)
  4. 半径 std > 0.05 (三层)
  5. 无 NaN

用法:
  python3 m_arm_ema_step1p5v2_verify.py [GPU_ID]
"""
import os, sys, glob, subprocess
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
DATA = os.path.join(REPO, "HG-Rec/dataset/Instruments/item_emb.parquet")
OUTDIR = os.path.join(REPO, "products/m_arm/m_radius_spread_step1p5v2_ema")
LOG = os.path.join(OUTDIR, "train.log")
GPU_ID = sys.argv[1] if len(sys.argv) > 1 else "0"

os.makedirs(OUTDIR, exist_ok=True)

# 跟 Step 2 一致, 但 LR_LOG_R 降到 1.0 (用户 2026-07-27 第 1.5 步修正版).
COMMON = (
    f"--data_path {DATA} "
    f"--lr 1e-3 --batch_size 256 --epochs 1 "
    f"--num_emb_list 64 128 256 --e_dim 34 "
    f"--loss_type poincare --kmeans_init True --kmeans_iters 1000 "
    f"--sk_epsilons 0.0 0.0 0.0 --sk_iters 50 "
    f"--beta 0.5 "
    f"--product_manifold --angular_dim 2 --radial_dim 32 "
    f"--r_target_list 2.0,2.7,3.4 --norm_target 1.0 1.35 1.70 --gamma_norm 5.0 "
    f"--alpha 1.0 --beta_radial 1.0 "
    f"--use_normcap --normcap_target 0.3 --normcap_euc_only "
    f"--w_ent 0.1 --anti_collapse dead_revive "
    f"--r_spread_list 0.3 0.3 0.3 "
    f"--eval_step 5 --num_workers 0 "
    f"--device cuda:0 "
    f"--ckpt_dir {OUTDIR}"
)

print(f"[step1.5v2 verify] starting 1-epoch training on GPU {GPU_ID}")
print(f"[step1.5v2 verify] log → {LOG}")

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = GPU_ID
env["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_radius_spread_step1p5v2"
os.makedirs(env["TRITON_CACHE_DIR"], exist_ok=True)
env["PYTHONUNBUFFERED"] = "1"
env["LR_LOG_R"] = "1.0"  # 用户 2026-07-27 第 1.5 步修正版: 降到 1.0

with open(LOG, "w") as f:
    proc = subprocess.run(
        ["python3", "-u", os.path.join(REPO, "HG-Rec/train_hrqvae.py")] + COMMON.split(),
        stdout=f, stderr=subprocess.STDOUT, env=env
    )
print(f"[step1.5v2 verify] training done, exit={proc.returncode}")
if proc.returncode != 0:
    print(f"❌ FAIL: training exited with code {proc.returncode}")
    sys.exit(1)

# ---- 找 ckpt ----
ckpts = sorted(glob.glob(os.path.join(OUTDIR, "**", "*.pth"), recursive=True))
print(f"\n[step1.5v2 verify] found {len(ckpts)} ckpt files:")
for c in ckpts:
    print(f"  - {c}")

if not ckpts:
    print("❌ FAIL: no ckpt found.")
    sys.exit(1)
ckpt_path = ckpts[-1]
print(f"\n[step1.5v2 verify] using ckpt: {ckpt_path}")

# ---- 加载 latent ----
df = pd.read_parquet(DATA)
emb_full = np.stack(df['embedding'].values, axis=0)
x_full = torch.from_numpy(emb_full).float()
N = x_full.shape[0]

# ---- 加载 ckpt ----
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ck['state_dict'] if 'state_dict' in ck else ck

# ---- 重建 encoder ----
HYP_DIM = 2; EUC_DIM = 32
layers = [512, 256, 128, 64]
encode_layer_dims = [x_full.shape[1]] + layers + [HYP_DIM + EUC_DIM]

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
              for k, v in sd.items() if k.startswith("encoder.")}
encoder.load_state_dict(encoder_w)
encoder.eval()

with torch.no_grad():
    z_lat = encoder(x_full)
    # NormCap (normcap_euc_only) — 只压 euc 子空间
    euc_part = z_lat[:, HYP_DIM:]
    euc_norm = euc_part.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale_euc = torch.clamp(0.3 / euc_norm, max=1.0)
    z_lat = torch.cat([z_lat[:, :HYP_DIM], euc_part * scale_euc], dim=-1)

z_hyp_raw = z_lat[:, :HYP_DIM]  # (N, 2)

# ---- 应用 layer 0 的 EMA 中心化 (训练时模型对 L0 实际做的事) ----
# L0 是 residual 0, 它的 hyp_raw 就是 encoder 输出 hyp 部分.
hyp_mean_l0 = sd["hrq.vq_layers.0.hyp_mean"]  # (hyp_dim,)
hyp_scale_l0 = sd["hrq.vq_layers.0.hyp_scale"]  # (1,)
print(f"\n[step1.5v2 verify] layer 0 hyp_mean = {hyp_mean_l0.tolist()}")
print(f"[step1.5v2 verify] layer 0 hyp_scale = {hyp_scale_l0.item():.4f}")
z_c = (z_hyp_raw - hyp_mean_l0) / (hyp_scale_l0 + 1e-6)  # 用户公式

# ---- 测 5 个数 ----
R_TARGET_NORMS = [1.0, 1.35, 1.70]
R_SPREAD = 0.3

print("\n=== Step 1.5v2 用户验证 (5 个数) ===\n")

# ---------- 数 1: z_c 归一化前范数 ∈ [0.5, 5] ----------
z_c_norm = z_c.norm(dim=-1)
norm_min = z_c_norm.min().item()
norm_max = z_c_norm.max().item()
norm_mean = z_c_norm.mean().item()
norm_ok = 0.5 <= norm_min and norm_max <= 5.0
print(f"[数 1] z_c 范数: min={norm_min:.4f} max={norm_max:.4f} mean={norm_mean:.4f}")
print(f"        范围要求 ∈ [0.5, 5]  {'✅' if norm_ok else '❌'}")
norm_pass = norm_ok

# ---------- 数 2: 1000 样本两两余弦 std > 0.3 ----------
rng = np.random.default_rng(42)
sample_idx = rng.choice(N, size=1000, replace=False)
z_sample = z_c[sample_idx]
z_sample_n = F.normalize(z_sample, dim=-1, eps=1e-6)
gram = z_sample_n @ z_sample_n.t()
triu_idx = np.triu_indices(1000, k=1)
cosines = gram[triu_idx[0], triu_idx[1]].cpu().numpy()
cos_std = float(np.std(cosines))
cos_mean = float(np.mean(cosines))
cos_std_ok = cos_std > 0.3
print(f"\n[数 2] 1000 样本两两余弦: std={cos_std:.4f}  mean={cos_mean:.4f}")
print(f"        要求 std > 0.3  {'✅' if cos_std_ok else '❌ 方向仍坍缩'}")
cos_pass = cos_std_ok

# ---------- 数 3: 两两余弦 MEAN < 0.3 ----------
cos_mean_ok = cos_mean < 0.3
print(f"\n[数 3] 1000 样本两两余弦 mean={cos_mean:.4f}  要求 < 0.3  {'✅' if cos_mean_ok else '❌ 仍集中于 1 附近'}")
cos_mean_pass = cos_mean_ok

# ---------- 数 4: 半径 std > 0.05 (三层) ----------
print(f"\n[数 4] 半径 std (per-codeword log_r → r_per_k):")
all_radius_pass = True
for li in [0, 1, 2]:
    log_r_key = f"hrq.vq_layers.{li}.log_r"
    if log_r_key not in sd:
        print(f"        L{li}: ❌ log_r not in state_dict")
        all_radius_pass = False
        continue
    log_r = sd[log_r_key]
    r_target = R_TARGET_NORMS[li]
    r_per_k = r_target + R_SPREAD * (2 * torch.sigmoid(log_r) - 1)
    r_std = r_per_k.std().item()
    r_min = r_per_k.min().item()
    r_max = r_per_k.max().item()
    r_in_range = (r_min >= r_target - 0.3 - 1e-6) and (r_max <= r_target + 0.3 + 1e-6)
    std_ok = r_std > 0.05
    ok = r_in_range and std_ok
    if not ok:
        all_radius_pass = False
    print(f"        L{li}: r_std={r_std:.4f} r_min={r_min:.4f} r_max={r_max:.4f}  "
          f"target={r_target}  {'✅' if ok else '❌'}")
radius_pass = all_radius_pass

# ---------- 数 5: 无 NaN ----------
nan_pass = True
nan_loc = []
for li in [0, 1, 2]:
    log_r_key = f"hrq.vq_layers.{li}.log_r"
    if log_r_key in sd and torch.isnan(sd[log_r_key]).any():
        nan_pass = False
        nan_loc.append(f"L{li}.log_r")
for k in ["hrq.vq_layers.0.hyp_mean", "hrq.vq_layers.0.hyp_scale"]:
    if k in sd and torch.isnan(sd[k]).any():
        nan_pass = False
        nan_loc.append(k)
if torch.isnan(z_hyp_raw).any():
    nan_pass = False
    nan_loc.append("z_hyp_raw")
if torch.isnan(z_c).any():
    nan_pass = False
    nan_loc.append("z_c")
print(f"\n[数 5] 无 NaN:  {'✅' if nan_pass else f'❌ NaN 在 {nan_loc}'}")

# ---------- 综合判定 ----------
print(f"\n=== 综合 ===")
all_pass = norm_pass and cos_pass and cos_mean_pass and radius_pass and nan_pass
print(f"  数 1 (z_c 范数 ∈ [0.5, 5]):       {'✅' if norm_pass else '❌'}")
print(f"  数 2 (两两余弦 std > 0.3):        {'✅' if cos_pass else '❌'} (核心)")
print(f"  数 3 (两两余弦 mean < 0.3):       {'✅' if cos_mean_pass else '❌'} (核心)")
print(f"  数 4 (半径 std > 0.05):           {'✅' if radius_pass else '❌'}")
print(f"  数 5 (无 NaN):                    {'✅' if nan_pass else '❌'}")
print(f"\n  {'✅ PASS — Step 1.5v2 EMA 中心化生效' if all_pass else '❌ FAIL — 见上方诊断信息'}")