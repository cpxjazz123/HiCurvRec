#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用户 2026-07-27 第 1.5 步验证: LayerNorm 修双曲分支尺度.

跑 1 epoch, 报告 5 个数 (对应用户要求):
  1. z_hyp_raw 归一化前范数 ∈ [0.1, 10] (不能再是 1e6)
  2. 1000 样本两两余弦 std > 0.3
  3. 前 5 个样本方向肉眼不同
  4. 半径 std > 0.05 (Step 1 性质保留)
  5. 无 NaN

用法:
  python3 m_arm_layernorm_step1p5_verify.py [GPU_ID]
"""
import os, sys, glob, subprocess
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
DATA = os.path.join(REPO, "HG-Rec/dataset/Instruments/item_emb.parquet")
OUTDIR = os.path.join(REPO, "products/m_arm/m_radius_spread_step1p5_layernorm")
LOG = os.path.join(OUTDIR, "train.log")
GPU_ID = sys.argv[1] if len(sys.argv) > 1 else "0"

os.makedirs(OUTDIR, exist_ok=True)

# 跟 Step 2 完全一致的配置 (含 LayerNorm fix in utils.py).
# LR_LOG_R=10.0 维持 (用户 "只做这一步" 指示).
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

print(f"[step1.5 verify] starting 1-epoch training on GPU {GPU_ID}")
print(f"[step1.5 verify] log → {LOG}")
print(f"[step1.5 verify] args = {COMMON}\n")

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = GPU_ID
env["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_radius_spread_step1p5"
os.makedirs(env["TRITON_CACHE_DIR"], exist_ok=True)
env["PYTHONUNBUFFERED"] = "1"
env["LR_LOG_R"] = "10.0"  # 维持 Step 2 的值, 用户 "只做这一步"

with open(LOG, "w") as f:
    proc = subprocess.run(
        ["python3", "-u", os.path.join(REPO, "HG-Rec/train_hrqvae.py")] + COMMON.split(),
        stdout=f, stderr=subprocess.STDOUT, env=env
    )
print(f"[step1.5 verify] training done, exit={proc.returncode}, log={LOG}")
if proc.returncode != 0:
    print(f"❌ FAIL: training exited with code {proc.returncode}")
    sys.exit(1)

# ---- 找 epoch_0 ckpt ----
ckpts = sorted(glob.glob(os.path.join(OUTDIR, "**", "*.pth"), recursive=True))
print(f"\n[step1.5 verify] found {len(ckpts)} ckpt files:")
for c in ckpts:
    print(f"  - {c}")

if not ckpts:
    print("❌ FAIL: no ckpt found, training did not save.")
    sys.exit(1)

# 训练 1 epoch 的 ckpt 可能是 epoch_1_* 或最后落盘的那个
ckpt_path = ckpts[-1]
print(f"\n[step1.5 verify] using ckpt: {ckpt_path}")

# ---- 加载真实 latent ----
df = pd.read_parquet(DATA)
emb_full = np.stack(df['embedding'].values, axis=0)
x_full = torch.from_numpy(emb_full).float()
N = x_full.shape[0]
print(f"[step1.5 verify] latent N={N}, dim={x_full.shape[1]}")

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

# ---- 测 5 个数 ----
R_TARGET_NORMS = [1.0, 1.35, 1.70]
R_SPREAD = 0.3

print("\n=== Step 1.5 用户验证 (5 个数) ===\n")

# ---------- 数 1: z_hyp_raw 归一化前范数 ∈ [0.1, 10] ----------
z_hyp_raw_norm = z_hyp_raw.norm(dim=-1)
norm_min = z_hyp_raw_norm.min().item()
norm_max = z_hyp_raw_norm.max().item()
norm_mean = z_hyp_raw_norm.mean().item()
norm_ok = 0.1 <= norm_min and norm_max <= 10.0
print(f"[数 1] z_hyp_raw 归一化前范数: min={norm_min:.4f} max={norm_max:.4f} mean={norm_mean:.4f}")
print(f"        范围要求 ∈ [0.1, 10]  {'✅' if norm_ok else '❌ 仍是 1e6 量级 — LayerNorm 没接上'}")
norm_pass = norm_ok

# ---------- 数 2: 1000 样本两两余弦 std > 0.3 ----------
rng = np.random.default_rng(42)
sample_idx = rng.choice(N, size=1000, replace=False)
z_sample = z_hyp_raw[sample_idx]
z_sample_n = F.normalize(z_sample, dim=-1, eps=1e-6)
gram = z_sample_n @ z_sample_n.t()
# 取上三角 (去掉对角线)
triu_idx = np.triu_indices(1000, k=1)
cosines = gram[triu_idx[0], triu_idx[1]].cpu().numpy()
cos_std = float(np.std(cosines))
cos_mean = float(np.mean(cosines))
cos_ok = cos_std > 0.3
print(f"\n[数 2] 1000 样本两两余弦: std={cos_std:.4f}  mean={cos_mean:.4f}")
print(f"        要求 std > 0.3  {'✅' if cos_ok else '❌ 方向仍坍缩 (encoder 真学成一样)'}")
cos_pass = cos_ok

# ---------- 数 3: 前 5 个样本方向肉眼不同 ----------
print(f"\n[数 3] 前 5 个样本方向 (z_hyp_raw 归一化后):")
z_n_full = F.normalize(z_hyp_raw, dim=-1, eps=1e-6)
for i in range(5):
    print(f"        [{i}] {z_n_full[i].tolist()}")
# 检查前 5 个样本两两不同
seen = set()
all_diff = True
for i in range(5):
    key = tuple(z_n_full[i].round(decimals=2).tolist())
    if key in seen:
        all_diff = False
    seen.add(key)
print(f"        前 5 个样本两两不同 (按 2 位小数)  {'✅' if all_diff else '❌ 仍有样本方向相同'}")
diff_pass = all_diff

# ---------- 数 4: 半径 std > 0.05 (Step 1 性质保留) ----------
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
    if log_r_key in sd:
        if torch.isnan(sd[log_r_key]).any():
            nan_pass = False
            nan_loc.append(f"L{li}.log_r")
# 也查 encoder 输出
if torch.isnan(z_hyp_raw).any():
    nan_pass = False
    nan_loc.append("z_hyp_raw")
print(f"\n[数 5] 无 NaN:  {'✅' if nan_pass else f'❌ NaN 在 {nan_loc}'}")

# ---------- 综合判定 ----------
print(f"\n=== 综合 ===")
all_pass = norm_pass and cos_pass and diff_pass and radius_pass and nan_pass
print(f"  数 1 (范数 ∈ [0.1, 10]):          {'✅' if norm_pass else '❌'}")
print(f"  数 2 (两两余弦 std > 0.3):        {'✅' if cos_pass else '❌'}")
print(f"  数 3 (前 5 个样本方向不同):       {'✅' if diff_pass else '❌'}")
print(f"  数 4 (半径 std > 0.05):           {'✅' if radius_pass else '❌'}")
print(f"  数 5 (无 NaN):                    {'✅' if nan_pass else '❌'}")
print(f"\n  {'✅ PASS — Step 1.5 修复生效' if all_pass else '❌ FAIL — 见上方诊断信息'}")