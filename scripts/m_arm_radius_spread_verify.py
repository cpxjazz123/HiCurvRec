#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用户 2026-07-27 第一步验证: per-codeword learnable radius spread.

跑 1 epoch, 打印每层 hyp 码字切空间范数的 min/max/std, 检查:
  1. epoch 0 (初始化时) min = max = r_target_norm (证明起点等于 M2 baseline)
  2. epoch 1 之后 std > 0.05 (半径确实在分化)
  3. 范围: 全部落在 [r_target - r_spread, r_target + r_spread] 内
  4. 数值: 无 NaN

用法:
  python3 m_arm_radius_spread_verify.py [GPU_ID]
"""
import os, sys, glob, subprocess
import numpy as np
import torch

REPO = "/home/wlia0047/wenyu/GeneRec"
REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
DATA = os.path.join(REPO, "HG-Rec/dataset/Instruments/item_emb.parquet")
OUTDIR = os.path.join(REPO, "products/m_arm/m_radius_spread_step1")
LOG = os.path.join(OUTDIR, "train.log")
GPU_ID = sys.argv[1] if len(sys.argv) > 1 else "0"

os.makedirs(OUTDIR, exist_ok=True)

# 跟 M2 完全一致的配置, 只加 --r_spread_list 0.3 0.3 0.3
# epochs=1 (用户 spec: 跑 1 epoch 看分化).
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

print(f"[verify] starting 1-epoch training on GPU {GPU_ID}")
print(f"[verify] log → {LOG}")
print(f"[verify] args = {COMMON}\n")

# 写一个包装脚本, 让 Python 输出写到 log
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = GPU_ID
env["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_radius_spread_step1"
os.makedirs(env["TRITON_CACHE_DIR"], exist_ok=True)
env["PYTHONUNBUFFERED"] = "1"

with open(LOG, "w") as f:
    proc = subprocess.run(
        ["python3", "-u", os.path.join(REPO, "HG-Rec/train_hrqvae.py")] + COMMON.split(),
        stdout=f, stderr=subprocess.STDOUT, env=env
    )
print(f"[verify] training done, exit={proc.returncode}, log={LOG}")

# 读 ckpt 测半径
import torch
ckpts = sorted(glob.glob(os.path.join(OUTDIR, "**", "*.pth"), recursive=True))
print(f"\n[verify] found {len(ckpts)} ckpt files:")
for c in ckpts:
    print(f"  - {c}")

if not ckpts:
    print("❌ FAIL: no ckpt found, training did not save.")
    sys.exit(1)

# 取 epoch 0 (初始化后第一个 ckpt) 和最终 ckpt
def get_layer_radii(state, n_e_list, hyp_dim=2):
    """从 state_dict 提取每层 hyp 切空间范数 (应用 r_spread 公式)."""
    out = {}
    for li, n_e in enumerate(n_e_list):
        # 找 vq_layers.{li}.log_r
        log_r_key = f"hrq.vq_layers.{li}.log_r"
        if log_r_key in state:
            log_r = state[log_r_key]
            # r_k = r_target_norm + r_spread * (2*sigmoid(log_r) - 1)
            r_target = [1.0, 1.35, 1.70][li]
            r_spread = 0.3
            r_per_k = r_target + r_spread * (2 * torch.sigmoid(log_r) - 1)
            out[li] = r_per_k.detach().cpu().numpy()
        else:
            print(f"  ❌ L{li}: log_r not in state_dict")
            out[li] = None
    return out

n_e_list = [64, 128, 256]

print("\n=== 验证 (用户 2026-07-27 第一步) ===\n")

# epoch 0 (初始化后)
ckpt_e0 = [c for c in ckpts if "epoch_0" in os.path.basename(c) or "epoch_1_" in os.path.basename(c)]
print(f"[epoch 0] checking {ckpt_e0}")
if ckpt_e0:
    state = torch.load(ckpt_e0[0], map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    radii_e0 = get_layer_radii(state, n_e_list)
    print(f"\n[epoch 0 / 1 初始化]")
    for li in [0, 1, 2]:
        if radii_e0[li] is None:
            continue
        r = radii_e0[li]
        target = [1.0, 1.35, 1.70][li]
        print(f"  L{li}: min={r.min():.4f} max={r.max():.4f} std={r.std():.4f}  "
              f"target={target}  "
              f"{'✅ min≈max≈target' if abs(r.min()-target) < 0.01 and abs(r.max()-target) < 0.01 else '❌'}")

# 最终 ckpt
ckpt_final = ckpts[-1]
print(f"\n[final] checking {ckpt_final}")
state = torch.load(ckpt_final, map_location="cpu", weights_only=False)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
radii_final = get_layer_radii(state, n_e_list)

all_pass = True
print(f"\n[final]")
for li in [0, 1, 2]:
    if radii_final[li] is None:
        all_pass = False
        continue
    r = radii_final[li]
    target = [1.0, 1.35, 1.70][li]
    in_range = (r.min() >= target - 0.3 - 1e-6) and (r.max() <= target + 0.3 + 1e-6)
    no_nan = not np.isnan(r).any()
    std_ok = r.std() > 0.05
    init_ok = True  # epoch 0 already checked above
    pass_all = in_range and no_nan and std_ok
    print(f"  L{li}: min={r.min():.4f} max={r.max():.4f} std={r.std():.4f}  "
          f"target={target}  range=[{target-0.3:.1f},{target+0.3:.1f}]  "
          f"{'✅' if pass_all else '❌'}  "
          f"(in_range={in_range}, std>0.05={std_ok}, no_nan={no_nan})")
    if not pass_all:
        all_pass = False

print(f"\n=== 综合 ===")
print(f"  {'✅ PASS: 半径分化 + 范围 + 无 NaN + 起点 = M2' if all_pass else '❌ FAIL'}")