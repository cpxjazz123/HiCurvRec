#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""30 秒验证: 双曲码字方向是否在训练中被学到.

对比最早 (epoch_X_collision_Y) vs 最晚 (epoch_49_collision_0.5518) ckpt
的 hyp 子空间方向余弦. 应 < 0.99 (方向确实在学).
若 ≈ 1.0 → 梯度被切断, 回去修.

用法:
  python3 m_arm_direction_check.py [arm_dir]
"""
import sys, os, glob, re
import torch
import torch.nn.functional as F

ARM_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/m2_went0.1_antidead_revive"

# 找 ckpt 子目录
subdirs = sorted(glob.glob(os.path.join(ARM_DIR, "Jul-27*")))
if not subdirs:
    print(f"ERROR: no ckpt subdir under {ARM_DIR}")
    sys.exit(1)
CKPT_DIR = subdirs[-1]
print(f"[ckpt_dir] {CKPT_DIR}")

# 找最早 + 最晚 epoch ckpt
epoch_ckpts = []
for f in os.listdir(CKPT_DIR):
    m = re.match(r"epoch_(\d+)_collision_([\d.]+)_model\.pth", f)
    if m:
        epoch_ckpts.append((int(m.group(1)), float(m.group(2)), f))
epoch_ckpts.sort()
if len(epoch_ckpts) < 2:
    print(f"ERROR: need ≥2 epoch_ckpts, got {len(epoch_ckpts)}")
    sys.exit(1)

EPOCH_EARLY = epoch_ckpts[0]
EPOCH_LATE = epoch_ckpts[-1]
print(f"[compare] epoch_{EPOCH_EARLY[0]}_collision_{EPOCH_EARLY[1]} "
      f"vs epoch_{EPOCH_LATE[0]}_collision_{EPOCH_LATE[1]}\n")

state_early = torch.load(os.path.join(CKPT_DIR, EPOCH_EARLY[2]),
                          map_location="cpu", weights_only=False)
state_late = torch.load(os.path.join(CKPT_DIR, EPOCH_LATE[2]),
                         map_location="cpu", weights_only=False)
if isinstance(state_early, dict) and "state_dict" in state_early:
    state_early = state_early["state_dict"]
if isinstance(state_late, dict) and "state_dict" in state_late:
    state_late = state_late["state_dict"]

print("=== Per-layer hyp direction cosine (early vs late) ===\n")
results = {}
for li in [0, 1, 2]:
    key = f"hrq.vq_layers.{li}.embeddings.weight"
    if key not in state_early or key not in state_late:
        print(f"  L{li}: weight key not found")
        continue
    w_e = state_early[key][:, :2]
    w_l = state_late[key][:, :2]
    dir_e = F.normalize(w_e, dim=-1)
    dir_l = F.normalize(w_l, dim=-1)
    cos = (dir_e * dir_l).sum(-1)
    mean_cos = cos.mean().item()
    min_cos = cos.min().item()
    max_cos = cos.max().item()
    print(f"  L{li}: cos_sim mean={mean_cos:.4f}  min={min_cos:.4f}  max={max_cos:.4f}  "
          f"Δ_dir={1.0 - mean_cos:.4f}")
    results[li] = mean_cos

print("\n=== 闸门判定 ===")
all_below_99 = all(c < 0.99 for c in results.values())
print(f"  {'✅ PASS: 所有层 cos < 0.99 (方向确实在学)' if all_below_99 else '❌ FAIL: 至少一层 cos ≥ 0.99 (梯度被切断, 回去修)'}")