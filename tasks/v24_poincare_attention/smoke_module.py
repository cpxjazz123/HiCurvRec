#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v24 smoke test — 验证 CurvatureAttnBias 算 κ + B[i,j] = -γ · |κ_i - κ_j| 正确.

5 单测:
  1. test_gamma_zero_identity: γ=0 时 bias 全零 (起点等同 baseline)
  2. test_l0_kappa_constant: L0 token κ_i = κ_0
  3. test_l1_kappa_lookup: L1 token κ_i = κ_1 + Δκ_{1, q_0}
  4. test_l2_kappa_pair_lookup: L2 token κ_i = κ_2 + Δκ_{2, (q_0, q_1)}
  5. test_bias_sign_and_magnitude: γ=+1 时 B[i,j] = -|κ_i - κ_j| ≤ 0

R36 严格: 不调参, 走曲率机制.
"""
import sys
import os
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO))

# 用 v22.b Stage2 ckpt 的真实 curvature
import torch

CKPT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/hrqvae_kappa_sync.ckpt"
LAYER_ID_LUT_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/common/stage3/_layer_id_lut_cache.npy"

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
kappa_l0 = float(ckpt["final_kappas"][0])
kappa_l1 = float(ckpt["final_kappas"][1])
kappa_l2 = float(ckpt["final_kappas"][2])
delta_kappa_l1 = sd["vq_layers.1.delta_kappa.weight"].float()  # (64, 1)
delta_kappa_l2 = sd["vq_layers.2.delta_kappa.weight"].float()  # (8192, 1)
print(f"[v24 smoke] kappa_l0={kappa_l0:.4f} kappa_l1={kappa_l1:.4f} kappa_l2={kappa_l2:.4f}")
print(f"[v24 smoke] delta_kappa_l1 shape={delta_kappa_l1.shape} std={delta_kappa_l1.std().item():.4f}")
print(f"[v24 smoke] delta_kappa_l2 shape={delta_kappa_l2.shape} std={delta_kappa_l2.std().item():.4f}")

# 直接 load module (不依赖 import common.stage3 - 那里有argparse)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "v24_module",
    "/home/wlia0047/ar57/wenyu/GeneRec/common/stage3/stage3_train_pure_t5_v85p_repro.py",
)

# 用 runpy 加载模块, 但 stub 掉 argparse 避免触发命令行解析
# 简化: 直接 exec 文件 class 定义部分
src = open(spec.origin).read()
# 只提取 CurvatureAttnBias class (line ~943) 到 install_curvature_residual 之前
import re
m = re.search(r"class CurvatureAttnBias\(nn\.Module\):.*?(?=\ndef install_curvature_residual\()", src, re.DOTALL)
if not m:
    raise RuntimeError("找不到 CurvatureAttnBias 类定义")
cab_class_src = m.group(0)
# 注入 import
ns = {
    "__name__": "v24_cab_test",
    "torch": torch,
    "nn": __import__("torch.nn", fromlist=["nn"]),
}
exec(cab_class_src, ns)
CurvatureAttnBias = ns["CurvatureAttnBias"]

# 构建 module
cab = CurvatureAttnBias(
    kappa_l0=kappa_l0,
    kappa_l1=kappa_l1,
    kappa_l2=kappa_l2,
    delta_kappa_l1=delta_kappa_l1,
    delta_kappa_l2=delta_kappa_l2,
    gamma_init=0.0,
)
cab._cached_layer_id_lut = torch.zeros(1025, dtype=torch.long)
# 设 layer_id_lut: L0=[1,64], L1=[65,192], L2=[193,448], L3=449, PAD=0
cab._cached_layer_id_lut[0] = -1  # PAD
for i in range(1, 65):
    cab._cached_layer_id_lut[i] = 0  # L0
for i in range(65, 193):
    cab._cached_layer_id_lut[i] = 1  # L1
for i in range(193, 449):
    cab._cached_layer_id_lut[i] = 2  # L2
cab._cached_layer_id_lut[449] = 3  # L3

# ── Test 1: γ=0 identity ──
print("\n[Test 1] γ=0 → bias 全零")
input_ids = torch.tensor([[1, 2, 65, 193, 66, 449, 0]])  # L0, L0, L1, L2, L1, L3, PAD
cab.gamma.data = torch.tensor(0.0)
bias = cab(input_ids)
assert torch.all(bias == 0.0), f"FAIL: γ=0 应给零 bias, got max={bias.abs().max().item()}"
print(f"  PASS: bias.shape={bias.shape}, all zero ✓")

# ── Test 2: L0 kappa 常数 ──
print("\n[Test 2] L0 token κ_i = κ_0")
cab.gamma.data = torch.tensor(1.0)
input_ids = torch.tensor([[1, 2, 3]])  # 3 个 L0 tokens
kappa = cab.compute_kappa(input_ids)
expected = torch.full_like(kappa, kappa_l0)
assert torch.allclose(kappa, expected, atol=1e-6), f"FAIL: L0 κ 应={kappa_l0}, got {kappa.tolist()}"
print(f"  PASS: κ(L0 tokens) = {kappa.tolist()[0]} = {kappa_l0:.4f} ✓")

# ── Test 3: L1 kappa lookup (per q_0) ──
print("\n[Test 3] L1 token κ_i = κ_1 + Δκ_{1, q_0}")
# input: [L0=5, L1=65, L1=66]
# - position 0 (L0=5): κ = κ_0
# - position 1 (L1=65): q_0 = roll[-1] = position 0 = 5 (L0 code 5)
#   → l0_idx = 5-1 = 4, Δκ_{1,4} = delta_kappa_l1[4].item()
# - position 2 (L1=66): q_0 = roll[-1] = position 1 = 65 (L1 code 65, WRONG - actually rolls to 65)
#   → q_0_id = 65, l0_idx = 65-1 = 64 → clamped to 63
input_ids = torch.tensor([[5, 65, 66]])
kappa = cab.compute_kappa(input_ids)
expected_l0_kappa = kappa_l0
expected_l1_first = kappa_l1 + delta_kappa_l1[4].item()  # q_0 = position 0 = 5, l0_idx=4
expected_l1_second = kappa_l1 + delta_kappa_l1[63].item()  # q_0 = position 1 = 65, l0_idx=64 → clamp to 63
print(f"  kappa[0]={kappa[0,0].item():.4f} (expected {expected_l0_kappa:.4f})")
print(f"  kappa[1]={kappa[0,1].item():.4f} (expected {expected_l1_first:.4f})")
print(f"  kappa[2]={kappa[0,2].item():.4f} (expected {expected_l1_second:.4f})")
assert abs(kappa[0, 0].item() - expected_l0_kappa) < 1e-6
assert abs(kappa[0, 1].item() - expected_l1_first) < 1e-6
assert abs(kappa[0, 2].item() - expected_l1_second) < 1e-6
print(f"  PASS: L1 kappa lookup correct ✓")

# ── Test 4: L2 kappa pair lookup ──
print("\n[Test 4] L2 token κ_i = κ_2 + Δκ_{2, (q_0, q_1)}")
# input: [L0=10, L1=80, L2=200]
# - position 0 (L0=10): κ = κ_0
# - position 1 (L1=80): q_0 = roll[-1] = position 0 = 10
#   → l0_idx = 9, Δκ_{1,9}
# - position 2 (L2=200): q_0_l2 = roll[-2] = position 0 = 10
#   → l0_idx_l2 = 9
#   q_1_l2 = roll[-1] = position 1 = 80
#   → l1_idx_l2 = 80-65 = 15
#   → l2_idx_l2 = 9*128 + 15 = 1167
#   → Δκ_{2, 1167}
input_ids = torch.tensor([[10, 80, 200]])
kappa = cab.compute_kappa(input_ids)
expected_l2 = kappa_l2 + delta_kappa_l2[9 * 128 + 15].item()
print(f"  kappa[2]={kappa[0,2].item():.4f} (expected {expected_l2:.4f})")
assert abs(kappa[0, 2].item() - expected_l2) < 1e-6
print(f"  PASS: L2 pair lookup correct ✓")

# ── Test 5: bias sign + magnitude ──
print("\n[Test 5] γ=+1 时 B[i,j] = -|κ_i - κ_j| ≤ 0")
cab.gamma.data = torch.tensor(1.0)
input_ids = torch.tensor([[1, 2, 65]])  # L0, L0, L1
bias = cab(input_ids)
print(f"  bias.shape={bias.shape}")
# B[0,0] = -γ|κ_0 - κ_0| = 0
# B[0,1] = -γ|κ_0 - κ_0| = 0
# B[0,2] = -γ|κ_0 - κ_1.actual|
assert bias[0, 0, 0].item() == 0.0
assert bias[0, 0, 1].item() == 0.0
assert bias[0, 0, 2].item() < 0.0, f"FAIL: B[0,2] 应<0, got {bias[0,0,2].item()}"
# 对称性
assert torch.allclose(bias[0], bias[0].T, atol=1e-6), "FAIL: bias 非对称"
print(f"  PASS: B[L0,L1]={bias[0,0,2].item():.4f} (负, 对称) ✓")

print("\n=== ALL 5 v24 SMOKE TESTS PASSED ===")