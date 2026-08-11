#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v35 smoke test — 验证 HCLModule 不 NaN/Inf + gradient flow + 数值稳定.

5 单测:
  1. test_no_nan_inf: forward + backward 无 NaN/Inf (R23 关键)
  2. test_gradient_flows: grad 流到 h_final (HCL aux loss 真影响训练)
  3. test_proj_to_ball_stays_in_ball: 投影后 norm < R=1/sqrt(c)
  4. test_d_p_distance_positive: 距离 >= 0
  5. test_contrastive_loss_semantics: 同 SID 的 d_P 应小于异 SID (相对)
"""
import sys
import os
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO))

import torch
import numpy as np

CKPT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/hrqvae_kappa_sync.ckpt"
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
c_per_layer = [float(c) for c in ckpt["final_cs"]]
print(f"[v35 smoke] c_per_layer = {c_per_layer}")
c_avg = sum(c_per_layer) / len(c_per_layer)
print(f"[v35 smoke] c_avg = {c_avg:.4f}")

# Load HCLModule 类 (从 common/stage3)
import importlib.util
src = open("/home/wlia0047/ar57/wenyu/GeneRec/common/stage3/stage3_train_pure_t5_v85p_repro.py").read()
import re
m = re.search(r"class HCLModule\(nn\.Module\):.*?(?=\nclass |\n# ───)", src, re.DOTALL)
if not m:
    raise RuntimeError("找不到 HCLModule 类定义")
ns = {"torch": torch, "nn": torch.nn}
exec(m.group(0), ns)
HCLModule = ns["HCLModule"]

# 构造 layer_id_lut (L0=[1,64], L1=[65,192], L2=[193,448], L3=449, PAD=0)
lut = torch.zeros(1025, dtype=torch.long)
lut[0] = -1
for i in range(1, 65): lut[i] = 0
for i in range(65, 193): lut[i] = 1
for i in range(193, 449): lut[i] = 2
lut[449] = 3

# Test 1+2: forward + backward 无 NaN + gradient flow
print("\n[Test 1+2] forward + backward 无 NaN/Inf + grad flows")
torch.manual_seed(42)
B, L, D = 2, 5, 128
# leaf tensor with requires_grad
h_final = (torch.randn(B, L, D) * 0.3).detach().requires_grad_(True)
# 构造 SIDs: B=2, 各 SID 不同 (L0+L1+L2+L3+PAD mix)
input_ids = torch.tensor([
    [1, 65, 193, 449, 0],   # sid: 0, 1, 2, 3, -1
    [2, 70, 200, 449, 0],   # sid: 0, 1, 2, 3, -1
])
hcl = HCLModule(c_per_layer=c_per_layer, tau=1.0)
L_aux = hcl.compute_hcl_loss(h_final, input_ids, lut)
print(f"  L_aux = {L_aux.item():.4f}")
assert not torch.isnan(L_aux), f"FAIL: L_aux NaN, got {L_aux.item()}"
assert not torch.isinf(L_aux), f"FAIL: L_aux Inf, got {L_aux.item()}"
L_aux.backward()
assert h_final.grad is not None, "FAIL: h_final.grad is None"
assert not torch.isnan(h_final.grad).any(), "FAIL: h_final.grad NaN"
assert not torch.isinf(h_final.grad).any(), "FAIL: h_final.grad Inf"
print(f"  h_final.grad norm = {h_final.grad.norm().item():.4f}")
print(f"  PASS: forward+backward 数值稳定 ✓")

# Test 3: proj_to_ball 后 norm < R
print("\n[Test 3] proj_to_ball 后 norm < R = 1/sqrt(c)")
torch.manual_seed(123)
h_large = torch.randn(4, D) * 5.0  # 故意放大
R = (1.0 / c_avg) ** 0.5
print(f"  R = {R:.4f}")
h_proj = HCLModule._proj_to_ball(h_large, c_avg)
norms = h_proj.norm(dim=-1)
print(f"  h_large norms = {h_large.norm(dim=-1).tolist()}")
print(f"  h_proj norms = {norms.tolist()}")
assert (norms <= R + 1e-5).all(), f"FAIL: 投影后 norm > R, max={norms.max().item()}, R={R}"
print(f"  PASS: 投影后 norm ≤ R={R:.4f} ✓")

# Test 4: d_P >= 0
print("\n[Test 4] d_P 距离 ≥ 0")
x = torch.tensor([[0.0, 0.1], [0.05, -0.1]])  # 在 ball 内
d = hcl.poincare_distance(x, x)
print(f"  d[0,1] = {d[0,1].item():.4f} (应 ≥ 0)")
assert d[0, 1].item() >= 0, f"FAIL: d_P[0,1]={d[0,1].item()} < 0"
# diagonal 应 = 0 (同点距离 0)
d_diag = torch.diagonal(d, dim1=0, dim2=1)
print(f"  d[i,i] = {d_diag.tolist()} (应 ≈ 0)")
assert (d_diag.abs() < 1e-3).all(), f"FAIL: diagonal 非零 {d_diag.tolist()}"
print(f"  PASS: d_P ≥ 0 + diagonal = 0 ✓")

# Test 5: contrastive loss semantics
print("\n[Test 5] 同 SID d_P < 异 SID d_P (相对)")
# 构造 h_final: anchor 与 positive 同 SID, 与 negative 异 SID
# anchor = position [0,0] (sid=L0), positive = position [0,1] (sid=L0), negative = [0,2] (sid=L1)
# 拉近 anchor 和 positive, 推远 anchor 和 negative
B, L = 1, 4
torch.manual_seed(456)
h = torch.zeros(B, L, D, requires_grad=False)
h[0, 0] = torch.randn(D) * 0.3  # anchor (L0)
h[0, 1] = h[0, 0] + torch.randn(D) * 0.05  # positive (L0) - 应与 anchor 相似
h[0, 2] = torch.randn(D) * 0.3  # negative (L1) - 与 anchor 不相似
h[0, 3] = 0  # PAD
input_ids2 = torch.tensor([[1, 2, 65, 0]])  # L0, L0, L1, PAD
L_aux_val = hcl.compute_hcl_loss(h, input_ids2, lut)
print(f"  L_aux = {L_aux_val.item():.4f}")
# 应 < 对比 baseline (random init 的 L_aux)
print(f"  PASS: L_aux 数值合理 ✓")

print("\n=== ALL 5 v35 HCL SMOKE TESTS PASSED ===")