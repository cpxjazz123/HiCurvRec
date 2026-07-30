#!/usr/bin/env python3
"""Task #167 — Issue #58 Step 2: 梯度单元测试 + Bug 根因分析

🚨 关键发现: α_l_raw 和 scale_l 都是 DEAD PARAMETER
- α_l_raw: forward 中 _per_component_dist_sq 用 alpha 但 argmin 截断梯度; commitment loss 用 kappa_m() = fixed kappa (跟 Issue #56 的 MixedCurv 子类不匹配)
- scale_l: forward 中完全未使用 (代码注释说要校准 codebook 但未实现)
"""
from __future__ import annotations
import sys, os
_HGREC_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec"
sys.path.insert(0, _HGREC_ROOT)

import torch
import torch.nn as nn
from model.hrqvae_issue55_56 import (
    FreeCurvVectorQuantizationMixedCurv,
    FreeCurvVectorQuantizationMixedCurvWithScale,
    RiemannianAdamW,
)
from model.hrqvae_free_curv import FreeCurvVectorQuantization

torch.manual_seed(42)


print("=" * 75)
print("T1: FreeCurvVectorQuantizationMixedCurv α_l_raw gradient")
print("=" * 75)

vq = FreeCurvVectorQuantizationMixedCurv(
    n_e=64, e_dim=32, M=1, kappa_fixed=0.74,
    kmeans_init=False, sk_eps=0.003,
)
print(f"  alpha_l_raw init = {vq.alpha_l_raw.item():.6f}")
print(f"  alpha_l (sigmoid) = {vq.alpha_l.item():.6f}")
print(f"  alpha_l_raw.requires_grad = {vq.alpha_l_raw.requires_grad}")

x_input = torch.randn(16, 32) * 0.3
x_q, vq_loss, indices = vq(x_input, use_sk=False)
recon_loss = ((x_input - x_q) ** 2).mean()
total_loss = recon_loss + vq_loss
total_loss.backward()

print(f"\n  Forward: x_q shape={x_q.shape}, vq_loss={vq_loss.item():.6f}")
print(f"  After backward:")
print(f"    alpha_l_raw.grad = {vq.alpha_l_raw.grad}")
print(f"    embeddings.weight.grad.norm() = {vq.embeddings.weight.grad.norm().item():.6f}")

if vq.alpha_l_raw.grad is None:
    print(f"\n  🚨 BUG CONFIRMED: α_l_raw.grad = None")
    print(f"     即使 forward 用 alpha (sigmoid), 反向传播到 alpha_l_raw 路径被切断")


print("\n" + "=" * 75)
print("T2: FreeCurvVectorQuantizationMixedCurvWithScale scale_l gradient")
print("=" * 75)

vq_full = FreeCurvVectorQuantizationMixedCurvWithScale(
    n_e=64, e_dim=32, M=1, kappa_fixed=0.74,
    kmeans_init=False, sk_eps=0.003, scale_init=1.0,
)
print(f"  scale_l init = {vq_full.scale_l.tolist()}")

x_q, vq_loss, indices = vq_full(x_input, use_sk=False)
recon_loss = ((x_input - x_q) ** 2).mean()
total_loss = recon_loss + vq_loss
total_loss.backward()

print(f"  After backward:")
print(f"    scale_l.grad = {vq_full.scale_l.grad}")
print(f"    alpha_l_raw.grad = {vq_full.alpha_l_raw.grad}")

if vq_full.scale_l.grad is None:
    print(f"\n  🚨 BUG CONFIRMED: scale_l.grad = None (dead parameter)")
    print(f"     代码 docstring 说 'codebook = embeddings.weight * scale_l.unsqueeze(-1)'")
    print(f"     但 forward 路径中没有这个实现")


print("\n" + "=" * 75)
print("T3: 根因分析 — 梯度流追踪")
print("=" * 75)

print("""
  Forward 路径 (FreeCurvVectorQuantization.forward):

    d = self._per_component_dist_sq(latent, codebook)   # ← 用了 alpha
      ↓
    indices = torch.argmin(d, dim=-1)                    # ← argmin 截断梯度!
      ↓
    x_q_hard = codebook.index_select(0, indices)         # ← index_select 无梯度
      ↓
    x_q_st = x + (x_q - x).detach()                       # ← straight-through, x_q 无梯度
      ↓
    recon_loss = ‖x - x_q_st‖²                            # ← 不依赖 alpha (因为 detach)

    kappa = self.kappa_m()  # 在 MixedCurv 子类返回 fixed kappa (不用 alpha!)
      ↓
    commitment_loss = d_κ(x_q, x)²  +  beta * d_κ(x, x_q)²
      ↓
    total_loss = recon_loss + vq_loss

  🚨 **死路径 1: α_l 只影响 d, 但 d 只用于 argmin (无梯度)**
  🚨 **死路径 2: commitment/codebook loss 用 kappa_m()=fixed_kappa, 不依赖 alpha**
  🚨 **死路径 3: x_q_st 通过 detach 让 recon loss 不依赖 d**

  **结果**: total_loss 对 α_l_raw 的梯度严格为 0
""")


print("\n" + "=" * 75)
print("T4: 反向验证 — 手动构造可微路径, alpha 应该有梯度")
print("=" * 75)

# Construct a path where alpha is differentiable
vq3 = FreeCurvVectorQuantizationMixedCurv(
    n_e=64, e_dim=32, M=1, kappa_fixed=0.74,
    kmeans_init=False, sk_eps=0.003,
)
x_input3 = torch.randn(16, 32) * 0.3

# Directly use _per_component_dist_sq output as loss (NOT argmin)
codebook = vq3.embeddings.weight
d = vq3._per_component_dist_sq(x_input3, codebook)  # (B, K)
# Use distance directly as loss (no argmin)
dist_loss = d.min(dim=-1)[0].mean()  # min has gradient
dist_loss.backward()

print(f"  Distance-direct loss: {dist_loss.item():.6f}")
print(f"  alpha_l_raw.grad = {vq3.alpha_l_raw.grad}")
if vq3.alpha_l_raw.grad is not None:
    print(f"  alpha_l_raw.grad.norm() = {vq3.alpha_l_raw.grad.norm().item():.8f}")
    print(f"  ✅ Confirming α_l_raw IS differentiable in principle")
    print(f"     The bug is in the LOSS CONSTRUCTION (uses argmin + straight-through)")
else:
    print(f"  ❌ Even direct dist loss has no gradient (deeper bug)")


print("\n" + "=" * 75)
print("T5: RiemannianAdamW 公式 bug 验证 (1+ vs 1-)")
print("=" * 75)

# Code formula: grad_riem = grad / (1 + ‖x‖²)²
# Standard Poincaré: grad_riem = grad * (1 - κ‖x‖²)² / 4

print("  边界行为对比 (κ=0.74):")
print(f"  {'‖x‖':<8} {'Code (1+)²':<15} {'Std (1-κ·r²)²/4':<20} {'Ratio':<10}")
for norm in [0.0, 0.3, 0.5, 0.7, 0.866, 0.95, 1.0]:
    code_f = (1 + norm**2)**2
    std_f = (1 - 0.74 * norm**2)**2 / 4
    ratio = code_f / std_f if std_f > 0 else float('inf')
    print(f"  {norm:<8.3f} {code_f:<15.4f} {std_f:<20.4f} {ratio:<10.2f}")


print("\n" + "=" * 75)
print("T6: Issue #58 §疑点 1 验证 — verdict 描述是否准确")
print("=" * 75)

print("""
  Issue #55 verdict 写道:
    "投影公式(1/(1+‖x‖²)²)在‖x‖≪1时接近1(无信号)，
     在‖x‖→1时接近0(梯度消失)"

  验证: ‖x‖=1 时 1/(1+1)² = 1/4 = 0.25 (不接近 0!)
  ❌ verdict 数学描述错误!

  实际公式 1/(1+‖x‖²)²:
    - ‖x‖=0:  factor = 1.0  (无投影效果)
    - ‖x‖=0.5: factor = 0.5
    - ‖x‖=1:  factor = 0.25
    - ‖x‖→∞: factor → 0  (但 Poincaré 球 ‖x‖ < 1)

  真正在 ‖x‖→1 时趋于 0 的是 (1-κ‖x‖²)²/4:
    - ‖x‖=0:    factor = 0.25
    - ‖x‖=0.5:  factor = (1-0.74·0.25)²/4 = 0.171
    - ‖x‖=1:    factor = (1-0.74)²/4 = 0.0169  ✓ 趋于 0
    - ‖x‖→1/√κ ≈ 1.16: factor = 0  (球面边界)

  **这是 #47 同款公式 bug**: 1+ 应该是 1-, 且漏了 κ 和 /4
""")


print("\n" + "=" * 75)
print("SUMMARY — Issue #58 Step 2 完整结论")
print("=" * 75)

print("""
  BUG #1 (致命): α_l_raw 是 dead parameter
    - 原因: commitment loss 用 kappa_m() = fixed_kappa (不用 alpha)
    - 路径: _per_component_dist_sq 用 alpha 但 argmin 截断; loss 路径用 fixed kappa
    - 影响: Issue #56 α_l 静止 = 实现 bug, 不是 saturation
    - 修复方向: 让 commitment/codebook loss 也用 mixed_curv_dist

  BUG #2 (致命): scale_l 是 dead parameter
    - 原因: forward 中完全未使用 scale_l
    - 影响: Issue #55 s_l 静止 = 实现 bug (docstring 撒谎)
    - 修复方向: 在 _per_component_dist_sq 中加入 scale_l 应用

  BUG #3 (中等): RiemannianAdamW 公式 1+ vs 1- 数学错误
    - 原因: 代码用 (1+‖x‖²)², 标准 Poincaré 球用 (1-κ‖x‖²)²/4
    - 影响: codebook 学习方向偏离正确 Riemannian 流形
    - 修复方向: 替换为正确公式

  BUG #4 (次要): Issue #55 verdict 数学描述错误
    - 原因: "‖x‖→1 时趋于 0" 的描述跟实际公式不符
    - 影响: verdict 论证链断裂 (用户 Issue #58 指出)

  **结论**:
    - Issue #55 + #56 早期 NO-GO 结论是基于 buggy 实现
    - 修复 Bug #1+#2+#3 后需要重新跑 Stage 1-3 验证
    - 跟 Issue #47 同模式 (公式 bug 被误判为方向问题)
    - 建议重新开放 Issue #55 + #56 修复后重测
""")
