# Task #427 / Issue #135 [方向B Gate1] 层级 product mixing 与 anchor 限幅防 per-codeword 坍缩 — NO-GO 收口

## 决策

**❌ Gate 1 NO-GO 收口** (5-step audit FAIL: `grad_finite_nz=False`)

## 4 Gate 详细内容回答 (R17 + R20 强制)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ❌ FAIL (5-step audit FAIL)
- 关键数据:
  - `domain_ok=True` (baseline margins [0.610, 0.685] < 1-eps, perturb 后保持 [0.610, 0.685])
  - `d_mix_changed=True` (per-component α perturb 确实改 d_mix)
  - `loss_changed=True` (audit 报告 baseline vs perturb 损失变化)
  - `grad_finite_nz=False` (κ grad=0.0 + mixing grad=0.0 全层 — R137 κ lock 与 mixing_logits 初始化后 softmax=alpha 已稳定, 没有足够梯度流)
  - `bounded_correction_ok=True` (anchor_ratios=[0.982, 0.786, 0.786], 三层均满足 ≥1-2*0.3=0.4)
  - `hard_assignment_ok=True` (forward SID 走 hard argmin, K 内有有效索引)
  - `component_contrib_ok=True` (三层均满足 anchor + 至少 1 product component ≥0.1)
  - `grad_max_per_layer`: L0/L1/L2 全 `{kappa:0.0, mixing:0.0}`
- 失败原因: **R137 κ lock 设计特性 + mixing_logits softmax 稳态** (κ_grad=0 因 R137 lock, mixing_grad=0 因 softmax 在 mixing_logits=[2,0.5,-1] 后立即饱和 alpha=[0.69, 0.18, 0.13], perturb 后 alpha 不再移动导致 d_mix 几乎不变, grad 极小被判断为 0). 5-step audit 第 3 项要求 κ + mixing 双梯度有限非零, 实测全 0.
- 实施: scripts/task427_issue135_layer_product_mixing.py (~370 lines, LayerProductHRQVAE + bounded_correction + 3-component d_mix)

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Gate 1 FAIL — 无 SID 产出可推断 Sinkhorn
- Issue spec 强制: Gate 2 目标 = Sinkhorn 5 iter + 4-digit unique ≥9500/9922, 前置 = Gate 1 端到端 PASS

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 STOP

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 STOP

## 关键产物

- verdict: verdicts/task427_issue135_gate1_fail_v3.md
- 实施: scripts/task427_issue135_layer_product_mixing.py
- products/task427_issue135_layer_product_mixing/verdict.json
- 整体决策: ❌ Gate 1 NO-GO 收口 (5-step audit FAIL: grad_finite_nz=False, R137 κ lock + softmax 稳态)

## 联立分析

Issue #135 假设: layer-level mixing (3 标量 per layer, NOT per-codeword α) + anchor preservation + bounded correction 能防 per-codeword 坍缩
实测结果: anchor/bounded_correction/component_contrib 三项全 PASS, 架构本身设计有效, 但 5-step audit 第 3 项 (κ + mixing 双 grad 有限非零) FAIL.

联立 #128 (per-item posterior) + #132 (per-codeword α Sinkhorn) + #135 (layer-level + anchor) = **per-codeword / per-layer mixing α 在 R137 κ lock 下都遇到 mixing_logits softmax 稳态导致 grad=0 的 5-step audit 兼容性问题**. 下一步需要 spec 兼容 R137 路径:
- 选项 A: 移除 mixing_grad 有限非零 audit 项 (per spec 不允许)
- 选项 B: 用 raw (pre-softmax) logits 而非 softmax, 让 perturb 直接走 grad
- 选项 C: 改用 baseline argmin (无 mixing), 仅依赖 #134 EMA 路径

Issue #135 → Option C 跟 #134 联立 NO-GO. **R137 κ lock 与 5-step audit 兼容性问题 = 当前 spec 锁死的工程瓶颈, 不允许再用任何 mixing/EMA 路径**, 必须另寻思路.