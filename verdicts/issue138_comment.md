## Issue #138 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### 5 件套审计 (Issue #138 spec 强制) — 全部已落地, commit 600c1a3
1. **config**: `descriptions/task427_issue135_layer_product_mixing.md` (LayerProductHRQVAE + bounded_correction + 3-component d_mix)
2. **sha256**: scripts 落地 (R20 audit), 跟 #132 同数据 / 同 budget / seed=42 (per Issue spec)
3. **raw_log**: `logs/task427_issue135_layer_product_mixing.log` (5-step audit 完整输出)
4. **verdict**: `verdicts/task427_issue135_gate1_fail_v3.md` + `products/task427_issue135_layer_product_mixing/verdict.json`
5. **commit**: `600c1a3` (tracked, pushed origin/main)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ❌ FAIL (5-step audit FAIL)
- 关键数据:
  - `domain_ok=True` (margins [0.610, 0.685])
  - `d_mix_changed=True`
  - `loss_changed=True`
  - `grad_finite_nz=False` (κ grad=0.0 + mixing grad=0.0 全层 — R137 κ lock + mixing_logits softmax 稳态)
  - `bounded_correction_ok=True` (anchor_ratios=[0.982, 0.786, 0.786])
  - `hard_assignment_ok=True`
  - `component_contrib_ok=True`
  - `grad_max_per_layer`: L0/L1/L2 全 `{kappa:0.0, mixing:0.0}`
- 失败原因: **R137 κ lock 设计特性 + mixing_logits softmax 稳态** (mixing_logits=[2,0.5,-1] softmax 饱和 alpha=[0.69,0.18,0.13], perturb 后 alpha 不移动导致 mixing_grad=0 + κ_grad=0). Issue #138 spec 要求"kappa 和 mixing 梯度有限非零并有更新" + Issue #135 layer-level mixing 跟 R137 κ lock **不兼容**.

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec

### 关键产物
- commit hash: 600c1a3
- push: origin/main (pushed 2026-08-01)
- verdict: verdicts/task427_issue135_gate1_fail_v3.md
- 实施: scripts/task427_issue135_layer_product_mixing.py
- 整体决策: ❌ Gate 1 FAIL (5-step audit FAIL: grad_finite_nz=False, R137 κ lock + mixing_logits softmax 稳态)

### 联立 #128 + #132 + #135 + #138
4 方向共同结论 = **per-codeword / per-layer mixing α 在 R137 κ lock 下都遇到 mixing_logits softmax 稳态导致 grad=0 的 5-step audit 兼容性问题**. 下一方向必须明确"是否用 R137 κ lock 路径":
- 选项 A: 解除 R137 κ lock → mixing_logits grad 自由 → per layer-level mixing 才有可能 (但 R137 是历史 13+ task 的 stability anchor, 解除风险大)
- 选项 B: 修改 spec 兼容 R137 κ lock (per Issue #138 spec 强制 κ grad 有限非零) → κ 不动 → 任何 mixing 路径 5-step audit FAIL
- 选项 C: 改用 baseline argmin (无 mixing), 仅依赖 #134 EMA 路径 → Issue #137 仍 NO-GO

Issue #138 → 选项 A/B/C 必须 owner 拍板, AI 不能自主决定 (R11.4 critical decision).