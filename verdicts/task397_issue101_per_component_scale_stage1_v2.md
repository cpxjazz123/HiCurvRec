# Task #397 / Issue #101 [方向B Gate1] per-component scale 修复 — ❌ Gate 1 NO-GO (agree3 FAIL)

**日期**: 2026-07-31
**触发**: Issue #101 [方向B Gate1] 修复product聚合尺度并复验训练后agree3
**前置**:
- task391 #98 audit FAIL: synthetic agree3=0%, init agree3=4.20% (<95% PASS 阈值)
- 根因诊断: `FreeCurvVectorQuantization._per_component_dist_sq` 没 per-component std normalization + aggregate-sum argmin bias toward component 0

**任务**: 实施 per-component std normalization + Stage 1 50 epoch 实证复跑 (per Issue #101 spec "复验训练后agree3")
**结果**: ❌ **Gate 1 NO-GO** — agree3 L0/L1/L2 = 0.5245/0.0023/0.1905 (target ≥ 0.95, FAIL all layers)

---

## 1. R17 Gate 决策 (R20 强制详细)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE product path): ❌ FAIL (agree3 全部层 FAIL)
- **关键数据**:
  - **L0 util = 96.88%** (target ≥ 90%, PASS) — 修复对 L0 有效
  - **L1 util = 14.84%** (target ≥ 90%, **FAIL**, -75.16pp)
  - **L2 util = 12.11%** (target ≥ 90%, **FAIL**, -77.89pp)
  - **agree3 L0 = 0.5245** (target ≥ 0.95, **FAIL** -42.55pp)
  - **agree3 L1 = 0.0023** (target ≥ 0.95, **FAIL** -94.77pp, near zero)
  - **agree3 L2 = 0.1905** (target ≥ 0.95, **FAIL** -75.95pp)
  - 训练时长: 50 epoch (~3 min, GPU 1 parallel to task396 GPU 0)
  - ckpt: `products/task397_issue101_per_component_scale_stage1/ckpt/issue101_ckpt.pt` (R12 强制落盘)
- **失败根因**:
  1. **per-component std normalization 不解决维度坍缩**: L0 util 96.88% 显示 L0 修复有效 (block_dim 11), 但 L1/L2 util 12-15% 说明 per-component scale 仍偏
  2. **agree3 L1 = 0.0023 (几乎零)**: per-component argmin 跟 full argmin 完全不相关 — mean aggregation 没用, per-component distance 仍主导 component 0
  3. **agree3 L0 = 0.5245 (近 random)**: 跟 L0 util 96.88% 矛盾 — 高 util 但低 agree3 → per-component 选错 codeword, 但 full path 偶然选对 (Sinkhorn / kmeans 副作用)
  4. **κ-Stereographic 在 per-component 维度的归一化被 mean aggregation 削弱**: 单 component 维度的距离量级差异无法通过 mean 消除
- **未实施部分**:
  - 仍需探索: product_manifold scale regularization / per-component learnable κ / Riemannian block-coordinate descent
- **实施**: `scripts/task397_issue101_per_component_stage1_train.py` (R4 py_compile OK)
- **evidence**: `products/task397_issue101_per_component_scale_stage1/evidence.json`
- **verdict 路径**: `verdicts/task397_issue101_per_component_scale_stage1_v2.md` (本文件)
- **commit**: (see git log)

### Gate 2/3/4: ⏸ STOP per Issue #101 spec
- **原因**: Issue #101 spec 仅 Gate 1 ("复验训练后agree3"), Gate 2/3/4 不在本 issue 范围
- **后续**: Gate 2 已由 Issue #102 [方向C Gate2] 处理 (task395 PASS), Gate 3 已由 task396b 启动 (PID 660598 GPU 0 训练中), Gate 4 由 task398 下一轮跑

---

## 2. 跨 #391/#397/#394 联立对比 (R18 实证)

| Task | Issue | Path | agree3 L0 | agree3 L1 | agree3 L2 | util L0/L1/L2 | 结果 |
|------|-------|------|-----------|-----------|-----------|----------------|------|
| task391 | #98 audit (synthetic) | product_manifold | 0% | 0% | 0% | - | ❌ FAIL |
| task391 | #98 audit (init) | product_manifold | 4.20% | 4.20% | 4.20% | - | ❌ FAIL |
| **task397** | **#101 + std norm fix** | **product_manifold** | **52.45%** | **0.23%** | **19.05%** | **96.88/14.84/12.11%** | **❌ FAIL** |
| task394 | #97 patch (HVectorQuant, c=1.0) | HRQ-VAE | - | - | - | 100/100/100% | ✅ PASS |

→ **per-component scale fix 不解决 product 聚合坍缩**: task391 → task397 agree3 没本质提升 (除 L0 偶然涨到 52% 之外, L1/L2 仍 ≈ 0%)

→ **结论**: product_manifold 架构层面 agree3 不达标, 即使 per-comp std norm + mean aggregation 也不通过. Issue #101 路径 (per-component scale 修复) **NO-GO 收口**.

---

## 3. 关键新发现 (跨 19 方向 × 25 verdict)

1. **per-component std norm + mean aggregation 不构成杠杆**: agree3 L1/L2 ≪ 0.95 → 修复范围不够, 需更深的 product 架构重设计
2. **L0 util 高 ≠ agree3 高**: 96.88% util 但 52.45% agree3 说明 codeword selection 在 per-comp 维度仍错
3. **κ-Stereographic per-component 几何 ≠ full-product 一致**: per-comp argmin ≠ full argmin, 维度坍缩是几何结构问题不是 scale 问题
4. **HG-Rec c=1.0 单层架构 (#97 patch) 是当前唯一 PASS**: 1.56%/0.78%/0.39% baseline → 100/100/100% via 1-line patch. product path 仍 FAIL
5. **后续方向 (per R18)**:
   - 方向 1: Riemannian block-coordinate descent (per-component 独立更新)
   - 方向 2: product_manifold 改用共享 κ + per-component scale 但 mean over geometric mean
   - 方向 3: 放弃 product_manifold 路径, 改 HG-Rec c=1.0 (已 PASS)

---

## 4. 关键产物 (R21 强制具体 hash)

- **commit hash**: 27265e6 (Issue #101 [方向B Gate1] per-component scale 修复 Gate 1 NO-GO, R17+R20 强制) — 已 push origin/main (R15 + R21 v2 强制, 不允许 pending 占位)
- **verdict**: verdicts/task397_issue101_per_component_scale_stage1_v2.md (本文件)
- **实施**: scripts/task397_issue101_per_component_stage1_train.py
- **ckpt**: products/task397_issue101_per_component_scale_stage1/ckpt/issue101_ckpt.pt (R12 强制)
- **evidence**: products/task397_issue101_per_component_scale_stage1/evidence.json
- **run log**: products/task397_issue101_per_component_scale_stage1/nohup.out

---

result: Issue #101 [方向B Gate1] per-component scale 修复 Gate 1 FAIL (agree3 L0/L1/L2 = 0.5245/0.0023/0.1905 ≪ 0.95 阈值). 跨 19 方向 × 25 verdict 联立: per-component std norm + mean aggregation 不解决 product 聚合坍缩, Issue #101 路径 NO-GO 收口. 唯一 PASS 仍是 task394 #97 patch (HVectorQuant c=1.0, util 100/100/100%). 后续 task396b Stage 3 T5 + task398 Stage 4 R@K 仍走 HG-Rec baseline recipe.