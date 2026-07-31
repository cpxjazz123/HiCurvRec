# Task #400 / Issue #107 [方向B Gate1] anchored residual product — ❌ Gate 1 NO-GO (L1/L2 util FAIL)

**日期**: 2026-07-31
**触发**: Issue #107 [方向B Gate1] anchored residual product 避免component主导坍缩
**前置**:
- task397 #101 per-comp std norm + mean agg NO-GO (L1/L2 14.84%/12.11%)
- task399 #106 staged unfreeze NO-GO (L1/L2 10.16%/5.47%)
- task391 #98 audit FAIL: component-0 主导

**任务**: 实施 anchored residual product (per-comp centroid + std norm + mean aggregation) + Stage 1 50 epoch
**结果**: ❌ **Gate 1 NO-GO** — L1/L2 util 31.25%/15.62% (target ≥ 50%, FAIL)

---

## 1. R17 Gate 决策 (R20 强制详细)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE product path): ❌ FAIL (L1/L2 util FAIL)
- **关键数据**:
  - **L0 util = 57.81%** (target ≥ 50%, PASS, +7.81pp margin)
  - **L1 util = 31.25%** ❌ (target ≥ 50%, **FAIL** -18.75pp)
  - **L2 util = 15.62%** ❌ (target ≥ 50%, **FAIL** -34.38pp)
  - **注意** (历史性发现): epoch 35 L0/L1/L2 = 1.000/1.000/0.984, 全部 PASS! 但 epoch 49 退化到 0.578/0.312/0.156 → 训练后期坍缩回 component-0
  - 训练时长: 50 epoch (~3 min, GPU 1 parallel to task396b GPU 0)
  - ckpt: `products/task400_issue107_anchored_residual/ckpt/issue107_ckpt.pt` (R12 强制落盘)
- **失败根因**:
  1. **anchored residual product 在中期 (ep35) 真的解锁**: L1/L2 util 100%/98.4% 说明 per-comp centroid subtraction 短期有效
  2. **后期 (ep 40→49) 退化**: L0 从 100% → 57.81%, L1 从 93.8% → 31.25%, L2 从 84% → 15.62% → 训练过程 component-0 仍逐渐主导
  3. **lr=1e-4 50 epoch 长训导致 component-0 re-collapse**: 修复只在前 35 epoch 稳定, 后续被梯度推动到 component-0
  4. **per-comp centroid subtraction 治标不治本**: 短期避免 component-0, 长期训练仍被 SGD 推回坍缩态
- **修复实施**:
  - `FreeCurvVectorQuantization._per_component_dist_sq`: per-comp centroid subtraction + std norm + mean aggregation
- **实施**: `scripts/task400_issue107_anchored_residual.py` (R4 py_compile OK)
- **evidence**: `products/task400_issue107_anchored_residual/evidence.json`
- **verdict 路径**: `verdicts/task400_issue107_anchored_residual_v2.md` (本文件)
- **commit**: (pending push)

### Gate 2/3/4: ⏸ STOP per Issue #107 spec
- **原因**: Issue #107 spec 仅 Gate 1 ("anchored residual product 避免component主导坍缩"), Gate 2/3/4 不在本 issue 范围

---

## 2. 跨 #391/#397/#399/#400 联立对比 (R18 实证)

| Task | Issue | Path | L0 util | L1 util | L2 util | 结果 |
|------|-------|------|---------|---------|---------|------|
| task391 | #98 audit (synthetic) | product_manifold | - | - | - | ❌ FAIL (audit) |
| task397 | #101 + std norm + mean agg | product_manifold | 96.88% | 14.84% | 12.11% | ❌ FAIL |
| task399 | #106 + staged unfreeze | product_manifold | 100.00% | 10.16% | 5.47% | ❌ FAIL |
| **task400** | **#107 + anchored residual** | **product_manifold** | **57.81%** | **31.25%** | **15.62%** | **❌ FAIL** |
| task394 | #97 patch (HVectorQuant c=1.0) | HRQ-VAE | 100% | 100% | 100% | ✅ PASS |

→ **anchored residual product 是 3 个 product 变体中最好的**: L1 31.25% vs 14.84%/10.16%, L2 15.62% vs 12.11%/5.47% (+16pp/+21pp / +3.5pp/+10pp)
→ **但仍 FAIL final util** (target ≥ 50%, 不达标). **epoch 35 是真 PASS 但 epoch 49 退化**, 训练时长是关键变量
→ **结论**: Issue #107 路径 **NO-GO 收口**, 但留下"短期 PASS + 长期退化"信号. 跨 21 方向 × 27 verdict 联立: product_manifold 架构 4 方向 (audit + std_norm + staged_unfreeze + anchored_residual) 全部 NO-GO 收口, product 路径彻底锁死.

---

## 3. 关键新发现 (跨 21 方向 × 27 verdict)

1. **anchored residual product 是真杠杆 (短期)**: epoch 35 L0/L1/L2 = 100/100/98.4% 全部 PASS. 但 50 epoch 后退化到 57/31/15%
2. **训练时长是 product path 的隐性变量**: 30 epoch 可能 PASS, 50 epoch FAIL. 之前 task397/399 都没测中间 epoch
3. **per-comp centroid subtraction 短期有效**: anchor 把 component-0 推离, 但 lr 推动仍让其回归
4. **HG-Rec c=1.0 单层架构 (#97 patch) 是当前唯一 PASS**: 100/100/100% final, 无退化
5. **后续方向 (per R18 强制实验)**:
   - 方向 1 (优先): 放弃 product_manifold, 走 HG-Rec baseline (#97 + task395 + task396 + task398)
   - 方向 2: 探索 epoch ≤ 30 early-stop 训练协议 (per task400 ep 35 信号)
   - 方向 3: 探索 Riemannian block-coordinate descent (per-comp 独立优化)

---

## 4. 关键产物 (R21 强制具体 hash)

- **commit hash**: 6f6b095 (Issue #107 [方向B Gate1] anchored residual product Gate 1 NO-GO, R17+R20 强制) — 已 push origin/main (R15 + R21 v2 强制, 不允许 pending 占位)
- **verdict**: verdicts/task400_issue107_anchored_residual_v2.md (本文件)
- **实施**: scripts/task400_issue107_anchored_residual.py
- **ckpt**: products/task400_issue107_anchored_residual/ckpt/issue107_ckpt.pt (R12 强制)
- **evidence**: products/task400_issue107_anchored_residual/evidence.json
- **run log**: products/task400_issue107_anchored_residual/nohup.out

---

result: Issue #107 [方向B Gate1] anchored residual product Gate 1 FAIL (L1/L2 util 31.25%/15.62% ≪ 0.5 阈值, 但 epoch 35 短期 PASS 100/100/98%). 跨 21 方向 × 27 verdict 联立: product_manifold 架构 4 方向全部 NO-GO 收口, Issue #107 路径 NO-GO. 后续只走 HG-Rec baseline (#97 patch path).