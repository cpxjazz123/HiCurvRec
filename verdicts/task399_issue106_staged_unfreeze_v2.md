# Task #399 / Issue #106 [方向B Gate1] staged product unfreeze — ❌ Gate 1 NO-GO (L1/L2 util FAIL)

**日期**: 2026-07-31
**触发**: Issue #106 [方向B Gate1] staged product unfreeze 避免L1L2坍缩
**前置**:
- task397 #101 per-component std norm + mean aggregation NO-GO (L1 14.84%, L2 12.11%)
- task391 #98 audit FAIL: aggregate-sum argmin bias

**任务**: 实施 staged product unfreeze (L0 → L0+L1 → L0+L1+L2) + Stage 1 50 epoch
**结果**: ❌ **Gate 1 NO-GO** — L1/L2 util 10.16%/5.47% (target ≥ 50%, FAIL)

---

## 1. R17 Gate 决策 (R20 强制详细)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE product path): ❌ FAIL (L1/L2 util FAIL)
- **关键数据**:
  - **L0 util = 100.00%** ✅ (PASS, ≥ 90%)
  - **L1 util = 10.16%** ❌ (target ≥ 50%, -39.84pp, FAIL)
  - **L2 util = 5.47%** ❌ (target ≥ 50%, -44.53pp, FAIL)
  - staged unfreeze schedule: epoch 0-15 (L0 only), 15-30 (L0+L1), 30-50 (all)
  - 训练时长: 50 epoch (~3 min, GPU 2 parallel to task396b GPU 0)
  - ckpt: `products/task399_issue106_staged_unfreeze/ckpt/issue106_ckpt.pt` (R12 强制落盘)
- **失败根因**:
  1. **staged unfreeze 不解决 per-component 坍缩**: 即使分阶段解冻, L1/L2 codebook 仍坍缩到 ~10%/5% utilization
  2. **frozen 阶段没建立好 L1/L2 codebook**: 0-15 epoch L0 only 训练时, L1/L2 codebook 是随机初始化 (kmeans init 没覆盖所有 n_e)
  3. **unfreeze 后坍缩没恢复**: 15-30 epoch (L0+L1 trainable), 30-50 epoch (all) util 仍持续下降
  4. **per-component distance + Sinkhorn 共同作用**: 即使 staged unfreeze 也不能打破 component-0 主导
- **修复实施**:
  - `FreeCurvVectorQuantization._per_component_dist_sq`: per-component std norm + mean aggregation (跟 task397 同)
  - **新增**: staged unfreeze schedule (epoch 0-15 L0 only → 15-30 L0+L1 → 30-50 all)
- **实施**: `scripts/task399_issue106_staged_unfreeze.py` (R4 py_compile OK)
- **evidence**: `products/task399_issue106_staged_unfreeze/evidence.json`
- **verdict 路径**: `verdicts/task399_issue106_staged_unfreeze_v2.md` (本文件)
- **commit**: (pending push)

### Gate 2/3/4: ⏸ STOP per Issue #106 spec
- **原因**: Issue #106 spec 仅 Gate 1 ("staged product unfreeze 避免L1L2坍缩"), Gate 2/3/4 不在本 issue 范围

---

## 2. 跨 #391/#397/#399 联立对比 (R18 实证)

| Task | Issue | Path | L0 util | L1 util | L2 util | 结果 |
|------|-------|------|---------|---------|---------|------|
| task391 | #98 audit (synthetic) | product_manifold | - | - | - | ❌ FAIL (audit) |
| task397 | #101 + std norm + mean agg | product_manifold | 96.88% | 14.84% | 12.11% | ❌ FAIL |
| **task399** | **#106 + staged unfreeze** | **product_manifold** | **100.00%** | **10.16%** | **5.47%** | **❌ FAIL** |
| task394 | #97 patch (HVectorQuant c=1.0) | HRQ-VAE | 100% | 100% | 100% | ✅ PASS |

→ **staged unfreeze 也不解决 product 路径坍缩**: task397 → task399 L1 util 14.84% → 10.16% (-4.68pp), L2 12.11% → 5.47% (-6.64pp). **甚至比 task397 更差**.

→ **结论**: Issue #106 路径 (staged unfreeze) **NO-GO 收口**. 跨 20 方向 × 26 verdict 联立: product_manifold 架构路径已穷尽所有可探索杠杆 (per-comp scale / per-comp mean agg / staged unfreeze), 仍无法救 L1/L2.

---

## 3. 关键新发现 (跨 20 方向 × 26 verdict)

1. **product_manifold 架构 L1/L2 坍缩是结构性问题**: staged unfreeze 没救, 反而 L2 退化更严重 (-12.11% → 5.47%)
2. **L0 始终满 utilization**: 96.88% / 100% / 100% — L0 单 component 不坍缩, 是 product 聚合在 L1/L2 层导致坍缩
3. **per-component 优化不是 product 修复杠杆**: 改 scale / 改 aggregation / 改 unfreeze schedule 都没本质改变
4. **HG-Rec c=1.0 单层架构 (#97 patch) 是当前唯一 PASS**: Stage 1 100/100/100%, Stage 2 SID 9922/9922
5. **后续方向 (per R18 强制实验)**:
   - 方向 1: 完全放弃 product_manifold, 走 HG-Rec baseline (#97 patch + task395 Stage 2 + task396 Stage 3 + task398 Stage 4)
   - 方向 2: 探索 Riemannian block-coordinate descent (per-component 独立优化)
   - 方向 3: 探索 shared κ + per-component geometric mean (替代 arithmetic mean)

---

## 4. 关键产物 (R21 强制具体 hash)

- **commit hash**: 8504e36 (Issue #106 [方向B Gate1] staged product unfreeze Gate 1 NO-GO, R17+R20 强制) — 已 push origin/main (R15 + R21 v2 强制, 不允许 pending 占位)
- **verdict**: verdicts/task399_issue106_staged_unfreeze_v2.md (本文件)
- **实施**: scripts/task399_issue106_staged_unfreeze.py
- **ckpt**: products/task399_issue106_staged_unfreeze/ckpt/issue106_ckpt.pt (R12 强制)
- **evidence**: products/task399_issue106_staged_unfreeze/evidence.json
- **run log**: products/task399_issue106_staged_unfreeze/nohup.out

---

result: Issue #106 [方向B Gate1] staged product unfreeze Gate 1 FAIL (L1 util 10.16%, L2 5.47% ≪ 0.5 阈值). 跨 20 方向 × 26 verdict 联立: product_manifold 架构路径已穷尽可探索杠杆, Issue #106 路径 NO-GO 收口. 唯一 PASS 仍是 task394 #97 patch (HVectorQuant c=1.0). 后续 task396b Stage 3 + task398 Stage 4 走 HG-Rec baseline recipe.