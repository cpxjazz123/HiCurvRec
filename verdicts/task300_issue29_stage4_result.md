# Task #300 / Issue #29 — Gate 4 Stage 4 NO-GO

**日期**: 2026-07-30
**状态**: ❌ **Gate 4 Stage 4 NO-GO — per-layer K_l=[128,64,32] R@10=0.0979 (-4.0% vs HG-Rec baseline 0.1020)**
**决定**: Issue #29 关闭, per-layer 异构 K_l 路径 NO-GO, 不构成 R@10 增益杠杆.

---

## 1. Stage 4 评估结果

| 指标 | Issue #29 | HG-Rec baseline (#84) | Δ |
|------|-----------|----------------------|---|
| Recall@5 | 0.0789 | 0.0816 | -3.3% |
| **Recall@10** | **0.0979** | **0.1020** | **-4.0%** ❌ |
| Recall@20 | 0.1199 | 0.1279 | -6.3% |
| NDCG@5 | 0.0677 | 0.0690 | -1.9% |
| NDCG@10 | 0.0738 | 0.0755 | -2.3% |
| NDCG@20 | 0.0793 | 0.0821 | -3.4% |

**Issue #29 §Gate 3 通过条件** `R@10 > 0.1020` **未满足** → hard-stop, 关闭 Issue #29.

---

## 2. 跨 Gate 综合

| Gate | 内容 | 状态 |
|------|------|------|
| Gate 0 | per-layer K_l 接入 baseline (single-variable) | ✅ PASS (reg test) |
| Gate 1 | Stage 1 100 epoch 训练 | ✅ PASS (L0/L1/L2 100% util, collision=0.1465) |
| Gate 2 | Sinkhorn 5 iter 推断 | ✅ PASS (9922 unique, collision=0.1427) |
| Gate 3 | T5-mini 200 epoch 训练 | ✅ PASS (HG_Rec_best.pth 落盘) |
| **Gate 4** | **Stage 4 Test R@10 eval** | **❌ NO-GO (-4.0%)** |

**关键观察**: **过 Gate 1+2 ≠ 过 Gate 4**. task276 已实证 "过 Stage 1 ≠ 过 Stage 4", Issue #29 是 13 方向中第 4 个实证.

---

## 3. 失败根因 (R11.5 反向推断)

### 3.1 K_l 异构的边际效应

Issue #29 走 K_l 维度异构 (L0 K=128 翻倍 / L1 K=64 减半 / L2 K=32 减半), 但 R@10 = -4.0% NO-GO.

**假设**: T5-mini 在 Stage 3 训练时把 K_l 异构映射成 token-level 频率分布异构, 但 200 epoch 内 T5 还没学到 K_l 异构的语义含义, 把 K_l 异构当成噪声 → R@10 跌.

**对比 baseline K=[64,128,256]**:
- baseline K=64 (L0) → Issue #29 K=128 (L0): L0 容量翻倍但缺乏足够的训练时间让 T5 学到 L0 多出来的码字语义
- baseline K=128 (L1) → Issue #29 K=64 (L1): L1 容量减半, T5 可能学到了 L1 缺码字的低质编码
- baseline K=256 (L2) → Issue #29 K=32 (L2): L2 容量 -87.5%, T5 必然丢失大量 L2 细粒度信息 → R@10 跌

### 3.2 跟 Issue #30 对比

| 维度 | Issue #29 | Issue #30 |
|------|-----------|-----------|
| 单一变量 | K_l 异构 [128,64,32] | codebook transforms r_l + s_l |
| 共享根因处理 | ❌ (不动 encoder / commit) | ✅ (把码字 norm 推到 ‖x‖_E ≈ 0.85 健康区) |
| Stage 1 L0 util | 100% (突破 Phase 0) | 100% (突破 Phase 0) |
| Gate 4 R@10 | 0.0979 (-4.0%) NO-GO | **0.1022 (+0.2%) GO** |

**关键 insight**: **不动 encoder 的 per-layer 异构 (#30) GO**, **动 codebook 容量 (#29) NO-GO**. 这进一步验证 Phase 0 mode collapse 的真杠杆是**码字几何** (Issue #30 路径), 不是**码字容量** (Issue #29 路径).

---

## 4. 关键决策 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Issue #29 关闭 | ✅ NO-GO verdict | 重跑 / 调 K_l 数值 | Issue #29 §Gate 3 hard-stop 明确, R@10 ≤ 0.1020 → STOP |
| 2 | 不申请 K_l 数值扫描 | ✅ NO-GO 收口 | K=[64,64,64] / K=[256,256,256] 等变体 | task294 + Issue #29 实证 K_l 异构不能突破 R@10 |
| 3 | Issue #29 不上 paper | ✅ Issue #30 是 paper 锚点 | Issue #29 也写 paper section | Issue #29 NO-GO, 写入 paper 但作为对照基线, 不是获胜配置 |

---

## 5. 物理产物

- `verdicts/task300_issue29_gate1_gate2_result.md` (Gate 1+2 PASS)
- `verdicts/task300_issue29_stage4_metrics.json` (Stage 4 metrics)
- `scripts/task300_issue29_gate1_stage1_train.sh` (Stage 1 launcher)
- `scripts/task300_issue29_gate2_stage2_codebook.py` (Stage 2 Sinkhorn 推断)
- `scripts/task300_issue29_gate3_stage3_train.sh` (Stage 3 T5 launcher)
- `scripts/task300_issue29_gate4_stage4_eval.sh` (Stage 4 eval launcher)
- `products/task300/hrqvae_issue29_gate1/Jul-30-2026_00-03-53_*/` (Stage 1 ckpt)
- `products/task300/ckpt_hgrec_issue29/Instruments/Jul-30-2026_00-17-08/HG_Rec_best.pth` (Stage 3 ckpt)
- `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue29_per_layer_k.npy` (Stage 2 SID)

---

## 6. 关联

- [[task301-issue30-stage4-result]]: Issue #30 GO (R@10=0.1022 +0.2%) — 首个 per-layer 可变曲率机制击败 baseline
- [[task300-issue29-gate1-gate2-result]]: Issue #29 Gate 1+2 PASS (100% util, collision=0.1427)
- [[phase0-mode-collapse]]: 6 任务 Phase 0 mode collapse, Issue #29 K_l 异构突破 Phase 0 但 Gate 4 跌
- [[cross-task-c-k-range-no-go-exhausted]]: 8 方向 NO-GO 收口, K_l 异构方向新增 NO-GO

---

result: Task #300 / Issue #29 Gate 4 Stage 4 NO-GO. R@10=0.0979 (-4.0% vs baseline 0.1020). 5 Gate 全跑完 (Gate 0/1/2/3 PASS, Gate 4 NO-GO). 关键 insight: K_l 异构 [128,64,32] 突破 Phase 0 mode collapse (L0 100% util), 但 Gate 4 R@10 反跌, 实证 "过 Stage 1 ≠ 过 Stage 4". 跟 Issue #30 对比 = 不动 encoder 的 per-layer 异构 (Issue #30) GO, 动 codebook 容量 (Issue #29) NO-GO. Issue #29 关闭.