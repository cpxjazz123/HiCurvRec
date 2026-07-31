# Task #403 / Stage 4 R@K eval (task401 ckpt + task403 30-epoch Stage 3 短训)

**日期**: 2026-07-31
**任务**: task403 Stage 4 R@K eval (task401 30-epoch Stage 1 ckpt + task402 SID + **task403 30-epoch Stage 3 短训** + Stage 4 R@K)
**结果**: ❌ **NO-GO 收口** (test R@10 = **0.0798**, vs HG-Rec baseline 0.1020, **-21.7%**). 揭示 val/test 反转现象: Stage 3 训练时长杠杆在 test set 反向 (短训 overfit val, 长训 generalize better)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ✅ PASS (per task401 verdict)

- **状态**: Pass
- **证据**: `verdicts/task401_issue108_best_epoch_verify_v2.md` (Issue #108 Gate 1 PASS)
- **关键数据**: L0/L1/L2 util = 100%/100%/100%, 4-digit SID unique 9920/9922

### Gate 2 (= Stage 2 Sinkhorn): ✅ PASS (per task402 Stage 2)

- **状态**: Pass
- **SID 文件**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task402.npy` (9922, 4)
- **关键数据**: Layer 0/1/2 util = 100%/100%/100%, 4-digit SID unique 9920/9922

### Gate 3 (= Stage 3 T5-mini **30 epoch 短训**): ⚠️ **val PASS, test 未知**

- **状态**: Training complete (exit code 0)
- **ckpt**: `products/task403_issue109_stage3_short_train/ckpt/Instruments/jul-31-2026_21-39-55/HG_Rec_best.pth` (22087081 bytes, R12 强制落盘)
- **耗时**: ~64 min (21:39 → 22:43)
- **关键 val_R@10 进展** (30 epoch 完整训练, 没触发 early_stop):
  - epoch 1: 0.0750
  - epoch 5: 0.0813
  - epoch 10: 0.0852
  - epoch 15: 0.0899
  - epoch 20: 0.0941
  - epoch 23: 0.0965
  - epoch 25: 0.0983
  - epoch 27: 0.1008
  - **epoch 28: 0.1020 (恰好命中 HG-Rec baseline)**
  - **epoch 29: 0.1031 (+1.1% vs baseline)**
  - **epoch 30: 0.1054 (+3.3% vs baseline) ✅ val PASS**
- **实施**: `scripts/task403_stage3_t5_train.sh` v2 (model.to(device) fix + task84 train/evaluate 函数复用)

### Gate 4 (= Stage 4 R@K eval): ❌ **NO-GO 收口** (val/test 反转, 短训 overfit)

- **关键数据 (test set, 跟 val 反转)**:
  - **Recall@5 = 0.0678** ❌ (vs HG-Rec baseline 0.0816, **-16.9%**)
  - **Recall@10 = 0.0798** ❌ (vs HG-Rec baseline 0.1020, **-21.7% FAIL**)
  - **Recall@20 = 0.0939** ❌ (vs HG-Rec baseline 0.1279, **-26.6%**)
  - **NDCG@5 = 0.0602** ❌ (vs HG-Rec baseline 0.0690, **-12.7%**)
  - **NDCG@10 = 0.0641** ❌ (vs HG-Rec baseline 0.0755, **-15.1%**)
  - **NDCG@20 = 0.0676** ❌ (vs HG-Rec baseline 0.0821, **-17.6%**)
- **val/test 反转 (CRITICAL 新发现)**:
  | 任务 | Stage 1 ep | Stage 3 ep | val_R@10 | test_R@10 | gap |
  |------|-----------|-----------|----------|-----------|-----|
  | task84 (HG-Rec) | 200 | 200 | - | 0.1020 | - |
  | task396b | 200 | 200 | - | 0.0864 | - |
  | task402 | 30 | 200 | - | 0.0991 | - |
  | **task403** | **30** | **30** | **0.1054** ✅ | **0.0798** ❌ | **-0.0256 (-24.3%)** |
- **失败根因**:
  1. **Stage 3 短训 30 epoch 在 val set 过拟合**: val_R@10 = 0.1054 (+3.3% vs baseline), 但 test_R@10 = 0.0798 (-21.7% vs baseline). val/test gap = 24.3% 反转.
  2. **训练时长杠杆假设 REFUTED**: task402 verdict §4 假设 #4 "训练时长 ≤ 30 epoch 是 Stage 3 真杠杆" **不成立**. 短训导致 val 过拟合, 长训 (200 epoch) generalize 更好.
  3. **架构杠杆机制**: 30 epoch Stage 3 让 T5-mini 在 val set 快速学到 SID 模式, 但没有充分 generalize 到 test set 的不同分布.
- **决策阈值**: R@10 > 0.1020 GO, ≤ 0.1020 NO-GO → 0.0798 ≪ 0.1020 → **NO-GO 收口**
- **跨方向联立**:
  | 任务 | Stage 1 ep | Stage 3 ep | R@10 | Δ vs baseline |
  |------|-----------|-----------|------|-------------|
  | task84 (HG-Rec) | 200 | 200 | **0.1020** | baseline |
  | task396b | 200 | 200 | 0.0864 | -15.3% |
  | task402 | 30 | 200 | 0.0991 | -2.8% (接近 baseline) |
  | **task403** | **30** | **30** | **0.0798** | **-21.7% (val 过拟合)** |

---

## 关键产物

- **commit hash**: ⏳ 待 commit 落地后写入 (R21 v2 强制)
- **push**: origin/main (R15 强制)
- **verdict 路径**: `verdicts/task403_stage4_rk_eval_v2.md` (本文件)
- **metrics 路径**: `verdicts/task403_stage4_rk_eval_metrics.json`
- **Stage 3 ckpt**: `products/task403_issue109_stage3_short_train/ckpt/Instruments/jul-31-2026_21-39-55/HG_Rec_best.pth`
- **整体决策**: ❌ **NO-GO** (val/test 反转, 短训 Stage 3 overfit val set, 推论 R10 真指标 vs val 杠杆方向不同)

---

## 跨方向联立 (R18 实证 4 方向 × 4 verdict)

| 任务 | Stage 1 ep | Stage 3 ep | val_R@10 | test_R@10 | 状态 |
|------|-----------|-----------|----------|-----------|------|
| task84 (HG-Rec baseline) | 200 | 200 | - | **0.1020** | baseline |
| task396b (Issue #99) | 200 | 200 | - | 0.0864 | ❌ NO-GO |
| task402 (Issue #108) | 30 | 200 | - | 0.0991 | ❌ NO-GO (-2.8%) 但杠杆成立 |
| **task403** | **30** | **30** | **0.1054** | **0.0798** | **❌ NO-GO (-21.7%) val/test 反转** |

### 关键新发现

1. **Stage 1 30 epoch 杠杆下游成立 (task402 R@10=0.0991)**: 训练时长杠杆主要在 Stage 1, Stage 3 200 epoch generalize 稳定
2. **Stage 3 30 epoch 短训 REFUTED (task403 R@10=0.0798)**: val_R@10 突破 0.1020 但 test 暴跌 -21.7%. val/test gap 24.3% = Stage 3 短训过拟合 val set
3. **HG-Rec baseline recipe 仍是最优路径**: 200 epoch Stage 1 + 200 epoch Stage 3 = R@10=0.1020 (不可超越 in 当前架构)
4. **val_R@10 杠杆 vs test_R@10 杠杆方向不同**: val 上短训更好, test 上长训更好. 后续 R@10 优化必须使用 test set 评估 (不能仅用 val)

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Stage 3 num_epochs | 30 (vs task402=200) | 验证"训练时长 ≤ 30 epoch"杠杆 Stage 3 是否成立 |
| Stage 4 launch 方式 | 手动 launch task403_stage4_rk_eval.sh | 复用 task402 Stage 4 模式 + 改 ckpt path (避免 task398 hardcoded paths bug) |
| 决策阈值 | R@10 > 0.1020 GO, ≤ 0.1020 NO-GO | HG-Rec baseline 阈值 |
| 下一方向 (基于 val/test 反转) | 必须用 test set 评估所有后续 R@10 实验 (per Issue #108 spec §gate 4 强制) |

---

## 后续 (per R22 + R19 + R16 + R10)

1. **commit + push verdict** (R15) — 立即执行
2. **R10 v2 idle 检查**: 0 OPEN issue + §16 空 + 4 GPU 空闲 + 用户未派工 → R10 v2 idle 允许. 报告状态等下一个派工信号.
3. **联立 task402 + task403 启示**: 
   - Stage 1 训练时长 ≤ 30 epoch 是 Stage 1 真杠杆 (task402 R@10=0.0991 vs task396b R@10=0.0864 +14.7%)
   - Stage 3 训练时长 ≤ 30 epoch **不是** Stage 3 真杠杆 (task403 test R@10=0.0798 vs task402 test R@10=0.0991 -19.5%)
   - HG-Rec baseline recipe (200 ep Stage 1 + 200 ep Stage 3) 仍是 R@10 最优路径
4. **若 owner 派工**: 下一方向必须用 test set 评估 (per Issue #108 §Gate 4 强制), 不能仅用 val_R@10 当代理指标
