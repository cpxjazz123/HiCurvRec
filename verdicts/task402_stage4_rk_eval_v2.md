# Task #402 / Stage 4 R@K eval (task401 ckpt 下游验证)

**日期**: 2026-07-31
**任务**: task402 Stage 4 R@K eval (task401 30-epoch Stage 1 ckpt + Sinkhorn Stage 2 + 200 epoch Stage 3 + Stage 4 R@K)
**结果**: ❌ **NO-GO 收口** (R@10 = 0.0991, vs HG-Rec baseline 0.1020, **-2.8%**), 但 task401 训练时长 ≤ 30 epoch 杠杆下游效应成立 (vs task396b R@10 = 0.0864, **+14.7%**)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ✅ PASS (per task401 verdict)

- **状态**: Pass
- **证据**: `verdicts/task401_issue108_best_epoch_verify_v2.md` (Issue #108 Gate 1 PASS)
- **关键数据**: L0/L1/L2 util = 100%/100%/100%, 4-digit SID unique 9920/9922 (3-digit collision 部分由 4-digit dedup 解决)

### Gate 2 (= Stage 2 Sinkhorn): ✅ PASS (per task402 Stage 2 evidence)

- **状态**: Pass
- **证据**: `products/task402_ckpt_downstream/stage2_evidence.json`
- **SID 文件**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task402.npy` (9922, 4)
- **关键数据**: Layer 0/1/2 util = 100%/100%/100%, 4-digit SID unique 9920/9922

### Gate 3 (= Stage 3 T5-mini 200 epoch): ✅ PASS (per task402 Stage 3 训练)

- **状态**: Training complete, exit code 0
- **ckpt**: `products/task402_ckpt_downstream/ckpt/Instruments/Jul-31-2026_20-06-14/HG_Rec_best.pth` (22087081 bytes, R12 强制落盘)
- **耗时**: ~80 min (20:06 → 21:27)
- **关键数据**: T5-mini 5.5M params, num_layers=6, num_decoder_layers=4, d_model=128, codebook_size=64/128/256/1, num_epochs=200, batch_size=256, lr=1e-4, seed=42, early_stop=20, beam_size=20, infer_size=96
- **实施**: `scripts/task84_hgrec_stage3_train.py` + Issue #97 patch (poincare_distance proj_to_ball)
- **ckpt 演变**:
  - epoch 0/9/72/86/92/99/.../199: 持续 val save best ckpt (R12 + early_stop=20)
  - 最终 best ckpt saved at epoch ~199

### Gate 4 (= Stage 4 R@K eval): ❌ **NO-GO 收口** (但 task401 训练时长杠杆下游成立)

- **关键数据**:
  - **Recall@5 = 0.0823** ✅ (vs HG-Rec baseline 0.0816, **+0.8%**)
  - **Recall@10 = 0.0991** ❌ (vs HG-Rec baseline 0.1020, **-2.8%**)
  - **Recall@20 = 0.1206** ❌ (vs HG-Rec baseline 0.1279, **-5.7%**)
  - **NDCG@5 = 0.0716** ✅ (vs HG-Rec baseline 0.0690, **+3.8%**)
  - **NDCG@10 = 0.0770** ✅ (vs HG-Rec baseline 0.0755, **+2.0%**)
  - **NDCG@20 = 0.0825** ✅ (vs HG-Rec baseline 0.0821, **+0.4%**)
- **失败根因**:
  1. **task401 训练时长 ≤ 30 epoch 是下游 Stage 3 R@10 部分杠杆**: task402 R@10=0.0991 vs task396b R@10=0.0864, **+14.7% 提升**
  2. **但仍略低于 HG-Rec baseline -2.8%**: 200 epoch Stage 3 可能稀释 task401 训练时长优势. Stage 1 30 epoch + Stage 3 200 epoch 组合仍有差距
  3. **NDCG 全部略优于 baseline** (+0.4% to +3.8%): 但 Recall 略低, 说明 ranking 质量好但 top-K 覆盖不够
- **跨方向联立**:
  | 任务 | Stage 1 训练时长 | Stage 3 训练时长 | R@10 | Δ vs HG-Rec |
  |------|-----------------|-----------------|------|-------------|
  | task84 baseline (HG-Rec) | 200 epoch | 200 epoch | 0.1020 | baseline |
  | task396b | 200 epoch (Issue #99) | 200 epoch | 0.0864 | -15.3% |
  | **task402** | **30 epoch (Issue #108 task401)** | **200 epoch** | **0.0991** | **-2.8%** |
  | 下一方向 task403 (待) | 30 epoch | 30 epoch (短训) | ? | ? |

- **决策阈值**: R@10 > 0.1020 GO, ≤ 0.1020 NO-GO → 0.0991 < 0.1020 → **NO-GO 收口** (vs baseline)
- **但杠杆成立**: R@10 较 task396b 提升 +14.7%, 验证 task401 训练时长杠杆下游传导有效

---

## 关键产物

- **commit hash**: ⏳ 待 commit 落地后写入 (R21 v2 强制, 不允许 pending)
- **push**: origin/main (R15 强制)
- **verdict 路径**: `verdicts/task402_stage4_rk_eval_v2.md` (本文件)
- **metrics 路径**: `verdicts/task402_stage4_rk_eval_metrics.json`
- **Stage 3 ckpt**: `products/task402_ckpt_downstream/ckpt/Instruments/Jul-31-2026_20-06-14/HG_Rec_best.pth`
- **整体决策**: ❌ **NO-GO vs baseline** (R@10 = 0.0991 vs 0.1020), 但 task401 训练时长 ≤ 30 epoch 杠杆下游成立 (+14.7% vs task396b)

---

## 跨方向联立 (R18 实证 22 方向 × 30 verdict)

| 任务 | Issue | 修复路径 | R@10 | 状态 |
|------|-------|---------|------|------|
| task84 | (HG-Rec baseline) | 200 epoch Stage 1 + 200 epoch Stage 3 | **0.1020** | **baseline** |
| task398 | #104 + #105 | Issue #99 + 200 epoch Stage 1 + 200 epoch Stage 3 | 0.0864 | ❌ NO-GO (-15.3%) |
| **task402** | **下游验证** | **task401 30 epoch Stage 1 + 200 epoch Stage 3** | **0.0991** | **❌ NO-GO (-2.8%) 但杠杆成立 (+14.7% vs task396b)** |

### 关键新发现

1. **训练时长 ≤ 30 epoch 是 Stage 1 真杠杆 (ep 1 → ep 30 避开退化带)**
2. **杠杆下游传导到 Stage 3 部分有效**: task402 (30 ep Stage 1) vs task396b (200 ep Stage 1) R@10 +14.7%
3. **但 200 epoch Stage 3 仍稀释杠杆**: R@10 仍略低于 HG-Rec baseline (-2.8%)
4. **下一方向 task403**: 30 epoch Stage 1 + 30 epoch Stage 3 短训, 看 R@10 是否突破 0.1020
   - 若 task403 R@10 > 0.1020 → "训练时长 ≤ 30 epoch" 是 Stage 1 + Stage 3 双杠杆
   - 若 task403 R@10 ≤ 0.1020 → 训练时长杠杆仅 Stage 1 真, 下游仍需 HG-Rec baseline recipe

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Stage 4 launch 方式 | 手动 launch task402_stage4_rk_eval.sh (wait helper 错误调用 task398 script) | task402 wait helper 跑 task398 script 实际评测 task396b (硬编码 paths), 必须独立 launch 真正 task402 Stage 4 |
| Stage 4 bug 修复 | `_extract_metrics` 函数兼容 dict/list/tuple | 之前 line 73 crash 因为 evaluate() 返回类型不一致 (dict vs list), 修复后 robust |
| 决策阈值 | R@10 > 0.1020 GO, ≤ 0.1020 NO-GO | 跟 task398 一致 (HG-Rec baseline R@10=0.1020) |
| 下一方向 | task403 (30 epoch Stage 1 + 30 epoch Stage 3) | 验证训练时长杠杆在 Stage 3 是否也成立 |

---

## 后续 (per R22 + R19 + R16)

1. **commit + push verdict** (R15)
2. **task403 (新方向)**: task401 Stage 1 30 epoch ckpt 复用 → Stage 2 Sinkhorn (复用 task402 SID) → Stage 3 T5-mini **30 epoch** 短训 → Stage 4 R@K eval. 验证"训练时长 ≤ 30 epoch"杠杆在 Stage 3 也成立.
3. **若 task403 R@10 > 0.1020**: 推翻"下游 T5 仍需 200 epoch"假设, 训练时长杠杆双 Stage 都成立
4. **若 task403 R@10 ≤ 0.1020**: HG-Rec baseline recipe 仍是 Stage 4 R@K 最优路径, task402 verdict 收口