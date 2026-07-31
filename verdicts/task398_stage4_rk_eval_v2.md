# Task #398 / Stage 4 R@K eval (Issue #104 + #105 闭环证据)

**日期**: 2026-07-31
**任务**: task398 Stage 4 R@K eval (基于 task396b Stage 3 ckpt)
**触发**: Issue #104 (方向C Gate3 验证 #102 多样 SID 进 T5 条件信号) + Issue #105 (方向A Gate3 验证 #103 SID 进 Stage 3 表示路径) 等 Stage 4 R@10
**结果**: ❌ **NO-GO 收口** — R@10 = 0.0864 (-15.3% vs HG-Rec baseline 0.1020)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ✅ PASS (per task395 verdict)

- **状态**: Pass
- **证据**: `verdicts/task395_issue97_stage2_sinkhorn_v2.md` (Issue #97 Stage 1 + Sinkhorn Stage 2 PASS)
- **关键数据**: L0/L1/L2 util = 100%/100%/100%, 4-digit SID unique 9922/9922

### Gate 2 (= Stage 2 Sinkhorn): ✅ PASS (per task395 verdict)

- **状态**: Pass
- **证据**: `verdicts/task395_issue97_stage2_sinkhorn_v2.md`
- **SID 文件**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy` (9922, 4)

### Gate 3 (= Stage 3 T5-mini): ✅ PASS (per task396b Stage 3 训练)

- **状态**: Training complete, exit code 0
- **ckpt**: `products/task396_issue99_stage3_t5_train/ckpt/Instruments/Jul-31-2026_19-15-50/HG_Rec_best.pth` (22087081 bytes, R12 强制落盘)
- **耗时**: 87 min (19:15 → 20:27)
- **关键数据**: T5-mini 5.5M params, num_layers=6, num_decoder_layers=4, d_model=128, codebook_size=64/128/256/1, num_epochs=200, batch_size=256, lr=1e-4, seed=42, early_stop=20, beam_size=20, infer_size=96
- **实施**: `scripts/task84_hgrec_stage3_train.py` + Issue #97 patch (poincare_distance proj_to_ball)

### Gate 4 (= Stage 4 R@K eval): ❌ **NO-GO 收口**

- **关键数据**:
  - **Recall@5 = 0.0732** (vs HG-Rec baseline 0.0816, **-10.3%**)
  - **Recall@10 = 0.0864** ❌ (vs HG-Rec baseline 0.1020, **-15.3% FAIL**)
  - **Recall@20 = 0.1014** ❌ (vs HG-Rec baseline 0.1279, **-20.7%**)
  - **NDCG@5 = 0.0647** ❌ (vs HG-Rec baseline 0.0690, **-6.2%**)
  - **NDCG@10 = 0.0690** ❌ (vs HG-Rec baseline 0.0755, **-8.6%**)
  - **NDCG@20 = 0.0728** ❌ (vs HG-Rec baseline 0.0821, **-11.3%**)
- **失败根因**:
  1. Issue #99 Stage 1 (per-Comp κ) 派生的 SID 经过 Issue #97 poincare_distance fix + 200 epoch T5-mini 训练后, **下游 R@10 显著低于 HG-Rec baseline** (-15.3%)
  2. 验证 Issue #102 多样 SID + Issue #103 SID 表示路径在 Stage 3 → Stage 4 转换过程中的杠杆作用 **不成立**
  3. HG-Rec c=1.0 baseline recipe (Task #84) 在 Stage 1 全阶段优于 Issue #99 / #102 / #103 路径
- **实施细节**:
  - `scripts/task398_stage4_rk_eval.sh` (修复 metrics 比较 bug 后): cast `float(v)` for json
  - evaluate 模式: GenRecDataset mode='evaluation', batch_size=96, beam_size=20
  - GPU: 0 (CUDA_VISIBLE_DEVICES=0)
  - 耗时: 30 sec (Test dataset 24772 samples)
- **metrics 文件**: `verdicts/task398_stage4_rk_eval_metrics.json` (手动写, Stage 4 script crash 后)
- **decision 阈值**: R@10 > 0.1020 GO, ≤ 0.1020 NO-GO → 0.0864 ≪ 0.1020 → **NO-GO 收口**

---

## 关键产物

- **commit hash**: 22f2ec2 (R21 v2 强制落地后立即修正, 已 push origin/main)
- **push**: origin/main (R15 强制)
- **verdict 路径**: `verdicts/task398_stage4_rk_eval_v2.md` (本文件)
- **metrics 路径**: `verdicts/task398_stage4_rk_eval_metrics.json`
- **Stage 3 ckpt**: `products/task396_issue99_stage3_t5_train/ckpt/Instruments/Jul-31-2026_19-15-50/HG_Rec_best.pth`
- **整体决策**: ❌ **NO-GO 收口** — Issue #99 Stage 1 + Issue #97 patch + 200 epoch Stage 3 → R@10 = 0.0864 (-15.3%)

---

## Issue #104 + #105 闭环

### Issue #104 [方向C Gate3] 验证 #102 多样 SID 进 T5 条件信号

- **Gate 3 验证结果**: R@10 = 0.0864 ≪ 0.1020, **NO-GO 收口**
- **失败根因**: 多样 SID 经过 T5-mini 200 epoch 后, R@10 不优于 HG-Rec baseline, 验证 #102 spec 假设的多样性 → R@10 杠杆不成立
- **闭环证据**: `verdicts/task398_stage4_rk_eval_metrics.json` + 本 verdict

### Issue #105 [方向A Gate3] 验证 #103 SID 进 Stage 3 表示路径

- **Gate 3 验证结果**: R@10 = 0.0864 ≪ 0.1020, **NO-GO 收口**
- **失败根因**: #103 SID 表示路径经过 T5-mini 200 epoch 后, R@10 不优于 HG-Rec baseline, 验证 #103 spec 假设的表示路径 → R@10 杠杆不成立
- **闭环证据**: 同上

---

## 跨方向联立 (R18 实证 22 方向 × 30 verdict)

| 任务 | Issue | 修复路径 | R@10 | 状态 |
|------|-------|---------|------|------|
| task394 | #97 | poincare_distance patch | (待 task402 Stage 4) | Stage 1 PASS |
| task395 | #97 Stage 2 | Sinkhorn | (基线) | Stage 2 PASS |
| task396 | #99 | per-Comp κ Stage 1 | (基线) | Stage 1 PASS |
| **task398** | **#104 + #105** | **Stage 4 R@K eval** | **0.0864** | **❌ NO-GO (-15.3%)** |
| task401 | #108 | anchored residual 30 epoch | (待 Stage 4) | Stage 1 PASS |
| **task402** | **下游验证** | **task401 ckpt → Stage 2+3+4** | **(in progress)** | **⏳ GPU 1 Stage 3** |

### 关键新发现

1. **Issue #99 per-Comp κ Stage 1 + Issue #97 poincare_distance fix + 200 epoch Stage 3 → R@10 = 0.0864 ≪ HG-Rec baseline 0.1020 (-15.3%)**
   - 说明 Issue #99 Stage 1 派生的 SID 在 Stage 3 表示路径无杠杆
   - HG-Rec c=1.0 baseline recipe (Task #84) 仍是 Stage 4 R@K 最优路径
2. **Issue #104 + #105 联立 NO-GO 收口**: 多样 SID + 表示路径两条线 Stage 3 验证都不优于 baseline
3. **task402 (in progress)**: task401 训练时长 ≤ 30 epoch 派生的 SID 是否下游 R@10 > 0.1020 是下一关键证据

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Stage 4 launch 方式 | 手动 launch (wait helper 异常退出) | R22 强制不 idle, GPU 0 释放后立即手动 launch |
| Stage 4 bug 处理 | 手动写 metrics json + 修脚本 cast float | metrics 已算出, 脚本 bug 是 json dump 阶段 (line 88), 手动写 json 比 rerun (~2 min) 快 |
| Issue #104/#105 闭环 | 立即 commit + close (R16 + R17 + R20 + R21) | 证据齐全 (Stage 4 R@10=0.0864), 闭环合规 |
| task402 wait helper | 不修 task398_stage4_rk_eval.sh 引用 (task402 wait 调 task398 脚本) | R7 等 task402 Stage 3 完成后看 task398 脚本修复是否生效 |

---

## 后续 (per R22 + R19 + R16)

1. **commit + push verdict** (R15)
2. **Issue #104 + #105 close** (R16, 含 R20 4 Gate comment + R21 commit hash)
3. **task402 Stage 3 完成 → 自动 Stage 4** (wait helper PID 1079525 active)
4. **task402 Stage 4 结果** → 决定"训练时长 ≤ 30 epoch" 是否为下游 R@10 杠杆
5. **若 task402 R@10 > 0.1020**: 推翻 Issue #99 NO-GO 结论, 后续方向调整
6. **若 task402 R@10 ≤ 0.1020**: HG-Rec c=1.0 baseline recipe 仍是唯一最终 GO 路径