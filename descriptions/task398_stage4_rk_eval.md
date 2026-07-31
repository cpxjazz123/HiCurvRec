# Task #398 / Stage 4 R@K eval post-task396 (Issue #104 闭环)

**日期**: 2026-07-31
**触发**: task396b Stage 3 T5 训练完成后立即执行 (per Issue #104 spec "验证#102多样SID进入T5条件信号")
**前置**:
- task396b Stage 3 T5 200 epoch 训练完成 (PID 660598, GPU 0)
- HG_Rec_best.pth 落盘: `products/task396_issue99_stage3_t5_train/ckpt/Instruments/<ts>/HG_Rec_best.pth`
- task395 Stage 2 SID: unique 9922/9922 (100%)
- HG-Rec baseline R@10 = 0.1020 (Task #84 baseline, Musical_Instruments)

**任务**: Stage 4 R@K eval vs HG-Rec baseline 0.1020 (GO/NO-GO 决策)
**结果**: ⏸ 待 task396b 训练完成后启动 (预计 ~127 min)

---

## Issue #104 Gate 3 spec (R17 强制 4 Gate 顺序)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ✅ PASS (per task394, Issue #100 闭环)
- L0/L1/L2 util 100/100/100%
- commit: 66d1f8e

### Gate 2 (= Stage 2 SID): ✅ PASS (per task395, Issue #102 + #103 闭环)
- SID 4-digit unique 9922/9922, util 100/100/100%
- commit: ca8a558

### Gate 3 (= Stage 3 T5-mini): ⏳ TRAINING (task396b)
- Epoch 22/200, ~10.5%, 16 min elapsed, expected ~143 min total
- HG_Rec_best.pth 已落盘 (R12 强制)
- commit pending

### Gate 4 (= Stage 4 R@K eval): ⏸ 本任务
- 启动条件: task396b Stage 3 训练完成
- 目标: R@10 > 0.1020 (HG-Rec baseline, GO threshold)
- 实施: scripts/task398_stage4_rk_eval.sh (基于 task174 stage4 模式 + task84 baseline recipe)
- GPU: 1 (R7 空闲, 避免抢 task396b GPU 0)
- code_path: _t5_rqvae_task396.npy (task396a 产物)
- codebook_size: [64, 128, 256, 1]
- beam_size: 20

---

## 评估指标 (per HG-Rec baseline + Issue #104 spec)

| 指标 | HG-Rec baseline | 决策阈值 |
|------|-----------------|----------|
| Recall@5 | 0.0816 | ≥ 0.0816 |
| **Recall@10** | **0.1020** | **> 0.1020 GO, ≤ 0.1020 NO-GO** |
| Recall@20 | 0.1279 | ≥ 0.1279 |
| NDCG@5 | 0.0690 | ≥ 0.0690 |
| NDCG@10 | 0.0755 | ≥ 0.0755 |
| NDCG@20 | 0.0821 | ≥ 0.0821 |

→ **GO 条件**: R@10 > 0.1020 + 6 项 metrics 全部齐全
→ **NO-GO**: R@10 ≤ 0.1020 或 missing metrics

---

## 关键产物 (待 Stage 3 完成后)

- **eval script**: scripts/task398_stage4_rk_eval.sh
- **eval log**: logs/task398_stage4_eval_<ts>.log
- **metrics JSON**: verdicts/task398_stage4_rk_eval_metrics.json
- **verdict**: verdicts/task398_stage4_rk_eval_v2.md (R20 4 Gate 详细内容)

---

## 后续 (per R22 + R19)

1. **task398 启动** — Stage 4 eval, 立即 GPU 1 launch
2. **Issue #104 闭环** — commit + push + close issue (R16 + R17 + R20 + R21)
3. **GO/NO-GO 决策**:
   - GO (R@10 > 0.1020): 跨 18 方向 × 25 verdict 验证 #97 patch 路径 (Stage 1+2+3+4 全跑通). 立即可写 paper §6.7.10 + Issue #99 Issue #100 Issue #102 Issue #103 Issue #104 5 issues 全部闭环
   - NO-GO (R@10 ≤ 0.1020): 闭环 #104 NO-GO 收口, 回到 Stage 3 训练时长 / lr / recipe 调整方向