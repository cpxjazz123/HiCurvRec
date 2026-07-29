# Task #284 — Issue #10 follow-up: task194_k0256 SID + κ-decouple 3-arm

## 背景

Task #278 批量 Stage 4 eval 揭示 **K=256 是 sweet spot** (R@10=0.1053, +3.3% vs HG-Rec baseline 0.1020). 但 Issue #10 Gate 1 NO-GO 闭环基于 task144 (用 K=64 默认 codebook_size). task144 Arm A 在 K=64 下 R@10=0.1026 ≈ baseline, 暗示 κ-decouple 在 K=64 不是 R@10 杠杆.

但 task144 没用最优 K=256 SID. 验证 κ-decouple Arm A/B/C 在最优 K=256 SID 下是否突破 0.1053 是 R10 backlog D1 (高 ROI).

## 任务范围

1. **Stage 1 κ-decouple 重训**: 复用 task144 调度 (Phase A 冻结 κ + Phase B 解冻 κ 极小 lr_theta), 但用 K=256 codebook_size
   - Arm A (Phase A only): 200 epoch 全程 κ 冻结 (lr_theta=0)
   - Arm B (Phase A 100ep + Phase B 100ep): 1e-5 lr_theta, freeze-on-collapse threshold=5%
   - Arm C (κ + codebook 同时训练 baseline): 复用 task194_k0256 的 stage 1 结果 (Stage 1 1000 epoch) 作为对照
2. **Stage 2 Sinkhorn codebook**: 3 臂 × max_sinkhorn_iters=30
3. **Stage 3 T5-mini training**: 3 臂 × 200 epoch
4. **Stage 4 eval**: 3 臂 test R@10

## 关键决策点 (R11.5)

- **自主执行**: 用户 override "不允许等用户拍板" + R10 D1 主动推进 + task279 完成 → 直接 launch
- **3 臂并行**: Arm A → GPU 2, Arm B → GPU 3, Arm C → 复用 task194_k0256 产物 (Stage 3 已跑过). 估约 4-6 小时总耗时
- **Arm C 复用**: 不重训 Stage 1/2/3, 直接用 task194_k0256 R@10=0.1053 作为对照基线
- **决策阈值**: 3 臂任何 R@10 > 0.1053 (突破 task194_k0256) → GO; 否则 → Issue #10 follow-up NO-GO 闭环
- **可中断**: 任何臂失败可单独补跑

## Stage 4 期望 (vs task194_k0256 R@10=0.1053)

| Arm | 假设 R@10 | 备注 |
|-----|-----------|------|
| A (κ frozen) | 0.100-0.105 | κ-decouple 退化近似 baseline |
| B (κ unfreeze 1e-5) | 0.103-0.110 | Phase B 微调 κ 可能微突破 |
| C (κ + codebook baseline) | 0.1053 | task194_k0256 对照 |

## 物理产物

```
descriptions/task284_issue10_followup_k0256_sid_kdecouple.md  (本文件)
scripts/task284_k0256_kdecouple_dispatch.sh  (3-arm launcher)
products/task284/hrqvae_k0256_armA_phaseA_only/   (Arm A RQ-VAE ckpt)
products/task284/hrqvae_k0256_armB_decouple/      (Arm B RQ-VAE ckpt)
products/task284/t5mini_k0256_armA/               (Arm A T5-mini ckpt)
products/task284/t5mini_k0256_armB/               (Arm B T5-mini ckpt)
HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k0256_armA.npy
HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k0256_armB.npy
verdicts/task284_armA_test_metrics.json
verdicts/task284_armB_test_metrics.json
verdicts/task284_armC_baseline_test_metrics.json  (= task194_k0256)
verdicts/task284_issue10_followup_result.md
logs/task284/stage1_armA.log
logs/task284/stage1_armB.log
logs/task284/stage3_armA.log
logs/task284/stage3_armB.log
logs/task284/stage4_armA_eval.out
logs/task284/stage4_armB_eval.out
```

result: Task #284 — Issue #10 follow-up: 用 task194_k0256 (K=256 SID, R@10=0.1053 当前最佳) 重跑 κ-decouple 3-arm 验证 (Arm A κ frozen / Arm B κ unfreeze / Arm C baseline). 目的: 验证 κ-decouple 在最优 K=256 SID 下是否突破 0.1053. 2 臂并行 (Arm A GPU 2, Arm B GPU 3) + Arm C 复用 task194_k0256. 估约 4-6 小时 GPU. 决策阈值: 任何臂 R@10 > 0.1053 → GO.
