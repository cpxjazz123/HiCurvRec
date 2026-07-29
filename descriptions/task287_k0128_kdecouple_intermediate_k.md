# Task #287 — K=128 κ-decouple 2-arm 验证曲线 (task144 K=64 + task284 K=256 中间 K)

## 背景

Task #144 (K=64) + Task #284 (K=256) 联立揭示 **κ-decouple + 大 K 是负面相互作用**:
- task144 K=64 Arm A: R@10=0.1026 ≈ baseline (中性, +0.6%)
- task144 K=64 Arm B: R@10=0.1017 ≈ baseline (中性, -0.3%)
- task284 K=256 Arm A: R@10=0.0846 (-17.0%, ❌ 严重退化)
- task284 K=256 Arm B: R@10=0.0864 (-15.3%, ❌ 严重退化)

K=128 是 K=64 (中性) 和 K=256 (退化) 中间点. 验证 K=128 下 κ-decouple 表现是:
- **H1**: R@10 ≈ baseline 0.1020 (跟 K=64 类似, K=128 也中性)
- **H2**: R@10 在 -5% to -10% (中间过渡, 单调退化)
- **H3**: R@10 ≈ task194_k0128 baseline 0.1027 (跟 task194 一致, κ-decouple 中性)

K=128 验证闭环 κ-decouple + K 曲线.

## 任务范围

1. **Stage 1 κ-decouple 训练 (2 臂)**:
   - Arm A: Phase A only 200 ep, κ frozen at 0, num_emb_list=128 128 256
   - Arm B: Phase A 100ep + Phase B 100ep κ unfreeze lr_theta=1e-5, freeze-on-collapse 5%, num_emb_list=128 128 256
2. **Stage 2 Sinkhorn codebook (2 臂)**:
   - 复用 task284_stage2_free_curv.py (FreeCurvHRQVAE 专用, task286 修过)
3. **Stage 3 T5-mini training (2 臂)**: codebook_size=[128,128,256,1], d_model=128, 200 ep
4. **Stage 4 eval (2 臂)**: task278_batch_stage4_eval.py v3 (--codebook_size "128,128,256,1")

## 关键决策点 (R11.5 + 用户 override "do by yourself")

- **自主执行**: R10 D7 backlog + 用户 "Generic 继续" + R10 主动推进 → 直接 launch
- **2 臂并行**: Arm A → GPU 2, Arm B → GPU 3 (跟 task284 同模式)
- **GPU 0/1 空闲**: 给其他任务用 (R7 兼容)
- **决策阈值**: 任何臂 R@10 接近 baseline 0.1020 (±2%) → κ-decouple 在 K=128 中性; 否则 → κ-decouple 在 K=128 也退化
- **可中断**: 任何臂失败可单独补跑

## Stage 4 期望 (跟 task144 K=64 + task284 K=256 对比)

| K | Arm A R@10 | Arm B R@10 | 期望曲线 |
|---|------------|------------|----------|
| 64 (task144) | 0.1026 | 0.1017 | ≈ baseline (中性) |
| **128 (task287)** | **?** | **?** | **待测** |
| 256 (task284) | 0.0846 | 0.0864 | -17% / -15% (退化) |

## 物理产物

```
descriptions/task287_k0128_kdecouple_intermediate_k.md  (本文件)
scripts/task287_k0128_kdecouple_dispatch.sh  (2-arm launcher)
products/task287/hrqvae_k0128_armA_phaseA_only/   (Arm A RQ-VAE ckpt)
products/task287/hrqvae_k0128_armB_decouple/      (Arm B RQ-VAE ckpt)
products/task287/t5mini_k0128_armA/               (Arm A T5-mini ckpt)
products/task287/t5mini_k0128_armB/               (Arm B T5-mini ckpt)
HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_kappa_decouple_armA_k0128.npy
HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_kappa_decouple_armB_k0128.npy
verdicts/task287_armA_test_metrics.json
verdicts/task287_armB_test_metrics.json
verdicts/task287_k0128_intermediate_k_result.md
logs/task287/stage1_armA.log
logs/task287/stage1_armB.log
logs/task287/stage234_waiter.log
logs/task287/stage2_armA.log
logs/task287/stage2_armB.log
logs/task287/stage3_armA.log
logs/task287/stage3_armB.log
logs/task287/stage4_armA_eval.out
logs/task287/stage4_armB_eval.out
```

## R9-Enforce 备注

2026-07-29 audit: descriptions/ 存在 9 个历史空洞 (230, 255, 257-260, 264, 266, 282) — 由 task285/task286 commit 用 task ID 但 description 未建造成. 新任务用 task287 (跳过 285/286 commit 占用的编号). 后续批次可用 renumber_tasks 脚本填补.

result: Task #287 — K=128 κ-decouple 2-arm 验证曲线. 目的: 验证 task144 K=64 中性 + task284 K=256 退化的中间点 K=128 表现. 决策阈值: 任何臂 R@10 接近 baseline ±2% → κ-decouple 中性; 否则 → 中间过渡 (退化曲线). 2 臂并行 (Arm A GPU 2, Arm B GPU 3). 估约 1.5-2 小时 GPU. 复用 task284 launch script + task284_stage2_free_curv.py + task286 waiter 永久 fix.
