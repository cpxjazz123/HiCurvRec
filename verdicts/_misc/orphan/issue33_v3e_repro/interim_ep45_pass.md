# Issue #33 v3e pureT5 复现 valid_R@10=0.1083 — interim PASS at ep45

**日期**: 2026-08-09  
**训练 PID**: 3436130 (单卡 Stage3 主脚本)  
**Status**: 训练中 (ep45/200, ES=20), ep45 **首次超过** 0.1083 用户目标

## 用户目标

`/loop 5m 帮我复现0.108的结果 must to do that you can chan ge the framework`  
- 0.108 = valid_R@10=0.1083 from `taskA_stage3_pureT5_4e5abe` ep10 (Issue #55/v2 2026-08-03)
- ckpt 已丢失, 需重训
- 用户授权改 framework (R31 解除)

## 配置 (Issue #33 v3e, 用户授权改 framework)

| 参数 | v3e 值 | pureT5_4e5abe 原值 | v77 路径值 |
|------|--------|-------------------|-----------|
| GPU | 单卡 (cuda:0) | 单卡 | DDP 4 卡 |
| BATCH_SIZE | 256 | 256 | 1024 |
| LR | 1e-4 | 1e-4 | 4e-4 |
| NUM_EPOCHS | 200 | 200 | 200 |
| EARLY_STOP | 20 | 20 | 10 |
| NUM_WORKERS | 0 | 0 | 2 |
| INFER_SIZE | 96 | 96 | 256 |
| EVAL_INTERVAL | 5 | 5 | 5 |
| SID | hyp_v2 capmatch (sha=06af0fed) | v3e poincare (sha=5c058531, lost) | hyp_v2 capmatch (sha=06af0fed) |

**关键差异**: v3e 用 hyp_v2 SID 替换丢失的 v3e poincare SID. hyp_v2 是当前 v85 路径 SOTA (test_R@10=0.1031).

## 训练轨迹 (ep5→ep45)

| epoch | valid_R@10 | ΔR@10 | train_loss | NDCG@20 |
|-------|-----------|-------|-----------|---------|
| 5 | 0.0637 | — | 4.6628 | 0.0360 |
| 10 | 0.0726 | +0.0089 | 3.8945 | 0.0532 |
| 15 | 0.0857 | +0.0131 | 3.6083 | 0.0685 |
| 20 | 0.0959 | +0.0102 | 3.4575 | 0.0755 |
| 25 | 0.1006 | +0.0047 | 3.3536 | 0.0792 |
| 30 | 0.1021 | +0.0015 | 3.2770 | 0.0807 |
| 35 | 0.1040 | +0.0019 | 3.2135 | 0.0820 |
| 40 | 0.1062 | +0.0022 | 3.1608 | 0.0834 |
| **45** | **0.1087** | **+0.0025** | **3.1167** | **0.0850** |

## 4 Gate 验收

### Gate 1 (训练 Setup)
✅ PASS — 单卡 batch=256 lr=1e-4 ES=20 NUM_WORKERS=0 INFER_SIZE=96 严格按 Issue #55/v2 pureT5_4e5abe 配置 (用户授权 framework 改动: 替换 SID 为 hyp_v2). 数据加载 OK (131837 train, 24772 valid). T5 5.5M 参数.

### Gate 2 (训练健康)
✅ PASS — loss 单调下降 (ep1=6.42 → ep45=3.12), valid_R@10 单调上升 (0.0637 → 0.1087), 无 NaN/Inf, 无 wrapper broken, GPU 0 25-35% util 3.9GB, 单 epoch 43-45s 稳定.

### Gate 3 (用户目标达成)
✅ **PASS at ep45** — valid_R@10=**0.1087** > 0.1083 用户目标 (+0.0004). NDCG@20=0.0850. 训练仍继续 (ep50-200 ES=20), 后续可能继续上升至 0.11+.

### Gate 4 (test_R@10)
⏳ **待 Stage4 eval** — 训练完成后用 `taskA/stage4/taskA_stage4.py` (或新建 `taskA_stage4_pureT5_v3e.py`) 跑 Stage4 eval 出 test_R@10. 估算基于 v85 hyp_v2 路径 test_R@10=0.1031 + 单卡 small-batch regularization, 估 test_R@10 ≈ 0.090-0.095 (valid/test ratio ~1.18, 跟 pureT5_4e5abe 历史 0.1015 类似).

## 改动文件

1. `/fs04/ar57/wenyu/GeneRec/common/stage3/stage3_train_pureT5_v3e.py` (新建, fork 自 stage3_train_pure_t5.py)
   - 顶部 CONSTANTS: NUM_EPOCHS=200, EARLY_STOP=20, BATCH_SIZE=256, INFER_SIZE=96, LR=1e-4, NUM_WORKERS=0
   - docstring + 注释更新: Issue #33 v3e + 复现 pureT5_4e5abe
2. `/fs04/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_pureT5_v3e.py` (新建)
   - 顶部 V3E_CONFIG 硬编码: SID=hyp_v2 (sha=06af0fed), product_dir=stage3_pureT5_v3e_hyp_v2
   - 单卡 CUDA_VISIBLE_DEVICES=0 (无 torchrun)
   - HAB 关 (False, False) — pureT5 路径无 HAB
3. `/home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_hyp_v2/` (新产物目录)
   - HG_Rec_best.pth (ep45 best valid_R@10=0.1087)
   - train_pure_t5.log (完整 trace)
   - run.log (wrapper 启动日志)
   - _TRAINING_PID (PID 3436130)

## 已知与目标差异

| 维度 | v3e vs pureT5_4e5abe |
|------|---------------------|
| SID | hyp_v2 capmatch (sha=06af0fed) vs v3e poincare (sha=5c058531, lost) |
| 目标数字 | valid_R@10=0.1087 vs 0.1083 (+0.0004, **达到**) |
| 训练 epoch 达峰 | ep45 vs ep10-31 (pureT5_4e5abe 历史 0.1083@ep10, 0.1197@ep31) |
| 训练时延 | 单卡 43-45s/ep vs 原 30s/ep (bf16 + 4 卡 DDP vs 单卡) |

## 后续

1. 让训练继续 (ep50-200), 看能否突破 0.110
2. 训练完成后跑 Stage4 eval 出 test_R@10, 写最终 verdict
3. commit + push (R15) + 关闭 Issue #33 internal task

## Why & How to apply

**Why**: 用户持续触发"复现 0.108", 我前 12 轮误以为不存在 (实际 = valid_R@10=0.1083 from pureT5_4e5abe), 后用户授权改 framework, 立即 fork Stage3 主脚本 + 启动单卡重训, ep45 达 0.1087.

**How to apply**: 用户再说"复现 0.108"时, 直接理解为 valid_R@10=0.1083 目标, 用 v3e 配置 (单卡 batch=256 lr=1e-4 + hyp_v2 SID) 可复现. 真实本环境 SOTA test_R@10=0.1031 (我们 v4 ckpt), pureT5_4e5abe 实际 test_R@10=0.0956 (valid/test ratio 1.181) — 0.108 valid ≠ 0.108 test.