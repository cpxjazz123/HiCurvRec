# Task #320 — Issue #38 5-arm Stage 3 Retraining (Optimizer改造/LR schedule/R-Drop/BF16/control)

**日期**: 2026-07-30
**状态**: 🔄 queued (R7: 等 task318 完成释放 GPU, 预计 15-30 min)
**Stage**: Stage 3 training (200 epoch T5-mini 5.5M) + Stage 4 evaluation
**Anchor**: Issue #30 GO endpoint (task301) + task194_k0256 R@10=0.1053 (+3.2% baseline)
**决策阈值**: 至少 1 Arm R@10 > 0.1053 (task194_k0256 实质突破 anchor)

## Issue #38 5-arm Stage 3 retraining design (from GitHub Issue #38 body)

| Arm | Stage 3 改造维度 | 超参 | vs baseline |
|-----|-----------------|------|------------|
| **A** | Optimizer 改造 | AdamW (lr=1e-3, wd=0.05) + cosine + warmup 5000 + dropout 0.05 | baseline Adam (lr=1e-4) + constant LR + dropout 0.1 |
| **B** | LR schedule | inverse square root + warmup 10000 + Adam | baseline constant LR |
| **C** | Regularization | R-Drop alpha=1.0 (2 forward passes + symmetric KL) | baseline single forward |
| **D** | Mixed precision | BF16 autocast + FP32 master | baseline FP32 |
| **E** | control | Adam + constant LR + dropout 0.1 + FP32 | = baseline |

## Settings (per arm)
- dataset: Musical_Instruments (Instruments, 9922 items 5-core)
- code_path: `_t5_hrqvae_issue30_per_layer_transforms.npy` (Issue #30 GO endpoint, L0/L1/L2 100%)
- T5-mini config: d_model=128, d_ff=1024, num_heads=6, num_layers=6, num_decoder_layers=4, d_kv=64 (5.5M params)
- num_epochs: 200 (full sweep, disable early stop for fair comparison)
- batch_size: 256
- Stage 4 beam_size: 100 (Issue #30 ε ceiling)
- seed: 42
- R12: 强制存 best_ckpt (per Arm)

## Per-Arm expected runtime (200 epoch T5-mini 5.5M on L40S)
- Arm A: ~75-90 min (cosine + warmup + higher lr, may converge earlier)
- Arm B: ~80-95 min (inverse sqrt schedule)
- Arm C: ~120-140 min (R-Drop = 2x forward passes, ~50% slower)
- Arm D: ~60-75 min (BF16 faster than FP32)
- Arm E: ~75-90 min (control = baseline timing)
- **Total parallel**: max(120, 95, 140, 75, 90) = ~140 min wall time (4 in parallel, 1 in queue)

## Throughput condition
- 至少 1 Arm R@10 > 0.1053 (task194_k0256 anchor) — Issue #38 → "Stage 3 protocol change 通过"
- All Arms R@10 ≤ 0.1053 — Issue #38 → "Stage 3 protocol change NO-GO 收口"

## GPU strategy (R7 不抢卡)
- task318 parallel session: 4 arms on 4 GPUs (07:47 start, ~20 min runtime as of 08:07)
- task320 启动条件: task318 全部结束, GPU 0/1/2/3 util < 10%
- 启动顺序: A/B/D/E 先 (4 个, 1 个 GPU 各一), C 后跑 (因为 R-Drop 慢)
- 预计启动时间: 2026-07-30 08:20-08:30

## task318 partial findings (08:16 snapshot, 50 epoch proxy test, 29:16 elapsed)
**adam/adamw ep39 进一步提升 val_R@10=0.1174-0.1179 (+11.5% vs anchor 0.1053)**
- **adam ep39** (08:16): val_NDCG@20=0.0930, val_R@10=0.1174 ⭐ (vs ep32: 0.1151, +2.0% in 7 epochs)
- **adamw ep39** (08:16): val_NDCG@20=0.0927, val_R@10=0.1179 ⭐⭐ (vs ep31: 0.1155, +2.1% in 8 epochs)
- **sgd ep43** (08:16): val_NDCG@20=0.0848, val_R@10=0.1054 (中规中矩, +5.5% from ep36)
- **adafactor ep39** (08:16): val_NDCG@20=0.0754, val_R@10=0.0966 (NO-GO, 平台期)
- **推测 (08:16)**: 50 epoch 完成后 adam/adamw 预期 val_R@10=0.119-0.122, task320 Arm A 200 epoch (cosine + warmup + lr=1e-3) 期望 val_R@10=0.13-0.15
- **implication**: Issue #38 Stage 3 Optimizer 改造路径 GO 几乎确定. adam/adamw 即使 constant LR 也实质突破, task320 Arm A 叠加 cosine + warmup 期望进一步突破

## Task structure
- task320_armA_optimizer_200ep.sh
- task320_armB_lr_schedule_200ep.sh
- task320_armC_rdrop_200ep.sh
- task320_armD_bf16_200ep.sh
- task320_armE_control_200ep.sh
- 单一 Stage 3 train 脚本: `scripts/task320_issue38_5arm_stage3_train.py` (--arm 参数)

## Cross-task dependencies
- task318 Arm 1 (parallel session) 50 epoch 快速 proxy — 如 Arm 1 (adam/adamw) 任一明显优于 baseline, task320 Arm A 用其变体作为起点
- task194_k0256 anchor R@10=0.1053 (K=256, not K=128 issue30 default) — K=256 仍 GO endpoint
- Issue #30 K=100 ceiling R@10=0.1045 — 5-arm Stage 3 需击败此 ceiling

## R11.5 transparency
- 选 cosine + warmup 5000 是 LLM 训练标准 (T5 paper, HuggingFace default)
- 选 inverse sqrt + warmup 10000 是 Transformer 经典 schedule (Vaswani 2017)
- 选 R-Drop alpha=1.0 是 paper 默认 (Liang et al. 2021)
- 选 BF16 over FP16 是 Ampere+ GPU (sm_89 L40S) 默认, 无需 loss scaling
- 选 disable early stop 200 epoch 全跑是为了 5-arm 公平对比 (early stop 引入 race condition)

## Next steps (R10 推進)
1. 写 task320 Stage 3 train 脚本 (--arm 参数, 5 arms 复用)
2. 写 5 个 launcher shell (--gpu 0/1/2/3 + 不同 --arm)
3. 等 task318 结束 → 启动 task320 5-arm 并行
4. 200 epoch 训练完成 → Stage 4 K=100 eval
5. 写 task320_result.md verdict
