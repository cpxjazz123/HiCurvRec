# Task #242 / Issue #11 Gate 1 (Arm A) — NO-GO

**日期**: 2026-07-29
**状态**: Gate 1 FAIL → 进入 Gate 1b (Arm A+ dead-codeword revival)

## 配置

| 项 | 值 |
|----|-----|
| 配置 | Task #222 baseline + `--c_k_range_list "1:5,0.5:20,0.5:20"` |
| L0 c_k | U(1.0, 5.0) (Phase 0 OPEN 82.68%) |
| L1 c_k | U(0.5, 20.0) (Phase 0 OPEN 67.15%) |
| L2 c_k | U(0.5, 20.0) (Phase 0 OPEN 75.69%) |
| Stage 1 ckpt | 新跑 (40 epoch 早停协议) |
| Gate 1 通过条件 (Issue #11) | L0 util ≥ 90% **AND** collision ≤ 0.3706 |
| GPU | 0 (4 卡空闲) |
| Wall clock | 32.5 s (10 batches × 40 epoch × 0.32 s/batch) |

## Gate 1 数值 (best_collision_model.pth, epoch 14 collision=0.8938)

| Layer | K | Used | Utilization | 通过 (≥90%)? |
|-------|---|------|-------------|---------------|
| L0 | 64 | 15 | **23.44%** | ❌ FAIL (-66.56pp) |
| L1 | 128 | 59 | 46.09% | ❌ FAIL |
| L2 | 256 | 98 | 38.28% | ❌ FAIL |

| 指标 | 实测 | 阈值 | 通过? |
|------|------|------|------|
| collision_rate | **0.9385** (task223 口径, raw argmin) | 0.3706 | ❌ FAIL (+0.5679) |
| best_collision (训练日志, 含 Sinkhorn) | 0.8938 (epoch 14) | 0.3706 | ❌ FAIL |

**两条 Gate 1 通过条件全部不满足。**

## 跟历史对照

| Run | L0 c_k range | L0 util | L1 util | L2 util | collision | 来源 |
|-----|--------------|---------|---------|---------|-----------|------|
| task220 (200 ep, ep29) | U(0.5, 5) 全层 | 20.31% | 96.09% | 93.75% | 0.3835 | task220 |
| task222 (40 ep, ep29) | U(0.5, 5) 全层 | 20.31% | 98.44% | 91.02% | 0.3706 | task222 |
| **task242 Arm A (40 ep)** | **U(1, 5) L0 / U(0.5, 20) L1/L2** | **23.44%** | **46.09%** | **38.28%** | **0.9385** | **本任务** |

**Arm A 改变了什么**：
- L0 c_k 区间 U(0.5, 5) → U(1, 5): L0 利用率小幅上升 20.31% → 23.44% (+3.13pp)
- L1/L2 c_k 区间 U(0.5, 5) → U(0.5, 20): L1/L2 利用率**灾难性下跌** 98.44%→46.09%, 91.02%→38.28%
- collision_rate 上升 0.3706 → 0.9385 (+0.5679, 灾难性)

**Arm A 没有解锁 L0, 反而把 L1/L2 推到更严重的坍缩。**

## 机制解读

Issue #11 推荐 Arm A 的核心论点是：L1/L2 用更宽区间 (U(0.5, 20)) 释放几何参与 → 反向梯度结构变化 → L0 利用率回升。**实测这个反传机制不存在**：
- L1/L2 利用率从 98%/91% 跌到 46%/38%, 宽区间直接把这些层的码字推到边界附近的少数几个 (跟 task203 / task204 路径相同的 c‖x‖² 饱和假设 falsified 现象)
- L0 利用率只回升 3.13pp, **远低于 Issue #11 假设的"显著提升"**
- 整体 collision 0.9385 说明三层 SID 在 argmin 阶段几乎全是 dominant codeword 1-2 个, 互相冲突

## 决策

按 Issue #11 §阶段闸门:
> Gate 1 未过 → 禁止进入 Stage 3, 不接受任何形式的绕过 — 不作为 fallback option, 不作为 sanity check, 不以"先跑跑看"为由, 不因为 Gate 0 漂亮而放行. 唯一允许的后续动作是启动 Gate 1b (Arm A+).

→ 启动 Gate 1b (Arm A+): 同配置 + `--anti_collapse dead_revive`, 期望死码字复活改善 L0 utilization.

## 上游改动保留

- `HG-Rec/model/utils.py`: HVectorQuantization.c_k_range_list 单元素 list 存储 + _ensure_c_k layer_idx 透传 + HResidualVectorQuantization.forward enumerate 设 layer_idx=0
- `HG-Rec/model/hrqvae.py`: c_k_range_list 透传到 HResidualVectorQuantization
- `HG-Rec/train_hrqvae.py`: `--c_k_range_list` CLI flag (格式 "low1:high1,low2:high2,...")
- 5/5 sanity test 通过 (三层 c_k 在正确区间, default fallback 到 (c_k_min, c_k_max) 兼容上游)

这些上游改动保留, Issue #11 #242 Gate 1b / 后续 per-layer κ 实验可复用.

## 产物

- `products/task242/hrqvae_perlayer_ck/Jul-29-2026_02-32-39_*/best_collision_model.pth` (epoch 14, L0 util 23.44%)
- `products/task242/hrqvae_perlayer_ck/Jul-29-2026_02-32-39_*/best_loss_model.pth` (epoch 39 by recon loss)
- `products/task242/hrqvae_perlayer_ck/Jul-29-2026_02-32-39_*/epoch_*_collision_*_model.pth` (epochs 9/14/19/24/29/34/39, R12 save_limit 5)
- `logs/task242/perlayer_ck_stage1_train.out`
- `scripts/task242_issue11_gate1_perlayer_stage1.sh`
