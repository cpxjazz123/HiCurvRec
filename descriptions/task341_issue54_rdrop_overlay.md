# Task #341 — Issue #54 — Issue #49 Stage 1/2 + Issue #38 R-Drop Stage 3 叠加

**状态**: 待启动 (2026-07-30)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/54

## 目的

Issue #49 沿用了 baseline recipe (Adam, lr=1e-4, dropout=0.1), 没有引入针对性训练协议优化。但 Issue #38 R-Drop α=1.0 已验证在别的配置上跑出 +1.4% 独立增益 (test R@10=0.1034 vs baseline 0.1020)。本 issue 检验: 在 #49 的曲率配置基础上, 叠加 R-Drop 等训练协议优化, 能否填平 -1.5% 差距甚至反超。

## 假设

- **H1**: #49 的 -1.5% 差距跟曲率学习无关, 只是 Stage 3 用 baseline 配方没优化, 叠加 R-Drop 后能填平甚至反超 (R-Drop +1.4% 应能覆盖曲率 -1.5%)。
- **H2**: 训练协议优化和曲率学习存在**负交互** (e.g. R-Drop 双 forward+一致性 loss 跟 κ 参数梯度更新耦合), 叠加后比 #49 原始结果更差。

## 实验设计 (复用 Issue #49 Stage 1/2 产出, 只改 Stage 3)

**Stage 3 配方** (跟 Issue #38 R-Drop Arm C 一致):
- R-Drop α=1.0 (KL 一致性 loss 在同一 batch 两次 forward 之间)
- Optimizer: AdamW (lr=1e-4, 默认参数)
- LR scheduler: linear warmup 20 ep + linear decay
- Dropout: 0.1
- 其他: 跟 Issue #49 训练协议一致

**Stage 1/2 复用**:
- Stage 1: Issue #49 Arm B 最佳 ckpt (products/task340/arm_minus/stage1/best_collision_model.pth, κ=[-0.128, -0.110, -0.123], codebook 健康)
- Stage 2: Issue #49 SID (products/task340/arm_minus/sid.npy, 9922/9922 unique, collision=0.1906)

**Stage 3 + 4 重新跑**:
- 训练 200 epoch, early stop patience=20 (跟 #49/#51 一致)
- Beam=20 test eval

## 决策阈值 (vs Issue #49 Arm B R@10=0.1005 + Issue #40 正确 baseline 0.1020)

| 条件 | 含义 | 决策 |
|------|------|------|
| **R@10 > 0.1020** | 填平 + 超过 baseline | **GO** (H1 成立, 协议杠杆独立于曲率) |
| 0.1005 < R@10 ≤ 0.1020 | 缩小差距但未超 baseline | 边际 GO, 协议杠杆部分有效 |
| R@10 ≤ 0.1005 | 跟 #49 持平或更差 | **H2 成立** (负交互), 关闭 R-Drop+曲率叠加方向 |

## 风险与不确定性

1. **GPU 占用**: Stage 3 重跑 ~1.5h (T5-mini 200 epoch). 等 Issue #51 Stage 3 跑完后或 Issue #49 Stage 3 跑完后启动。
2. **Stage 1/2 复用风险**: Issue #49 Stage 1 ckpt 是否与新 Stage 3 协议兼容 — codebook/SID 都是 input 不变, 应该 OK

## 产物

| 文件 | 路径 |
|------|------|
| Stage 3 T5 训练 | `products/task341/rdrop_overlay/t5_*/` |
| Stage 4 eval | `verdicts/task341_issue54_stage4_beam20.json` |
| Verdict | `verdicts/task341_issue54_result.md` |

## R11.5 自主决策

- 直接复用 Issue #49 Stage 1/2 产出, 无需新训练 HRQ-VAE (成本节约 ~5h Stage 1)
- Stage 3 协议 = Issue #38 Arm C R-Drop α=1.0 + Issue #49 其他默认
- 启动时间: 等 Issue #51 Stage 3 (PID 2585388) 跑完后, 或 Issue #49 Stage 3 已完成复用其 ckpt