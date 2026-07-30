# Task #340 — Issue #53 — κ-codebook 解冻节奏对照

**状态**: 待启动 (2026-07-30)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/53

## 目的

Issue #49 用的 κ-codebook 解耦调度是**硬性两阶段**: Phase A 完全冻结 κ 训练 codebook 至稳定, 第 200 epoch 整体一次性解冻 κ 进入 Phase B。这个"何时解冻、怎么解冻"的具体节奏是拍脑袋定的, 从未做过对照实验。本 issue 检验: 换更平滑/更早/振荡式解冻节奏, 能否让 κ 和 codebook 更好地协同适应。

## 假设

- **H1**: 硬性两阶段切换在切换点附近对 codebook 造成不必要的冲击, 用更平滑调度 (e.g. lr_theta 从 0 随 epoch 指数增长, 不设明确解冻时间点) 能减少冲击, 让 κ 学习过程中 codebook 始终保持更好健康度。
- **H2**: 解冻节奏对最终结果影响有限, -1.5% 差距主要来自别的原因 (Issue #52 per-layer 趋同 / Issue #51 信号质量)。

## 实验设计

复用 Issue #49 统一公式 + HRQ-VAE 基础设施, 4 臂对照:

| 臂 | 配置 | 描述 |
|----|------|------|
| **Arm A** | =#49 原配置 | 硬性两阶段, 第 200 epoch 整体解冻 (对照组) |
| **Arm B** | 渐进式解冻 | lr_theta 从 epoch 0 开始按预设曲线 (指数增长) 平滑上升, 无明确解冻时间点 |
| **Arm C** | 更早解冻 | 第 50 epoch 解冻, 但用更小初始 lr_theta + 更长总训练时间 |
| **Arm D** | 振荡式 | 多次在"冻结"和"小幅解冻"之间切换, 不是一次性单向解冻 |

每臂全程记录:
- codebook utilization 轨迹 (不只是终值)
- κ_m 训练轨迹
- Stage 2 SID 唯一性

## Stage 1 Gate 1 标准

- codebook utilization ≥ 90% **全程** (不只是终态)
- three-digit collision_rate ≤ 0.20
- **额外**: 任何一臂 util 轨迹明显优于 Arm A 对照组 (特别是切换点附近) → 进入 Stage 2-4

## Stage 2-4 (任一臂 Gate 1 PASS)

- Stage 2: Sinkhorn 推断 + 4-digit dedup → (9922, 4) SID
- Stage 3: T5-mini 200 epoch
- Stage 4: test eval beam=20

## 决策阈值

| 条件 | 含义 | 决策 |
|------|------|------|
| 任何臂 util 轨迹明显优于 Arm A | 节奏影响 codebook 健康 | H1 部分成立 |
| 任何臂 R@10 > 0.1005 (Issue #49 Arm B) | 节奏能缩小 -1.5% 差距 | GO |
| 任何臂 R@10 > 0.1020 | 真正超过 baseline | **GO** |
| 4 臂 util 轨迹无显著差异 | 调度节奏不是关键变量 | H2 成立, 应转向 #52/#51 |

## 风险与不确定性

1. **GPU 占用**: 4 臂并行 Stage 1 — 但只有 2 张卡空闲 (GPU 2/3), 需要分批
2. **时间预算**: Stage 1 总 ~10-15 min + Stage 3 总 ~2h + Stage 2-4 各 ~5 min = ~3h

## 产物

| 文件 | 路径 |
|------|------|
| Stage 1 训练 | `products/task340/{arm_a,arm_b,arm_c,arm_d}/{best_collision_model,phase_b_final}.pth` |
| κ 轨迹 | `products/task340/{arm}/kappa_log.json` |
| Stage 2 SID | `products/task340/{arm}/sid.npy` |
| Stage 4 eval | `verdicts/task340_issue53_stage4_beam20.json` |
| Verdict | `verdicts/task340_issue53_result.md` |

## R11.5 自主决策

- θ_init=-0.02 (跟 Issue #49 Arm B 一致)
- 4 臂并行问题: 实际只有 2 张卡空闲 (GPU 2/3), 所以分批: GPU 2 跑 Arm A + B, GPU 3 跑 Arm C + D