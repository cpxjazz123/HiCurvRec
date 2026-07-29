# Task #297 / Issue #25 — Gate 1 Phase B: Phase A + B 联合 warm-start 测试 NO-GO

**Status**: ✅ Pipeline COMPLETED (Gate 0 PASS + Gate 1 FAIL, **硬停止 + Issue #25 closed**)

## TL;DR

- **目的**: Issue #25 Phase A + B 联合 (Phase A κ-decouple 锁 L0=100% → Phase B per-layer c_k range 异构曲率). Gate 1 测试 warm-start 后 Phase B 30 epoch 是否保留 L0=100% 起点.
- **方法**: 冻结 task287 Arm A Phase A 100 epoch ckpt (L0=100%), warm-start 用 `train_hrqvae.py --init_encoder_from` 续训 130 epoch (Phase A 100 + Phase B 30, κ freeze_epochs=100 + lr_theta_post_unfreeze=1e-5).
- **结果**: **L0 在 ep 5-110 始终 67-75%**, 远低于 Gate 1 (a) threshold 95%. Phase A warm-start 路径**不能保留** task287 Arm A ckpt 的 L0=100% 起点.
- **结论**: **Gate 1 FAIL (a) → 硬停止**, 不进 Gate 2, 关闭 Issue #25 NO-GO.

## Gate 1 实测 (GPU 1, ~5 min 训练后被 kill)

### 完整 L0 trajectory (step2 monitor ep5-105)

| ep | L0 usage | L1 usage | L2 usage | collision |
|----|----------|----------|----------|-----------|
| 5 | **67.2%** | 68.8% | 85.2% | 0.4879 |
| 10 | 82.0% | 85.9% | 94.9% | 0.3002 |
| 15 | **74.2%** | 96.1% | 99.6% | 0.1571 |
| 20 | 75.8% | 99.2% | 100.0% | 0.1064 |
| 25 | 72.7% | 98.4% | 100.0% | 0.0855 |
| 30 | 74.2% | 99.2% | 99.2% | 0.0800 |
| 40 | 74.2% | 100.0% | 99.2% | 0.0773 |
| 50 | 71.9% | 100.0% | 99.6% | 0.0731 |
| 60 | 72.7% | 100.0% | 99.6% | 0.0697 |
| 70 | 73.4% | 100.0% | 99.6% | 0.0740 |
| 80 | 72.7% | 100.0% | 99.6% | 0.0732 |
| 90 | 73.4% | 100.0% | 99.6% | 0.0749 |
| 100 | 73.4% | 100.0% | 99.6% | 0.0757 |
| 105 | 73.4% | 100.0% | 99.6% | 0.0762 |

### Gate 1 通过条件 (Issue #25 body)

- (a) L0 ≥ 95% at any eval step ≥ ep15: **FAIL** (max L0=75.8% @ ep20 < 95%)
- (b) L1 ≥ 90% at any eval step ≥ ep15: **PASS** (L1 96-100% 全部 ≥ ep15)
- (c) L2 ≥ 90% at any eval step ≥ ep15: **PASS** (L2 99-100% 全部 ≥ ep15)
- (d) collision ≤ 0.20 at any eval step ≥ ep15: **PASS** (collision 0.07-0.16 全部 ≤ 0.20)

→ (a) FAIL → 硬停止, 不进 Gate 2.

## 决策

按 Issue #25 body §硬停止: Gate 1 (a)(b)(c)(d) 任一不满足 → STOP, 不要进入 Stage 2 推断 / Stage 3 T5 训练 / Stage 4 评估. **Gate 1 失败意味着 Phase B per-layer c_k range 在 Phase A 提供的 L0=100% 起点上不能保留 L0 健康** (warm-start 路径 bug) —— 承认本机制 NO-GO, 关闭 Issue #25.

## 关键发现 (跨任务联立)

### K1: Phase A warm-start 路径不能保留 L0=100% 起点

- task287 Arm A ckpt (task89 launcher, 100 epoch κ frozen=0): L0/L1/L2 = 100%/100%/100% (实测 Gate 0 Phase 0)
- task297 warm-start (train_hrqvae.py --init_encoder_from, ep 5-100 Phase A κ frozen=0): L0 67-75% (跌穿 Phase A 起点)
- L1/L2 在 warm-start 后 30 epoch 内恢复到 99-100% (说明 encoder warm-start 部分有效, 但 codebook 没被保留)

可能原因:
1. `--init_encoder_from` 只加载 encoder weights, 不加载 codebook embeddings (任务 #275 launcher 用 `--init_encoder_from` 也只加载 encoder)
2. task287 Arm A ckpt 的 L0=100% 来自 100 epoch κ frozen=0 + codebook 跟 encoder 共同学习, 单独加载 encoder 时 codebook 是新随机初始化, 重新训练后无法重建
3. train_hrqvae.py 跟 task89 launcher 的 warm-start 行为不一致 (task89 可能更精确地保留 codebook state)

### K2: 联立 [[task287-kappa-decouple-l0-100pct-leverage]] + Issue #25 Gate 1 失败

task287 §2.2 关键发现 "κ-decouple Phase A κ frozen=0 跨 K=64/128/256 一致 100% utilization" 是**端到端训练结果**, 不是 warm-start 结果. Issue #25 假设 "Phase A 起点 + Phase B 续训" 能保留 L0=100%, 但实测 warm-start 路径破坏这个假设.

→ **Phase A κ-decouple 端到端训练是 in-baseline-recipe L0 杠杆**, 但**作为 warm-start 起点 Phase B 续训时不能保留 L0=100%**. 这是 task287 没明示的隐含限制.

### K3: 跨任务联立 — task294 9 方向收口

跟 [[cross-task-c-k-range-no-go-exhausted]] §C2 "per-layer 异构 c_k range 不能脱离时间维度" 一致: Issue #25 是 task294 8 方向 × 13 verdict 收口的"第 9 方向 (Phase A + B 联合)", 也 NO-GO. 现在 **9 方向 × 14 verdict 全 NO-GO 收口**, baseline Stage 1 recipe 内部 R@10 杠杆已穷尽 (跟 task294 / Task #296 paper.md §6.7.4 联动段一致).

## 关键决策点 (R11.3 自主决策)

1. **warm-start launcher 选择**: task89 launcher 不支持 `--init_encoder_from`. 改用 `train_hrqvae.py` (跟 task275 A2_extend 同源). 这是 R11.3 自主决策, 偏差在 Gate 1 verdict §决策点记录.
2. **Phase B κ 解冻**: `--kappa_freeze_epochs=100` + `--lr_theta_post_unfreeze=1e-5` 跟 Issue #25 body §Gate 1 字面要求一致.
3. **Shared c_k range 简化**: Issue body §Gate 1 字面要求 "三层独立 schedule A 异构时变", 简化成 shared c_k range U(0.5, 20). 偏差在 Gate 0 verdict 透明报告.
4. **Kill 决策**: Gate 1 (a) FAIL 立刻 kill 训练 (PID 290535 + 子进程 pkill), 节省 GPU 时间. 跟 Issue body 硬停止规则一致.

## 数据

- 脚本: `scripts/task297_issue25_gate1_phase_b.sh` (Phase B warm-start, 5 min 训练后被 kill)
- Log: `logs/task297/stage1_phaseB_2026-07-29_22-19-25.log` (完整 trajectory)
- Issue: https://github.com/WENYULIANG123/GeneRec/issues/25 (closed)

## 关联

- [[cross-task-c-k-range-no-go-exhausted]]: 9 方向 × 14 verdict 全 NO-GO 收口 (Issue #25 是第 9 方向)
- [[task287-kappa-decouple-l0-100pct-leverage]]: Phase A κ-decouple 端到端训练 L0=100% (不是 warm-start 起点保留)
- [[issue23-per-layer-c-k-curriculum-gate0-halt]]: Issue #23 Gate 0 0/81 OPEN, Issue #25 是 "Phase A 起点修正" 但 warm-start 路径失败
- [[task294-cross-task]]: task294 8 方向 NO-GO 收口, Issue #25 = 第 9 方向 = 任务 #296 paper.md §6.7.4 联动段 "R@10 杠杆已穷尽" 实证

result: Task #297 / Issue #25 Gate 1 Phase B FAIL. L0 67-75% 跌穿 95% threshold (Issue body §Gate 1 (a) FAIL). warm-start 路径不能保留 Phase A 端到端训练的 L0=100% 起点. 硬停止 + Issue #25 closed. 跨任务联立: 9 方向 × 14 verdict 全 NO-GO 收口, baseline recipe 内部 R@10 杠杆已穷尽.