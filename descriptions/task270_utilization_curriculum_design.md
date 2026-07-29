# Task #270 — Stage 1 L0 utilization ≥ 90% curriculum 方案设计 (Issue #17 修复后新方向)

## 背景

Task #263 直接测量 task253 Stage 1 L0=**73.44%** < 90% (§6.7.4 stop-loss (i) 触发). 同机制族 6/6 历史 task (task178/179/180/199/201/203/204 等) 也都是 L0 < 90%. 这说明 baseline recipe 在 50 epoch 默认 β=0.5 下无法把 L0 utilization 推到 ≥ 90%.

Task #265 修了 step2 monitor 让 utilization 真打印. 工具链就绪, 但**没有机制把 L0 utilization 推到 ≥ 90%**. 这是当前 §6.7.4 stop-loss (i) 始终触发的根因.

候选机制 (控制变量单 layer / 单一变量):

| 机制 | 文件 / CLI | 假设原理 | 风险 |
|---|---|---|---|
| **A: β curriculum** | env `BETA_CURRICULUM=0.0:30,0.5:50` (epoch) | β=0 时是纯欧氏 loss, 码字自由散开; 30 epoch 后提到 β=0.5 进入双曲约束. 让码字先用欧氏散开, 再用双曲距离精修 | 可能 β 突变导致 loss 跳变 |
| **B: encoder freeze epoch** (已有, Task #193) | `--freeze_encoder_epoch 20` | 20 epoch 后冻结 encoder, 只让 codebook+decoder 重新分配. 让前期学到的 L0 不被 encoder 重新打乱 | 同 task193 评估 |
| **C: Sinkhorn 端点 rebalance** | stage2 codebook (新增) | Stage 1 不动, Stage 2 推断阶段加 Sinkhorn rebalance. 但 L0 utilization 是 Stage 1 量, Stage 2 不能事后改 | **机制错配, 弃** |
| **D: lr_log_r 提升** (已有, 但默认 10.0) | env `LR_LOG_R=50.0` | log_r 半径参数用更大 lr 加速半径分化, 让码字在双曲空间里"伸出"更远 | 可能坍缩成几个 high-norm 大码字 |
| **E: dead_revive 频率** (已有, hook) | `--revive_freq 5` (假设存在) | 每 N epoch 重新 init dead code 到 batch mean. **但 Issue #17 Gate 1 (c) verifier 实测 utilization pre-revive = 73%, post-revive 已经 100%**. 所以 dead_revive 在 hidden layer 是工作的, **真正问题在 L0 argmin 时的 unique count** | 需要看代码确认 revive 算法 |

**推荐组合**: 候选 2 (β curriculum + encoder freeze + lr_log_r↑), 因为这是 baseline recipe 的"自然推广", 不引入新算法.

## 任务范围 (本 cron tick)

1. **设计三种 curriculum 配方**: 候选 A1 (β 0.0→0.5 @ epoch 20)、A2 (β 0.1→0.5 @ epoch 30)、A3 (β 0.0→0.3 不达 β=0.5)
2. **Stage 1 验证计划**: 50 epoch × 3 配方 × 1 GPU / 配方 = ~5 min/配方, 总 15 min
3. **通过条件**: 至少一个配方 L0 utilization ≥ 90% (≥58/64 unique codes) 在 epoch 30 以后
4. **写 launcher 脚本** `scripts/task270_utilization_curriculum.sh` (default 启动 A1)
5. **风险**: curriculum 配方如果 L0 utilization 仍 < 90% → 闭环 NO-GO, 跟 Issue #17 §1 step 1 链同步

**关键决策点 (R11.3)**:
- **A vs B vs D**: 单一变量设计 — 只能一次测一个机制, 否则违反 §6.7.4 越闸教训
- **本 cron tick 仅写 launcher 脚本 + 设计文档, 不启动训练** (后续 cron tick 启动 A1, 5 min 总训练量, 风险可控)
- **不写 Stage 2/3/4 计划**: Issue #17 §H3 已明示本方向"不申请任何 Stage 3/4 预算"

## 物理产物

```
descriptions/task270_utilization_curriculum_design.md  (本文件)
verdicts/task270_utilization_curriculum_design_result.md
scripts/task270_utilization_curriculum.sh  (3 配方 launcher, 0 GPU 设计)
```

## 后续 (后续 cron tick)

按 R10 + R11.5:
- Stage 1 launch A1 (5 min, GPU 0)
- 如果 L0 ≥ 90% PASS → 进 Stage 2 推断 (Sinkhorn 5 min) + Stage 3 训练 (~1h) + Stage 4 eval (~5 min). 端到端 ~1.5h
- 如果 NO-GO → 试 A2 → A3 → 标 issue "L0 ≥ 90% 在 baseline recipe 不可能", 资源转其他 backlog 候选

result: Task #270 — Stage 1 L0 ≥ 90% curriculum 3 配方 (A1/A2/A3 β 课程) 设计 + launcher 脚本预写, 0 GPU 本 tick. 等下个 tick 启动 A1 Stage 1 验证 (~5 min).
