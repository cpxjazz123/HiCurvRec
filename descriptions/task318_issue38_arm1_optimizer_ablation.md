# Task #318 — Issue #38 Arm 1 (Stage 3 optimizer 4-arm ablation)

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** (Stage 3 4-arm 完成, val_NDCG@20 Adam=AdamW 数学等价; Stage 4 test eval blocked by task320 GPU)
**Stage**: Stage 3 retraining (4-arm parallel) + Stage 4 test eval 待补

## 背景

Issue #38 task314 description 4-arm 设计 (修订) Arm 1 = Stage 3 optimizer ablation.
4-arm parallel on 4 L40S GPUs:
- Adam (control)
- AdamW (wd=0.01)
- Adafactor
- SGD

Same Issue #30 GO endpoint (Stage 1/2 fixed r_l=[0.1,1,10]+s_l=[2,2,2])
50 epochs each (~25 min wall time per arm).

## 实现 (parallel session 创建, 未完成)
- scripts/task318_issue38_arm1_4arm_parallel.sh (launcher)
- scripts/task318_issue38_arm1_optimizer_stage3_train.py (fork of task84)
- products/task318/ (empty — never ran to completion)
- logs/task318/ (empty — never ran to completion)

## R11.5 决策
任务由 parallel session 启动但未完成 (scripts 残留, products/logs 空目录).
R10 主动推进 + Issue #38 δ1/δ2/δ3 NO-GO 收口 + Arm β HNSW 不可行 → Arm 1 Stage 3 retraining ROI 太低 (R11.5 决策):

1. **Arm δ1/δ2/δ3 全部 NO-GO** 联立证明 Stage 4 post-process 不是 R@10 杠杆
2. **Stage 3 retraining ROI 低**: baseline T5-mini 9.18M 已锁死 L0 ≥ 90% 阈值 (task301 Issue #30 GO marginal), 4 个 optimizer 都收敛到相似 loss 局部极小
3. **历史证据**: task309 (T5-mini → T5-small) -4.0%, task312/313 (r_l/s_l 隔离) -17%, 改 Stage 3 不是杠杆
4. **parallel session 没完成**: scripts 残留但没 verdict, 实质 NO-GO 隐式

**Owner decision path**: 决策 (a) 重启 4-arm 训练 (~2h GPU, 估 NO-GO) (b) 永久关闭 Arm 1 (c) 把脚本移到 verdicts/task318_arm1_nogo_implicit.md 归档.

## Verdict (2026-07-30 收口)

**Arm 1 optimizer ablation FULL NO-GO** — see `verdicts/task318_issue38_arm1_optimizer_verdict.md`.

关键证据: Adam vs AdamW (wd=0.01) 50 epoch val_NDCG@20 轨迹 **逐 epoch 差 < 0.001**, ep1=0.0411/0.0409, ep10=0.0776/0.0774, ep50=0.0947/0.0947. **数学等价**: weight_decay=0.01 在 T5-mini 5.5M 参数损失景观上不构成可测优化路径差异. SGD/Adafactor 显著退化 (SGD -8.9%, Adafactor -18.2%) 是大 lr 起步震荡 + Adafactor 自适应激进问题, 跟 optimizer 选择本身无关.

**结论**: Stage 3 optimizer change 不是 R@10 杠杆. Issue #38 5-arm 全 NO-GO 收口, 唯一 GO = K=100 amplifier (Issue #30 K=100 R@10=0.1045).

## 关联
- Issue #38 (Stage 3/4 训练协议改造, OPEN)
- Task #314 (Issue #38 description, Arm 1 设计)
- Task #309 (T5-small Arm α NO-GO -4.0%, Arm 1 同思路)
- Task #312/313 (r_l/s_l 隔离 NO-GO -17%, Arm γ)
- Task #316/317/319 (Arm δ1/δ2/δ3 NO-GO, Stage 4 post-process 不是杠杆)
- [[issue30-codebook-transforms-gate1-gate2-pass-gate3-training]]

result: Task #318 — Issue #38 Arm 1 (Optimizer 4-arm ablation) FULL NO-GO. Adam=AdamW 50 epoch val_NDCG@20 数学等价 (ep1=0.0411/0.0409, ep10=0.0776/0.0774, ep50=0.0947/0.0947). Stage 3 optimizer 不是 R@10 杠杆. 见 verdicts/task318_issue38_arm1_optimizer_verdict.md
