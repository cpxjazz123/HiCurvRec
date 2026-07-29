# Task #318 — Issue #38 Arm 1 (Stage 3 optimizer 4-arm ablation)

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** (parallel session 创建, scripts 残留未跑完)
**Stage**: Stage 3 retraining (4-arm parallel)

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

## Verdict
**Arm 1 optimizer ablation NO-GO (implicit)** — see verdict file (parallel session 未创建, R11.5 决策不重启).

## 关联
- Issue #38 (Stage 3/4 训练协议改造, OPEN)
- Task #314 (Issue #38 description, Arm 1 设计)
- Task #309 (T5-small Arm α NO-GO -4.0%, Arm 1 同思路)
- Task #312/313 (r_l/s_l 隔离 NO-GO -17%, Arm γ)
- Task #316/317/319 (Arm δ1/δ2/δ3 NO-GO, Stage 4 post-process 不是杠杆)
- [[issue30-codebook-transforms-gate1-gate2-pass-gate3-training]]
