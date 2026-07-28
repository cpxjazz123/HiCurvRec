# Task #236 — Issue #10 Gate 0: collision 指标口径统一 (PASS)

## 来源
- GitHub Issue #10 (2026-07-28): [Validation] 3-arm converged collision 设计 + collision 指标口径统一 (task223 baseline 0.99 vs task225 §5 baseline 9.07% 冲突) — 承接 Task #233
- Issue #10 §H0: 零 GPU, 阻塞项. 锁定单一 collision 定义, 重述历史数字

## 关键发现
1. task225 §5 列「Stage 2 collision」实际记的是 uniqueness (uniqueness 9.07% baseline / 5% Sinkhorn / 10% TIGER / 12% Letter)
2. 权威定义 (`HG-Rec/model/hrqvae_trainer.py:246` + `scripts/task200_stage2_codebook.py:166`):
   `collision_rate = (N_items - N_unique_SID) / N_items = 1 - uniqueness_rate`
3. task200 dual_v5 collision 0.8387 = uniqueness 16.13% (跟 task223 0.3706 = 62.94% 不在同一架构族)
4. `<= 12%` 真实意图不明, 待 Issue #10 Gate 2 Arm B 跑完才能定

## 产物
- verdicts/task236_collision_metric_unification_result.md (5906 bytes)

## 决策
- 选 task223 权威 collision_rate 定义 (1 - uniqueness), 因为源码直读
- task225 §5 retro-label: 列「Stage 2 collision」= uniqueness, 数字全部重述
- task200 -10.3% retro-label: **confounded by Stage 3 truncation + codebook convergence**

## 状态
Gate 0 PASS. 后续 Gate 1 (Arm B Sinkhorn 部分配置 1 次 Stage 3 训练) 等下一 tick 启动.
