# Task #252 — Issue #15 Gate 1: Phase 0 一致率带预测效力判定 (PASS WITH CAVEAT)

## 来源
- GitHub Issue #15 (2026-07-28): [Gate 0 校准] Phase 0 一致率带的预测效力审计
- 承接 Task #251 (Gate 0 PASS): 4 带 + 2 灰区分类锁定; 4 机制 × 6 配置配对档案已重建
- 任务 #15 Gate 1 (零 GPU 分析)

## 任务目的

Issue #15 §阶段闸门 Gate 1:
- (a) 同号检验: PC κ 内部 (task222 vs task242 ArmA) Δ一致率 vs Δ利用率同号比例 ≥ 5/6?
- (b) 命中率检验: 三层全 OPEN 配置的 Stage 1 存活率?
- (c) 跨机制检验: Gromov 等其他机制是否同号 / 迁移?
- 通过条件: (a) ≥ 5/6 且 (b) 命中率明确写出
- 硬停止: (a) < 5/6 → STOP 写 verdict / 数据不足 → STOP

## 决策 (R11.3 自主决策)
- 数据: verdicts/task251_pair_table.json (上任务已生成)
- 不补新数据, 全用现有配对档案
- 写 verdict 报告同号 / 命中率 / 跨机制三组 + Gate 1 → Gate 2 建议

## 产物
- verdicts/task252_issue15_gate1_predictive_validity_result.md
- Issue #15 GitHub 评论 (Gate 1 结果)

## 状态
零 GPU 分析, ~5 min wall.