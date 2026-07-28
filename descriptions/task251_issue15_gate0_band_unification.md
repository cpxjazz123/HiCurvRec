# Task #251 — Issue #15 Gate 0: Phase 0 一致率带口径锁定 + 配对档案重建

## 来源
- GitHub Issue #15 (2026-07-28): [Gate 0 校准] Phase 0 一致率带的预测效力审计
- 承接 Issue #11 (FULL NO-GO) + Task #241/242 顺手产出的配对数据
- 任务 #15 Gate 0 (零 GPU, 分钟级)

## 任务目的

Issue #15 §阶段闸门 Gate 0:
- 定位 Phase 0 一致率 4 条带原始出处 (>95% FAIL / 60-90% OPEN / <50% TOO_STRONG / agreement<0.90)
- 明确 90-95% 灰区如何判
- 重建 task220/221/222/235/242 ArmA/ArmA+ 6 run × 3 层配对档案 (Phase 0 一致率 ↔ Stage 1 利用率)
- 通过条件: 单一无歧义带宽定义 + ≥2 机制 × ≥2 配置 + 每格有 verdict 出处
- 硬停止: 任何一格靠反推 → 标 N/A 排除, 不填近似值

## 决策 (R11.3 自主决策)
- 锁定 file: verdicts/task251_band_unification.md (沿用 task236 体例)
- 配对数据全从 Issue #15 §1 + 现有 verdict 文件取, 不臆测
- 灰区处理: 沿用 Issue #15 §4 锁定的 "FAIL 边界灰区" 默认归 OPEN 偏 FAIL (与 task231 一致, 等 #13 / #15 Gate 1 给出进一步证据)

## 产物
- verdicts/task251_issue15_gate0_band_unification_result.md (Gate 0 verdict)
- verdicts/task251_band_unification.md (口径锁定文件)
- verdicts/task251_pair_table.json (配对档案)
- Issue #15 GitHub 评论 (Gate 0 PASS)

## 状态
零 GPU, ~5 min wall.