# Task #246 — paper-aligned fixes 增量更新综合 ranking (v3)

## 来源
- Task #87 (2026-07-23): 综合 ranking 22 行 baseline + 5 新 baseline, 16 闭环
- Task #87 v2 (2026-07-24): 增量 HGN (#95) + per-layer curvature (#88) + free-curv (#89)
- Task #141 (2026-07-24): **Caser paper-aligned fix** — RecBole default yaml + config_dict override (lr=0.001, wd=0.0). R@10=0.0378 (vs paper 0.0392, Δ -3.6%, ∈ paper ±5% 下限)
- Task #143 (2026-07-24): **FDSA paper-aligned fix NO-GO** — 撤回, 用 Task #85 R@10=0.0594 作为正式 baseline (paper-aligned `selected_features=[]` 让 FDSA 失去 class 信号 → 训练不收敛)
- Task #150 (2026-07-24): **LETTER paper-aligned fix** — 用 paper-faithful recipe (lr=2e-5, batch=8, epochs=4). R@10=0.0509 (vs paper 0.0581, Δ -12.4%, ∈ paper ±25%)

## 任务目的

增量更新 task87 v2 综合 ranking:
1. 用 Task #141 paper-aligned Caser 0.0378 替换 task87 报告的 "RecBole default yaml" Caser 0.0463 (Task #87 当时误标 paper-aligned, 实际是 over-trained yaml)
2. 用 Task #150 paper-aligned LETTER 0.0509 替换 task87 报告的 over-trained LETTER 0.0997
3. Task #143 FDSA 撤回, 沿用 Task #85 baseline (无变化)
4. retro-label: Task #87 Caser 0.0463 / LETTER 0.0997 都是 over-trained "+ve outlier", 不是 paper-aligned 真实值
5. 输出 v3 ranking 表 + paper-aligned systematic bias 重述

## 决策 (R11.3 自主决策)
- 不重跑 paper-aligned 闭环任务 (Task #141/143/150 已闭环, R12 ckpt 落盘)
- 直接基于 task141/143/150 verdict 数字 + task87 v2 增量更新 ranking
- 保留 v2 HG-Rec c555 0.1051 排名 (Task #88 闭环, 未变更)
- Task #140/148/151 paper-aligned fixes in-progress, 不在本次增量范围

## 产物
- verdicts/task246_paper_aligned_ranking_increment_result.md (增量 ranking 表 + retro-label)
- verdicts/task246_ranking_v3.json (机器可读)

## 状态
零 GPU 数据整理. 复用 task141/143/150 verdict 数字. 不写脚本, 直接编辑 verdict.