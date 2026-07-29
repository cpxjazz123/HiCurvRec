# Task #269 — Musical_Instruments paper-aligned LETTER/FDSA R@10 重测 (横向校准 HG-Rec baseline 0.10204)

## 背景

Task #267 关闭 Issue #10 后, 把 backlog 高 ROI 候选 1 (paper-aligned 重测) 选为下一推进方向, 展开为 Task #269. 目标是用 paper-aligned recipe 在 Musical_Instruments 上重测 LETTER 和 FDSA, 跟 HG-Rec baseline R@10=0.10204 横向校准 (排除数据集差异).

## 任务范围

1. 核对 paper-aligned baselines 在 Musical_Instruments 上的实际完成度
2. 确认是否需要新 Stage 3 训练
3. 评估横向校准 ROI

## 关键决策点 (R11.3)

- **NO-OP 路径**: 若 paper-aligned baselines 已经在 Musical_Instruments 上跑过且 verdict 完整 → 不重测, 仅整理现有产物
- **重测路径**: 若有 baseline 缺 paper-aligned R@10 → 启动 Stage 3 训练 (~1h GPU)

## 物理产物

```
descriptions/task269_letter_fdsa_paper_aligned_retest.md  (本文件)
verdicts/task269_letter_fdsa_paper_aligned_retest_result.md
```

result: Task #269 — paper-aligned Musical_Instruments baselines (LETTER / FDSA / Caser) R@10 横向校准重测, 跟 HG-Rec baseline R@10=0.10204 比较. 评估 NO-OP 还是重测.
