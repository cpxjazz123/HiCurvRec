# Task #50 — 战线一 S6 item2vec 训练

> **任务目的**: 用行为共现 (Skip-Gram + Negative Sampling) 训 embedding, 作为 QMP 矩阵的对照源
> **完成日期**: 2026-07-20
> **状态**: ✅ 已完成

---

## 1. 背景

承接 Task #48 S4 AE (纯重构) + Task #49 QMP 矩阵需求. S6 item2vec 用 SGNS 训, 与 margin ranking / AE 都不同的训练目标.

## 2. 实验设计

Skip-Gram w=5, neg_k=5, embed=64, 5000 steps, lr=1e-3. 数据: Toys partition_0 tfrecord `sequence_data` 字段.

## 3. 决策触发

| 指标 | 范围 | 判定 |
|------|------|------|
| norm ρ_max ≈ 1.0 | L2 归一 | ✅ norm 健康 |
| loss < 0.1 | SGNS 收敛 | ✅ 训练成功 |

## 4. 预算

~5 min (5000 steps)

## 5. 结果 (回填)

- final_loss = 0.0653 (SGNS 收敛)
- norm: mean=1.000, max=1.000
- erank = 60 (QMP 中最高)
- NP@10 = 0.40

## 6. 产物

`scripts/task158_s6_item2vec.py` + `products/task158_s6_item2vec/{entity_embedding.pt, task158_summary.json}`
