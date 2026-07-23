# Task #49 — 战线一 QMP 6 指标 4 源对照

> **任务目的**: 用 QMP (Quantizability Metric Package) 测 4 种 embedding 源的可量化性
> **完成日期**: 2026-07-20
> **状态**: ✅ 已完成

---

## 1. 背景

承接 Task #46 (G1 PASS) + Task #48 (S4 AE). 用 QMP 矩阵系统比较 S1 MCKG / S4 AE / S5 T5 / S6 item2vec 在 norm shape / SCR / NP@10 / erank 上的差异.

## 2. QMP 6 指标

- M1: norm shape (ρ_max, CV, γ_1)
- M2: SCR (Split/Concat Ratio)
- M3: D_rel
- M4: codebook utilization
- M5: NP@k (邻域保持)
- M6: erank (有效秩)

## 3. 决策触发

QMP 矩阵用于选 embedding 源. S1 MCKG 多项指标最差, S4/S5/S6 各有所长.

## 4. 预算

~10 min (4 源 × 6 指标)

## 5. 结果 (回填)

| 源 | ρ_max | SCR | NP@10 | erank |
|---|-------|-----|-------|-------|
| S1 MCKG | 485 | 4.21 | 0.53 | 28 |
| **S4 AE** | 1.0 | (1sub) | **0.80** | 45 |
| S5 T5 | 1.67 | (1sub) | 0.42 | 46 |
| S6 item2vec | 1.0 | (1sub) | 0.40 | **60** |

**S4 AE 综合最优 (NP@10)**.

## 6. 产物

`scripts/task157_qmp_measure.py` + `products/task157_qmp_measure/task157_summary.json`
