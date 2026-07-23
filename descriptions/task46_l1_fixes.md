# Task #46 — 战线二 L1 后处理修复 + 2×2 消融 (G1 门控)

> **任务目的**: 测 log1p / quantile / whitening 2×2 消融是否能把 SCR 从 4.22x 压到 <1.5x
> **完成日期**: 2026-07-20
> **状态**: ✅ 已完成 (G1 PASS)

---

## 1. 背景

承接 Task #44 V5: MCKG margin ranking 训出的 norm 长尾是 root cause. L1 后处理是否能在不动训练目标的前提下压回 SCR?

## 2. 实验设计

4 种 fused 输入处理 + 6 种 L1 变体, 在 StandardRQ (K=64, 3-layer) 上跑 2000 steps 测 SCR.

## 3. 决策触发 (G1 门控)

| SCR (vs fused_L2) | 决策 |
|-------------------|------|
| < 1.5x | G1 PASS → GO-浅 |
| 1.5-3.0x | 边界 → 需 L2/L3 |
| > 3.0x | G1 FAIL → GO-深 |

## 4. 预算

~15 min (6 变体 × 2000 steps)

## 5. 结果 (回填)

| 变体 | SCR | 判定 |
|------|-----|------|
| fused_raw_baseline | 2.96x | ❌ |
| **fused_log1p** | **0.31x** | ✅✅✅ |
| **fused_quantile** | **1.05x** | ✅ |
| fused_whiten | 220x | ❌ 反而恶化 |
| fused_whiten_log1p | 4.09x | ❌ |

**G1 门控 PASS** — log1p 是 winner.

## 6. 产物

`scripts/task154_l1_fixes.py` + `products/task154_l1_fixes/task154_summary.json`
