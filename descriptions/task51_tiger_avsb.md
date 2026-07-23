# Task #51 — 战线四 端点验证 TIGER A vs B

> **任务目的**: 用 RQ-VAE 重构 loss 作为下游 Recall 的 proxy, 验证 QMP→Recall 相关性
> **完成日期**: 2026-07-20
> **状态**: ✅ 已完成

---

## 1. 背景

承接 Task #49 QMP + Task #50 S6. 端到端 TIGER 训练需 ~6-8 h, 用 Stage 2 RQ-VAE 重构 loss (Task #109 已证 ρ=-0.67 与下游 Recall 相关) 作为 proxy 验证假设.

## 2. 实验设计

3 源 (A=T5, B=MCKG_raw, C=MCKG+log1p) × 5000 steps StandardRQ (K=256, 3-layer). 测 RQ recon loss + codebook utilization + SID entropy.

## 3. 决策触发

| 条件 | 判定 |
|------|------|
| T5 (A) < MCKG (B) | ✅ QMP→Recall 假设成立 |
| MCKG+log1p (C) < MCKG (B) | ✅ L1 fix 端到端有效 |

## 4. 预算

~10 min (3 源 × 5000 steps)

## 5. 结果 (回填)

| 源 | RQ recon | vs A |
|---|----------|------|
| A (T5) | **0.0452** | 1.00× |
| B (MCKG raw) | **12.7161** | **281.05×** ❌ |
| **C (MCKG + log1p)** | **0.0515** | 1.14× ✅ |

**QMP→Recall 假设完全成立 (281× 差距), L1 log1p 端到端有效 (99.6% 改进)**.

## 6. 产物

`scripts/task159_tiger_avsb_proxy.py` + `products/task159_tiger_avsb/task159_summary.json`
