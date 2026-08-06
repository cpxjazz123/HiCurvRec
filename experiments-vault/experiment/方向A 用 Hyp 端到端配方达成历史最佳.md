---
type: experiment
category: "experiment"
zh: "方向A 用 Hyp 端到端配方达成历史最佳"
direction: "方向A"
status: "BREAKTHROUGH"
topic: "hyp_routes"
created: 2026-08-05
tags:
  - status-breakthrough
  - topic-hyp-routes
  - experiment
---

# 方向A 用 Hyp 端到端配方达成历史最佳

> **情况**: [[Hyp 路线端到端配方方法 (历史最佳)|Hyp 路线端到端配方方法 (历史最佳)]] (BREAKTHROUGH)
> **主题**: [[Hyp 路线方法 (per-item radius + 自由 κ)|Hyp 路线方法 (per-item radius + 自由 κ)]] (hyp_routes)

## 用什么方法

方向A per-item radius + capmatch REL_STRUCT + 自由 κ (Hyp 路线唯一超基线)

## 方法描述

**方向 A 做了什么**: 把 Hyp 路线 (per-item radius + 自由 κ + capmatch REL_STRUCT) 端到端跑通，test R@10 = **0.1048** (+0.0024 vs baseline)，是历史唯一稳定超基线案例。

**方法描述**:
- **Stage 1 (per-item radius)**: 不再固定 c=1，每个 item 自适应半径 —
`r = R_MAX · sigmoid(3·(||t5_emb|| − 0.7))`，R_MAX=0.99，浅层 r 大、深层 r 小。
- **Stage 2 (capmatch REL_STRUCT)**: 
  - `REC_LAYER_W = 1:3:9` 损失反向促进深层码本充分训练
  - `REL_STRUCT=1` 加入关系结构先验
  - bf16 + bs1024 + 1000 epoch
  - κ 三层自由学习 (final κ ≈ [0.024, 0.044, 0.055])
- **Stage 3 (T5-mini + 早停)**: 
  - bf16 + bs1024 + lr=3e-4 + 95 epoch 早停 (early_stop=20)

**结果**: test R@10 = **0.1048** (+0.0024)，valid R@10 = 0.1269 (+0.0002)，NDCG@20 = 0.0860 (+0.0105)。

**5% 超基线 (0.1075) 在 Hyp 路线物理不可达**已 9 次确认 (Issue #39/#41/#43 + 历史多次)。

## 关键指标

- **test_r10**: 0.1048
- **delta_r10**: +0.0024
- **ndcg20**: 0.086
- **valid_r10**: 0.1269

## 相关实验

- [[EXPERIMENT_GUIDE|Vault 入口]]
