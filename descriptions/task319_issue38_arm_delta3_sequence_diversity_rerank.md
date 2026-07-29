# Task #319 — Issue #38 Arm δ3 (Stage 4 sequence-level diversity reward rerank)

**日期**: 2026-07-30
**状态**: 待启动 (R9 max+1 = 318)
**Issue**: Issue #38 Stage 3/4 训练协议改造 (OPEN)
**Arm**: δ3 — Sequence-level diversity reward

## 背景

Issue #38 task314 description Arm δ3:
> δ3: Sequence-level rerank — 同一 sequence 内不同 beam 之间的多样性奖励

不依赖 item_emb.parquet (跟 δ2 不同), 纯基于 beam 输出的 item diversity。

## 简化决策 (R11.5 自主决策)

**做法**: 对每个 test sample, 取 top-K beams, 计算 beam-pair diversity (e.g., unique items ratio 或 item pair distance), 用 combined score = beam_logit * (1-alpha) + diversity * alpha rerank.

简化方案: **Maximal Marginal Relevance (MMR)** 风格的 rerank:
1. 对每个 beam, 计算它到所有其他 beams 的 "diversity" (1 - 共享 item 比例 / 或者基于 distinct item count)
2. combined_score = 1/(rank+1) * (1-alpha) - diversity * alpha
3. Greedy 选 beam: 第一名是 top-1 beam, 然后每次选跟已选集合 diversity 最大的下一个

Anchor = Issue #30 GO ckpt + K=50 (沿用 task316)。

具体 metric:
- diversity_score(beam_i) = sum_j != i (1 - shared_items / total_items_in_pair)
- normalized: max(1.0)
- alpha=0: pure beam order (baseline)
- alpha=1: pure diversity (max diverse subset)

## 通过条件
- 任一 alpha R@10 > 0.1022 (Issue #30 端点 +0.2pp) → Stage 4 多样性奖励杠杆发现
- 任一 alpha R@10 > 0.1045 (Issue #30 K=100 ceiling) → 超越 ceiling
- 全 alpha ≤ 0.1022 → Arm δ3 NO-GO

## R11.3 透明
- 选 MMR-style diversity 因为 (a) 不需要额外 model load (b) 计算轻量 (c) 跟 task316 + δ2 模式一致 (alpha sweep)
- diversity 用 unique item count / total item count (item-level diversity, 跟 paper DPP 类似思想)
- alpha sweep [0.0, 0.3, 0.5, 0.8, 1.0] (比 δ1 多 0.8 因为 diversity 是跟 top-1 距离远的事)
- 不重训 Stage 3, 不下载额外 model

## 执行顺序
1. Load Issue #30 ckpt + K=50 (沿用 task316)
2. 跑 Stage 4 eval with MMR diversity rerank, alpha sweep
3. 写 metrics JSON + verdict

## 关联
- Issue #38 (Stage 3/4 训练协议改造, OPEN)
- Task #314 (Issue #38 description, Arm δ3 多样性奖励)
- Task #316 (Arm δ1 NO-GO, 同 rerank 模式)
- Task #317 (Arm δ2 embedding centroid rerank, 同步启动)
- [[issue30-codebook-transforms-gate1-gate2-pass-gate3-training]]
