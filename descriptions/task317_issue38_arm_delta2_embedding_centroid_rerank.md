# Task #317 — Issue #38 Arm δ2 (Stage 4 embedding centroid similarity rerank)

**日期**: 2026-07-30
**状态**: 待启动 (R9 max+1 = 317)
**Issue**: Issue #38 Stage 3/4 训练协议改造 (OPEN)
**Arm**: δ2 — Embedding centroid similarity rerank

## 背景

Issue #38 task314 description 4-arm 设计 (修订):
- ~~Arm α T5-small NO-GO~~ (task309 R@10=-4.0%)
- ~~Arm γ Issue #35 NO-GO~~ (task312/313 R@10=-17%)
- **Arm δ (Stage 4 post-process rerank)** — 不重新训练, 立即可跑
  - δ1 Frequency prior: ❌ NO-GO (task316, alpha=0.10 ≈ baseline 0.1041, alpha≥0.3 NDCG crash)
  - **δ2 Embedding centroid similarity** (本任务) — 用 item_emb (Stage 1) 余弦相似度
  - δ3 Sequence-level diversity reward (task318)
- Arm β HNSW + rerank: 不可行 (SID discrete setting)
- Arm ε K=100 ceiling: ✅ done

## 简化决策 (R11.5 自主决策)

**做法**: 沿用 task316 模式 (Stage 4 post-process rerank hook), 但 rerank prior 从 log_freq 改成 embedding centroid similarity.

具体:
1. 对每个 test sample: history item IDs → item_emb[history] → (seq_len, 768) → centroid = mean → (768,)
2. 对每个 beam: beam item IDs (non-pad) → item_emb[beam] → (beam_seq_len, 768) → mean → (768,)
3. cosine similarity = centroid @ beam_emb / (||centroid|| * ||beam_emb||)  (形状 (B, K))
4. 重新评分: combined_score = beam_index * (1-alpha) - cosine_sim * alpha  (lower better)
5. 排序 + calculate_pos_index

Anchor = Issue #30 GO ckpt + K=50 (沿用 task316), alpha sweep [0.0, 0.1, 0.3, 0.5, 1.0]。

item_emb.parquet 已确认存在 (HG-Rec/dataset/Instruments/item_emb.parquet), 9922 items × 768-dim float64.

## 通过条件
- 任一 alpha R@10 > 0.1022 (Issue #30 端点 +0.2pp, GO marginal) → Stage 4 后处理杠杆发现
- 任一 alpha R@10 > 0.1045 (Issue #30 K=100 ceiling) → Issue #30 ceiling 也可超越
- 全 alpha ≤ 0.1022 → Arm δ2 NO-GO, Issue #38 δ2 收口

## R11.3 透明
- 选 alpha sweep [0.0, 0.1, 0.3, 0.5, 1.0] 因为跟 task316 (Arm δ1) 对齐, 易比较 delta
- 选 cosine similarity 不是 L2 distance 是因为 cosine 是 embedding 相似度标准度量
- 不重训 Stage 3 (节省 ~6 GPU-hour, R11.5 接受这是低成本探针)
- 不下载 item_emb (已存在, 直接 load parquet)
- 不用 per-layer 权重 (单层 item_emb centroid 足够)

## 执行顺序
1. Load item_emb.parquet (768-dim float64, 9922 items)
2. Load Issue #30 ckpt (T5-mini, 9.18M)
3. 跑 Stage 4 eval with cosine sim rerank, alpha sweep
4. 写 metrics JSON + verdict

## 关联
- Issue #38 (Stage 3/4 训练协议改造, OPEN)
- Task #314 (Issue #38 description, Arm δ 立即可跑)
- Task #316 (Arm δ1 frequency prior, NO-GO R@10=0.1042 max)
- Task #301 (Issue #30 GO anchor, ckpt + SID)
- Task #309b (Issue #30 K=50 ceiling R@10=0.1041)
- [[issue30-codebook-transforms-gate1-gate2-pass-gate3-training]]
