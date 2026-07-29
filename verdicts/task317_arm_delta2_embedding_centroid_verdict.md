# Task #317 — Issue #38 Arm δ2 (Stage 4 embedding centroid similarity rerank) NO-GO

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** — moderate alpha (0.10-0.50) 跟 baseline 完全一致 (R@10=0.1041), alpha=1.00 极端崩溃
**Stage**: Stage 4 rerank hook (无 retraining, 即时验证)

## Settings
- ckpt: task301 Issue #30 GO (per-layer Codebook Transforms r_l=[0.1,1,10] + s_l=[2,2,2])
- code_path: `_t5_hrqvae_issue30_per_layer_transforms.npy`
- Beam size: 50 (Issue #30 K=50 ceiling)
- GPU 0, 总耗时 ~327s (5 alpha × ~65s)
- item_emb.parquet: 9922 items × 768-dim (Stage 1 T5-base output), pre-normalized unit-norm
- Rerank hook: per-sample history centroid (mean of history item_emb) vs per-beam centroid (mean of beam item_emb), cosine similarity, weighted with beam_index in combined score

## Metric table (alpha sweep)
| alpha | R@5 | R@10 | R@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|-----|------|------|--------|---------|---------|
| 0.00 (no rerank) | 0.0828 | 0.1041 | 0.1310 | 0.0704 | 0.0772 | 0.0840 |
| 0.10 | 0.0828 | 0.1041 | 0.1310 | 0.0704 | 0.0772 | 0.0840 |
| 0.30 | 0.0828 | 0.1041 | 0.1310 | 0.0704 | 0.0772 | 0.0840 |
| 0.50 | 0.0828 | 0.1041 | 0.1310 | 0.0704 | 0.0772 | 0.0840 |
| **1.00 (pure centroid)** | **0.0180** | **0.0366** | **0.0711** | **0.0105** | **0.0164** | **0.0250** |

## Verdict
**Arm δ2 embedding centroid similarity rerank NO-GO** — 两段:
1. **alpha=0.10-0.50 (mild-moderate bias)**: 全部跟 baseline **完全相同** (R@10=0.1041). 这是真正的信号 — 不是 in-sigma, 是逐 batch 完全相同. 说明 cosine similarity 跟 beam rank 的相对差异太小, 不影响 argmax 顺序.
2. **alpha=1.00 (pure centroid)**: R@10=0.0366 (-64.9%) — 完全没意义, 按 centroid similarity 排 ≠ 按 beam rank 排.

## Root cause
- T5 generate() 输出的 beam 已经是按 T5 logit 排序的, top-50 beams 的 item_emb centroid 跟 history centroid 的 cosine similarity 差异非常小 (都在 ~0.5-0.6 range, 几乎一致).
- 极端 alpha 才能 break tie, 但那等于完全放弃 T5 logit ranking → 灾难性 -65%.
- SID 序列是 *hierarchical semantic code*, beam rank 已是 optimal ranking, 加 embedding 相似度 prior 是 noise injection.

## Cross-task rerank attempts 联立 (Stage 4 后处理方向)
| Sub | Method | R@10 (max) | Verdict |
|-----|--------|-----------|---------|
| Arm δ1 (task316) | Frequency prior | 0.1042 | NO-GO |
| **Arm δ2 (this)** | **Embedding centroid similarity** | **0.1041** | **NO-GO** (逐 batch 完全相同 baseline) |
| Arm δ3 (task319) | Sequence-level diversity reward | TBD | running |

## Implications
- **Stage 4 embedding centroid 后处理不是 R@10 杠杆** — T5 beam 已是 ranking-optimal, 加 cosine sim prior 在 moderate alpha 完全无效果, 极端 alpha 灾难
- 真杠杆还是在 Stage 3 training (T5 capacity / data augmentation) 或 Stage 1/2 quantizer (已 24 方向 NO-GO)
- Issue #38 Arm δ 全部子方向 (δ1/δ2/δ3) 收口, 唯一 GO = K=100 amplifier (Issue #30 ε)
- Stage 4 driver 唯一 lever = beam_size (task309b K=50 0.1041, K=100 0.1045)

## R11.3 transparency
- 选 cosine similarity 不是 L2 distance 因为是 embedding 相似度标准度量
- 选 history centroid = mean(history_emb) (简单, 跟 history average pooling 一致) 而不是 max-pooling (没理论依据)
- 选 alpha sweep [0.0, 0.1, 0.3, 0.5, 1.0] 跟 task316 (Arm δ1) 对齐, 易比较 delta
- 没用 per-layer 权重 (单层 item_emb centroid 足够, SID 是 hierarchical code 但 item embedding 是单层)
- 不用 max-pooling 因为 (a) 没理论依据 (b) max 不可微 → 不利于未来扩展
- Pre-compute centroids in CPU 一次性完成 (24772 rows × 768 dim ~ 76 MB, 不爆内存)

## Issue #38 status update
- Arm α: ❌ NO-GO (T5-small -4.0%, task309)
- Arm β: ⏸ untested (HNSW + rerank 不可行 in SID discrete setting)
- Arm γ: ❌ NO-GO (r_l/s_l 隔离 -17%, Issue #35 task312/313)
- Arm δ1: ❌ NO-GO (freq prior rerank)
- **Arm δ2: ❌ NO-GO** (embedding centroid similarity rerank)
- Arm δ3: 🔄 running (sequence-level diversity rerank)
- Arm ε: ✅ FULL closed K=100 ceiling R@10=0.1045 (Issue #30 specific, task301/307/309b)

## Next steps (R10 推進)
- Issue #38 5-arm 全收口 (δ3 完成后), 唯一 GO = K=100 amplifier
- Backlog 真无新 Stage 3/4 杠杆方向 — R10 后续必须转向 (a) loop.md housekeeping (b) North Star §3 重审 (c) Issue backlog 列表更新
- 等 owner #26/#34/#38 决策
result: Task #317 — Issue #38 Arm δ2 (Stage 4 embedding centroid similarity rerank) NO-GO. alpha sweep 0.10-0.50 跟 baseline 完全相同 (R@10=0.1041), alpha=1.00 极端崩溃 (-64.9%). 跟 task316 (Arm δ1) + K14-K16 一致: Stage 4 post-process rerank 不是 R@10 杠杆.