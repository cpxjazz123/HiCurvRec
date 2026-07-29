# Task #319 — Issue #38 Arm δ3 (Stage 4 sequence-level diversity reward rerank) NO-GO

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** — moderate alpha (0.30-0.80) 跟 baseline 完全一致 (R@10=0.1041), alpha=1.00 极端崩溃
**Stage**: Stage 4 rerank hook (无 retraining, 即时验证)

## Settings
- ckpt: task301 Issue #30 GO (per-layer Codebook Transforms r_l=[0.1,1,10] + s_l=[2,2,2])
- code_path: `_t5_hrqvae_issue30_per_layer_transforms.npy`
- Beam size: 50 (Issue #30 K=50 ceiling)
- GPU 1, 总耗时 ~344s (5 alpha × ~69s)
- Rerank hook: diversity_score = 0.5 * intra_beam (unique/total items per beam) + 0.5 * inter_beam (1 - pairwise shared fraction), combined with beam_index

## Metric table (alpha sweep)
| alpha | R@5 | R@10 | R@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|-----|------|------|--------|---------|---------|
| 0.00 (no rerank) | 0.0828 | 0.1041 | 0.1310 | 0.0704 | 0.0772 | 0.0840 |
| 0.30 | 0.0828 | 0.1041 | 0.1310 | 0.0704 | 0.0772 | 0.0840 |
| 0.50 | 0.0828 | 0.1041 | 0.1310 | 0.0704 | 0.0772 | 0.0840 |
| 0.80 | 0.0828 | 0.1041 | 0.1310 | 0.0704 | 0.0772 | 0.0840 |
| **1.00 (pure diversity)** | **0.0208** | **0.0455** | **0.0844** | **0.0116** | **0.0195** | **0.0291** |

## Verdict
**Arm δ3 sequence-level diversity rerank NO-GO** — 两段:
1. **alpha=0.30-0.80 (mild-moderate-strong bias)**: 全部跟 baseline **完全相同** (R@10=0.1041). 这跟 task317 (Arm δ2) 完全一致 — moderate alpha 不影响 beam argmax 顺序.
2. **alpha=1.00 (pure diversity)**: R@10=0.0455 (-56.3%) — 完全没意义, 按 diversity 排 ≠ 按 beam rank 排.

## Root cause
- 跟 task317 (Arm δ2 embedding centroid) 完全一致: T5 generate() top-50 beams 的 diversity 跟 beam rank 的相对差异太小 (inter-beam diversity 集中在 ~0.5-0.6 range, intra-beam diversity 集中在 ~0.7-1.0 range).
- 极端 alpha 才能 break tie, 但放弃 T5 logit ranking 灾难性.
- 跟 Arm δ1 (freq prior) + Arm δ2 (centroid) 联立: **3 种不同 prior (freq / embedding / diversity) 都不能在 moderate alpha 下显著 reorder top-50 beams**. Stage 4 post-process 不是 R@10 杠杆.

## Cross-task rerank attempts 联立 (Stage 4 后处理方向)
| Sub | Method | R@10 (max) | Verdict |
|-----|--------|-----------|---------|
| Arm δ1 (task316) | Frequency prior | 0.1042 | NO-GO |
| Arm δ2 (task317) | Embedding centroid similarity | 0.1041 | NO-GO (逐 batch 完全相同 baseline) |
| **Arm δ3 (this)** | **Sequence-level diversity reward** | **0.1041** | **NO-GO** (逐 batch 完全相同 baseline) |

## Implementation notes (R11.5 透明)
- v1 (initial Python set-based diversity, O(K²) per beam-pair): per-batch 13s, total ~100 min — 太慢, kill
- v2 (vectorized via index_put_ + chunked L1 distance, O(B×K×K×V) chunked at 32 batch): per-batch 1.4s, total ~5.8 min — 可接受
- v2 architecture: bag[bk, v] = count of item v in beam bk, then chunked (32 batch) compute pairwise |bag_i - bag_j|_1 → intersect_count = (count_i + count_j - L1) / 2 → shared_fraction = intersect / max(count_i, count_j)
- v2 diversity_score = 0.5 * intra + 0.5 * inter (R11.5 自选 weight, 因为 intra 和 inter 都 [0,1] 同 scale)
- alpha sweep [0.0, 0.3, 0.5, 0.8, 1.0] — 跟 task316 (δ1) 起点 0.0 不同, 起点 0.0 是 sanity baseline

## Implications
- **Stage 4 diversity reward 后处理不是 R@10 杠杆** — 跟 freq / centroid prior 一样
- 3 种 prior (freq / embedding / diversity) 在 moderate alpha 全部无法 reorder top-50 beams, 极端 alpha 全部灾难性
- Stage 4 driver 唯一 lever = beam_size (task309b K=50 0.1041, K=100 0.1045)
- Issue #38 5-arm 全收口 (δ1/δ2/δ3 NO-GO + ε GO marginal), 真杠杆还是 Stage 3 training / Stage 1-2 quantizer (已 24 方向 NO-GO)

## Issue #38 status update
- Arm α: ❌ NO-GO (T5-small -4.0%, task309)
- Arm β: ⏸ untested (HNSW + rerank 不可行 in SID discrete setting)
- Arm γ: ❌ NO-GO (r_l/s_l 隔离 -17%, Issue #35 task312/313)
- Arm δ1: ❌ NO-GO (freq prior rerank)
- Arm δ2: ❌ NO-GO (embedding centroid similarity rerank)
- **Arm δ3: ❌ NO-GO** (sequence-level diversity reward rerank)
- Arm ε: ✅ FULL closed K=100 ceiling R@10=0.1045 (Issue #30 specific, task301/307/309b)

## Next steps (R10 推進)
- Issue #38 5-arm 全收口, 唯一 GO = K=100 amplifier
- Backlog 真无新 Stage 3/4 杠杆方向 — R10 后续必须转向 (a) loop.md housekeeping (b) North Star §3 重审 (c) Issue backlog 列表更新
- 等 owner #26/#34/#38 决策
result: Task #319 — Issue #38 Arm δ3 (Stage 4 sequence-level diversity reward rerank) NO-GO. alpha sweep 0.30-0.80 跟 baseline 完全相同 (R@10=0.1041), alpha=1.00 极端崩溃 (-56.3%). 跟 task316 (δ1) + task317 (δ2) 一致: 3 种 prior 全部 NO-GO. Stage 4 post-process 不是 R@10 杠杆.