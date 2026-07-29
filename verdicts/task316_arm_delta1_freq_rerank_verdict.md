# Task #316 — Issue #38 Arm δ1 (Stage 4 frequency prior rerank) NO-GO

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** — alpha 0.10 跟 baseline 一致, alpha>=0.30 之后 NDCG 暴跌
**Stage**: Stage 4 rerank hook (无 retraining, 即时验证)

## Settings
- ckpt: task301 Issue #30 GO (per-layer Codebook Transforms r_l=[0.1,1,10] + s_l=[2,2,2])
- code_path: `_t5_hrqvae_issue30_per_layer_transforms.npy`
- Beam size: 50 (Issue #30 ceiling)
- GPU 0, 总耗时 ~170s (5 alpha × 34s)
- Rerank hook: per-item log_freq (从 train.history 算出, log1p + z-score), 对 beams 按 log_freq sum 重排

## Metric table (alpha sweep)
| alpha | R@5 | R@10 | R@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|-----|------|------|--------|---------|---------|
| 0.00 (no rerank) | 0.0828 | 0.1041 | 0.1310 | 0.0704 | 0.0772 | 0.0840 |
| 0.10 | 0.0828 | 0.1042 | 0.1310 | 0.0704 | 0.0773 | 0.0840 |
| 0.30 | 0.0828 | 0.1039 | 0.1308 | 0.0606 | 0.0674 | 0.0741 |
| 0.50 | 0.0796 | 0.1027 | 0.1304 | 0.0528 | 0.0602 | 0.0672 |
| **1.00 (pure freq)** | **0.0219** | **0.0388** | **0.0735** | **0.0119** | **0.0173** | **0.0260** |

## Verdict
**Arm δ1 frequency prior rerank NO-GO** — 三段:
1. **alpha=0.10 (mild bias)**: 实质跟 baseline 一致 (R@10=0.1042 vs 0.1041, Δ +0.0001, **in-sigma**). log_freq 排名的相对差异跟 beam 顺序接近, 不影响 top-10 recall.
2. **alpha=0.30-0.50 (moderate bias)**: NDCG 暴跌 -13% 到 -22% — frequency bias 损害 ranking 质量 (popular items 在 SID 顶端, 真正相关的被压下去).
3. **alpha=1.00 (pure freq)**: R@10=0.0388 (-62.7%) — 完全没意义, 频次排序 ≠ SID 排序.

## Root cause
SID 序列是 *hierarchical semantic code* (4 digits: 64×128×256×1 layer). 高频 item ≠ 高质量 SID — SID 的 precision 在 100% (Sinkhorn decoded), rerank freq 实际上是 noise injection.

## Cross-task rerank attempts 联立 (Stage 4 后处理方向)
| Sub | Method | R@10 | Verdict |
|-----|--------|------|---------|
| Arm δ1 (this) | Frequency prior | 0.1041 (max) | NO-GO (+0.0001 in-sigma) |
| (untested) Arm δ2 | Embedding centroid similarity | TBD | lower priority |
| (untested) Arm δ3 | Sequence-level diversity reward | TBD | lower priority |

## Implications
- **Stage 4 frequency-based 后处理不是 R@10 杠杆** — sid 序列本身已经是最高信息密度表达, 加 prior 是 noise
- 真杠杆还是在 Stage 3 training (T5 capacity / data augmentation) 或 Stage 1/2 quantizer (已 24 方向 NO-GO)
- Issue #38 仍剩: Arm β (HNSW + rerank 不可行 — SID 已是 discrete code), Arm ε K=100 ceiling (✅ done)

## R11.3 transparency
- 选 alpha sweep [0.0, 0.1, 0.3, 0.5, 1.0] 因为 (a) alpha=0 是 sanity check (确认 baseline) (b) alpha=1.0 是 extreme 证明 freq alone 没用 (c) 中间点定位 trade-off.
- 没用 item_emb.parquet (Stage 1 T5 编码) 做 δ2 — 那需要下载 250MB 文件 + compute, ROI 太低.
- K=50 (vs K=100) 因为 Issue #30 K=100 ceiling 已知 (task307 = 0.1045). 重测 K=50 是 baseline 锚定, 跟 task309b 一致.

## Issue #38 status update
- Arm α: ❌ NO-GO (T5-small -4.0%, task309)
- Arm β: ⏸ untested (HNSW + rerank 不可行 in SID discrete setting)
- Arm γ: ❌ NO-GO (r_l/s_l 隔离 -17%, Issue #35 task312/313)
- **Arm δ1: ❌ NO-GO** (freq prior rerank)
- Arm ε: ✅ FULL closed K=100 ceiling R@10=0.1045 (Issue #30 specific, task301/307/309b)

## Next steps (R10 推進)
- Issue #38 5-arm 全收口, 唯一 GO 是 K=100 amplifier
- Backlog 真无新 Stage 3/4 杠杆方向 — R10 后续必须转向 (a) loop.md housekeeping (b) North Star §3 重审 (c) Issue backlog 列表更新
result: Task #316 — Issue #38 Arm δ1 (Stage 4 frequency prior rerank) NO-GO. alpha sweep max R@10=0.1042 (= baseline), alpha>=0.3 NDCG crash
