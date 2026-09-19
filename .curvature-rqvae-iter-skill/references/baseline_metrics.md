# Baseline Metrics for Promotion Gates

Last measured on 2026-09-15, Amazon-2023 Instruments dataset.

## SID Quality (Stage 2 output)

| Method | Collision Rate | Occupancy Gini | Embedding HitRate@500 |
|---|---|---|---|
| TIGER_RQ-VAE  | 7.04%  | 0.0672 | 0.4472 |
| HG_RQ-VAE     | 1.59%  | 0.0157 | 0.4472 |
| curvature_RQ-VAE (current) | 3.93% | 0.0381 | 0.4472 |
| LETTER_RQ-VAE | 30.48% | 0.2705 | 0.4472 |

**Note**: HitRate is identical because all 4 share the same input embedding (stage1 output). It measures embedding health, NOT SID quality. The differentiators are Gini + collision.

**Promotion gate**:
- `Gini <= 0.038 + 0.005 = 0.043`
- `collision_rate <= 3.93% × 1.10 = 4.32%`
- `HR@500 >= 0.4472 - 0.005 = 0.4422`

## Downstream GR (Stage 3) — from previous v318 / v317 runs

| Metric | Value |
|---|---|
| test_R@10   | 0.1179 (v318) — well above 0.065 floor |
| test_R@20   | 0.1533 |
| test_NDCG@20| 0.0932 |
| valid_ndcg@20 (Stage 3 epoch ~100) | 0.1005 |

These were achieved with the **v318 cyclic-c curriculum**. Any iteration must beat or match these.

**Success target**:
- `test_R@10 > 0.065`
- `valid_ndcg@20 >= 0.070`

Since v318 baseline already exceeds both, the bar is in practice: **don't regress** the v318 numbers. If a new mechanism lands within 5% of v318 valid_ndcg@20 (≥0.095), it is a viable candidate for downstream stage3 full run.
