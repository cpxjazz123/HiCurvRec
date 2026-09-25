# stage3_outcome_iter15 (Agent E — downstream eval)

**Run**: `results/stage3_T5Train/curvature_RQ-VAE_iter15/logs/Amazon_2023_Instruments/Sep-25-2026_22-46-35`

| Field | Value |
|-------|-------|
| n_eval | 57439 |
| train epochs | 150 (final train loss ≈ 1.880) |
| best_checkpoint | `.../Sep-25-2026_22-46-35/HG_Rec_best.pth` |
| test_recall@5 | 0.03787 |
| test_recall@10 | **0.05824** |
| test_ndcg@5 | 0.02538 |
| test_ndcg@10 | 0.03194 |

**Gate**: hard `test_recall@10 > 0.065` → **FAIL**.

**Interpretation**: Frozen residual `u_l` (iter15) ranks **below** iter11 (learnable `u_l`, 0.0598) and iter14 (behavior `b_l`, 0.0594) on the paper proxy metric. Mechanism was cleanly implemented but did not improve the Stage-1 ceiling.
