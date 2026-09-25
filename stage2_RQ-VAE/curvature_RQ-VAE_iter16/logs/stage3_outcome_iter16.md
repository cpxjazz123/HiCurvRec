# stage3_outcome_iter16 (Agent E — downstream eval)

**Run**: `results/stage3_T5Train/curvature_RQ-VAE_iter16/logs/Amazon_2023_Instruments/Sep-26-2026_00-03-22`

| Field | Value |
|-------|-------|
| n_eval | 57439 |
| train epochs | 150 (final train loss = 1.898) |
| best_checkpoint | `.../Sep-26-2026_00-03-22/HG_Rec_best.pth` |
| test_recall@5 | 0.03794 |
| test_recall@10 | **0.05688** |
| test_ndcg@5 | 0.02551 |
| test_ndcg@10 | 0.03159 |

**Gate**: hard `test_recall@10 > 0.065` → **FAIL** (gap = -0.00812).

**Interpretation**: Closed-form fixed `[c0=0.663, c1=0.435, c2=0.433]` is the **worst Stage-2 mechanism** so far. Even iter15 (0.0582) outperformed it by +0.0014. The mid-cycle cyclic c0 (~0.05) is much smaller than the formula's c0=0.663, which suggests fixed higher L0 curvature loses the sharp geometry advantage at the assignment step.
