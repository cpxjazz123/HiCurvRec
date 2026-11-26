## Stage3 result: relative-ranking arm is a negative result

One frozen 150-epoch Stage3 run, four-rank DDP, seed42, beam20, no retry:

| metric | #267 new | parent #265 (109) |
|---|---:|---:|
| test_recall@5 | 0.03300893 | — |
| **test_recall@10** | **0.05078431** | 0.05259493 |
| test_ndcg@10 | 0.02724479 | — |
| n_eval | 57439 | 57439 |

Delta `-0.00181062` (`-3.4426%`). Run directory:
`results/stage3_T5Train/TIGER_relative_curvature_ranking_rq/logs/Amazon_2023_Instruments/Nov-26-2026_17-58-19`.

### Stage2 and mechanism diagnostics

- Run `stage2_88816eeca29442c7b3284f4d4a68f610`, 3000 epochs, four ranks, seed42:
  final collision `0.04534917`, wall `811.659 s` (parent `813.357 s`).
- §63 layer record: L3 collision `0.04547118`, L1 Hard ER `8.3595751`,
  L1 positive hit `0.06100385`, 57439 valid rows.
- Independent full comparison vs #265: q Pearson `+0.02130` vs `-0.03310`;
  q Spearman `-0.03373` vs `-0.09342`; q MAE `2.20395` vs `2.24595`;
  within-centre q median `0.09119` vs `0.08695`; q prediction variance
  `0.88073` vs `0.88857` (target `2.87297`).

The relative term moved ordering in the intended direction but did not recover
full-table positive association, variance, or recommendation gain.

### Prospective score check (frozen before Stage3)

S `1.2921335084032926` was again above the ten-arm range and therefore predicted
the highest score / last recall rank. Actual recall rank is **6/11**. Combined
with the ten historical arms: Pearson `0.513532`, Spearman `0.863636`,
Kendall `50/5` (tau `0.818182`), versus `0.928526` / `1.0` / `45/0` on the ten.
Five inversions (`101, 104, 100, 107, 106`). Reported as-is, no refit or clipping.

### Delivery

- Evidence: `results/research/issue267/REPORT.md`, `VERDICT.md`,
  `analysis/STAGE3_RECALL_11.json`, `analysis/FULL_COMPARISON_109_NEW.json`,
  `analysis/LAYER_RECORD.json`.
- R2: `r2:genrec/110/` (115 files, four-part layout, recall keys 23).
- Infra end diff: PASS (0 changed, 0 missing).
