# INVALIDATED — superseded by the #49 bugfix round

Everything in this directory is a **historical snapshot** of the analysis committed at
`b5d7bc4` (issue #48). It is preserved verbatim and is **not** to be quoted as a
result.

Five concrete implementation defects were found and fixed in issue #49. The most
severe invalidated the headline numbers:

| # | defect | consequence for this directory |
|:-:|:--|:--|
| 1 | `run_arm` scored `torch.arange(n_tr, n_tr + n_dev)` as "dev", but the split is a scattered set of source rows | the "held-out" score was computed on **34,856 training rows (90%)** and only 3,978 real dev rows; 5,155 dev users were never scored |
| 2 | the frequency prior was fitted on `targets[dev_idx]` | the baseline used the labels it was being tested against, so `G_H1` measured memorisation rather than history information |
| 3 | `evaluate_probe` summed `-log` over a whole 256-wide prior row, and ranked the batch axis at L1 | prior CE was ~1,500 nats instead of a cross-entropy, and prior Top-K was meaningless |
| 4 | `load_valid_target_ids` compared an **int** anchor against a set of **(anchor, target)** tuples | nothing was filtered; F3 ran on all 57,439 validation rows instead of the 52,855 frozen held-out positives |
| 5 | the "catalogue-uniform" ambiguity averaged over occupied SID **groups**, not catalogue **items** | for arms with huge groups this was off by a large factor (e.g. 0.715 vs the true 3.723 for `fix_carry_euclid_raw`) |

Consequently the `M4 +0.000985`, `M5 +0.000584` and `M6 -0.000182` MAE results
reported in this directory's `report.md` are **pre-fix numbers and are withdrawn**.
The `L2`/`L3` prior CE columns in `probe_scores.csv` are likewise wrong.

`M0` (the ER-only baseline) and `F2` `D2` were not implicated and are used unchanged
as invariance controls.

The corrected rerun lives in the sibling directory
[`history_hierarchy_predictors_fix48/`](../history_hierarchy_predictors_fix48/), with a
before/after table and the regression tests in
`scripts/test_history_hierarchy_fix48.py`.
