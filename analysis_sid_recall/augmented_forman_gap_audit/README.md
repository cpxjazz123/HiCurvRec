# Audit of #136: corrected inference and AF3 negative-edge semantics (Issue #137)

Re-runs the #136 measurements on the same 108 content-hash-matched SID–Stage3
records with the two statistical defects fixed and the negative-curvature
semantics split into the quantities it had conflated. The archived
`analysis_sid_recall/augmented_forman_gap/` artifacts are left untouched and
their p/Holm columns and the mislabelled negative-edge field are **SUPERSEDED**.

## Corrections

| defect | old behaviour | corrected behaviour |
|---|---|---|
| **B1 permutation null** | resampled outcomes **with replacement** (`integers`) and did not re-rank the resampled Spearman sample -- a bootstrap null with duplicated/missing observations | draws a uniform **bijection** of all `n` outcome indices for every draw, evaluates the fixed-rank Spearman and Pearson statistics on the permuted pairing, and reports `p = (1 + #{|stat_perm| >= |stat_obs|}) / (B + 1)`; bootstrap stays a **with-replacement paired** CI and is labelled as such |
| **B2 Holm family** | one adjustment over `min(p_r, p_rho)` across 260 metric/outcome entries, written as a single `holm_p` | `p_pearson`, `p_spearman`, `holm_pearson`, `holm_spearman` adjusted **within each coefficient column** over the full declared family: 66 metric-levels × 4 outcomes × 2 coefficients = **528 potential**, with the actual defined/undefined counts published |
| **B3 negative fraction** | `(per-group AF3 mean < 0).mean()` was called "negative-edge fraction" | three separately named and separately computed fields, plus ordinary-Forman analogues: `group_mean_negative_fraction` (groups whose **mean** is negative), `edge_negative_fraction` (negative **edges** over all eligible induced edges), `mean_group_negative_edge_fraction` (unweighted mean of per-group edge fractions) |

## Frozen inputs

Same TRAIN-only graph, item mapping, SID join, quotient construction, gap
definition and Stage3 outcomes (R@5, R@10, NDCG@5, NDCG@10) as #136; one cached
global edge-curvature pass; `E_within + E_cross + E_unlabelled = E_total` is
checked per row; `gap_af3 − gap_forman = 3(T_within − T_cross)` is checked per
row. R@20/NDCG@20 stay `NOT_EVALUATED`.

## Files

| file | contents |
|---|---|
| `stats_core.py` | bijection permutation null, exhaustive oracle, paired bootstrap, per-column Holm, midranks, partial correlation, VIF, Cook's distance |
| `audit_features.py` | corrected intra/quotient/gap features with the distinct negative-fraction measures |
| `run_measure.py` | 108-arm measurement with checkpoint/resume; semantics and new-vs-old tables |
| `run_stats.py` | both coefficient families with corrected p/Holm/CI, controls, models, null sensitivity, influence, verdict |
| `feature_semantics_audit.csv` | per SID/level: eligible groups, edges, both negative fractions and their Forman analogues |
| `correlations_pearson_spearman.csv` | every test with individual `p_*`, `holm_*`, both CIs, n, coverage |
| `within76.csv`, `model_controls.csv`, `null_sensitivity.csv`, `new_vs_old_correlations.csv` | stability, resolution/controls models, null variants, old-vs-new |
| `tests/` | exhaustive permutation oracle (n=5/6), scipy agreement, perfect ±1, Holm oracle and 520-test gate, edge-vs-group negative fixture, graph invariants, null histogram/degree buckets, join and edge partition, restart determinism |

## Reproduce

```bash
python3 analysis_sid_recall/augmented_forman_gap_audit/run_measure.py --resume
python3 analysis_sid_recall/augmented_forman_gap_audit/run_stats.py
python3 -m unittest discover -s analysis_sid_recall/augmented_forman_gap_audit/tests
```

## Verdict vocabulary

`GAP_ASSOCIATION_ROBUST`, `GAP_REDUNDANT_WITH_FORMAN`, `MIXED/NULL_SENSITIVE`,
`NO_STABLE_GAIN`, `BLOCKED`. The gap remains observational: it does not prove
hyperbolic manifold necessity and does not justify a Stage2 curvature loss.
