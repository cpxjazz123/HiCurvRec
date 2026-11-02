# Issue #130 - interaction strength, SID prefixes and Recall

110 (SID, run) rows, joined by SID content hash. Provenance: HASH_PROVEN 0, RECORD_HASH_MATCH 110, MISSING 0. Only the admitted 110-pair table was measured, so no wider-tier or strict-only *measurement* is claimed; the strict subset is an analysis subset of the same rows.

## Q1 - does interaction intensity affect SID prefix organisation?

Mass-weighted retention inside the same prefix, minus a size-preserving label-shuffle null (and the same null restricted to activity strata):

| feature | value |
|---|---|
| L1 retention excess | +0.524 (p 0.000, Holm 0.005, n 110) |
| L2 retention excess | +0.016 (p 0.872, Holm 1.000, n 110) |
| L3 retention excess | -0.141 (p 0.137, Holm 0.823, n 110) |
| L1 activity-stratified | +0.525 (p 0.000, Holm 0.005, n 110) |
| L2 activity-stratified | +0.030 (p 0.762, Holm 1.000, n 110) |
| L3 activity-stratified | -0.133 (p 0.162, Holm 0.823, n 110) |

Same-prefix share by pre-declared TRAIN strength tertile (median across arms):

| level | weak | medium | strong | median trend |
|---|---|---|---|---|
| L1 | 0.0410 | 0.0626 | 0.0879 | +0.0401 |
| L2 | 0.0017 | 0.0041 | 0.0081 | +0.0080 |
| L3 | 0.0002 | 0.0006 | 0.0018 | +0.0040 |

Per-arm excess over the null (median across arms): L1 +0.0838, L2 +0.0093, L3 +0.0023. The same-prefix share rises with strength at every level, but only the L1 association with Recall survives the Holm correction: **strong interactions are concentrated inside the first-level prefix, and that concentration is the part that tracks Recall.**

## Q2/Q3 - does strength add beyond binary alignment, and does weighted ORC add more?

Nested models on the matched rows (n=82 for all four), leave-one-mechanism-series-out RMSE:

| model | features | n | RMSE | in-sample R2 |
|---|---|---|---|---|
| `A_controls` | 3 | 82 | 0.00359 | +0.7055 |
| `B_plus_weighted_alignment` | 6 | 82 | 0.00386 | +0.7585 |
| `C_plus_strength_gap` | 9 | 82 | 0.00412 | +0.7938 |
| `D_plus_weighted_orc` | 12 | 82 | 0.00366 | +0.8572 |

| step | n | RMSE before | RMSE after | relative |
|---|---|---|---|---|
| A_controls->B_plus_weighted_alignment | 82 | 0.00359 | 0.00386 | -7.5% |
| B_plus_weighted_alignment->C_plus_strength_gap | 82 | 0.00386 | 0.00412 | -6.7% |
| C_plus_strength_gap->D_plus_weighted_orc | 82 | 0.00412 | 0.00366 | +11.2% |

**Q2.** The weighted-alignment block does not improve out-of-series prediction (-7.5% at the A->B step), and adding the strength gap makes it worse still.

**Q3.** The weighted ORC block recovers part of that loss (+11.2% at the C->D step) but the full model does not beat the controls-only model (D 0.00366 vs A 0.00359).

## Q4 - are the level patterns distinct and stable?

| tier | rows | R1 | R2 | R3 | g1(w) | g2(w) | g3(w) | g1 gap |
|---|---|---|---|---|---|---|---|---|
| strict_A2_73 | 73 | +0.608 (p 0.00) | +0.198 (p 0.09) | +0.040 (p 0.73) | +0.024 (p 0.85) | -0.043 (p 0.72) | -0.242 (p 0.04) | +0.531 (p 0.00) |
| admitted_110 | 110 | +0.524 (p 0.00) | +0.016 (p 0.87) | -0.141 (p 0.14) | -0.159 (p 0.10) | -0.433 (p 0.00) | -0.394 (p 0.00) | +0.616 (p 0.00) |

Sign reversal check: L1 retention is positive in both tiers (+0.524 (p 0.00) vs +0.608 (p 0.00)), while L3 retention is negative in both and L2 is null in both; the direction is stable, the magnitudes are not.

Do the associations survive the resolution controls? (partial Spearman given log quotient edges, the level's prefix count, A_l and collision):

| feature | partial given resolution | Spearman vs log edges |
|---|---|---|
| `R_l1_excess` | +0.346 | -0.262 |
| `gw_l1_exposure` | +0.046 | -0.038 |
| `gap_l1` | +0.482 | +0.393 |
| `R_l2_excess` | +0.314 | -0.823 |
| `gw_l2_exposure` | -0.060 | -0.913 |
| `gap_l2` | +0.285 | +0.776 |
| `R_l3_excess` | +0.322 | -0.868 |
| `gw_l3_exposure` | -0.293 | -0.447 |
| `gap_l3` | -0.015 | +0.850 |

## Verdict

**Q1 (organisation) - yes, and it is level-specific.** Strong training interactions are much more likely to share a prefix: the median per-arm retention excess is +0.0838 at L1 and the same-prefix share climbs across the strength tertiles (0.041 -> 0.063 -> 0.088 at L1). Across arms, that excess tracks Recall at L1 only (+0.524 (p 0.00); L2 +0.016 (p 0.87), L3 -0.141 (p 0.14)).

**Q2 (prediction over binary alignment) - no.** Adding the weighted alignment block to the binary alignment, resolution and collision controls worsens the leave-one-series-out RMSE.

**Q3 (weighted ORC) - no incremental value.** The weighted-ORC block improves on the model with the strength gap but not on the controls-only model, and its own association with Recall is indistinguishable from the resolution proxy in the confound table. Weighted curvature is reported separately from the binary `g_l`/`z_l` of #129 and from any model hyperbolic curvature, which is not trained here.

**Q4 (levels) - distinct in organisation, not in prediction.** L1 is the only level whose retention excess is non-null after Holm, and its in-sample association survives the resolution controls (partial +0.346); the L1 strength gap behaves the same way (partial +0.482). Neither adds out-of-series predictive value (Q2/Q3), and the L2/L3 associations are null univariately, so no level-specific predictive pattern is claimed.

**Scope.** Observational and offline: no Stage 2 loss, tokenizer, codebook or Stage 3 training was touched, and the historical Recall is read-only. A within-prefix concentration of strong interactions is an *organisation* fact about this SID, not a causal geometry benefit.
