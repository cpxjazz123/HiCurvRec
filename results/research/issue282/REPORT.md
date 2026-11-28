# Issue #282 — REPORT

Analysis-only, `INFRA_MODE=FROZEN`, no training, no Stage3. Training graph only; validation rows and the
legal-negative definition are the frozen #277 ones.

## C1 — data

57 439 validation rows; **92.41 % unseen**; 129 labeled SIDs (120 strong / 9 weak; 13 high-recall with 6
strict). The baseline `A` reproduces the #277 corrected Ridge6 exactly: **MAE 0.002972, ρ 0.667**
(`inner_alpha_fixed`, #277-AUDIT procedure).

## C2/C3 — pair-level types and coverage

| type | definition | positive rows (of 57 439) | share |
|---|---|---|---|
| `chain` | a→m→b | 22 524 | 39.2 % |
| `revchain` | b→m→a | 23 045 | 40.1 % |
| `merge` | a→m←b (shared successor) | 22 130 | 38.5 % |
| `fork` | a←m→b (shared predecessor) | 23 164 | 40.3 % |
| `closed` | only on observed a→b | 4 358 (seen rows) | — |

Multi-label: most positives carry several types at once (the sets overlap heavily). 4-node graphlets are
**excluded** (sampled asset, no pair-level certification); node-level Orbit totals are not used;
curvature is **excluded** (coverage 12.9 % / 2.65 % from #279).

## C5 — per-SID type-conditioned L1 fidelity

Mean over the 253 SIDs (support ≥ 1 row):

| type | mean support rows | mean P | mean N | mean ER |
|---|---|---|---|---|
| chain | 22 524 | 0.1699 | 0.0708 | 9.99 |
| revchain | 23 045 | 0.1693 | 0.0708 | 9.89 |
| merge | 22 130 | 0.1688 | 0.0708 | 9.79 |
| fork | 23 164 | 0.1678 | 0.0708 | 9.61 |

The type-conditioned ERs differ by **≈4 %** at the mean level; the P values by ≈1 %. Matched-negative
levels (5 draws per row, 287 195 cells): **level 2 (type ∩ + degree-matched) 1.35 %**, level 1 (degree)
11.2 %, level 0 (any legal) 87.4 % — the type-matched negative population is thin, so the type-conditioned
`N_t` is dominated by degree-matched/unmatched negatives (stated as a coverage limit; the #277 uniform
negative probability is the separate, unchanged reference).

Frozen SID contrasts: F1 = ER_merge − ER_chain (mean −0.198), F2 = ER_fork − ER_chain (mean −0.375):
merge/fork retention is slightly **lower** than chain retention across SIDs.

## C6/C7/C8 — model comparison (129 rows, same folds)

| model | OOF MAE | ρ | >0.06 tp/fp/fn |
|---|---|---|---|
| A (277 corrected) | 0.002972 | 0.667 | 13/0/0 |
| D1 = A + F1 | 0.002982 | 0.659 | 13/0/0 |
| D2 = A + F2 | 0.002965 | 0.663 | 13/0/0 |
| D3 = A + both | 0.002984 | 0.666 | 13/0/0 |

Paired SID-group bootstrap (`MAE(A) − MAE(model)`): D1 −9.6e-6 [−4.6e-5, +2.3e-5]; D2 +6.8e-6
[−7.1e-6, +2.2e-5]; D3 −1.2e-5 [−7.3e-5, +3.9e-5]. **All CIs cover 0; no candidate passes.** D1/D2
also lower ρ (0.659/0.663 vs 0.667).

Residual associations (cross-fitted): F1 +0.076, F2 +0.056 — negligible.

Similar-overall contrast (frozen tolerance |Δl1_pos| ≤ 0.01 and relative ER within 20 %): **477 SID
pairs**. Spearman of ΔF1 vs Δrecall = **−0.198**, ΔF2 vs Δrecall = **−0.154**: a weak *negative*
association (higher merge/fork retention relative to chain goes with slightly *lower* recall), in the
opposite direction of the "merge/fork is valuable" hypothesis, and the pairs are not independent
(477 pairs from 129 SIDs). Descriptive only; no claim.

## Answers

1. **With overall L1 fidelity close, do chain/fork/merge/closed retention differences correlate with
   Recall?** Only weakly and negatively (ρ ≈ −0.2/−0.15 on matched pairs; the type-conditioned ERs
   differ by ~4 % at the mean and the matched-negative coverage is 1.35 %). No supported correlation;
   the closed type exists only on 4 358 seen rows and is not a feature.
2. **Which types deserve the next mechanism?** None on this evidence: the type contrasts are nearly
   collinear with the overall rates, the matched-negative sample is dominated by non-type-matched draws,
   and no type contrast improves the prediction.
3. **Reproducible predictive increment?** No — every paired CI covers 0; D2's +6.8e-6 is not significant
   and D1/D2 degrade ρ. #277 remains frozen; no new predictor, no new prospective predictions.
