# Issue #279 — REPORT

Analysis-only, `INFRA_MODE=FROZEN`, no training, no Stage3. Inputs pinned: #276 `SID_TABLE.json`
`4c927914…`, `FOLDS.json` `1a8fb81d…`; #277 `PREDICTOR.json` `ec339f4a…`; #278 tables.

## C1 — data and identity

129 labeled SIDs (120 strong / 9 weak; 13 high-recall with 6 strong), 253 SIDs with a readable SID file,
folds unchanged. Training graph (from `train.parquet` only): 2 698 387 edges, 323 633 distinct;
**92.41 % of the 57 439 validation relations are unseen** (`c(a,b)=0`), matching #278's 92.4 %.

## C4/C6 — higher-order relations (training graph only)

| score | AUC | 95 % CI (anchor-clustered) | degree-matched AUC | AUC seen | AUC unseen |
|---|---|---|---|---|---|
| `two_hop` (weighted a→m→b) | 0.641 | [0.638, 0.644] | 0.660 | 0.891 | **0.624** |
| `two_hop_norm` | 0.640 | [0.636, 0.643] | 0.654 | 0.886 | 0.623 |
| `common_succ` | 0.642 | [0.638, 0.645] | 0.661 | 0.900 | 0.624 |
| `cs_jaccard` | 0.642 | [0.637, 0.646] | 0.656 | 0.908 | 0.624 |
| `common_pred` | 0.649 | [0.645, 0.652] | 0.667 | 0.902 | **0.631** |
| `cp_jaccard` | 0.649 | [0.645, 0.653] | 0.662 | 0.913 | 0.631 |
| `direct` (reference) | 0.536 | [0.533, 0.539] | 0.544 | 0.996 | 0.499 |

Positives = all validation relations; negatives = one fixed legal-uniform negative per row (seed 42).
**Higher-order topology does discriminate unseen transitions** (AUC 0.62–0.63) and the degree-matched
AUC is not lower than the raw one (0.66), so the signal is not a degree/popularity proxy (H2 refuted).

Curvature (fixed 10 000-pair sample, asset lookups): coverage 12.9 % of positives, 2.65 % of negatives;
only 100 pairs have both sides covered; AUC 0.707 on 100 pairs — **not identifiable**, no claim.

## C7 — per-SID higher-order fidelity

All 253 SIDs computed (L1 and L1+L2). Frozen stratum threshold: median `log1p(two_hop)` over unseen
rows with `two_hop>0` = 4.5326 → high 9 347 rows, low 9 291 rows, unseen 53 081 rows.
Per-SID table: `SID_FEATURES.json` (unnormalised L1/L2 positive/negative rates + weighted ERs).

## C9/C10 — pre-registered model comparison (129 rows, same folds)

| model | features | OOF MAE | ρ | >0.06 tp/fp/fn |
|---|---|---|---|---|
| A control | 6 #277 | 0.002972 | 0.667 | 13/0/0 |
| C1 | A + `er_unseen_2hop` | 0.002892 | 0.730 | 13/1/0 |
| C2 | A + `dp_unseen_degmatch` | 0.002898 | 0.704 | 13/1/0 |
| C3 | A + both | 0.002866 | 0.711 | 13/1/0 |
| E278 (analysis control) | A + `P_w,N_w` | 0.002769 | 0.729 | 13/1/0 |

Paired SID-group bootstrap (`MAE(A) − MAE(candidate)`, 2000 reps):

| candidate | Δ MAE | 95 % CI | passes? |
|---|---|---|---|
| C1 | +0.0000797 | [−0.0000601, +0.0002374] | no (CI covers 0; +1 FP) |
| C2 | +0.0000744 | [−0.0001999, +0.0003725] | no |
| C3 | +0.0001057 | [−0.0001085, +0.0003262] | no |
| E278 | +0.0002034 | [−0.0000069, +0.0004717] | no |

Every candidate adds one false positive at the 0.06 threshold and none has a CI lower bound above 0.
**No candidate passes; #277 remains frozen; no new predictor and no new prospective predictions.**

Residual diagnostics (cross-fitted): residual vs `er_unseen_2hop` ρ −0.085, vs `dp_unseen_degmatch`
+0.146, vs `P_w` +0.030, vs `N_w` +0.119. Stratified residuals: low-recall arms over-predicted by
0.00335 (A) → 0.00316 (C1) → 0.00313 (C3); high-recall under-prediction shrinks from 0.00167 to 0.00138.
The candidate features move the known regression-to-the-mean pattern only marginally.

## Answers

1. **Which higher-order relations predict unseen transitions?** Directed 2-hop paths and shared
   predecessors/successors do (AUC 0.62–0.63 on unseen, 0.66 after degree matching); a direct edge is
   uninformative on unseen relations (AUC 0.499).
2. **Does the SID retain them?** Only weakly: the per-SID weighted ER differs from uniform, but adding
   it to the six features lowers OOF MAE by ≤0.0001 with a CI covering zero and a new false positive.
3. **Is there a Recall-prediction increment beyond the six metrics?** No — the six metrics already
   absorb the usable information (H4); the relation-level signal is real but not transferable through
   SID sharing at this sample size.
4. **Does curvature add independent value?** Not identifiable here (12.9 %/2.65 % coverage, 100 pairs);
   no claim either way.
