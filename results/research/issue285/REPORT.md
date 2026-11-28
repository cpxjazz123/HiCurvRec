# Issue #285 — REPORT

Analysis-only, INFRA_MODE=FROZEN, no training, no Stage3. References fixed: #284
`b106d01e…`, #277 predictor, #283 `801048ca…`; the #284/#277/#283 artifacts are hash-recorded in
`AUDIT_285.json`. The 129-SID cohort, the 5 folds and the 127 near-duplicate clusters are unchanged.

## C1 — recomputed conditions and their overlap

- All numbers below were produced by the project interpreter
  (`genrec_env/bin/python3.10`) through `tools/iteration/execution.py`; logs under `logs/`.
- `ER = P/N` exactly: max |ER − P/N| = **0.0** over 129 SIDs (verified; ER carries no independent
  information beyond P and N). Median ER 6.67.
- **N, C, P are largely one source**: Spearman N–C **0.864**, N–P 0.801, C–P 0.739; the first
  principal component of (N,C,P) explains **88.8 %** of the variance (`eigvals` 2.665/0.288/0.047).
- Fold-local tertiles (training-fold quantiles only, the frozen rule; per-fold edges in
  `AUDIT_285.json`): lowN 44, lowC 41, highN 44, highC 45; the two low sets overlap heavily —
  intersection 34, **Jaccard 0.667**, lowC|lowN 0.773, lowN|lowC 0.829.
- Joint cells (frozen definitions, min support 10): **LL 34** (27 strong, 11 high-recall of which 4
  strong), **LN 10** (10 strong, 0 high), **LC 7** (5 strong, 2 high — **unidentifiable**, below the
  pre-registered support), **HH 39** (39 strong, 0 high), other 39. Cell recall means are close
  (0.050–0.055) but the ranges differ: LL [0.0260, 0.0647], HH [0.0353, 0.0585].
- Global-tercile cross-check reproduces the #284 bands exactly: lowN ρ(ALL,y)=0.907, lowC 0.846.
- Time-postordered sensitivity (2 labels, not blind): both have low N (0.00875/0.00889) and non-low C
  (0.151/0.134) → LN by the global bins; recalls 0.04795/0.04847 sit inside the LL range. Not used
  for any fit.

## C2 — the joint condition, interaction, monotonicity and counterexamples

| cell | n | strong | high | recall range | P ρ [CI] | ALL ρ [CI] | reversals P / ALL |
|---|---|---|---|---|---|---|---|
| LL | 34 | 27 | 11 | 0.0260–0.0647 | **0.951 [0.861, 0.987]** | 0.926 [0.818, 0.970] | 7.7 % / 10.9 % |
| LN | 10 | 10 | 0 | 0.0289–0.0555 | 0.480 [−0.460, 1.0] | 0.365 [−0.678, 0.969] | 29.5 % / 36.4 % |
| LC | 7 | 5 | 2 | 0.0427–0.0648 | 0.536 [−0.500, 1.0] | 0.429 [−0.620, 1.0] | 28.6 % / 28.6 % |
| HH | 39 | 39 | 0 | 0.0353–0.0585 | 0.240 [−0.128, 0.602] | 0.082 [−0.285, 0.465] | 39.5 % / 44.9 % |
| other | 39 | 39 | 0 | 0.0399–0.0563 | 0.434 [0.111, 0.675] | 0.286 [−0.053, 0.577] | 36.1 % / 40.5 % |

- **Interaction (LL − HH, the pre-registered test)**: P Δρ **+0.711 [0.334, 1.075]**; ER +0.828
  [0.483, 1.214]; chain ALL **+0.844 [0.429, 1.210]**; fork +0.858; merge +0.841 — all CIs exclude 0,
  both cells ≥ 10 → the conditional difference is **supported** (not the one-significant-one-not
  fallacy). But see the same-cell comparisons below.
- **P is at least as strong as chain ALL inside LL** (0.951 vs 0.926; fork/merge 0.926 identical),
  and the partial Spearman after rank-residualising N and C within LL is P **0.962 [0.750, 0.980]**,
  ALL **0.925 [0.660, 0.959]**. The ALL-gate model C (below) numerically equals the raw-ALL arm, so
  the gate adds nothing; H39 (ALL-specific value over P) is refuted.
- **No strict monotonic law**: within LL 43/560 comparable pairs (7.7 %) invert for P and 61/560
  (10.9 %) for ALL. The counterexample list is in `GRID_285.json`: e.g. within LL SIDs (11,81) have
  ΔP=+0.0300 with Δrecall=−0.00014; (81,96) have ΔALL=−0.00158 with Δrecall=+0.00355. The correct
  statement is a **probabilistic conditional rank rule**, not a mathematical law.
- **Low axes alone already separate the high tail**: AUC of the raw axis for `recall>0.06` is 0.0975
  for N (i.e. **−N AUC = 0.9025 [0.824, 0.965]**) and 0.0962 for C (−C 0.9038 [0.844, 0.955]);
  11/13 high recalls are in LL, 2 in LC, 0 in LN/HH/other.
- **Range restriction explains part**: truncating LL to the recall interval shared with HH
  [0.03527, 0.0585] leaves n=22 (0 high) with P ρ **0.878** and ALL 0.815 (vs full LL 0.951/0.926);
  the recall SD is 0.0090 (LL) vs 0.0056 (HH). The residual gap to HH (P 0.240, ALL 0.082) is still
  large, so H38 ("only range restriction") is refuted while the aggregation component is real.
- **Robustness**: strong-only LL (27 SIDs, 4 high) P 0.930 / ALL 0.890; without the high tail
  (n=23) P 0.886 / ALL 0.826; leave-one-out P 0.946–0.973, ALL 0.919–0.949 — no single-SID or
  weak-identity dominance.
- The joint cell is not an independent second axis: with N–C 0.864 and PC1 0.888, the "double
  condition" is essentially one low-activity/low-collision axis.

## C3 — pre-registered conditional predictors

| model | extra terms | OOF MAE | ρ | reversals | strict MAE | CI of MAE(A)−model | gate |
|---|---|---|---|---|---|---|---|
| A (#277) | — | 0.0029719 | 0.6666 | 0.2485 | 0.003096 | — | — |
| B (N gate) | G_N + P·G_N | 0.0030144 | 0.6184 | 0.2626 | 0.003151 | [−1.86e−4, +8.9e−5] | fail |
| C (ALL gate) | ALL + ALL·G_N | 0.0027032 | 0.7648 | 0.2023 | 0.002782 | [−7.63e−5, +6.36e−4] | fail |
| D (smooth double) | P·G_N·G_C | 0.0030102 | 0.6584 | 0.2514 | 0.003124 | [−9.89e−5, +1.5e−5] | fail |
| B_C alt (C gate) | G_C + P·G_C | 0.0031642 | 0.5696 | 0.2846 | 0.003294 | [−3.71e−4, −3.9e−5] | worse |
| ref #284 raw ALL | ALL | 0.0026990 | 0.7665 | 0.2029 | 0.002778 | [−7.65e−5, +6.41e−4] | fail |
| selected (C 4/5 folds, B 1) | — | 0.0028176 | 0.7087 | 0.2278 | 0.002921 | [−1.51e−4, +4.69e−4] | fail |

Exact reproductions: A 0.002971939/0.666609; the #284 raw-ALL arm 0.002699006/0.766546. Every
>0.06 confusion is 13/0/0; 0 rows > 0.065 → **0.065 NOT_EVALUABLE**. The ALL-gate arm is numerically
the #284 candidate plus a redundant gate (MAE difference 4.2e−6); the P-gate and smooth-double arms
are worse than A; the C-gate alternative is significantly worse. **All gate=true is false.**

**Power**: the best arm C has per-SID mean improvement 2.69e−4 with SD 2.11e−3 → **383 independent
SIDs for 80 % power** (MDE at 129 = 4.63e−4 > observed); the selected model needs 847. The
conditional stratum has only 34 SIDs, where the selected model's within-LL MDE is 2.2e−4 — far above
the within-LL mean improvement of 7.4e−5. Conditional-only validation is not feasible at this n.

**Selection optimism**: the 129 cohort has been reused across #276–#285; every number here is
exploratory/internal and does not prove future validity.

## C4 — frozen research candidate (not a replacement)

`PROSPECTIVE_285.json` freezes the best pre-registered arm (C, ALL + ALL·G_N) with
`G_N = 1[N ≤ 0.00890656]` (full-cohort q1/3), alpha 0.01 and the recorded raw-unit coefficients
(ALL −0.11248, ALL·G_N +0.02477, same six as #277). It carries predictions for the **122
label-pending SIDs**, side by side with the #277 and #284 predictions, and the 2 time-postordered
labels as sensitivity: #277 0.05319/0.05083, #285 0.05300/0.05009, actual 0.04795/0.04847 (both
over-predict; n=2, no formula change). The candidate is marked
`research_candidate_exploratory_redundant_with_284`; #277, the 124-row prospective file and the #284
candidate are unchanged. Replacement of #277 requires the historical gate **and** a future
separately-authorized independent-label validation.

## Answers

1. **Must low negative sharing and low collision both be present?** No. The two axes are one source
   (N–C 0.864, PC1 88.8 %, low-set Jaccard 0.667); low N alone contains 11/13 high recalls, low C
   adds 2, and the both-high cell contains none. The joint cell is a concentration, not an
   independent second condition.
2. **Under the condition, do P or ALL add predictive value?** Descriptively yes as a rank rule (P
   0.951, ALL 0.926 within LL; partial after N,C 0.962/0.925), but **P is at least as good as chain
   ALL**, and the ALL gate adds nothing over raw ALL; the P-gate linear model is worse than A. No
   linear predictor increment.
3. **Is there a refutable conditional monotonic law?** No strict monotonicity (7.7 %/10.9 %
   inversions with listed counterexamples); the **conditional rank interaction** is supported by the
   pre-registered Δρ test (P +0.711 [0.334, 1.075]) and is partly, not wholly, range restriction.
4. **Does it reduce the predictor error?** No: every pre-registered candidate's paired CI contains 0
   (the best, C, [−7.63e−5, +6.36e−4]); the conditional predictors are not adopted.
5. **How many truly independent SIDs are needed?** ~383 for the C-arm effect at 80 % power (the
   selected model needs 847); the conditional cell (34 SIDs) cannot be validated at this scale — a
   future independent-label campaign with ≥400 SIDs is needed for the global effect, and a much
   larger one for the conditional claim.
