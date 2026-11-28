# Issue #284 — DIAGNOSIS

## Observation from #283 (the failure phenomenon)

The fixed candidate `all1_chain` improved OOF MAE 0.002972 → 0.002699 (−9.1 %) and ρ 0.667 → 0.767,
but its paired SID-cluster bootstrap CI [−1.06e−4, +6.35e−4] contains 0; the inner-selected model
(0.002822/0.716) also fails. The fixed arm was observed post hoc (it was a secondary result of #283),
so its point estimate is not a pre-registered primary result. The marginal-matched chain contrast
(509 pairs, |Δ|≤0.005 on the three rates) is +0.191 [0.045, 0.325], i.e. a weak positive signal under
equal marginals.

## Candidate root causes (competing, distinguishable)

| id | explanation | discriminating measurement |
|---|---|---|
| R1 | Real conditional effect: chain ALL carries information beyond the three marginals and the six metrics. | primary matched set (marginals + six metrics + L2/collision/L3/occupancy); ALL_norm residual |
| R2 | Selection/overfitting on 129 SIDs: the fixed arm is the best of several observed candidates. | inner-selected vs fixed arms; BH correction; power analysis |
| R3 | The association is marginal-driven (ALL is collinear with the pairwise rates). | ALL_norm; matched marginals; collinearity with l1_pos/neg/Hard ER |
| R4 | A few high-recall or weak-identity SIDs dominate. | leave-one-out range of the matched Spearman; strong-only subset; recall-band splits |
| R5 | L1 sharing without L2/L3 refinement over-merges (the issue's hypothesis): the association should be conditional on `cond_refine`/`p3_split`. | conditional bands + candidate D |
| R6 | The effect is real but underpowered at n=129. | observed effect vs required n / minimum detectable effect |

R1 vs R3 is separated by the marginal-conditional residual and the primary matched set; R2 by
selection and BH; R4 by leave-one-out and strong-only; R5 by the conditional bands and D; R6 by the
power calculation. No infeasibility claim is made from #283's tested set.

## Evidence review (existing artifacts)

- #283 `MODEL_COMPARISON.json`: fixed `all1_chain` 0.002699/0.7665; selected 0.002822/0.7156;
  baseline 0.0029719/0.6666; lift variants significantly worse.
- #283 `MATCHED.json`: marginal-matched chain +0.191 [0.045, 0.325]; 146-pair matched `all3_chain_lift`
  +0.357 [0.043, 0.598]; other contrasts cover 0; residuals: triangle −0.306 (q=0.004), all3 +0.242
  (q=0.026).
- #283 `JOIN_TABLE`: per-SID chain counts, transitions (only 12 allowed cells), LCP; the conditional
  refinement metrics are computable exactly from them.
- Labels: 129 (120 strong / 9 weak; 13 high with 6 strong), 5 folds; 124 prospective SIDs. Two new
  Stage3 results are bindable by exact SID and run identity (R2 114 pairrank_enc, R2 115
  codebook_budget_j) → time-postordered labels, **not** blind test.

## Formal-training admission six questions (analysis-only; answered for completeness)

1. **Problem**: is the chain-ALL point estimate a real conditional effect or an artifact, and does it
   depend on subsequent refinement?
2. **Evidence**: #283's CI contains 0 while the marginal-matched contrast is positive; the fixed arm
   was post hoc.
3. **Approach**: exact marginal-conditional residual, stricter frozen matching, conditional
   refinement decomposition, leave-one-out, inner selection and power — all replayable from #283 data.
4. **Difference from #283**: adds the conditional residual (not just ALL), the L2/L3 refinement
   interaction, the strict primary matched set, and the power / required-n analysis; the selection
   budget is unchanged (3 candidates, inner-only).
5. **Metrics**: SID-cluster paired bootstrap CI of MAE improvement, secondary ranking/confusion
   criteria, conditional-band associations with CIs.
6. **Knowledge if it fails**: whether the signal is marginal-driven, refinement-dependent,
   few-SID-driven, or underpowered — each is a distinct boundary for the next round and is recorded.

No Stage2/Stage3 training is performed (issue §范围, §4).
