# Issue #286 — DIAGNOSIS

## Observation from #285

The joint low-N low-C cell (34 SIDs, 11 of the 13 high recalls) shows a very strong rank association
(P 0.951 vs 0.240 in the high-high cell; interaction Δρ +0.711 [0.334, 1.075]) but every linear
conditional predictor failed the paired MAE gate (best C: 0.0027032, CI [−7.63e−5, +6.36e−4]), and
P was at least as informative as the LLM chain ALL. The natural next question: the two regimes may
have **different response shapes**, not merely different slopes, so a two-branch (mixture-of-experts)
predictor could capture what the additive interaction could not.

## Candidate explanations (distinguishable)

| id | explanation | discriminating measurement |
|---|---|---|
| E1 (H51) | Two-regime response: a gated expert in the low-N low-C region plus a global expert generalizes. | main candidate E's paired CI and secondary metrics; per-fold gate/expert coefficients |
| E2 (H52) | The gate is redundant: N/C are 0.864-collinear and the six metrics already carry the information. | ablations (N-only, C-only gates; P-only/ER-only good expert); comparison arms B/C/D |
| E3 (H53) | Overfitting / regression to the mean: 34 LL SIDs (27 strong, 11 high) cannot support two experts. | effective support Σg, per-fold weight distribution, λ selection stability, LOO/strong subset |
| E4 | Range restriction and label aggregation: the LL cell's wider recall range and the protocol-majority labels inflate the apparent gain. | residual intervals, cell errors, the #285 truncation result (P 0.951→0.878) |
| E5 (H55) | Gate classification is not regression: a label-learned gate can have good AUC/Brier while the regression stays unchanged. | diagnostic label gate AUC/Brier vs the regression CI |

## Cheap evidence already available

#285: LL 34 / HH 39 / LN 10 / LC 7; P 0.951 [0.861,0.987] in LL vs 0.240 [−0.128,0.602] in HH;
strong-only 0.930; without the high tail 0.886; LOO 0.946–0.973; N–C 0.864 and PC1 0.888. The #277
baseline is 0.002971939/0.666609 with 13/0/0 and a 0.065 NOT_EVALUABLE label range [0.0260, 0.0648].

## Formal-training admission six questions (analysis-only; answered for completeness)

1. **Problem**: can a gated two-regime predictor with no new inputs beat the six-metric Ridge6 under
   the frozen paired gate, given the strong but non-linear conditional rank structure?
2. **Evidence**: #285's interaction is strong while its additive conditional models fail; a shape
   difference (not a slope difference) is the untested alternative.
3. **Mechanism**: a region probability gate (label-free) mixing a P/ER expert with a six-metric
   expert, implemented as residual experts shrinking to R6 with cross-fitted residuals and an inner
   λ selection; every parameter is training-fold-only.
4. **Difference from #285**: different experts and a cross-fitted residual architecture instead of
   additive gate terms; no new inputs; the same evaluation protocol.
5. **Metrics**: the paired SID-cluster MAE CI (main) plus ranking/confusion secondaries and gate
   calibration diagnostics.
6. **Knowledge if it fails**: whether the gate is redundant (ablations), the experts overfit (support
   and λ stability), or the conditional relation is already absorbed — each narrows the next round.

No Stage2/Stage3 training is performed.
