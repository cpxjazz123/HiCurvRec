# Issue #281 — REPORT

Analysis-only, `INFRA_MODE=FROZEN`, no training, no Stage3. All models use only the six #277 features.

## C1 — audit

129 labeled SIDs, folds unchanged, 120 strong / 9 weak, 13 high-recall (6 strict); label range
0.0260–0.0648 (0.065 NOT_EVALUABLE). ER = `l1_pos_hit / l1_neg_hit` is a deterministic ratio.
The #281 pipeline's control A reproduces the **#276 stored ridge6** (MAE 0.003059, ρ 0.616); the
#277-audit corrected variant (0.002972/0.667) differs because its inner α rule used the per-group error
list, while #281 freezes the macro-average inner rule with in-fold scalers. Both are documented; the
paired comparison uses the #281 A as the like-for-like control.

## C5 — pre-registered groups (inner selection: B wins 4/5 folds, D 1/5)

| model | design | OOF MAE | ρ | >0.06 tp/fp/fn |
|---|---|---|---|---|
| A | linear Ridge6 | 0.003059 | 0.616 | 13/0/0 |
| B | additive quadratics (x, x²) | 0.003004 | **0.704** | 13/0/0 |
| C | +2 pre-registered interactions | **0.002945** | 0.667 | 13/1/0 |
| D | +2 hinge terms (knots inside) | 0.002937 | 0.639 | 13/1/0 |
| F | linear calibration of cross-fitted A | 0.003105 | 0.614 | 13/1/0 |
| auto | per-fold inner-selected model | 0.003086 | 0.680 | 13/1/0 |

Paired SID-group bootstrap (`MAE(A) − MAE(model)`):

| model | Δ MAE | 95 % CI | passes? |
|---|---|---|---|
| B | +5.5e-5 | [−4.5e-4, +4.4e-4] | no |
| C | +1.14e-4 | [−3.8e-5, +2.86e-4] | no (closest) |
| D | +1.22e-4 | [−4.1e-5, +2.90e-4] | no |
| F | −4.6e-5 | [−1.2e-4, +2.7e-5] | no (worse) |
| **auto (main candidate)** | **−2.7e-5** | [−4.9e-4, +3.5e-4] | **no** |

The pre-registered main candidate (auto) fails: the inner selection is unstable (fold 3 picks D, whose
outer prediction is worse), and no group's CI excludes 0. C and D are the closest (+1.1e-4/+1.2e-4 with
lower bounds just below 0), B improves ρ the most (0.704) but adds no significant MAE gain.

## C6 — shape and calibration

* **Low-recall over-prediction is largely quadratic-additive**: signed error on the low tercile
  A −0.00356 → **B −0.00122** (auto −0.00118); high tercile +0.00186 → +0.00158 (auto +0.00184).
  The additive quadratic model removes ~2/3 of the low-end bias while leaving the high end nearly
  unchanged — the response is not a simple monotone linear effect.
* Piecewise (D) reduces the low-end bias only mildly (−0.00310) and worsens q90 residuals (0.00835 vs
  0.00743 for A); the hinge knots are fold-dependent — no stable breakpoint.
* C's two interactions (pos×neg, ER×collision) have fold-varying coefficients; the interaction model
  improves MAE slightly but not ρ (0.667 = A) — no stable interaction law.
* F (linear recalibration of cross-fitted A) is worse than A, so the bias is not a global linear
  miscalibration; it is concentrated at the low end and partly captured by the quadratic terms.
* Calibration bins (EXTRAS281.json) show the same pattern: A over-predicts the lowest bin by ~0.0035,
  B by ~0.0012.
* Residual q90: A 0.00743, B 0.00737, C 0.00652, D 0.00835.

## Answers

1. **Does nonlinearity add independent value?** Only as a description of shape: additive quadratics
   remove most of the low-end bias and raise ρ to 0.704, but the MAE gain is not significant (CI covers
   0). The strictly pre-registered main candidate (inner-selected mix) is not better than A.
2. **Where does the slope change?** In the low-recall region: the quadratic fit flattens the predicted
   response at low `l1_pos_hit`/high collision (implied partial curves in `CURVES281.json`), matching
   the over-prediction of high-fidelity low-recall arms (e.g. `cogeo_region_argmax`, `dul_rq` in
   #277/#278). No stable hinge breakpoint exists across folds.
3. **Which interactions are stable?** None: the two pre-registered interaction coefficients vary in
   sign/magnitude across folds; C's ρ equals A's.
4. **Can over/under-prediction be reduced?** The low-end over-prediction can be reduced by ~2/3 with
   additive quadratics, but the overall MAE improvement is within noise; the high-end under-prediction
   persists.
5. **Falsifiable conditional law?** Not established with this sample: the shape is suggestive
   (concave response at the low end), but no condition survives the paired CI. A next round would need
   more SIDs in the low-recall band, not another transform.

No new predictor; #277 remains frozen; no new prospective predictions.
