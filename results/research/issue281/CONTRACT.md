# Issue #281 — pre-registered contract (frozen before any fit)

Frozen 2026-11-28 15:3x. Only the six #277 features enter every model; no external indicator is added.

## 1. Data

The 129 labeled SIDs of #276 (all six features finite), the #277 `FOLDS.json` 5 SID-group folds, the
identity split 120 strong / 9 weak and the 13 high-recall (6 strict). Ridge6 reference numbers are
recomputed from the stored predictions before the new fits (C1).

## 2. Model groups (fixed)

| id | design | complexity |
|---|---|---|
| A | `[x1..x6]`, ridge, inner α | linear reference |
| B | `[x1..x6, x1²..x6²]` (additive quadratics, no cross terms), ridge, inner α | ≤ 6 extra terms |
| C | `[x1..x6, x1·x2, x3·x5]` where x1=`l1_pos_hit`, x2=`l1_neg_hit`, x3=`l1_hard_er`, x5=`l3_collision` (hierarchy kept), ridge, inner α | 2 interactions |
| D | `[x1..x6, hinge(x3 − q3), hinge(x5 − q5)]` with `q3,q5 ∈ {q25,q50,q75}` of the **training fold**, both sides ≥ 20 % support, knots selected by inner CV | ≤ 2 hinges |
| E | shallow tree (depth ≤ 2, min leaf 10 % of training rows) | diagnostic only, **not run** (budget excluded: the pre-registered criterion is decided by A–D/F; recorded as NOT_RUN) |
| F | linear calibration `y = a + b·p` where `p` are the **inner cross-fitted** predictions of A inside the outer training fold; applied to the outer test predictions of A | 2 parameters |

ER = `l1_hard_er = l1_pos_hit / l1_neg_hit` is a deterministic ratio; x1, x2, x3 are never interpreted as
independent causes.

## 3. Inner selection (one rule, no outer winner-picking)

For every outer training fold: leave-one-SID-group-out over its groups; α ∈ {0.01, 0.1, 1, 10} for
A–D; the inner OOF MAE of each model selects both α and the model; tie-break order A < B < C < D < F
(fewer parameters). The selected model is refit on the outer training fold and predicts the outer test
fold. This "auto-selected" OOF is the single main candidate.

## 4. Criterion

`MAE(A) − MAE(auto)` SID-group paired bootstrap 95 % CI **lower bound > 0**, Spearman/Kendall/
reversals not worse, >0.06 FP/FN not worse; strict 120-row subset separately; 0.065 has no positives
(NOT_EVALUABLE). If the criterion fails, #277 remains; no new search.

## 5. Descriptive outputs (not used for selection)

Single-variable response curves and local slopes at p10/p50/p90 from B (quadratic) and D (hinges),
interaction response from C at grid points inside the observed joint range, knot locations across folds,
calibration curve (binned), stratified signed errors, counterexamples.
