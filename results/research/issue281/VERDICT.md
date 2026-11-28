# Issue #281 — VERDICT

**Decision: negative against the pre-registered criterion, with a descriptive shape finding.**
No nonlinear/piecewise group (nor the inner-selected main candidate) improves the OOF MAE
significantly; #277 remains the frozen predictor.

## C1–C11

| C | evidence | verdict |
|---|---|---|
| C1 | six features verified, 129/120/9/13/6, label range 0.0260–0.0648; A reproduces the #276 ridge6 (0.003059/0.616); the #277 corrected 0.002972 differs by the inner α rule (documented) | PASS |
| C2 | model groups/transforms/inner rule/criterion frozen in `CONTRACT.md` before fitting; no outer selection | PASS |
| C3 | H24–H27 frozen in `DIAGNOSIS.md`; budget bounded by the fixed group list | PASS |
| C4 | only the six features; ER = pos/neg dependence stated; no external indicator | PASS |
| C5 | A/B/C/D/F (+E declared NOT_RUN with the budget reason); knots from training quantiles with 20 % side support | PASS |
| C6 | partial curves, fold interaction coefficients, knot instability, calibration bins, residual quantiles | PASS |
| C7 | 5 SID-group folds, all fits/preprocessing inside; one OOF per SID; single inner-selection rule for the main candidate | PASS |
| C8 | auto vs A: ΔMAE −2.7e-5, CI [−4.9e-4, +3.5e-4] → fail; strict subset consistent; 0.065 NOT_EVALUABLE | PASS (negative) |
| C9 | answers above; no law claimed | PASS |
| C10 | no V2, no new prospective predictions | PASS |
| C11 | deliverables + infra zero drift | PASS |

## Hypotheses

* **H24 INCONCLUSIVE**: the quadratic additive model removes ~2/3 of the low-end bias and raises ρ to
  0.704, but the MAE gain is not significant (CI [−4.5e-4, +4.4e-4]); the pre-registered main
  candidate is worse than A.
* **H25 INCONCLUSIVE**: the interaction model is directionally best on MAE (+1.14e-4, CI [−3.8e-5,
  +2.86e-4]) but its ρ equals A's and the coefficients are fold-unstable.
* **H26 INCONCLUSIVE**: hinges reduce the low-end bias only mildly and worsen q90; no stable knot.
* **H27 SUPPORTED**: the inner-selected complexity does not survive the outer paired CI; A remains the
  reference.

## Closure checklist (§6.4)

1. Production code: `models281.py`, `extras281.py` executed. **YES**
2. Verification: design-matrix and calibration hand tests (3/3), metrics recomputed from stored OOF
   predictions. **YES**
3. Training: none (issue forbids Stage2/Stage3). **YES (N/A)**
4. Infra: FROZEN; snapshot comparison added/removed/changed = 0. **YES**
5. Executable work left in scope: none — negative result, #277 preserved. **NO**
