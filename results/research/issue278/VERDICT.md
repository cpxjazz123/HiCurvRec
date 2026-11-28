# Issue #278 — VERDICT

**Decision: negative result.** No importance-weighted variant passes the pre-registered standard; the
#277 six-feature Ridge remains the frozen predictor and no V2 is fitted.

## Answers to the final questions

1. **Does importance provide independent predictive information?** No. The weighted fidelity is
   collinear with the uniform one (Spearman 0.87) and with group popularity (0.917); its correlations
   with the control OOF residual are −0.05 (ER_w), −0.08 (lift), −0.065 (ΔER) — no explanatory power.
   The best candidate E (A + `P_w`, `N_w`) lowers OOF MAE by 0.000203 with a CI covering zero and adds
   one false positive, which the criterion forbids.
2. **Which behaviour relations are worth keeping?** Descriptively, seen transitions preserve the L1
   prefix ~2.2–2.5× more often than unseen ones (high-frequency 17.9 % vs unseen 7.1 %), and the
   unseen stratum is 92 % of the validation rows. But the *which* dimension does not translate into a
   predictor improvement beyond the *how much* measured by the uniform rates: the stratum signal is
   absorbed by popularity and by the existing features.
3. **Does it stably reduce the original over-prediction, rather than fitting history labels?** No. For
   the #274 R2 case the weighted ER is *higher* than the uniform ER (22.9 vs 8.95 at L1), so an
   `ER_w` model would have increased the +0.00524 over-prediction. The residual pattern (low-recall
   arms over-predicted by 0.0034, high-recall under-predicted by 0.0017) is the regression-to-the-mean
   shape already visible in #277, not a weighting effect.

## C1–C12

| C | evidence | verdict |
|---|---|---|
| C1 | pinned #277 artifacts (`ec339f4a…`, `4486b64d…`, `3d007c76…`); R2 prediction 0.05319 / actual 0.04795 / +0.00524 recomputed; R2 SID `9f6bf673…` on disk, no `7923…` file/history exists | PASS |
| C2 | 253/253 SIDs readable; 120 strong / 9 weak; folds unchanged; strict high-recall 6 | PASS |
| C3 | controls A/mean/median/single ER; definitions from the frozen evaluator; same rows/protocol | PASS |
| C4 | training-only edge graph (2 698 387 edges, 323 633 distinct); `w_freq`/`w_cond`/`w_lift`; `h_succ`/`h_pred` defined; strata q33=3/q66=7; unseen 92.4 % | PASS |
| C5 | `P_w/N_w/ER_w` for L1 and L1+L2 under uniform/w_freq/w_lift/w_sqrt/clip; per-SID strata; NA handling; 253/253 coverage | PASS |
| C6 | R2 diagnostic: weighted ER higher than uniform → would worsen the over-prediction; no tuning | PASS |
| C7 | candidates A/B/D/E/F frozen in `CONTRACT.md` before fitting; leak-free scaler; same folds | PASS |
| C8 | E: ΔMAE +0.000203, CI [−0.000016, +0.000452] (covers 0) and 1 false positive → **fails**; D/F null; B worse; strict subsets consistent | PASS (negative) |
| C9 | residual correlations ≤0.08 for ER variants, −0.124 for popularity; `ER_w` ↔ popularity 0.917; stratified residuals reported | PASS |
| C10 | no V2: the standard failed; #277 predictor untouched; negative result delivered | PASS |
| C11 | no new prospective predictions (standard failed); #277 predictions remain; known labels not blind | PASS |
| C12 | DIAGNOSIS/HYPOTHESES/CONTRACT, importance/weighted tables, model comparison, degree diagnostic, code, tests, REPORT/VERDICT, infra drift | PASS |

## Hypotheses

* **H12 REFUTED**: importance-weighted fidelity does not explain the OOF residual beyond the six
  features (|ρ| ≤ 0.08; collinear with the uniform ER 0.87 and popularity 0.92).
* **H13 REFUTED**: no candidate meets the criterion (E's CI covers 0 and it adds one FP).
* **H14 INCONCLUSIVE**: one prospective label cannot separate noise from a systematic weighting
  failure; the model-level null is stronger evidence, and the R2 weighted-ER direction is adverse.
* **H15 INCONCLUSIVE**: popularity correlates with the residual (−0.124) slightly more than ER
  (−0.049), but the two are rank-collinear at 0.917, so they cannot be separated in this queue.

## Closure checklist (§6.4)

1. Production code: `importance278.py`, `model278.py`, `degree278.py` executed as written. **YES**
2. Tests/verification: candidate metrics recomputed from OOF predictions; confusion tables computed
   from the predictions; inputs hashed. **YES**
3. Training: none (issue forbids Stage2/Stage3). **YES (N/A)**
4. Infra: FROZEN; snapshot comparison added/removed/changed = 0. **YES**
5. Executable work left in scope: none — negative result, no V2, #277 preserved. **NO**
