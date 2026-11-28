# Issue #278 — REPORT

Analysis-only, `INFRA_MODE=FROZEN`, no training, no Stage3. Inputs pinned: #277 `PREDICTOR.json`
`ec339f4a…`, `PROSPECTIVE.json` `4486b64d…`, `PROSPECTIVE_OUTCOME.json` `3d007c76…`; #276
`SID_TABLE.json` `4c927914…`, `FOLDS.json` `1a8fb81d…`.

## Conventions

* Importance is computed **only from training behaviour** (`train.parquet`): consecutive history pairs
  plus `(history[-1], target)`; 2 698 387 edges, 323 633 distinct; `q33 = 3`, `q66 = 7` over the
  training edge counts. Strata: `unseen`, `low`, `mid`, `high` (`IMPORTANCE.json`).
* Weighted fidelity uses the exact frozen evaluator formula on the same 57 439 validation rows
  (`CONTRACT.md` §2); `P_w = Σw h⁺/Σw`, `N_w = Σw h⁻/Σw`, `ER_w = P_w/N_w`, same weights for positive
  and negative; NA is reported, never imputed (`WEIGHTED_TABLE.json`).
* Folds: #277 `FOLDS.json`; scaler inside every inner training fold; alpha by inner SID-group CV.

## C1 — pinned predictor and the #274 R2 case

* #277 frozen prediction for R2: **0.0531902589**; actual **0.0479465172**; over-prediction
  **+0.0052437**; inside the empirical interval; `< 0.06` call correct (`PROSPECTIVE_OUTCOME.json`).
* R2's SID on disk is `9f6bf673…` (`_runs/stage2_58090127…/result/item_sids.json`); the Stage3 record
  carries the same hash. **No file in the repository has a `7923…` hash prefix and git history has no
  deletion event for that SID** — the early `7923…` export is not recoverable from disk or git, so it
  cannot be re-bound; the `9f6bf673…` SID is the one that produced the label.
* Six-feature contributions for R2 (from #277): intercept 0.05266, ER +0.0086, `l3_collision` +0.0024,
  `l1_neg_hit` +0.0019, `l2_distinct` −0.0016, `l1_pos_hit` −0.0005 → 0.0532.

## C2 — coverage and identity

253/253 SID rows have a readable SID file (no coverage gap). Strict identity: 120 strong / 9 weak
(7 weak high-recall); strict high-recall coverage 6. The common queue for all model comparisons is the
129 labeled SIDs; folds are #277's.

## C4 — importance statistics (descriptive)

| stratum (`c(a,b)`) | validation rows | share | L1 positive | L1 negative |
|---|---|---|---|---|
| unseen (`c = 0`) | 53 081 | 92.4 % | 0.0709 | 0.00883 |
| low (`0 < c ≤ 3`) | 1 386 | 2.4 % | 0.1703 | 0.00841 |
| mid (`3 < c ≤ 7`) | 886 | 1.5 % | 0.1467 | 0.00822 |
| high (`c > 7`) | 2 086 | 3.6 % | 0.1788 | 0.00736 |

Seen transitions preserve the L1 prefix ~2.2–2.5× more often than unseen ones, but **92 % of the
validation relations are unseen in training**, so frequency weighting can only speak about 7.6 % of
the rows. `w_lift` (conditioned on popularity) is the popularity-corrected variant.

## C5 — weighted fidelity (L1)

* 253/253 SIDs have `er_w`; `corr(ER_w, ER_uniform)` = 0.87–0.95 (Spearman), `mean(ER_w − ER_uniform)`
  = +15.45 (ER is a ratio with small denominators; the shift is large but mostly proportional).
* `ER_w` is strongly collinear with the **group popularity** (mean training in-degree of the items in
  the anchor L1 group): Spearman **0.917** (`DEGREE_DIAGNOSTIC.json`).
* L1+L2 tables are in `WEIGHTED_TABLE.json` (unseen stratum positives are 3.7× rarer at L2).

## C7/C8 — pre-registered model comparison (129 rows, 5 SID-group folds)

| model | features | OOF MAE | ρ | >0.06 tp/fp/fn | strict MAE/ρ | strict conf. |
|---|---|---|---|---|---|---|
| **A control** | 6 #277 features | 0.002972 | 0.667 | 13/0/0 | 0.003096/0.613 | 6/0/0 |
| B replacement | `P_w,N_w,ER_w`+3 | 0.004167 | 0.444 | 3/2/10 | 0.004179/0.350 | 1/2/5 |
| D increment | A + `ER_w` | 0.002981 | 0.661 | 13/0/0 | 0.003097/0.604 | 6/0/0 |
| **E increment** | A + `P_w,N_w` | **0.002769** | **0.729** | **13/1/0** | **0.002872/0.694** | 6/1/0 |
| F sensitivity | A + lift-ER | 0.002998 | 0.665 | 13/0/0 | 0.003121/0.616 | 6/0/0 |
| fold mean / median | — | 0.004669 / 0.004422 | — | — | — | — |
| single ER | `l1_hard_er` | 0.003855 | 0.355 | — | — | — |

Paired SID-group bootstrap (`MAE(A) − MAE(candidate)`, 2000 reps):

| candidate | Δ MAE | 95% CI | passes (CI > 0)? |
|---|---|---|---|
| B | −0.001195 | [−0.001640, −0.000773] | no (worse) |
| D | −0.000009 | [−0.000048, +0.000031] | no |
| E | +0.000203 | [−0.000016, +0.000452] | **no (CI covers 0)** |
| F | −0.000026 | [−0.000066, +0.000015] | no |

**No candidate passes the pre-registered standard.** E is directionally the best (MAE −6.8 %,
ρ +0.062) but its CI covers zero **and** it adds one false positive at the 0.06 threshold (13/1 vs
13/0), which the criterion forbids. D/F are indistinguishable from A; B is worse.

## C9 — do importance statistics explain the control residual?

OOF residual of A vs importance-derived quantities (Spearman, cross-fitted predictions):

| covariate | ρ with residual |
|---|---|
| `ER_w` (L1, w_freq) | −0.049 |
| `ER_uniform` | −0.064 |
| `ER_w_lift` | −0.076 |
| `ΔER = ER_w − ER_uniform` | −0.065 |
| group popularity | −0.124 |

All correlations are ≤0.13 in magnitude: the weighted fidelity does not explain the residual beyond the
six features, and the popularity diagnostic (C9/H15) is weak but collinear with `ER_w` (0.917), so the
two cannot be separated. Residual by label stratum (A): low-recall arms are over-predicted by 0.00335,
high-recall arms under-predicted by 0.00167 — the known regression-to-the-mean pattern, not a
weighting effect.

## C6 — #274 R2 diagnostic (exploratory, no tuning)

For the R2 SID: L1 uniform `ER = 8.95`, `w_freq ER = 22.92`, `w_lift ER = 12.10`; L2 much larger.
The weighted feature is **higher** than the uniform one, so an `ER_w`-based model would push R2's
prediction **up** (0.0532 → higher), i.e. the importance weighting does not explain or fix the
over-prediction; it moves in the wrong direction. This case cannot be used to select weights.

## C11 — prospective readiness

The pre-registered standard failed, so **no V2 is frozen** and no new prospective predictions are made;
the #277 predictor and its frozen predictions remain authoritative. The remaining unlabeled SIDs keep
their #277 predictions; known labels (including R2) are never blind predictions.

## Limits

1. 92 % of validation relations are unseen in training; a frequency-weighted fidelity can only act on
   the 7.6 % seen stratum, and the "which relation" question is mostly about unseen relations.
2. `ER_w` is a ratio with small denominators; the mean shift +15.45 is dominated by a few SIDs, so it
   is reported with the collinearity caveat rather than interpreted as a scale.
3. E's improvement is not significant and adds a false positive; D/F are null.
4. Only one prospective label (R2) exists; the round therefore cannot separate noise from a systematic
   weighting failure on the basis of that case alone — the model-level null is the stronger evidence.
