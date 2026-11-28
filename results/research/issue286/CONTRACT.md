# Issue #286 — CONTRACT (frozen 2026-11-28, before any model fit)

## 1. Data and audit

Same 129 labelled SIDs, the five frozen folds and the 127 near-duplicate clusters as #277–#285. The
six inputs are the #277 features only (`l1_pos_hit` P, `l1_neg_hit` N, `l1_hard_er` ER, `l2_distinct`,
`l3_collision` C, `l3_refine_bits`); ER = P/N is functionally dependent and is never treated as
independent. No new SID-level metric (ALL, effective rank, curvature, ...) enters any model.

## 2. Gate (pre-registered definitions)

- Primary gate: `g(N,C) = sigmoid(b0 − b1·N_std − b2·C_std)`, `b1,b2 ≥ 0`, with `N_std, C_std`
  standardised by the **training fold** mean/sd. The gate is fitted by a constrained logistic
  regression (L-BFGS-B, bounds `b1,b2 ≥ 0`, L2 penalty C=1.0) to the **feature-region target**
  `lowN & lowC`, where low = the training-fold bottom tertile of N or C. This is a *region
  probability*, never called a high-recall probability.
- Diagnostic gate (secondary, not the main candidate): the same logistic form fitted to the
  training-fold label `y > 0.06`; its outer-fold AUC/Brier/calibration are reported to show whether a
  label-learned gate would even be calibrated.
- Alternative single-variable gates `g_N(N)`, `g_C(C)` (one covariate, same constraint) are
  low-capacity diagnostics/ablations, reported with the same calibration metrics.

## 3. Branches and the main candidate

- `f_good`: ridge on `[P, ER]`; `f_other`: ridge on the six features. Expert ridge alpha is fixed at
  **1.0** (low complexity, pre-registered); the R6 baseline uses the #277 procedure (inner
  SID-group CV over the alpha grid {0.01,0.1,1,10}).
- **Main candidate E (locked): soft-gated residual experts shrinking to R6**
  `Rhat = R6(x) + λ·[g·e_good(x) + (1−g)·e_other(x)]`, where `R6` is fitted per outer fold, the
  experts `e_good`, `e_other` are ridge models on the **cross-fitted** residuals `y − R6_crossfit`
  (inner cross-fitting: each inner group is predicted by the R6 model fitted on the other inner
  groups), and `λ ∈ {0.25, 0.5, 0.75, 1.0}` is selected by the inner SID-group CV (a single
  pre-registered selection strategy; no structure or threshold search).
- Pre-registered structural comparisons (fixed, no selection): A = R6; B = fixed single branch ridge
  on `[P, ER]`; C = hard-gated two-branch (`g_bin = 1[g≥0.5]`, experts fitted on the two training
  subsets); D = soft-gated two-branch `g·f_good + (1−g)·f_other`.
- Ablations (fixed): f_good with P only; f_good with ER only; gate with N only; gate with C only —
  each inside the D formulation. These test whether the double condition and the double expert are
  needed.
- Minimum support: 20 independent SIDs effective (`Σ g` on the training fold); below this the good
  branch is flagged **unidentifiable** (the model still runs, the claim is not made). The hard-gate
  branch requires ≥ 10 training SIDs on each side.

## 4. Evaluation (same 129 SIDs, same folds)

OOF MAE/RMSE/Spearman/Kendall/reversal proportion, low/high-end bias, the >0.06 confusion
(TP/FP/FN/TN), the LL/HH/mixed fold-local cell errors, the strict 120-strong subset, and the
empirical residual interval. The primary gate's calibration (AUC/Brier vs the feature region) and the
diagnostic label gate's calibration (AUC/Brier vs `y>0.06`) are reported separately: a good gate
classification is **not** evidence of accurate recall regression. Paired SID-cluster bootstrap 95 % CI
(2,000 reps, 127 clusters) of MAE(A) − MAE(model) for E and each comparison; BH over the five arm
tests is reported as context. 0.065 has no positive labels → NOT_EVALUABLE.

**Success criterion (main):** the E vs A paired CI lower bound > 0 **and** Spearman/Kendall,
reversal proportion and the >0.06 TP/FP/FN not worse. If it fails: no new version; one research
candidate may be frozen (clearly labelled, not claimed better than #277).
**Counterexamples** (largest |prediction error| changes vs A) are listed; #274 R2 is not used to tune
anything. **Power:** per-SID paired error differences (mean, SD, required n at 80 % power), plus the
effective support of the gate regions.

## 5. Prospective freeze

Only one clearly-labelled research candidate (the main E) with frozen gate coefficients, expert
coefficients, scaler, α, λ, domain, input hashes and predictions for the 122 truly unlabeled SIDs;
#277/#284/#285 predictions are shown side by side and never overwritten. Replacement of #277
requires the historical gate **and** a future separately-authorized independent-label validation.
No Stage3.
