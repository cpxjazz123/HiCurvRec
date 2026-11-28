# Issue #286 — RUN_LOG

- Issue: #286 "Analysis-only: gated two-regime SID quality predictor using discrimination and behavioral fidelity"
- Body sha256: `8e7a54179518ffac897119eeade1a9e3af211a02aeb5cacfc80dc6fed651c7c1` (2467 chars, frozen 2026-11-28).
- **INFRA_MODE = FROZEN** (issue: "INFRA_MODE=FROZEN …不得改公共infra、原资产、训练流程;不重训…不启动Stage2/Stage3").
- **Training required? NO** — analysis-only; all acceptance quantities are recomputable from the frozen
  129 SIDs, 5 folds and the #277/#284/#285 artifacts.
- References fixed: #285 `a433172f2774c5d633f3798b2151f01683ad7ec2`; #277 `d1cb5cbc8bf1cee998ce8e7ca7fb0e8afdf16b0d`;
  #284 `b106d01e…`; #283 `801048ca…`.
- Units: independent SID, 127 near-duplicate clusters; the 2 time-postordered labels are diagnostic only.

## Frozen acceptance checklist (C1..C5)

| C | § | requirement | criterion |
|---|---|---|---|
| C1 | §1 | verify #285/#277 commits, hashes, labels, folds; the LL 34 SIDs / 11-of-13 high / 6 strong high / 7.7 % reversals; reproduce Ridge6 0.002971939 / 0.666609 / 13-0-0; record #277 domain, empirical residual interval, label aggregation | all reconciliations exact or documented |
| C2 | §2 | gated two-branch mechanism: g(N,C)=sigmoid(b0−b1N_std−b2C_std), b1,b2≥0; locked primary gate target (feature region); label-learned gate only as a diagnostic; f_good=P,ER; f_other=six; soft-gated residual experts shrinking to R6 as the main candidate; one pre-registered selection (alpha/λ inner); structure comparisons and the four ablations; per-fold gate coefficients, weights, support, prediction ranges | main candidate + fixed comparisons executed; all parameters training-fold only; cross-fitted residuals |
| C3 | §3 | same 129/folds vs #277: MAE/RMSE/Spearman/Kendall/reversals/bias; gate calibration AUC/Brier; >0.06 confusion; LL/HH/mixed cell errors; strong subset; 0.065 NOT_EVALUABLE; main criterion = paired SID-cluster MAE CI lower >0 and secondaries not worse; counterexamples; power | gate distinction vs regression accuracy stated; no unbounded retry |
| C4 | §4 | only if the main criterion passes → new version pending prospective validation; otherwise one clearly-labelled research candidate with frozen formula/gate/experts/scaler/domain and predictions for the 122 truly unlabeled SIDs; #277/#284/#285 files untouched; no Stage3 | frozen PROSPECTIVE_286.json |
| C5 | §5 | CONTRACT/HYPOTHESES/DIAGNOSIS, identity/fold audit, gate/expert formulas and coefficients, per-fold predictions, OOF/CI/ablation/calibration and error counterexamples, research candidate/predictions, tests, logs, REPORT/VERDICT, infra drift 0, commit+push | all files + zero drift |
