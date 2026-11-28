# Issue #278 — RUN_LOG (frozen acceptance list)

| item | value |
|---|---|
| issue | #278 importance-aware SID fidelity and prospective Recall predictor correction |
| body sha256 | `943760828a2f8a719cd64624eb75b5bb606ddaa6c1743d29a3a1c26fccbc380d` |
| pinned inputs | #277 `PREDICTOR.json` `ec339f4a…`, `PROSPECTIVE.json` `4486b64d…`, `PROSPECTIVE_OUTCOME.json` `3d007c76…`; #276 `SID_TABLE.json` `4c927914…`, `FOLDS.json` `1a8fb81d…` |
| INFRA_MODE | **FROZEN** — issue text: "单Agent；analysis-only，INFRA_MODE=FROZEN。复用 #276/#277 全部历史独立SID、已有训练图、train/valid 数据、归档与评估器；只允许研究专属分析代码" |
| training needed | **NO** — issue text: "不得修改公共infra、重训TIGER/RQ-VAE/T5、启动Stage2/Stage3或新跑最终test" |

## Frozen acceptance list

| C | requirement |
|---|---|
| C1 | Freeze the #277 predictor/prospective/outcome and the 129-SID queue, FOLDS and the 120/9 identity split; verify #274 R2's actual SID sha `9f6bf673…` (the re-run SID, not the early `7923…` prefix); recompute its prediction, six contributions, label and time; match other finished prospective labels by SID hash + run identity; no fabricated labels |
| C2 | Reuse the full SID content where possible (git/R2 read-only), list coverage per source; fixed common-SID queue and same-SID-group folds; keep the #277 leak-free scaler/inner CV; strict 6 high-recall vs the 13 with 7 weak identities reported separately |
| C3 | Controls: original six-feature Ridge, fold mean, L1 Hard ER single metric; no baseline retraining; verify the #277 standardisation parameters and definitions; all new/old features on the same validation rows, legal-negative protocol and first-three-code SID |
| C4 | Importance from training behaviour only: c(a,b) (log1p / training quantiles), conditional P(b\|a) with pre-frozen smoothing, PMI/lift comparison; higher-order relations (common successors/predecessors/neighbours, chains/forks/merges/closures) using existing assets; curvature only as an identified diagnostic; strata high/mid/low/unseen with frozen thresholds; no test-based reweighting; descriptive coverage first, main model limited to 1–2 new features |
| C5 | Weighted fidelity on the same validation rows: h+ = 1[prefix(a)==prefix(b)], h− = exact legal-uniform negative probability; P_w/N_w/ER_w with the same frozen non-negative weights for positive and negative; zero-denominator handling; L1 and L1+L2; by-stratum macro/micro, support, coverage, effective sample size, extreme-weight sensitivity; per-SID tables |
| C6 | Direct diagnostic of #274 R2 vs SIDs with similar six metrics but different recall: which relations are kept/lost, residual explanatory power; exploratory only, no R2-specific weighting; separate original six features, extra information, statistical redundancy |
| C7 | Corrected model: freeze candidates/weights/hypothesis/folds/criterion before looking at associations; all preprocessing/alpha/new-feature selection inside outer training folds only; no near-duplicate across folds; compare original Ridge6, a weighted replacement (Ridge6-W), and at most 1–2 importance-layer increments; refit the original Ridge6 on the same common queue as the fair control; keep the original frozen predictor output |
| C8 | Primary criterion: OOF MAE decrease vs the refit Ridge6 with SID-group paired bootstrap 95% CI supporting the improvement; Spearman/Kendall not worse; >0.06 false-positive/false-negative not worse (strict identity and the full queue separately); high/low-recall stratified residuals, calibration bias, empirical intervals; 0.065 NOT_EVALUABLE without positives; if the CI covers 0 or only R2 improves, no correction is claimed |
| C9 | Check whether importance statistics explain OOF residuals beyond the six features: cross-fitted residuals, correlation/ablation with group CI, collinearity with L1 pos/neg/ER, popularity, SID-group size; time extrapolation only if available |
| C10 | Freeze a new version only if the pre-registered standard passes: one final fit, saved training statistics/weights, all feature definitions, coefficients, scaler, alpha, training SID/label hashes, domain, error interval, independent recomputation; otherwise deliver a negative result and keep #277 |
| C11 | Prospective readiness: freeze old and new predictions for SIDs still without public labels (identity/features complete) before any label; known labels (including R2) are never blind; no Stage3 in this issue |
| C12 | Deliverables: DIAGNOSIS/HYPOTHESES/CONTRACT, identity/coverage audit, full training importance stats and per-SID stratum table, unified table/manifest, analysis code/tests, persisted logs, OOF/CI/ablations/residual counterexamples, old/new formulas and prospective predictions (if it passes), VERDICT, infra zero drift, commit+push; final answer: does importance add independent predictive information, which relations are worth keeping, does it reduce the over-prediction stably rather than fitting history |
