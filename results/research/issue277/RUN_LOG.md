# Issue #277 — RUN_LOG (frozen acceptance list)

| item | value |
|---|---|
| issue | #277 audit, interpret and prospectively freeze the six-feature SID Ridge quality predictor |
| body sha256 | `1cfb9429f66d6efd5ee3f9d9237696ccdfebcf6277f52b54316170deeba6b7db` |
| pinned round | #276 commit `f62fd6caa6f6d1e7690e591bfd06a9308f3ebaf9`; SID_TABLE/FOLDS/ANALYSIS_276 hashes verified equal to that commit |
| INFRA_MODE | **FROZEN** — issue text: "单Agent，analysis-only，INFRA_MODE=FROZEN；不修改公共infra、训练代码或旧资产；只复用 #276 表、折分、训练图及现有评估器，允许研究专属分析脚本" |
| training needed | **NO** — issue text: "不训练 RQ-VAE/T5，不重训 TIGER，不启动 Stage3。统计 Ridge 拟合允许。" |

## Frozen acceptance list

| C | requirement |
|---|---|
| C1 | Recompute every #276 OOF metric and the >0.06 confusion table from the stored predictions; the Ridge6 13/13 must be recomputed from its own predictions (the #276 script printed only PySR's threshold) |
| C2 | Audit high-recall identity: the 6 collision-only rows are **weak** evidence (collision equality ≠ content hash); re-check R2/history/original run SIDs; report strong vs weak separately; evaluate the strictly bound subset; weak sensitivity is not a strict result |
| C3 | Audit SID grouping, duplicates/near-dups, code-permutation equivalence, same-checkpoint exports, label duplication and aggregation; no family splits; no cross-fold leakage; verify the TIGER reference 0.0469 vs canonical 0.055206 identity and fix one same-protocol reference |
| C4 | Audit preprocessing: `inner_ridge_alpha` standardised before the inner split (inner leak). Keep the original OOF, report a corrected variant that fits the scaler inside every inner training fold, outer fit only on outer-training data; no seed changes |
| C5 | Feature definitions with worked examples: l1_pos_hit, l1_neg_hit, l1_hard_er, l2_distinct, l3_collision, l3_refine_bits; distinguish item-weighted collision vs 1−unique/N; state the functional dependence of ER on the hit rates |
| C6 | Explain the formula: same 5 outer SID-group folds, inner alpha grid {0.01,0.1,1,10}; per-fold scaler, alpha, intercept, standardised and raw-unit coefficients; independent NumPy check of `sklearn.predict`; direction/size/fold variation, correlation matrix, collinearity; ER-only / hit-rate-only / single / mean-median ablations inside the same folds; contributions for high and low recall SIDs, counterexamples, residuals, extremes; stratified errors, group bootstrap CI, >0.06 TP/FP; 0.065 not evaluable |
| C7 | Freeze a usable predictor only if the strict audit supports it: one final Ridge6 fit on the frozen development queue (alpha by internal group CV); freeze feature contract, scaler, coefficients, intercept, alpha, training SID/label hashes, versions, applicability domain; report the full-data fit separately from OOF; export recomputable JSON + formula + research-only prediction script; empirical prediction interval, out-of-domain warning, missing-value rejection; if the high-recall result vanishes on the strict subset, freeze as a failed research result |
| C8 | Prospective predictions on recoverable Stage2 SIDs without public Stage3 labels: freeze formula and input hashes **before** reading new metrics; no Stage3 start; known results (e.g. #274 R3) only as historical diagnostics; if no qualified unknown candidate, say so |
| C9 | Deliverables: AUDIT (original/corrected metrics, identity and leak checks), FEATURE_CONTRACT, COEFFICIENTS.json, per-fold OOF + ablations/CIs, PREDICTOR.json (if it passes), recomputable formula/scripts/tests, example contribution table, timestamped prospective predictions, persisted logs/commands/exit codes, REPORT/VERDICT, infra zero drift, commit+push |
| C10 | Answer: what the six metrics are, what the coefficients do, whether 13/13 survives the audit, whether the predictor can be used on new SIDs and its limits |

## Freeze order (protocol)

1. Audit → 2. corrected OOF → 3. final Ridge6 + PREDICTOR.json → 4. **prospective predictions frozen
and committed** → 5. only then may the Stage3 label of a prospective SID be read (recorded as the
outcome). No Stage3 start is authorized by this issue.
