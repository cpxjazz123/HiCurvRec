# Issue #279 — RUN_LOG (frozen acceptance list)

| item | value |
|---|---|
| issue | #279 higher-order behavioural structure fidelity for SID Recall prediction on unseen transitions |
| body sha256 | `4cac3bda8ac78029141adc7631a9b84fa4fad9f600c7376c607d3b638762addc` |
| pinned inputs | #276 `SID_TABLE.json` `4c927914…`, `FOLDS.json` `1a8fb81d…`; #277 `PREDICTOR.json` `ec339f4a…`; #278 `WEIGHTED_TABLE.json` (freq-based), `MODEL_COMPARISON.json`, `IMPORTANCE.json` |
| INFRA_MODE | **FROZEN** — issue text: "单Agent，analysis-only，INFRA_MODE=FROZEN；只复用已有训练有向图、距离/Graphlet/Orbit/曲率资产、SID、#276/#277/#278数据及评估器；允许本Issue专属只读分析与派生结果，不建公共infra、不改原始资产" |
| training needed | **NO** — issue text: "不得训练RQ-VAE/T5、重训TIGER、启动Stage2/3或新test" |

## Frozen acceptance list

| C | requirement |
|---|---|
| C1 | Reuse #278's 129 labeled SIDs / 253 computable / fixed 5 folds; verify 120 strong / 9 weak and 13 high-recall with only 6 strong; new results bound by exact hash separately; state the training transition counts, edge direction and the 92.4 % unseen ratio with denominators |
| C2 | Freeze competing hypotheses H1 (higher-order topology has predictive value on unseen relations), H2 (degree/popularity proxy), H3 (information useful but SID sharing cannot retain it), H4 (the six metrics already absorb it); register metrics, significance, budget and data isolation before looking at new results |
| C3 | Relation unit = validation last-history item a → target b, with the frozen legal-uniform negative definition; all higher-order features from the training graph only; existing assets must not add non-training edges; check whether the graph distance is symmetric and never treat symmetric distances as directional information |
| C4 | Offline higher-order strengths: directed 2-hop a→m→b (counts and degree/probability-normalised scores), common successors, common predecessors (raw + Jaccard/degree normalised), direct vs distance-2 vs longer; graphlet/orbit only where existing assets define it; curvature from the real training-graph triplets, stratified by structure type, with missing marked NA and comparability checked |
| C5 | Compute the structural features for all validation rows and the legal negative population; fixed stratified negative samples allowed as a diagnostic but the #277 uniform negative probability is unchanged; record coverage, zeros, support, sparsity, runtime, hashes; unseen transitions reported separately; degree/popularity-matched comparisons |
| C6 | Relation-level discrimination: real next-item pairs vs fixed legal negatives, AUC/PR-AUC, ranking/support, anchor-clustered CI; no validation-label fitting; deterministic normalisation first; if the higher-order scores do not discriminate, do not claim predictive value |
| C7 | SID fidelity: per SID on the same validation rows, L1/L1+L2 sharing and the legal negative sharing probability; by pre-registered higher-order stratum, seen/unseen, structural class; macro/micro, ER, P−N, support, centre macro average; weighted positive and negative with the same non-negative weights; NA for zero denominators; high-minus-low structure differences inside the unseen stratum |
| C8 | Main candidates limited to 1–2 added features (e.g. unseen high-order weighted ER, degree-matched high-vs-low structure sharing difference), frozen before any Recall association; all other strata descriptive/inner candidates; #274 R2 residual diagnostic without R2-specific weighting |
| C9 | Prediction correction: #277 folds, #278 leak-free inner scaler; same queue/folds comparison of original Ridge6, #278's best 8-feature model (analysis control, does not replace the predictor), Ridge6 + at most 2 higher-order features; inner-only selection; no outer-label screening |
| C10 | Primary criterion: OOF MAE decrease vs the same-queue Ridge6 with SID-group paired bootstrap 95% CI lower bound > 0; Spearman/Kendall not worse; >0.06 FP/FN not worse; strict identity subset separately; 0.065 NOT_EVALUABLE without positives; cross-fitted residual analysis controlling degree/popularity and the six features; same-queue ablations; if relation-level discrimination succeeds but SID prediction does not improve, state both |
| C11 | Curvature has no independent-contribution obligation: if only 2-hop/common-neighbour works, say curvature has no extra evidence; if the unseen stratum has no signal or too few samples, list counterexamples and limits; no repeated tuning on historical test |
| C12 | Freeze a new version only if the pre-registered criterion passes (feature definitions/parameters/hashes/domain/uncertainty; old and new simultaneous timestamped predictions for truly unknown Stage3 labels); otherwise keep #277 and produce no new predictor; known labels (R2) never count as blind |
| C13 | Deliverables: DIAGNOSIS/HYPOTHESES/CONTRACT, graph relation definitions and coverage audit, full derived features/per-SID stratum table + manifest, relation-level discrimination and degree-matching results, OOF/CI/ablations/residual analysis, code/tests, persisted logs, new predictor/prospective only if it passes, REPORT/VERDICT, infra zero drift, commit+push; final answers: which higher-order relations predict unseen transitions, whether the SID retains them, whether they add Recall prediction beyond the six, whether curvature adds independent value |
