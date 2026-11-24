# Issue256 R01 report — complete, not adopted

R01 repaired the measured collapse and improved several registered Stage2 endpoints, but the legal complete final test did not meet the issue target. Issue256 remains OPEN; this is not an implementation-failure rerun or an issue-closing verdict.

## Exact generation

- Arm: `TIGER_module_budget_prefix_rq`; immutable run `issue256_r01_formal`.
- Production source commit: `0cdfec9b024e568b5b08d96e777033844d66635e`; checkpoint `680439b57bfdbec622a5a109b0d589ab21566f71236fc007eb19d59b6108fc0c`.
- SID: `201d2cc537dfdcc27060302489324d818356fd6327b88de89584943108a5758f`; every one of24587 first-three-code rows and the canonical fourth suffix reproduced exactly without rewriting.
- Stage3 test/validation checkpoint: `d8540c837dedffd8291615bb6d7578308f74f6469e3182887685a08571082a96`.
- Stage2:3000epochs/world4/global4096/seed42. Stage3:150epochs/world4/global16384/beam20/seed42/full57439test; frozen configuration and data-derived EOS/vocabulary independently match.
- Complete four-entry R2 archive: `r2:genrec/102/`, key `TIGER_module_budget_prefix_rq/201d2cc537df`, all117 remote files SHA-verified.

## Recommendation endpoints

|Metric|Final test, independently reduced|Same-checkpoint external validation|
|---|---:|---:|
|recall@5|0.034105746966346905|0.038162224272706699|
|recall@10|0.052734204982677273|0.057643761207541916|
|ndcg@5|0.022554059459244928|0.025688765803069922|
|ndcg@10|0.028523448079844319|0.031942815505712291|

Both populations contain57439rows but are different immutable splits. Parent external-validation inference was unavailable and remains NA. The test R@10 gap to0.065 is0.01226579501732273; strict acceptance is false. Historical #252/#253 test scores are not a matched causal control or verified TIGER Recall.

## Registered Stage2 forecasts — individually evaluated

|Endpoint|Parent255|R01|
|---|---:|---:|
|Distinct L2 prefixes|306|8559|
|Distinct L3 triples|457|22972|
|Three-code collision|0.9814129417985114|0.06568511815186884|
|Catalogue reconstruction MSE|0.00018233678135234408|7.7839280014166759e-05|
|Validation direct minus degree-matched L1 sharing|0.0015360242767251675|0.037791790922898125|
|Validation degree-matched control L1 sharing|0.9206943498978897|0.009530292716133424|

Each forecast holds; this is not an aggregate proof that geometry or recommendation succeeded. R01 changes the whole initial module-coverage/population calibration policy, not an isolated module-only factor.

## Complete hard-prefix training geometry

Common fixed-three-space metric, tangent scale0.08433422446250916, no fitting on evaluation. Each level includes28391948 lawful triple occurrences/24326centres with the full original component references and multiplicity. Missing lawful triples remain missing.

|Prefix|Macro MAE|Micro MAE|Balanced sign accuracy|Negative recall|Zero recall|Positive recall|
|---|---:|---:|---:|---:|---:|---:|
|L1|2.591165416762423|1.8518167726521699|0.32394365677174441|0.18670280658937802|0|0.78512816372585525|
|L2|2.5821380574896446|1.8740577157998617|0.32045446144368678|0.19600411474776575|0|0.76535926958329459|
|L3|2.5820128898398336|1.8814655906755702|0.32112382578018112|0.20180950274714868|0|0.7615619745933947|

All three macro MAEs improve versus parent255, but all micro MAEs and balanced accuracies worsen: mixed, not uniform geometric improvement. There are77zero targets per cell; zero recall is observed zero, not a missing-data fill. These are in-sample training-graph measurements, not held-out geometric generalization.

## Gates, gradient discrimination and validation pairing

- Mean Euclidean gate weights by level:0.95622194/0.97680432/0.98550516. Both other curvature spaces remain present; do not claim uniform three-space use.
- Post-training actual-loss replay:18/18 module combined gradients and fresh-state diagnostic optimizer updates are nonzero. It is not AdamW continuation.
- The final real global4096 diagnostic averages VJP vectors, not norms.13/17 groups exceed the initial budget; Query0 maximum ratio9.0859271222. Only sharedScore has global raw original-ascent alignment, normalized dot-0.6988326877.
- Counterexample: local rank0 sharedKey/Query0 ascent signs disappear globally. One fresh-RNG final batch is not all epochs or an optimizer-step/Recall causal proof. The root cause remains unknown.
- Validation misses partition into no-L1=41397, best-L1=11132, best-L2=1514, best-L3=85. No-L1 is76.4798256% of misses. No gold validation target is in seen_history, so that split does not support a seen-gold-filter explanation.
- Paired user/item arrays, training degree/reconstruction/curvature covariates and their missingness are saved under recommendation_validation/. Training covariates are not held-out geometry or training supervision. No test-error rows guide the next change.

## Performance and warnings

- Pre-dispatch full-data inclusive epochs31–60 mean0.7027374609683951s; authorized mean ceiling0.8 passes, desired0.4–0.5 does not. P95=0.91014997754246s is not below0.8.
- Actual formal epochs31–60 mean=0.70266381517673537s, median=0.67340663680806756s, P95=0.95290805585682392s; all interepoch work remains in the enclosing scope.
- Per-card module timing/peak allocated and reserved memory: PROFILE_SUMMARY.json, four actual L40S ranks, instrumented windows kept separate from the inclusive training gate.
- Quantization error0.0072272333547981436 and effective rank5.390244278887607 are diagnostics. The low-error warning is not a measured generalization failure; collision alone is the hard quality gate.
- The layered SID record was written before Stage3 access. Undefined legacy diagnostics remain explicitly missing, never invented as0.

## Verification and evidence

-126candidate+64framework tests, real entry and behavior graph, actual four-rank original/curvature VJP/update/prefix/export proof passed before the sole formal dispatch. Earlier loader/profiler failures and their exact source generations remain preserved.
- Independent norm-budget LP reference, hand answers and specified-assertion mutation records: reference.py, MUTATIONS.md, BUDGET_MUTATIONS.json, MUTATIONS.json and LAYER_MUTATIONS.json; actual model MODULE_CHECKS.json.
- Post-training source review confirmed the same hard rule, three spaces, complete four ego tokens, original losses and actual paths. rank_local_checks changes diagnostic guards, not hard assignment. Final source/checkpoint/SID is bound in FINAL_STAGE2.json and the archived runtime.
- Independent raw reductions: `python3.9 -B -m tools.issue256.reconcile_stage2` and `reconcile_stage3`, both exit0. External full validation and validation_diagnosis also exit0; evaluator population regression and prefix hand answers pass3/3.
- Every descriptor and command/exit evidence is collected in ROUND_RESULT.json; the original pre-dispatch ledger bytes survive in REGISTRATION.json with SHA e58b5ee5b657ceb35c8cdc82a4a4ed1e78e2e934ae47a36cda598577459b04c5. Later live-ledger updates do not masquerade as those original bytes.

## R7 decision and test-reuse limits

Do not adopt R01 and do not retrain it. Preserve the valid negative run, complete archive and all failures. Continue with another separately preregistered train/validation-driven hypothesis after the live tree archival recorded in LIVE_ARCHIVE.json.

The fixed test was already exposed in prior issues. R01 adds one authorized final model evaluation and one read-only reduction of persisted predictions, not an independent second model test. There is no genuinely unvisited confirmation dataset. Any future winner is an exploratory reused-test threshold, not an unbiased generalization estimate.
