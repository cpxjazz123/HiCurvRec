# Issue #262 — VERDICT

Analysis only. No Stage2/Stage3/TIGER training, no optimizer update, no SID rewrite,
no change to public infrastructure or frozen assets.

## Acceptance

| item | result | evidence |
|---|---|---|
| #262 identity: 8 R2 bundles | **PASS** | `AUDIT.json` — one row per completed arm; candidate source/checkpoint/SID/final hashes |
| #262 missing audit: Stage3 validation metrics | **FILLED for R01-R04; NA for 100/101/106/107** | `VALIDATION.csv; AUDIT.json` — identity-bound Stage3 validation shards independently reduced from rank*.jsonl |
| #262 missing audit: full predicted-vs-target curvature Pearson/Spearman/RMSE | **FILLED** | `CURVATURE_FULL_*.json` — 28,391,948 valid frozen triples per arm; full coverage check |
| #262 missing audit: role/degree-stratified curvature MAE | **FILLED** | `CURVATURE_FULL_*.json` — centre role table and directed total-degree quintiles |
| #262 missing audit: per-level top1/top2 margin, residual, commitment/codebook | **FILLED** | `QUANT_GRAD_SWITCH_*.json` — actual checkpoint books and exported hard SID |
| #262 missing audit: switch-off across all eight | **FILLED (CPU interface; exactness diagnostic reported)** | `QUANT_GRAD_SWITCH_*.json; GPU_SWITCH_102.json` — no-intervention, behavior-off, curvature-off, uniform gate, Euclidean+uniform; no retraining |
| #262 missing audit: endpoint per-module gradients | **FILLED** | `QUANT_GRAD_SWITCH_*.json` — one common fixed batch; optimizer_steps=0; state-dict checksum unchanged |
| #262 missing audit: historical per-module gradient trajectory | **NA** | `MISSING.csv` — training stream records aggregate and level gradients only; not fabricated |
| #262 missing audit: R2 source tar local copy | **NA** | `AUDIT.json` — #261 retained only recall.json; local source tree/candidate hashes checked instead |
| H1: directed shared-in target vs degree-matched negative | **PARTIALLY_SUPPORTED** | `SHARED_IN.json` — raw paired difference and relation-matched/no-direct controls |
| H2: visible vs true directed held-out shared-in feature | **SUPPORTED** | `SHARED_IN.json` — #261 +0.958 is a cross-arm correlation of the visible-graph feature; true directed label has a much larger mean and reverses sign |
| H3: curvature/reconstruction as architecture/capacity proxy | **SUPPORTED_AS_DESCRIPTIVE** | `CORRELATIONS_262.csv` — n=8 descriptive; no causal direction |
| no training: Stage2/Stage3/optimizer updates | **PASS** | `TRAINING.json; QUANT_GRAD_SWITCH_*.json` — zero optimizer steps; parameter checksums unchanged; no launcher/training entry |
| infra: public files and frozen assets | **PASS** | `INFRA_SNAPSHOT_START.json; INFRA_SNAPSHOT_END.json; INFRA_DIFF.json` — path:sha256 before/after |

## Main findings

1. **#261's `+0.958` is the wrong layer.** It is the cross-arm Pearson of a within-arm Spearman between code distance and a **visible-graph** Jaccard field. The per-arm values are only about −0.039 to −0.075. The strict bank pairs truly share an in-neighbour on the graph that generated them (true mean Jaccard 0.2035), while the #261 visible field has mean 0.0041. Recomputing against the directed bank label gives cross-arm Pearson -0.8682, not +0.958. This is a structure-test association, not a recommendation-test effect.

2. **Raw H1 is positive but not isolated.** Target common-in rate 0.4033 vs degree/popularity-matched negative 0.2695; paired difference 0.1338 [0.1300, 0.1378]. A relation-matched negative gives 0.1070; with direct/reverse/two-hop removed on both sides the difference is 0.0611 (n=30771). The target also carries direct, reverse and two-hop structure and much higher SID prefix sharing; the raw shared-in advantage therefore does not identify an independent topology effect.

3. **New full-table curvature is informative but not causal.** For each arm the complete 28,391,948 valid triples were re-evaluated. The new Pearson/Spearman, RMSE, MAE and sign metrics are in `CURVATURE_FULL_*.json`; MAE reproduces the #261 micro MAE as a sanity check (e.g. 102: 1.8815 vs 1.8815). The eight-arm associations remain descriptive and architecture/capacity-confounded; H3 is not a causal claim.

4. **Quantisation/gradient/switch evidence is complete at the endpoint.** The actual exported hard SID, actual books and one common fixed batch were used. The endpoint forward/autograd records original/reconstruction/curvature/combined losses and per-module gradients; no optimizer step occurred and the state-dict checksum is unchanged. Per-level margins, residual, commitment and codebook values are in `QUANT_GRAD_SWITCH_*.json`. Switch probes are read-only; CPU no-intervention replay does not exactly reproduce the GPU export on 102 and is reported, not hidden.

## Remaining limitations

- Historical per-module gradient trajectories are NA; only aggregate and level gradients were archived. Endpoint gradients are not a trajectory.
- R2 source tar is not present in the retained local R2 download; local source tree and candidate hashes were checked. The historical public-infra drift for 100–102 is reported in `AUDIT.json` and is not treated as candidate drift.
- The structure-test population is not the recommendation-test population. It is not used for model selection or as a test substitute.
- n=8 exploratory associations; no multiplicity-adjusted discovery or causal claim.

## Close checklist

1. All production/analysis code implemented? **Yes** — issue-local scripts and tests; no public infrastructure change.
2. All required tests executed? **Yes** — `python3 -m pytest -q results/research/issue262/tests` passes 8 tests; this is recorded in `TEST_RESULTS.json`.
3. Required training completed? **Not applicable / zero by contract** — this is an analysis issue; `TRAINING.json` and the endpoint receipts record zero optimizer steps.
4. Infrastructure boundary respected? **Yes** — FROZEN snapshot start/end identical.
5. Executable work left? **No** after the final report and evidence commit.
