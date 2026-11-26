# Issue #261 metric contract (frozen before correlation ranking)

Frozen 2026-11-26, before any cross-arm correlation was computed in this issue.
Scope: the eight completed R2 bundles 100–107, one row per Stage2 SID/run and its
single Stage3 result.  Read-only; no training, parameter update, point deletion,
imputation or re-evaluation of a different test population.

## Data populations

| item | definition | expected |
|---|---|---|
| catalogue | rows of each `item_sids.json`; only the first three codes are semantic | 24,587 |
| validation | `results/stage0_build_parquet/valid.parquet`; history→target rows | 57,439 |
| train graph | frozen `results/research/behavior_graph/edges.parquet` | 315,391 edges |
| curvature triples | frozen `assets/v8_runtime_layout/targets.npy` + offsets | 28,391,948 |
| Stage3 test | one frozen `test_final.json` per arm; predictions are never re-scored on a new split | 57,439 rows |
| frozen TIGER | reference only; not included in the eight-arm correlation | one frozen SID/record |

Missing data is `NA` with a reason and attempted path; it is never 0 and never
imputed.  A row is retained even when a metric is NA.

## A. SID and behaviour

Let `c_i=(c_i1,c_i2,c_i3)` be the first three codes.

- distinct prefixes: `D_l = |{c_i[:l]}|`, `l=1,2,3`.
- collision: `1 - D_3/N`.
- entropy: `H(X) = -sum p log2 p`; `H(L1)`, `H(L2|L1)=H(L1,L2)-H(L1)`,
  `H(L3|L1,L2)=H(L1,L2,L3)-H(L1,L2)`.
- group sizes: for each prefix level, counts per prefix; report mean/median/p90/max.
- occupancy: distinct code values per level (0–255); dead codes = 256 - occupancy;
  concentration = max group count / N and normalized entropy.
- behaviour positive hit: `P(prefix(anchor,l)=prefix(target,l))` over validation rows
  with non-empty history, anchor = last history item; negative expectation: exact
  legal uniform catalogue negative mass excluding history/seen/anchor/target, as
  already frozen in the shared layer record.  Report positive/negative, difference,
  ratio, ER, conditional prefix survival and log enrichment.
- graph relations: from the frozen undirected train graph, direct edge, reverse
  edge, no edge, common-successor, common-predecessor and 2-hop pairs.  For a
  relation `r`, retention `= P(prefix_l(u)=prefix_l(v) | (u,v) in r)`; report raw,
  degree-stratified, and lift over a degree-matched non-edge baseline.  Prefix
  sharing is not claimed to be causal.

## B. Quantisation and reconstruction

- `z = encoder(x)` using the checkpoint's actual encoder MLP including biases;
  `q = sum_l codebook_l[c_l]`; `quant_error = mean ||z-q||_2`.
- reconstruction MSE from the checkpoint decoder when the decoder state and input
  dimension can be matched; otherwise NA with the exact missing-key reason.
- per-level residual norm `||z_l - q_l||`, commitment/codebook loss only when the
  corresponding historical loss record exists; otherwise NA.
- effective rank `(sum lambda)^2 / sum lambda^2` of the 32-d covariance, separately
  for `z` and hard `q`; spectrum, top-1/top-2 margin and normalized assignment gap
  are computed on the same population.
- codebook spacing = mean pairwise Euclidean distance among codewords per level;
  never mixed with quant_error or reconstruction MSE.

## C. Curvature and geometry

- target is the frozen `targets.npy` discrete curvature proxy (not intrinsic
  curvature); target rows are never regenerated.
- hard-q prediction uses the SID codebook vectors and a frozen equal-weight
  three-expert distance (curvatures −1,0,+1; tangent scale frozen from the run's
  saved calibration when available).  Continuous-z prediction is reported only
  when the checkpoint encoder is loadable; otherwise NA.
- metrics: micro/macro MAE, RMSE, Pearson/Spearman, sign accuracy, per-sign
  precision/recall/balanced accuracy, predicted/actual sign distribution, distance
  distortion, quantiles.  Negative/zero/positive strata are separate.
- If a run's native geometry is five-expert or role-routed, the equal-weight
  three-expert value is a cross-arm comparability diagnostic, not the run's native
  loss.  Native values are labelled separately and never averaged with it.
- held-out structural asset results are reported as held-out evidence, not as
  train-graph fidelity.

## D. GNN and gates

- token norms/variance/effective rank, within-role and between-role cosine
  separation, degree-confounded checks: only if the checkpoint/source exposes
  tokens without training.
- per-layer expert weights: mean, quantiles, normalized entropy, max weight,
  expert frequency, collapse share, item variance, role-conditioned difference;
  three- and five-expert arms use normalized entropy and sign-grouped weights.
- switch diagnostics: existing frozen behavior-off/curvature-off/uniform-gate
  measurements are reused only when their semantics match; otherwise NA.  A
  read-only switch is an intervention, not a retrained causal effect.

## E. Loss, gradients, training dynamics

- from each full `training_metrics.jsonl`: 3000-epoch completeness, first/mid/last
  windows, reconstruction/curvature/weighted totals, gate entropy/weights,
  collision, structural coverage, wall time.  Raw values only; no reweighting.
- module gradient checks: historical records are used when present.  A single
  fixed final checkpoint forward/autograd diagnostic is allowed only if the
  encoder/GNN/gate/codebook are differentiable in that source; parameters are
  frozen and hashes are checked before/after.  Endpoint gradient is not presented
  as the historical optimizer trajectory.  Unrecorded history is NA.
- timing is auxiliary, not a quality substitute.

## F. Downstream

- `test_final.json` Recall/NDCG are read and independently reduced from the saved
  prediction shards using the existing `tools/iteration/analysis.py` reduction.
  A mismatch is a FAIL and is reported; no re-scoring.
- validation Recall/NDCG is included only when an identity-bound saved artifact
  exists; test is never used as a validation value.  Missing is NA.

## Correlations and sensitivity

- one row per arm; for each numeric metric with ≥3 non-NA pairs report Pearson and
  Spearman, n, direction and absolute ranking; constants are `NA`.
- test metrics: R@5, R@10, NDCG@5, NDCG@10; validation metrics only when identity
  bound.
- report scatter data, nominal p, Fisher-z/bootstrap CI where defined, leave-one-out
  ranges, exclude-107 and exclude-100 (five-expert) sensitivity.  Eight points are
  exploratory; no causal claim and no point removal from the primary matrix.
- expose the metric correlation matrix; flag mathematical aliases (e.g. collision
  and distinct-triple count) instead of counting them as independent evidence.

## Acceptance math knobs

All tolerances: entropy/correlation 1e-12 for stored recomputation; SID integers
exact; collision and rates reported at full precision.  Test metric match tolerance
1e-10 versus the saved test_final.  No model/optimizer update is permitted.
