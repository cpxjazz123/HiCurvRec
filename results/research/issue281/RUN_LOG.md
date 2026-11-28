# Issue #281 — RUN_LOG (frozen acceptance list)

| item | value |
|---|---|
| issue | #281 nonlinear interactions and piecewise response laws of the frozen six SID metrics |
| body sha256 | `5976bf2cafbea370a8681263a5ed1496ac64a9cfef9fac23bd3b60faf6f634f8` |
| pinned inputs | #276 `SID_TABLE.json` `4c927914…`, `FOLDS.json` `1a8fb81d…`; #277 `PREDICTOR.json` `ec339f4a…`, `ANALYSIS_276.json`; #278–#280 records |
| INFRA_MODE | **FROZEN** — issue text: "单Agent，analysis-only，INFRA_MODE=FROZEN；只使用#277冻结129 SID数据、原五折及既有证据" |
| training needed | **NO** — issue text: "禁止训练RQ-VAE/T5、重训TIGER、启动Stage2/3或新test；可拟合统计分析模型" |

## Frozen acceptance list

| C | requirement |
|---|---|
| C1 | Fix the six #277 features (L1 pos/neg sharing, Hard ER, L2 distinct, L3 collision, L3 refinement gain); verify units, data identity, the 120/9 split and the 13 high-recall (6 strict); recompute Ridge6 MAE 0.002972, ρ 0.667, 13/13, 0 FP; strict subset separately; ER = pos/neg is not three independent causes |
| C2 | Pre-register model groups, complexity, transforms, inner selection, evaluation and budget before any fit; the fixed 5 folds are reused for the paired historical comparison only; all thresholds/centering/spline nodes/interaction choices from inner training data only; outer labels never select; report the label range and 0.065 NOT_EVALUABLE |
| C3 | Hypotheses H24 (monotone linear insufficient; moderate ranges), H25 (L1 pos benefit depends on the negative rate and L2/L3), H26 (piecewise reduces high/low calibration bias), H27 (complex models only overfit); pre-register maximum degrees of freedom, search time and the failure criterion |
| C4 | Only the six features; first inspect training-fold single-metric/conditional responses, quantiles and interaction plots; check collinearity among pos/neg/ER and L2/L3; marginal trends are not conditional effects |
| C5 | Models: A linear Ridge6; B regularised additive with at most one low-complexity nonlinear transform per variable (chosen inside); C at most two pre-registered interactions (L1 pos×neg, ER×L3 collision, ER×L2 prefix) with hierarchy; D continuous piecewise/hinge with at most one knot per key variable from training quantiles {25,50,75} and a frozen minimum segment support; E optional shallow tree (depth ≤2, min leaf pre-frozen) as a diagnostic; F inner-calibration of the #277 predictions (linear or limited monotone piecewise) using cross-fitted training predictions |
| C6 | Curves, local slopes, interaction contours, reproducible knot intervals and bootstrap uncertainty; only inside the real joint distribution; counterexamples and extrapolation warnings; no unbounded formula search or exact monotone fit of 129 points |
| C7 | Same SID queue and the original 5-fold nested CV; all fitting/preprocessing inside inner folds; one OOF per SID; the main candidate is produced by a single inner-selection rule over the pre-registered groups (no outer winner-picking); report the multi-model comparison limitation |
| C8 | Primary criterion: OOF MAE decrease vs Ridge6 with SID-group paired bootstrap 95 % CI lower bound > 0, Spearman/Kendall/reversals not worse, >0.06 FP/FN not worse; strict identity queue separately; stratified signed errors, calibration curve, prediction interval, RMSE, confusion; 0.065 NOT_EVALUABLE; classification success is not exact ranking |
| C9 | Answer: does nonlinearity add independent value; where does the slope change; which interactions are stable; can over/under-prediction be reduced; can a falsifiable conditional monotone law be stated; only with cross-fold stability and data support; otherwise keep #277 and do not expand the search |
| C10 | If the criterion passes, fit once on the development queue and freeze expression/knots/coefficients/scaler/domain/interval/hashes with an independent recomputation; freeze old/new predictions for truly unknown SIDs; otherwise deliver the negative result and keep the original predictor |
| C11 | Deliverables: CONTRACT, HYPOTHESES, fixed folds/identity audit, per-group OOF, paired CI, interpretable curves/interaction data, counterexamples, code/tests, persisted logs/receipts, REPORT/VERDICT, infra zero drift, commit+push |
