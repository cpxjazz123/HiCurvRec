# Issue #275 — pre-registered analysis contract (frozen 2026-11-28 13:10, before any search)

## Hypothesis space (competing, not assumed)
H1 L1 behavioural fidelity (`l1_pos_hit`, `l1_hard_er`) is the strongest single predictor of Recall@10.
H2 Capacity (`l2_distinct`, `l3_refine_bits`) adds beyond fidelity.
H3 Vector distortion (`quant_error`, `effective_rank`) adds beyond the SID core (Q_VEC only).
H4 Curvature/distance geometry adds beyond SID (Q_GEO; unidentifiable with n=3 — reported as such).
Null N: no feature set beats the fold mean/median out of fold; any apparent fit is family confounding.

## Splits
* Outer A (family): leave-one-family-out over families with n ≥ 3. Singleton families are reported
  separately and never used as an outer fold.
* Outer B (time): chronological split by the Stage3 run date; if a single side has < 8 rows the split
  is declared underpowered and only reported, never used to claim generalisation.
* Inner selection: within each outer training set, grouped by family; the expression is chosen on
  training folds only. Outer labels are never used for feature screening, scaling, constants or
  expression selection.

## Models / controls (same folds)
1. Training-fold mean; training-fold median.
2. Best single feature chosen on the training folds by Spearman (ties -> first in the frozen order).
3. Ridge regression (standardised features, alpha selected on inner grouped CV over {0.01,0.1,1,10}).
4. A trivial prior formula `l1_pos_hit` alone (no fitting).
5. PySR symbolic regression (fixed: pysr==0.19.4, Julia via juliacall, binary `+ - * /`,
   unary `square`, maxsize 10, maxdepth 4, niterations 100, populations 8, timeout 60 s per fold,
   random_state 42, deterministic=True). Selection: from the training-fold Pareto/hall_of_fame,
   the simplest expression within 1 SE of the best training loss; constants must be finite and the
   expression finite on the training domain.

## Metrics (primary) and aggregation
Outer MAE, RMSE, Spearman, Kendall, rank reversals (pairs ordered oppositely), direction accuracy
relative to the frozen TIGER recall (sign of prediction − TIGER). Paired differences vs the strongest
control with family-cluster bootstrap CIs (2000 resamples, families as clusters, seeded 42).
Secondary: 0.06 threshold confusion; with zero positives ≥ 0.06 the threshold metric is
`NOT_EVALUABLE` (reported, not scored).

## Frozen success criterion
The formula counts as usable only if, out of fold: (a) MAE is lower than the strongest control with a
family-bootstrap CI excluding zero, (b) Spearman/Kendall do not degrade, (c) no clear failure under the
time split, (d) the expression is simple and finite. CI covering zero or clear counterexamples =>
insufficient evidence. The search space is not expanded to force monotonicity.

## Budget
Feature recomputation: CPU only. PySR: 9 outer folds × 60 s ≈ 10 min CPU; no GPU, no training.
Random seeds fixed (42) for every stochastic step; versions logged in PYSR_ENV.md.
