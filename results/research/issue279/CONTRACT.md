# Issue #279 — pre-registered contract (frozen before any relation-level or Recall association)

Frozen 2026-11-28 14:5x, before computing any higher-order score against targets or labels.

## 1. Training graph (identical to #278)

Edges from `train.parquet` only: consecutive `history` pairs plus `(history[-1], target)`; count
`c(a,b)`, out-degree `out(a)`, in-degree `in(b)`, `E`. No validation/test behaviour is ever added.
Validation relation = `(a = history[-1], b = target)`; legal negative population exactly as the frozen
evaluator (`history ∪ seen_history ∪ {target}` excluded, uniform).

## 2. Higher-order pair scores (training graph only)

| id | definition |
|---|---|
| `two_hop` | `Σ_{m ∈ succ(a) ∩ pred(b)} c(a,m)·c(m,b)` (weighted directed 2-paths) |
| `two_hop_norm` | `two_hop / (sqrt(out(a)·in(b)) + 1)` |
| `common_succ` | `\|succ(a) ∩ succ(b)\|`; `cs_jaccard` = intersection / union |
| `common_pred` | `\|pred(a) ∩ pred(b)\|`; `cp_jaccard` = intersection / union |
| `direct` | `c(a,b)` (reference) |
| `curvature_pair` | over common centres `m ∈ (pred(a)∩pred(b)) ∪ (succ(a)∩succ(b))`: mean training-graph curvature of the legal triples `(m; a, b)`, plus the structural-class histogram; **NA** when no legal triple exists |

Distance: the assets' distance histogram is graph-wide; a symmetric distance is never used as a
directional signal. Graphlet/orbit: existing 3-node structure classes from the curvature asset are used
for stratification only; no 4-node statistic is added unless its definition and coverage exist
(the sampled 4-node asset is documented as coverage-limited).

## 3. Relation-level discrimination (independent of labels)

Positives: all 57 439 validation relations. Negatives: one fixed legal-uniform negative per row, drawn
with seed 42 from the legal population (uniform, no label use). Scores: the six high-order quantities
above (curvature on a fixed 20 000-pair subsample: 10 000 positives + 10 000 negatives, seed 42).
Metrics: AUC (Mann–Whitney), PR-AUC, median scores, tie fraction; anchor-clustered bootstrap
(2000 resamples of anchors) for the AUC CI; splits by `seen` (`c(a,b)>0`) / `unseen` and by direct edge;
degree-matched AUC within cells of (anchor out-degree decile × target in-degree decile).

## 4. SID-level features (frozen main candidates, at most two)

Per SID on the same validation rows, L1 (and L1+L2 reported):
* `er_unseen_2hop` = `P_w/N_w` over rows with `c(a,b) = 0`, weights `w = 1 + log1p(two_hop)`; NA when
  `N_w = 0`.
* `dp_unseen_degmatch` = degree-matched high-minus-low sharing difference in the unseen stratum:
  high/low split at the median of `log1p(two_hop)` over unseen rows with `two_hop > 0`; within each
  (out-degree decile × in-degree decile) cell with ≥1 high and ≥1 low row, `Δ = mean h+ (high) − mean h+
  (low)`; average over cells weighted by their row counts; NA when no cell qualifies.
* Descriptive only: frequency-weighted ER (#278), curvature-aggregated ER, stratum macros, centre macro.

## 5. Models (same queue, folds and inner procedure as #277/#278)

`A` = Ridge6 refit; `C1` = A + `er_unseen_2hop`; `C2` = A + `dp_unseen_degmatch`;
`C3` = A + both; `E278` = A + `P_w,N_w` (frequency weights) as an **analysis control**, not a replacement.
All preprocessing/alpha/feature selection inside outer training folds only.

## 6. Criterion (identical to #278)

Pass = OOF MAE decrease vs A with the SID-group paired bootstrap 95 % CI **lower bound > 0**, Spearman
and Kendall not worse, >0.06 FP/FN not worse, on the full 129 and the strict 120-row subsets separately.
0.065 NOT_EVALUABLE (max label 0.0648). If no candidate passes, keep #277, no new predictor.
