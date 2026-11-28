# Issue #278 — pre-registered contract (frozen before any association with recall)

Written 2026-11-28 14:3x, before computing importance statistics against labels. All formulas below are
frozen; deviations require a dated amendment written before the affected computation.

## 1. Transition importance from training behaviour only

Training rows: `results/stage0_build_parquet/train.parquet` (columns `history`, `seen_history`,
`target`). Directed edges:
* consecutive pairs `(h_i, h_{i+1})` inside every `history`;
* the intended successor `(history[-1], target)`.

`c(a,b)` = edge count; `out(a) = Σ_j c(a,j)`; `in(b) = Σ_a c(a,b)`; `E = Σ c`; N = 24 587 items.

Frozen candidates (each computed from training rows only):
| id | definition |
|---|---|
| `w_freq` | `log1p(c(a,b))` (primary count weight) |
| `w_cond` | `P(b\|a) = (c(a,b)+α)/(out(a)+α·N)`, `α = 1` |
| `w_lift` | `P(b\|a)/P_node(b)` clipped to `[1e-3, 1e3]`, `P_node(b) = (in(b)+α)/(E+α·N)` |
| `h_succ` | `\|succ(a) ∩ succ(b)\|` from the same graph |
| `h_pred` | `\|pred(a) ∩ pred(b)\|` |

Frozen strata over validation rows (thresholds from the **training edge-count distribution**):
`unseen` (`c(a,b)=0`), `low` (`0 < c ≤ q33`), `mid` (`q33 < c ≤ q66`), `high` (`c > q66`), where
`q33/q66` are the 33rd/66th percentiles of `c` over all training edges.

## 2. Weighted SID fidelity

Same validation rows as the frozen evaluator (non-empty history, non-empty legal population).
For `k ∈ {1, 2}` (L1 and L1+L2 prefixes):
`h+_r = 1[prefix_k(anchor) = prefix_k(target)]`,
`h−_r = (count_k(prefix_k(anchor)) − excluded_in_group_k(r)) / population(r)` exactly as
`tools/issue252/layer_metrics.validation_metrics`.
For a frozen non-negative weight array `w` (the same weights for positive and negative):
`P_w = Σ w_r h+_r / Σ w_r`, `N_w = Σ w_r h−_r / Σ w_r`, `ER_w = P_w/N_w` (NA if `N_w = 0`).
Uniform weighting reproduces the #277 features (`l1_pos_hit`, `l1_neg_hit`, `l1_hard_er`).
Zero denominators are reported, never imputed.

## 3. Common queue and folds

The 129 labeled SIDs of #276 (all have the six features) are the common queue; folds are the #277
`FOLDS.json` assignment (5 SID groups, seed 42); near-duplicates cannot cross folds (verified in #277).
Strict subset = 120 strong-identity rows; the 9 weak rows are reported separately.

## 4. Model candidates (frozen)

All fits use the #277 procedure: `StandardScaler` fitted inside every inner training fold only,
`Ridge` with `alpha ∈ {0.01, 0.1, 1, 10}` selected by inner SID-group CV; OOF predictions only.

| id | features | role |
|---|---|---|
| A | `l1_pos_hit, l1_neg_hit, l1_hard_er, l2_distinct, l3_collision, l3_refine_bits` | control (refit on the common queue) |
| B | `P_w, N_w, ER_w, l2_distinct, l3_collision, l3_refine_bits` with `w = w_freq` | weighted replacement |
| D | A + `ER_w` (1 added feature) | increment (primary candidate) |
| E | A + `P_w, N_w` (2 added features) | increment |
| F | A + `w_lift`-based `ER_w` (1 added feature) | sensitivity only (reported, not selected) |

Controls: fold mean, fold median, and the single L1 Hard ER.

## 5. Selection rule and primary criterion (frozen)

The primary criterion (from the issue): **OOF MAE must decrease relative to A on the common queue and
the SID-group paired bootstrap 95% CI of `MAE(A) − MAE(candidate)` must exclude 0**, while Spearman and
Kendall must not worsen and the >0.06 confusion (FP and FN) must not worsen, both on the strict
120-row subset and on the full 129-row queue. `0.065` has no positives → `NOT_EVALUABLE`.
If several candidates pass, the winner is the one with the largest MAE improvement; ties go to the
fewer added features. If none passes, no correction is claimed and #277 remains the frozen predictor.
No weight/threshold search beyond this list; no fold/seed changes; no R2-specific tuning.

## 6. Deliverables

`IMPORTANCE.json` (coverage/strata), `WEIGHTED_TABLE.json` (per-SID P_w/N_w/ER_w for L1 and L1+L2 plus
stratum tables), `MODEL_COMPARISON.json` (OOF metrics, bootstrap, confusion, residuals, calibration),
`REPORT.md`, `VERDICT.md`, `HYPOTHESES.json`; `PREDICTOR_V2.json` + prospective predictions **only if**
the standard passes.
