# Issue #277 — FEATURE_CONTRACT (six SID-only features)

All six features are computed from the discrete SID alone by the frozen evaluator
`tools/issue252/layer_metrics.py` (sha recorded in every feature JSON) on
`results/stage0_build_parquet/valid.parquet` (columns `history`, `seen_history`, `target`) and the
SID rows `item -> (c1, c2, c3, counter)` of the frozen loader (`tools/issue252/final_evaluation.py`,
`load_sid(path, 24587, (256, 256, 256))`). No checkpoint, no recall value, no training input is read.

## Definitions (exact)

For an example row with non-empty `history`, `anchor = history[-1]`,
`excluded = set(history) ∪ set(seen_history) ∪ {anchor, target}`, `population = N − |excluded|`
(N = 24 587 items). Examples with empty history or empty legal population are excluded from the
denominator (`effective_rows`).

| feature | formula | unit | denominator / mask |
|---|---|---|---|
| `l1_pos_hit` | `#{examples: prefix1(anchor) == prefix1(target)} / effective_rows` | rate | effective validation rows |
| `l1_neg_hit` | `mean_examples( (#items with prefix1(anchor) − #excluded with that prefix) / population )` | expected probability | same rows |
| `l1_hard_er` | `l1_pos_hit / l1_neg_hit` | ratio (dimensionless) | same rows; `NA` when `l1_neg_hit = 0` |
| `l2_distinct` | `#distinct (c1, c2) pairs` over the N items | count (items) | all N items |
| `l3_collision` | `1 − #distinct (c1, c2, c3) / N` | fraction (duplicate fraction) | all N items |
| `l3_refine_bits` | `H3 − H2`, `Hk = −Σ_g (n_g/N) log2(n_g/N)` over level-k prefix groups | bits | all N items |

**`l3_collision` is the duplicate fraction `1 − unique/N`, not an item-weighted or pairwise rate.** The
pairwise coincidence rate `1 − Σ_g n_g(n_g−1)/(N(N−1))` and `1 − Σ_g (n_g/N)²` are different numbers;
they are never substituted (the stored `metrics_corrected.collision` equals this definition for all 73
cross-checked SIDs).

**Functional dependence:** `l1_hard_er = l1_pos_hit / l1_neg_hit` up to a constant factor — the ER is
a deterministic function of the two hit rates (for the same example set and negative definition).
Its regression coefficient is therefore **not** an independent causal effect; removing the ER from a
model that keeps both rates removes almost no information.

## Worked example (hand-computed, asserted in `tests/test_feature_examples.py`)

SID with N = 6: items `0:(1,1,1) 1:(1,1,2) 2:(1,2,1) 3:(1,2,2) 4:(2,2,2) 5:(2,2,3)`.

* `l2_distinct` = 3 (`(1,1)`, `(1,2)`, `(2,2)`); `l3_collision` = 1 − 6/6 = **0**;
  `H2` = 3 groups × 2 items → `−3·(2/6)·log2(2/6)` = **1.585 bits**; `H3` = 6 singletons → **2.585 bits**;
  `l3_refine_bits` = **1.0 bit**.
* Example `history=[0]`, `seen=[]`, `target=2`: `anchor=0`, `excluded={0,2}`, `population=4`.
  * L1: `prefix1(0)==prefix1(2)==(1)` → positive hit 1; L1 group `(1)` has 4 members, 2 excluded →
    expected negative probability `2/4 = 0.5`; `l1_hard_er = 1.0 / 0.5 = 2.0`.
  * L2: `(1,1) != (1,2)` → 0; L2 group `(1,1)` has 2 members, 1 excluded → `1/4 = 0.25`.
  * L3: 0; the single-member group is fully excluded → 0.
* Duplicate-fraction vs pairwise: N = 5 with items `0..3` sharing one triple and item 4 unique →
  `l3_collision = 1 − 2/5 = 0.6`, pairwise rate `1 − 12/20 = 0.4` (asserted different).

## Model input contract

The frozen predictor consumes exactly these six numbers in this order:
`[l1_pos_hit, l1_neg_hit, l1_hard_er, l2_distinct, l3_collision, l3_refine_bits]`.
Missing, non-finite, or ER = `NA` values are rejected (no imputation). Values outside the training
min/max of the development queue trigger the out-of-domain warning in `predict277.py`.
