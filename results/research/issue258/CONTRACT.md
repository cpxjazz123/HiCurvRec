# Issue #258 implementation contract — role-conditioned level-2 cost

This file does **not** re-register the research hypothesis. The frozen research contract is
`results/research/issue256/rounds/r05/CONTRACT.md`
(sha256 `1ebc1eef161e3cf29806db5689230fbbcefae5f653f4d07e89989e83a9cfd752`).
This file freezes the implementation-level objects, shapes, boundaries and tolerances that the R05 candidate
must satisfy, plus the paper/official-implementation comparison required by the skill.

## 1. Level indices and the single change

Levels are 0-based indices `l in {0, 1, 2}` over the three residual-quantisation layers
(1-based levels 1, 2, 3). For item `i`, codeword `c` and residual `x_i^l`:

```
w_{i,l} = m_{r(i)}        if l == 1          # R05: level-2 cost is role-conditioned
          g_{i,l}         otherwise          # l in {0, 2}: item's own gate
D_l(i, c) = sum_e w_{i,l,e} * d_e(x_i^l, c)
```

- `m_r = softmax(theta_r)` over the three experts: `theta` has shape `[R+2, 3]` (`R` roles, one row for
  items without a role, one global row), dtype float32, one row per role. Initialised to zeros, so every
  role starts at the uniform mixture.
- `g_{i,l}` are the gate weights: two learned mixtures per item, shape `[N, 2, 3]`, normalised over the
  expert axis; row 0 serves semantic level 1 and row 1 serves semantic level 3. Keeping two rows preserves
  R04's no-gradient-free-parameter property: the second level's mixture is the role table, so the gate does
  not own a row for it.
- `d_e` are the three fixed curvature experts `e in {-1, 0, +1}` in the common scaled tangent space
  (`model/geometry.py`): negative curvature is the Poincaré asinh formula, zero is squared Euclidean,
  positive is the sphere chord-angle formula. No expert owns a scale.
- `r(i)` is read from the frozen training-only role table; `r(i) = -1` uses the uncovered-items row.

Difference from R04, exactly: `level_weights(..., level)` routes the role mixture at index **1** instead of
index **2**, and the gate again produces three mixtures. No other formula, loss, assignment rule, capacity,
schedule or protocol changes.

## 2. Switches (present and wired; not trained)

| switch | code state | level-2 mixture | level-2 geometry |
|---|---|---|---|
| `BOTH_ON` (candidate) | `role_mixture_on=True`, `curvature_on=True` | `m_{r(i)}` | three experts |
| `BEHAVIOR_OFF` | `role_mixture_on=False`, `curvature_on=True` | one global row for every item | three experts |
| `CURVE_OFF` | `role_mixture_on=True`, `curvature_on=False` | `m_{r(i)}` | all experts collapse to Euclidean |

`BEHAVIOR_OFF` must leave the level-1 and level-3 paths bitwise unchanged. `CURVE_OFF` must make the level-2
cost equal to plain squared Euclidean in the frozen tangent scale, independent of the role mixture.

## 3. Invariants

1. The hard SID of an item is a global identifier: it must not depend on batch companions, chunking or rank.
2. `role_rows` are integer row indices; the level-2 cost must gather them, never average over the batch.
3. The Sinkhorn balanced assignment and the assignment floor are the baseline's; only the level-2 cost matrix
   supplied to them changes. No new assignment rule, no extra loss, no behaviour-neighbourhood constraint.
4. Levels 0 and 2 receive gate rows 0 and 1 respectively; they must not read the role table.
5. Shared role weights are not shared codewords: codewords stay per-level and global.
6. The role table is the *same bytes* as R04 (`role_table.json`, sha256
   `32f986dac9bf8e382c11aedd62cd54e7defbf1853ee0b930b6fe51d4f334d3eb`); its `split_rule` must contain
   `training` and must not contain `heldout`. No new graph, orbit or sampling cache is created.
7. No source file outside `stage2_RQ-VAE/TIGER_role_conditioned_l2_rq/` and this evidence directory is
   modified by the round.

## 4. Boundary conditions and tolerances

| case | required behaviour |
|---|---|
| item with `r(i) = -1` | row `R` of the table (uncovered), never an out-of-range gather |
| empty batch (`N = 0`) | empty mixture `[0, 3]`; no reduction over an empty dimension |
| repeated item ids | identical row and identical cost for every copy |
| zero residual | finite cost for every expert; no NaN from `log(0)` or `asinh(inf)` |
| coincident point | Euclidean and sphere experts return 0; hyperbolic returns 0 through the radial limit |
| role mixture with one logit dominating | softmax stays finite and sums to 1 within `1e-6` |
| assignment | hard codes are integers; no tie-breaking change relative to the baseline rule |
| numerical agreement with the independent reference | `max |D_ref - D_prod| <= 1e-5 * max(1, |D_ref|)` in float64/float32 comparison; hard codes equal exactly |

## 5. Paper / official-implementation comparison

| module | paper location | official implementation | this project's choice | why |
|---|---|---|---|---|
| residual quantization, 3 levels × 256 codewords | TIGER, “Recommender Systems with Generative Retrieval” (Rajput et al., NeurIPS 2023), §3.2 semantic ID construction over RQ-VAE (Lee et al., CVPR 2022) | TIGER's released training code is not vendored in this repository; the executable reference is the frozen baseline `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` | keep the baseline's encoder/decoder, three 256-code levels, 768→32 capacity and RQ losses unchanged | the research change is the level-2 cost only; changing capacity or losses would confound the comparison |
| balanced assignment | RQ-VAE uses nearest codeword; balanced assignment follows Cuturi's Sinkhorn (NeurIPS 2013) as used by the frozen baseline | not re-implemented | keep the baseline Sinkhorn at the last level plus the floor-constrained nearest codeword; innovation enters the level-2 cost matrix (CLAUDE.md §19h) | replacing the assignment rule is out of scope and was measured to collapse code usage |
| fixed-curvature experts | mixed-curvature product manifolds (Gu et al., ICLR 2019) and Poincaré embeddings (Nickel & Kiela, NeurIPS 2017) | no official code used | three fixed experts κ ∈ {-1, 0, +1} in one scaled tangent space, squared distances, as inherited from R01–R04 | the mechanism compares geometry placements under one frozen implementation; fitting curvature per expert would add untested degrees of freedom |
| role signal | graphlet-orbit / graphlet degree vector literature (Przulj, 2007) | project asset, not an external implementation | one frozen orbit-role label per item from the training behaviour graph only; learned per-role mixture | the issue requires a training-only behaviour signal and forbids new graph/orbit/cache construction |
| GNN gate | not in TIGER | project implementation `model/layer_gate.py` | shared directed GNN + two expert-query rows serving levels 1 and 3; R05 restores the level-3 row | the contract keeps the item-level gate on levels 1 and 3 without a gradient-free row |

The paper-side statements above are limited to the published equations and to what the repository actually
contains; no claim is made about unpublished details of the TIGER codebase.

## 6. Amendment 1 (2026-11-25, before any training)

Section 1 first described the gate as `[N, 3, 3]` with one row per level. That would leave the middle query
row without a consumer (the level-2 mixture comes from the role table), i.e. a parameter with an identically
zero gradient. The implementation and this contract now use two gate rows, `[N, 2, 3]`, with row 0 serving
level 1 and row 1 serving level 3 — the same number of gate parameters as R04, now mapped to the first and
third levels. No other formula, switch, invariant or prediction changes; the research contract in
`results/research/issue256/rounds/r05/CONTRACT.md` is untouched.

