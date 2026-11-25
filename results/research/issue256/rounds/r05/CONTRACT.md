# R05 preregistration: role-conditioned level-2 prefix margin

Status: frozen before any R05 training. One main mathematical change, one
preregistered candidate, one legal complete run. The run id is
`issue256_r05_formal`; the arm is `TIGER_role_conditioned_l2_rq`.

## 1. Why this change (measured evidence, not a guess)

R03 put the role-conditioned curvature mixture at the level-1 cost and R04 at
the level-3 cost; both left the three-code triple essentially unique (baseline
20425, R03 23248, R04 22995 triples of 24587 items). The R04 export was then
decomposed against the frozen role table:

| quantity | value |
|---|---|
| distinct L1 codes | 256 (7.4684 of 8 bits) |
| distinct L2 prefixes | 8781 -> 34.3 second codes per first code |
| items per L2 prefix (mean / median / p90) | 2.80 / 1 / 6 |
| roles per L2 prefix (mean / median) | 1.75 / 1 |
| distinct L3 codes per (prefix, role) cell | 1.568 |
| items per triple | 1.069 |

So level 1 has nothing left to coarsen, the role is already almost pure inside
a prefix (median one role), and the level-3 refinement adds only 1.57 distinct
codes per (prefix, role) cell. The remaining freedom is how many distinct
**second** codes one level-1 cell opens: 34.3, with a median prefix holding a
single item. That is the level no round has touched yet.

## 2. The single main change

The per-role curvature mixture `m_{r,e} = softmax(theta_r)` multiplies the
**level-2** cost; levels 1 and 3 keep the item's own learned gate.

Notation, for item `i`, level `l`, codeword `c`:

* `r(i)` - the item's graphlet orbit role (the same frozen training-only table
  as R03/R04);
* `m_{r,e}` - the learned role mixture, `softmax(theta_r)` (R05 changes only
  which level consumes it);
* `g_{i,l,e}` - the item's own learned mixture from the layer gate, which now
  serves levels 1 and 3;
* `d_e(x, c)` - the squared distance of residual `x` to codeword `c` in expert
  `e`'s geometry.

Level-2 cost (the change):

    D_2(i, c) = sum_e  m_{r(i), e} * d_e(residual_i^2, c)

Levels 1 and 3 cost (the R04 behaviour removed there):

    D_l(i, c) = sum_e  g_{i,l,e} * d_e(residual_i^l, c),  l = 1, 3

Everything else is inherited unchanged from R04: the Sinkhorn balanced
assignment and the assignment floor, the commitment and codebook losses, the
sectional-curvature supervision, the layer-wise curvature budget calibration
and its refresh schedule, the 768/32 capacity, the three 256-code levels, the
four-card 3000-epoch Stage2 recipe (global batch 4096, seed 42, no early stop)
and the frozen Stage3 protocol (150 epochs, beam 20, seed 42).

### 2.1 Scope check (the mechanism gate)

* **Behaviour signal**: `r(i)`, the graphlet orbit type from the training
  behaviour graph only (same frozen asset as R03/R04).
* **Curve operation**: the three-expert mixed-curvature distance `d_e` with the
  curvature weights `m_{r(i),e}`.
* **Coupling path**: `r(i) -> m_{r(i)} -> D_2(i,c) -> level-2 code -> second SID
  token -> Stage3 decoder`. Removing the behaviour side (one global mixture) or
  the curve side (Euclidean distance) removes the candidate's computation.
* The Sinkhorn/floor assignment rule is the baseline's; only the level-2 cost
  matrix changes.

### 2.2 The three switches (present in code, not trained)

| switch | behaviour | curve | meaning |
|---|---|---|---|
| `BEHAVIOR_OFF` | one global mixture | on | the role-to-geometry coupling is removed |
| `CURVE_OFF` | role mixture | off (`d_e` -> Euclidean) | the mixed curvature is removed |
| `BOTH_ON` | role mixture | on | the candidate |

## 3. Falsifiable predictions (fixed before the run)

1. **Level-2 branching (pretraining check, the mechanism's own signature)**:
   the exported SID has at most 6000 distinct two-code prefixes (R04: 8781),
   i.e. at least 4.1 items per prefix, while level 1 stays unchanged (256
   codes; absolute L1 prefix carry inside 0.0496..0.0696).
2. **Recommendation**: validation Recall@10 >= 0.0583.
3. **Failure criterion**: distinct two-code prefixes >= 7000 (less than a 20%
   reduction), or validation Recall@10 <= 0.0576, falsifies the
   role-conditioned level-2 coupling.
4. **Stage2 gate**: three-code collision < 0.40; quantisation error and
   effective rank are recorded as diagnostics only.
5. **One run**: exactly one 3000-epoch four-card Stage2 run and one frozen
   150-epoch Stage3 run; no ablation arm, no tuning sweep, no seed sweep.

## 4. What the comparison can and cannot show

R05 versus R04 isolates the level the role table drives (level 2 versus level 3)
with the mechanism, the table and the recipe fixed. R05 versus the frozen
baseline changes the mechanism family as well and is the weaker comparison.
Both prior placements are recorded as completed negative results, so a third
negative would say the per-level mixture is not the lever on this corpus at
any level.
