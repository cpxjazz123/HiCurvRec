# R04 preregistration: role-conditioned level-3 refinement margin

Status: frozen before any R04 training. One main mathematical change, one
preregistered candidate, one legal complete run. The run id is
`issue256_r04_formal`; the arm is `TIGER_role_conditioned_l3_rq`.

## 1. Why this change (measured evidence, not a guess)

R03 falsified its own mechanism signature and its recommendation: the
role-conditioned level-1 mixture produced 9257 distinct two-code prefixes
(prediction <= 6000), an absolute L1 prefix carry of 0.0596 (criterion < 0.08
= falsified), validation Recall@10 0.054301 (prediction >= 0.0583) and final
test Recall@10 0.049009 (R02: 0.052665).

The R03 export was measured directly (conditional entropies over the persisted
SID, 24587 items):

| quantity | value |
|---|---|
| H(L1) | 7.4596 bits (256/256 codes used, near-uniform) |
| H(L2 | L1) | 5.1067 bits |
| H(L3 | L1 L2) | 1.9028 bits |
| distinct three-code triples | 23248 of 24587 (collision 0.0545) |

So the first level is already as coarse as a 256-code level can be, and the
item-unique information that makes the triple an identity map sits in the
third level's refinement given the prefix: 2.02 bits of freedom over 256
codewords is the level that turns a coarse prefix into a near-unique triple.
Changing the level-1 metric therefore could not move the triple's
granularity, which is exactly what R03 measured. The coupling must act where
the freedom is.

## 2. The single main change

The R03 role table moves from the level-1 cost to the level-3 cost. The
per-role curvature mixture `m_{r,e} = softmax(theta_r)` now multiplies the
distance of the third level; levels 1 and 2 keep the item's own learned gate.

Notation, for item `i`, level `l`, codeword `c`:

* `r(i) in {0..R+1}` - the item's graphlet orbit role (the same frozen
  training-only table as R03, rows for the roles plus the uncovered-items row
  and the global row);
* `m_{r,e}` - the learned role mixture, `softmax(theta_r)`, R04 changes only
  which level consumes it;
* `g_{i,l,e}` - the item's own learned mixture from the layer gate, which now
  serves levels 1 and 2;
* `d_e(x, c)` - the squared distance of the residual `x` to codeword `c` in
  expert `e`'s geometry (`model/geometry.py::expert_squared`).

Level-3 cost (the change):

    D_3(i, c) = sum_e  m_{r(i), e} * d_e(residual_i^3, c)

Levels 1 and 2 cost (the R03 behaviour removed there):

    D_l(i, c) = sum_e  g_{i,l,e} * d_e(residual_i^l, c),  l = 1, 2

Everything else is inherited unchanged from R03: the Sinkhorn balanced
assignment and the assignment floor, the commitment and codebook losses, the
sectional-curvature supervision, the layer-wise curvature budget calibration
and its refresh schedule, the 768/32 capacity, the three 256-code levels, the
four-card 3000-epoch Stage2 recipe (global batch 4096, seed 42, no early stop)
and the frozen Stage3 protocol (150 epochs, beam 20, seed 42).

### 2.1 Scope check (the mechanism gate)

* **Behaviour signal**: `r(i)`, the graphlet orbit type of the item, computed
  from the training behaviour graph only (same frozen asset as R03,
  `results/research/graphlet_roles/instances.npz`, one label per item).
* **Curve operation**: the three-expert mixed-curvature distance `d_e` with
  the curvature weights `m_{r(i),e}`.
* **Coupling path**: `r(i) -> m_{r(i)} -> D_3(i,c) -> level-3 code -> third
  SID token -> Stage3 decoder`. Removing the behaviour side (one global
  mixture) or the curve side (Euclidean distance) removes the candidate's
  computation; both-on is the candidate.
* The balanced assignment rule itself is not replaced: the innovation is in
  the cost matrix of one level, the Sinkhorn/floor machinery stays the
  baseline's.

### 2.2 The three switches (present in code, not trained)

| switch | behaviour | curve | meaning |
|---|---|---|---|
| `BEHAVIOR_OFF` | one global mixture for every item | on | the role-to-geometry coupling is removed, the curvature stays |
| `CURVE_OFF` | role mixture | off (`d_e` -> Euclidean) | the mixed curvature is removed, the role stays |
| `BOTH_ON` | role mixture | on | the candidate |

Per the experiment policy, the off states exist as code with a cheap wiring
check only; no off-state arm is trained.

## 3. Falsifiable predictions (fixed before the run)

Selection uses the validation split only. The final test is read once, after
the candidate's decision is frozen.

1. **Level-3 refinement (pretraining check, the mechanism's own signature)**:
   the exported SID has `H(L3 | L1 L2) <= 1.2` bits and at most 12000 distinct
   three-code triples on the validation split (R03: 2.0193 bits, 23247
   triples), while the level-1 statistics stay put (absolute L1 prefix carry
   within +-0.01 of 0.0596, i.e. 0.0496..0.0696).
2. **Recommendation**: validation Recall@10 >= 0.0583, i.e. R01's 0.05764 plus
   the 0.000949 median repeat spread of the protocol.
3. **Failure criterion**: if `H(L3 | L1 L2) >= 1.5` bits, or validation
   Recall@10 <= 0.0576, the role-conditioned level-3 coupling is falsified and
   the round is recorded as a negative result.
4. **Stage2 gate**: three-code collision < 0.40 (owner threshold); quantisation
   error and effective rank are recorded as diagnostics only.
5. **One run**: exactly one 3000-epoch four-card Stage2 run and one frozen
   150-epoch Stage3 run; no ablation arm, no tuning sweep, no seed sweep.

## 4. What is held constant and what the comparison can and cannot show

Held constant: capacity, level widths, losses, assignment rule, schedule,
budget calibration, optimiser, seed, batch sizes, epochs, port, data splits and
the Stage3 protocol.

Causal limit, stated in advance: R04 versus R03 isolates **the level that the
role table drives** (level 3 versus level 1) with the mechanism, the table and
the recipe held fixed. R04 versus the frozen baseline `TIGER_RQ-VAE` changes
the mechanism family as well and is therefore the weaker comparison.

## 5. Admission before the first worker

The same machine gates as R03, on the R04 source: real one-epoch entry with
SID export, the behaviour-graph contract, the four-rank diagnostics (module
gradient/optimizer groups, anchors 1 and 51, stale-coefficient mutation
caught), the inclusive 31-60 epoch window under the 0.8s ceiling, the complete
candidate/asset/controller test suites with no skip, and a dispatch probe over
the exact current source inventory. The evaluation consumers (Stage3 protocol,
selection rule, bias disclosure) are frozen before the first worker, as in R03.

## 6. Amendment 1 (frozen before any R04 training)

Section 1 first quoted `H(L3 | L1 L2) = 2.0193` bits and 23247 triples from a reduction that
treated the export's fourth column as a semantic code. That column is the per-item collision
counter appended by `_extend_collisions`; it is not part of the three-code triple. Recomputing on
the three semantic codes gives `H(L1) = 7.4596`, `H(L2 | L1) = 5.1067`, `H(L3 | L1L2) = 1.9028`
bits and 23248 distinct triples of 24587 items (collision 0.0545, identical to the recorded export
quality). The diagnosis, the single change and the predictions of section 3 are unchanged; the
thresholds stay <= 1.2 bits and <= 12000 triples. `rounds/r04/DIAGNOSIS.json` carries the corrected
measurement with its method and the SID sha256.
