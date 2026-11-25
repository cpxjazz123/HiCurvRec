# R03 preregistration: role-conditioned level-1 curvature mixture

Status: frozen before any R03 training. One main mathematical change, one
preregistered candidate, one legal complete run. The run id is
`issue256_r03_formal`; the arm is `TIGER_role_conditioned_l1_rq`.

## 1. Why this change (measured evidence, not a guess)

The R02 round falsified its own hypothesis: refreshing the maximal global module
budget every 50 native epochs moved final Recall@10 by -0.000070 and validation
Recall@10 by -0.000331 against the single refresh of R01. The refresh schedule
is therefore not the binding constraint.

Pairing 14 arms by their SID file and recomputing the layer metrics gives a
strong, repeated correlate of the final Recall: the granularity of the level-1
commitment.

| SID property | correlation with final Recall@10 |
|---|---|
| absolute L1 prefix carry | +0.625 |
| absolute L2 prefix carry | +0.597 |
| absolute L3 prefix carry | +0.583 |
| L3 refinement gain (bits) | +0.648 |
| distinct two-code prefixes | -0.670 |
| three-code collision | +0.617 |

The #255/#256 lineage sits at the wrong end of that axis: 8840 distinct
two-code prefixes, three-code collision 0.0649 (the baseline TIGER_RQ-VAE is at
7374 / 0.1693, the best mixed arm at 4151 / 0.2572), and an absolute L1 prefix
carry of 0.0643 against 0.0925 for the baseline and 0.1599 for the best mixed
arm. Its final Recall is correspondingly the lowest of that group (0.0527).

The mechanism behind it is visible in the lineage: the level-1 assignment cost
is already a mixed, item-weighted three-expert cost
(`model/geometry.py::mixed_cost`, `model/mechanism.py::assignment_cost`), and
the weights are the *free learned* per-item gate over the direction-aware ego
tokens. A gate free enough to give every item its own curvature mixture can
make every item's distance to every codeword distinctive, so the balanced
assignment spreads the items nearly one per code: the SID becomes an item
identity map instead of a taxonomy. The measured quantisation error of the R02
SID (0.0077) says the same thing from the other side - the encoder fits each
item individually.

## 2. The single main change

The level-1 curvature mixture is taken from the item's **graphlet orbit type**
instead of from the item's own free learned mixture. Levels 2 and 3 keep the
learned gate unchanged.

Notation, for item `i`, level `l`, codeword `c`:

* `g_{i,l} ∈ Δ_3` - the learned expert mixture of item `i` at level `l`
  (three curvatures `{-1, 0, +1}`), exactly as in R02.
* `r(i) ∈ {1..R}` - the item's graphlet orbit type, computed from the
  **training-only** behaviour graph (no validation or test behaviour).
* `m_{r,e} = mean{ g_{i,1,e} : r(i) = r }` - the role mixture, taken over the
  items of that role in the current batch and detached from the graph.
* `d_e(x, c)` - the squared distance of the residual `x` to codeword `c` in
  expert `e`'s geometry (`model/geometry.py::expert_squared`).

Level-1 cost (the change):

    D_1(i, c) = Σ_e  m_{r(i), e} · d_e(residual_i, c)

Level 2 and 3 cost (unchanged):

    D_l(i, c) = Σ_e  g_{i,l,e} · d_e(residual_i, c)

Everything else is inherited unchanged from R02: the Sinkhorn balanced
assignment and the assignment floor, the commitment and codebook losses, the
sectional-curvature supervision, the layer-wise curvature budget calibration
and its refresh schedule, the 768/32 capacity, the three 256-code levels, the
four-card 3000-epoch Stage2 recipe (global batch 4096, seed 42, no early
stop) and the frozen Stage3 protocol (150 epochs, beam 20, seed 42).

### 2.1 Scope check (the mechanism gate)

* **Behaviour signal**: `r(i)`, the graphlet orbit type of the item, computed
  from the training behaviour graph only.
* **Curve operation**: the three-expert mixed-curvature distance `d_e` with the
  curvature weights `m_{r(i),e}`.
* **Coupling path**: `r(i) → m_{r(i)} → D_1(i,c) → level-1 code → first SID
  token → Stage3 decoder`. Removing the behaviour side (one global mixture)
  or the curve side (Euclidean distance) removes the candidate's computation;
  both-on is the candidate.
* The balanced assignment rule itself is not replaced: the innovation is in the
  cost matrix, the Sinkhorn/floor machinery stays the baseline's.

### 2.2 The three switches (present in code, not trained)

| switch | behaviour | curve | meaning |
|---|---|---|---|
| `BEHAVIOR_OFF` | one global mixture `m̄_e` for every item | on | the role-to-geometry coupling is removed, the curvature stays |
| `CURVE_OFF` | role mixture | off (`d_e` → Euclidean) | the mixed curvature is removed, the role stays |
| `BOTH_ON` | role mixture | on | the candidate |

Per the experiment policy, the off states exist as code with a cheap wiring
check only; no off-state arm is trained.

## 3. Falsifiable predictions (fixed before the run)

Selection uses the validation split only. The final test is read once, after
the candidate's decision is frozen.

1. **Level-1 granularity (pretraining check, the mechanism's own signature)**:
   distinct two-code prefixes ≤ 6000 (R02: 8840) and absolute L1 prefix carry
   ≥ 0.10 (R02: 0.0643) on the validation split.
2. **Recommendation**: validation Recall@10 ≥ 0.0583, i.e. R01's 0.05764 plus
   the 0.000949 median repeat spread of the protocol.
3. **Failure criterion**: if the L1 carry stays below 0.08, or validation
   Recall@10 ≤ 0.0576, the role-conditioned level-1 coupling is falsified and
   the round is recorded as a negative result.
4. **Stage2 gate**: three-code collision < 0.40 (owner threshold); quantisation
   error and effective rank are recorded as diagnostics only.
5. **One run**: exactly one 3000-epoch four-card Stage2 run and one frozen
   150-epoch Stage3 run; no ablation arm, no tuning sweep, no seed sweep.

## 4. What is held constant and what the comparison can and cannot show

Held constant: capacity, level widths, losses, assignment rule, schedule,
budget calibration, optimiser, seed, batch sizes, epochs, port, data splits and
the Stage3 protocol.

Causal limit, stated in advance: R01 and R02 carry the numerical repairs
(deterministic static CUDA means, highest source-owned float32 dense
arithmetic, weak layer-owned rotating coverage history), so an R03 versus R02
comparison is not a single-factor isolation of the role conditioning; R03 versus
the frozen baseline TIGER_RQ-VAE is the only clean single-factor comparison
available, and it changes the mechanism family as well.

## 5. Admission before the first worker

The same machine gates as R02, on the R03 source: real one-epoch entry with SID
export, the behaviour-graph contract, the four-rank diagnostics (18 module
gradient/optimizer groups, anchors 1 and 51, stale-coefficient mutation
caught), the inclusive 31-60 epoch window under the 0.8s ceiling, the complete
candidate/asset/controller test suites with no skip, and a dispatch probe over
the exact current source inventory. The evaluation consumers (Stage3 protocol,
selection rule, bias disclosure) are frozen before the first worker, as in R02.
