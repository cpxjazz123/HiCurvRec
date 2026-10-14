# Codeword-crowding programme: one shared crowding condition, ten single-mechanism trees

Status at the time of writing: the shared layer and the Crowding-only control are complete
(Stage2 + Stage3 + committed); M1 is complete; M2's Stage2 is complete and its Stage3 was
killed mid-run by the loss of the GPU Slurm allocation; M3-M10 are implemented, verified and
committed, with their runs pending the same allocation. Rows that are still missing say
`pending` and are never filled with estimates.

Baseline for the whole programme: `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample`,
Stage3 `test_recall@10 = 0.0637372`, `test_ndcg@10 = 0.0351103`.

## 1. Fixed protocol

* **Shared layer** (byte-identical in all eleven trees, checked with `diff -q` per tree):
  * `scripts/capacity_mask.py` - L1 keeps only its first 128 selectable codewords
    (`CROWDING_L1_ALLOWED = 128`); distance to a disabled codeword becomes `+inf`, which both
    the argmin and the Sinkhorn path treat as unreachable. Codebook shapes are untouched, so
    the Stage3 token vocabulary is unchanged. The mask is installed **after** every other
    `_distances` patch in every tree, so a mechanism can never relax the capacity constraint.
  * `crowding_diagnostics.py` - the measurements, identical definitions in every tree:
    `level{q}_unused`, `level{q}_entropy_norm` (`H / ln 256`), `level{q}_top1_share`,
    `collision_at_{q}` (`1 - unique(prefix_q)/N`, joint prefix), `beh_auc_at_{q}` and
    `beh_conflict_at_{q}` (Mann-Whitney over joint prefixes; 200k transition samples),
    `recon_mse` (full-item reconstruction MSE measured at that evaluation).
  * `train_start` records `crowding_l1_allowed` and `crowding_mechanism`; every
    `EVAL_INTERVAL = 50` epochs a `crowding` event is written to `training_metrics.jsonl`.
* **Unchanged in every tree**: `EPOCHS = 3000`, `SEED = 42`, `EVAL_INTERVAL = 50`,
  `BATCH_SIZE_PER_RANK`, `PAIR_BATCH_SIZE_PER_RANK`, codebook sizes (256/256/256), KMeans
  init, optimiser and schedule, the behaviour ramp (`RAMP_START_EPOCH = 1500`,
  `BEHAVIOR_WEIGHT_MAX = 0.2`), pair sampling, and the Stage3 protocol
  (`NUM_EPOCHS 150 / NO_EVAL True / SKIP_TEST False / SEED 42 / BEAM_SIZE 20`, checked by the
  MCP on every Stage3 start).
* **Tree layout**: `stage2_RQ-VAE/TIGER_<slug>` + `results/stage2_RQ-VAE/TIGER_<slug>/<arm>/`
  + `results/stage3_T5Train/TIGER_<slug>__<arm>/`. Each mechanism tree was copied from the
  *source* tree, never from another mechanism tree.
* **Gates per tree** (all before its run): `py_compile`; a targeted mechanism check that
  exercises the installed wrapper (not only the helpers) on the tree's own probe checkpoint;
  `scripts/grad_check.py` (same outcome as the unmodified source tree, see 4.3);
  `diff -ru` audit against the control; MCP `stage2_status` pre-flight (`idle`); then the MCP
  smoke must return `passed: true` with empty `dead_terms` / `nonfinite_terms` and non-zero
  checkpoint-probe gradients before training is allowed to start.

## 2. Unified table

Diagnostics are the last `crowding` event (epoch 3000); Stage3 numbers come from that run's
own `test_final.json` (all `n_eval = 57439`). `pending` = run not finished.

| row | mechanism | final_collision | L1 unused | L1 H_norm | L1 top1 | coll@2 | coll@3 | auc@1 | conf@1 | recon_mse | recall@10 | ndcg@10 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | none, L1 capacity 256 | 0.036279 | 15 | 0.978224 | 0.010290 | 0.343352 | 0.036279 | 0.555995 | 0.998980 | 0.000141 | **0.0637372** | **0.0351103** | reference |
| crowding-only | none, L1 capacity 128 | 0.058812 | 131 | 0.861078 | 0.021312 | 0.502623 | 0.058812 | 0.573100 | 0.998905 | 0.000141 | 0.0635979 | 0.0350768 | crowding costs ~0.2 % |
| M1 | fan-out, parent-conditioned angular penalty on L2/L3 | 0.076219 | 135 | 0.857044 | 0.017164 | 0.583194 | 0.076219 | 0.573145 | 0.999025 | 0.000141 | 0.0630060 | 0.0342835 | COMPLETED_UNDERPERFORMING |
| M2 | behaviour-conflict margin loss (0.005, margin 0.05) | 0.109977 | 181 | 0.765435 | 0.043031 | 0.697442 | 0.109977 | 0.587628 | 0.998985 | 0.000141 | pending | pending | Stage3 pending |
| M3 | density-adaptive curvature for the geodesic assignment | pending | | | | | | | | | pending | pending | run pending |
| M4 | popularity-aware radial allocation (L1/L2) | pending | | | | | | | | | pending | pending | run pending |
| M5 | multi-interest branching per L1 bucket (L2) | pending | | | | | | | | | pending | pending | run pending |
| M6 | Mobius residual correction (levels 2-3) | pending | | | | | | | | | pending | pending | run pending |
| M7 | tangent-space gradient projection (radial part to the L1 codebook) | pending | | | | | | | | | pending | pending | run pending |
| M8 | geodesic nearest-neighbour assignment, all three levels | pending | | | | | | | | | pending | pending | run pending |
| M9 | per-level usage-count load penalty (L2/L3) | pending | | | | | | | | | pending | pending | run pending |
| M10 | prefix-keyed codeword load penalty (L2/L3) | pending | | | | | | | | | pending | pending | run pending |

Baseline row: Stage3 numbers read from
`results/stage3_T5Train/tiger_sample_resample/logs/Amazon_2023_Instruments/Oct-12-2026_17-28-43/test_final.json`
(the source tree's resample arm under the earlier variant-naming scheme, whose `test_recall@10`
and `test_ndcg@10` are exactly the programme's reference constants); its crowding readings were
recomputed offline from that run's own exported `sids_for_hgrec.npy` and checkpoint with the
shared definitions (`results/stage2_RQ-VAE/TIGER_CROWDING_CTRL/_evidence/baseline_offline_readings.txt`).

## 3. Crowding premise (Phase 1.5), evaluated on the control

Pre-registered criteria, applied as written:

1. **Collapse exclusion** (`level1_unused <= 2` and `level1_entropy_norm >= 0.90`): structurally
   inapplicable here. The mask disables exactly 128 of 256 L1 codewords, so `level1_unused >= 128`
   always holds and `H/ln 256` cannot exceed `ln 128 / ln 256 = 0.8750`. Eligible-capacity
   readings instead: 125 of the 128 selectable codewords are in use (97.7 %) and the realised
   entropy is 0.861078 / 0.875000 = 98.4 % of the attainable maximum. No collapse.
2. **Capacity really tightened**: yes on every capacity-side reading (`level1_unused` 15 -> 131,
   `top1_share` x2.07, `collision_at_1` 0.9902 -> 0.9949, `collision_at_2` 0.3434 -> 0.5026,
   `collision_at_3` 0.0363 -> 0.0588), and `train_start.crowding_l1_allowed = 128` is recorded in
   the control's metrics.
3. **Harmful crowding as the pre-registered pair defines it**: the criterion
   (`beh_auc_at_1` down AND `beh_conflict_at_1` up AND recall@10 < 0.0637372) does **not** fire.
   `beh_auc_at_1` moves up (0.5560 -> 0.5731) and `beh_conflict_at_1` moves down by 7.5e-5
   (0.998980 -> 0.998905). With this definition `beh_auc_at_1` is dominated by the prefix
   collision rate (a random item shares an anchor's L1 codeword with probability ~1/capacity), so
   cutting capacity raises it mechanically: the metric cannot separate normal sharing from harmful
   confusion in the harmful direction. The crowding evidence in this setup is the capacity and
   occupancy side (item 2) plus the downstream metric.
4. **Sharing vs confusion**: `collision_at_3` rises 0.0363 -> 0.0588, so the SID space is not
   squashed; the crowding is a prefix-level concentration, which is what the mechanisms attack.

Downstream: the control scores 0.0635979 / 0.0350768 against the baseline's 0.0637372 / 0.0351103
(-0.22 % / -0.10 % relative). Under this protocol the L1 capacity cut alone is nearly free, so
any mechanism's gain or loss must be read against the control, not the baseline.

## 4. Mechanisms

Each tree differs from the control only by its own operator plus identity, as audited per tree
(4.3). The operators are the pre-registered ones; where the plan's prose and its formula
disagree, the formula is implemented and the measured direction is reported (4.4).

### M1 `TIGER_hyperbolic_fanout` - parent-conditioned angular penalty (L2/L3)

`d_P(x, c) + strength * (1 - cos(u_x, u_c))` with `u_x = normalise(x - p)`,
`u_c = normalise(c - p)`, `p` the row's parent codeword, `c = 1.0`, `strength = 0.5`, computed in
closed form without an `(N, K, d)` tensor. `RQLayer.forward` hands each level the previous
level's indices; the trace helpers were rewritten to use the same rule so the behaviour term is
trained on the path the mechanism changes. With no parent indices the wrapped distance is
bit-identical to the control's.

* Measured: level 2 became **more** concentrated (`top1_share` 0.0122 -> 0.0650, `unused` 1 -> 14,
  `collision_at_2` 0.5026 -> 0.5832, `collision_at_3` 0.0588 -> 0.0762); level 1 spread slightly
  (`top1_share` 0.0213 -> 0.0172).
* Reason: the term is smallest for the candidate pointing the same way as the anchor, so items of
  one direction pile onto one child. The prose intent ("spread shared-L1 items across
  sub-branches") would need the opposite sign; that would be a post-hoc mechanism change and was
  not made.
* Stage3: 0.0630060 / 0.0342835 - worse than the control by 0.93 % / 2.26 %.
  Verdict **COMPLETED_UNDERPERFORMING**.

### M2 `TIGER_behavior_conflict_separation` - behaviour-conflict margin loss

`L = mean_i softplus(d_pos(i) - d_neg(i) + margin)` per level, in the curvature-1 ball, with the
hardest admissible negative chosen under exactly the behaviour term's masking rules (duplicate
targets plus, when enabled, the anchor's other real successors). Added to the total loss with the
fixed `weight = 0.005`, `margin = 0.05`; no new sampler, no change to
`behavior_contrastive_loss`, no capacity change.

* MCP smoke probed the term directly: `hyperbolic_conflict_loss` loss 0.753, grad 22.24, with
  `dead_terms = []`, `nonfinite_terms = []`.
* Measured: crowding **increased** (`level1_unused` 131 -> 181, `level1_top1_share` 0.0213 ->
  0.0430, `collision_at_2` 0.5026 -> 0.6974, `collision_at_3` 0.0588 -> 0.1100), while
  `beh_auc_at_1` rose 0.5731 -> 0.5876.
* Stage3: attempt 1 was killed at epoch 38/150 by the lost GPU allocation (5.7); result pending.

### M3-M10 (implemented, verified, runs pending)

| tree | operator | fixed parameters | level scope |
|---|---|---|---|
| M3 `TIGER_density_adaptive_curvature` | `c = c_min + (c_max - c_min) * (1 - H_usage/ln 256)` recomputed at every evaluation and read by the assignment closure; broadcast to all ranks | `c_min = 0.25`, `c_max = 4.0` | L1-L3 assignment |
| M4 `TIGER_popularity_radial` | `r_i = 1 / (1 + ln(1 + freq_i))`; codewords scaled radially inside the ball before the geodesic distance; factors measured once from the KMeans-initialised crowded assignment | `c = 1.0`, exponent 1.0 | L1, L2 |
| M5 `TIGER_interest_branching` | per-L1-bucket axes from the transition graph (power iteration, deflation), projected into the level-2 residual space; candidate penalised by `1 - cos(u_c, axis_{j*(x)})` | `c = 1.0`, `strength = 0.5`, `K = 2`, 20 iterations | L2 |
| M6 `TIGER_hyperbolic_residual` | `r = logmap0(expmap0(x) +_c (-expmap0(q)))` for levels 2-3; level 1 keeps the flat residual | `c = 1.0` | residual rule |
| M7 `TIGER_gradient_decoupling` | encoder-output gradient projected orthogonal to the row's L1 codeword direction, the removed radial part transplanted into that codeword's own gradient; forward is the identity | none | gradient operator |
| M8 `TIGER_geodesic_voronoi` | `install_nearest_assignment(layer, 1.0, euclidean=False)` on all three levels, `_hyp_assignment_active` held `True` for the whole run | `c = 1.0` | L1-L3 assignment |
| M9 `TIGER_capacity_redistribution` | `d(x, i) + strength * ln(1 + usage_i)` with a per-level usage EMA updated from the penalised argmin in training mode only | `decay = 0.9`, `strength = 0.5` | L2, L3 |
| M10 `TIGER_prefix_aware_quantization` | `d(x, i) + strength * ln(1 + counts[prefix, i])`, prefix key `l1` at L2 and `l1 * 256 + l2` at L3 | `decay = 0.9`, `strength = 0.5` | L2, L3 |

Each of these passed `py_compile`, a targeted mechanism check and the audit below. Highlights of
the checks (all on the tree's own probe checkpoint, CPU): M3's curvature formula against hand
values and its live-closure behaviour; M4's factors and the discriminating test that the capacity
constraint survives only because the radial patch reads back the wrapped function's `+inf`
pattern; M5's recovery of planted interest axes and its penalty against brute force; M6's
`logmap0`/`expmap0` round trip and that level 1 stays flat while level 2 differs; M7's operator
item by item (projection, transplant, identity forward) plus encoder/codebook gradients for both
objectives; M8's three-level geodesic distances and mask binding; M9's exact EMA rule and
frozen-in-eval histogram; M10's per-prefix counters, key packing and selected-slot bookkeeping.
For M9/M10 the measured load term is 2.1-3.1x the median base-distance margin and shifts 13-27 %
of level-2 assignments at the pre-registered strength, so neither is numerically inert.

### 4.3 Verification and single-variable audit

* `diff -rq` control vs each tree lists only `train_rqvae.py` (all trees) plus
  `model/layers.py` (M1, M5, M6, M10), `hyperbolic.py` (M6) and exactly one new mechanism module
  (none for M8, which reuses the existing patch). `crowding_diagnostics.py` and
  `scripts/capacity_mask.py` are byte-identical everywhere (`diff -q` clean), so the shared layer
  cannot drift between trees.
* Line-level inspection of each `train_rqvae.py` diff shows only: arm name, guard, provenance
  comment, mechanism import, mechanism-name constant, mechanism install/call lines, and the
  documented trace-helper delegation for the mechanisms whose assignment path must match the
  trace.
* `scripts/grad_check.py` returns the same verdict in every tree as in the unmodified source
  tree: residual shape `[3, 1024, 32]`, contrastive loss ~6.237, encoder gradients non-zero,
  19 parameters with non-zero gradients, and the single pre-existing failure
  ("no anchor retrieves its true successor") that the script also reports for the control and
  the source tree because it runs on a randomly initialised model with no KMeans init. No tree
  regresses this check; the script's own random-init assertion is not a mechanism signal.
* MCP smoke: control and M1 passed with empty `dead_terms`/`nonfinite_terms`; M2 additionally
  probed its own loss term (loss 0.753, grad 22.24).

### 4.4 Decisions and deviations, each with its reason

1. `recon_mse` is the **full-item** reconstruction MSE computed at the evaluation step (one extra
   forward over all 24587 items), not the training batch's mean: the key's definition asks for the
   full item set, and the training-batch value is not comparable across trees.
2. `beh_auc_at_{q}` / `beh_conflict_at_{q}` use the **joint** prefix `l1..lq`, matching
   `collision_at_{q}`. A first implementation used only the q-th level for `q >= 2`; it was fixed
   before any run, and the check now derives the expected AUC independently from the definition
   for `q = 1, 2, 3` and asserts that the single-level variant would differ.
3. M1/M5/M6/M10's trace helpers are **private** (`_trace_*`) in both the module and the trainer's
   import alias. The MCP smoke wraps every public callable of the tree's modules as a gradient
   probe and converts the result to a Python float; a public helper returning a multi-element
   tensor crashes the probe. Only scalar-returning functions (loss terms) stay public, which is
   also what the smoke is for.
4. M9/M10 keep the pre-registered additive operator on the level's existing distance. Adding a
   ball metric to make the fixed strength "look" stronger was considered and rejected as a
   post-hoc mechanism change; the measured penalty/margin ratio above shows the operator does move
   assignments at the specified scale.
5. M9/M10 install **after** the KMeans codebook init so the one-shot initialisation pass cannot
   seed the usage/prefix counters; M4/M5 likewise, because their factors and axes are measured
   from the initialised codebook.
6. M3 broadcasts the refreshed curvature from rank 0 to the other ranks right after the evaluation
   barrier, so all four ranks assign with the same curvature.
7. M6 adds `logmap0` to the tree's own `hyperbolic.py` (the plan allows this file's minimal
   change) and imports the mechanism module from `model/layers.py`, which every entry point of the
   tree already has on `sys.path`.
8. The pre-registered collapse gate in 3.1 is reported as structurally inapplicable rather than
   silently re-thresholded.

## 5. Incidents

1. **Control attempt 1 (FAILED_TECHNICAL, no result lost).** The first dispatch ran the smoke and
   started training while `crowding_diagnostics.py` still used the single-level behaviour metric.
   The module was corrected and the run was stopped 19 s into training, before any evaluation
   boundary, so no crowding event existed yet. Recorded in
   `results/stage2_RQ-VAE/TIGER_CROWDING_CTRL/_evidence/attempt1/record.json`.
2. **M1 attempt 1 (FAILED_TECHNICAL, MCP refused).** The smoke crashed in its own probe
   (`ValueError: only one element tensors can be converted to Python scalars`) because M1 exposed
   public multi-element helpers; the fix is deviation 4.4.3 and the refused smoke log is kept in
   `results/stage2_RQ-VAE/TIGER_hyperbolic_fanout/_evidence/attempt1_smoke_refused/`. No training
   was started by the refused attempt.
3. **M2 Stage3 attempt 1 (exited_incomplete, external).** The GPU Slurm job hosting this session
   (`61018313`, node `m3g119`) was terminated at 08:29:48 cluster time
   (`slurm-61018313.out`: "CANCELLED ... DUE to SIGNAL Terminated"). The run stopped at epoch
   38/150 with no traceback and no `test_final.json`; the launcher log is kept in
   `results/stage2_RQ-VAE/TIGER_behavior_conflict_separation/_evidence/`. The session now sits in
   job `60821920` on `m3j002`, which has no GPU devices, so the remaining Stage2/Stage3 runs
   (M2 retry, M3-M10) cannot be launched through the MCP until this session is inside a GPU
   allocation again; `61022707` (gpu partition, 7-day limit) is queued.
4. **Lingering Stage3 process groups.** Completed Stage3 runs leave their ranks alive after
   writing `test_final.json` (the NCCL teardown warning appears at process exit). Each such group
   was terminated after its verdict was on disk, purely to free the GPUs; no artifact, log or
   metric was removed.

## 6. Research questions

**Q1 - Does the crowding condition actually create harmful codeword crowding?** Capacity-wise yes,
behaviour-metric-wise as pre-registered no; see section 3. The control is 0.22 % below the
baseline in recall@10, so under this protocol L1 capacity 256 -> 128 is nearly free downstream and
the mechanisms must be judged against the control.

**Q2 - Which single hyperbolic mechanism mitigates it?** Partially answered: M1 does not - its
pre-registered operator concentrates level 2 and costs 0.93 % recall@10 versus the control. M2's
loss raises crowding further (section 4.2) and its Stage3 verdict is pending; M3-M10 are pending.

**Q3 - Does any mechanism beat the original 0.0637372 baseline?** No mechanism that has finished
does: the control reaches 0.0635979 and M1 0.0630060, both below the baseline. Rows M2-M10 are
pending and will be filled from their own `test_final.json` files when their runs complete.

## 7. Reproduction

Stage2: `write xd://mcp__genrec_stage2_start {"source_dir": ".../stage2_RQ-VAE/TIGER_<slug>"}`,
then poll `stage2_status` on the same tree until `finished` (three artifacts present). Stage3:
`write xd://mcp__genrec_stage3_start {"item_sids": ".../results/stage2_RQ-VAE/TIGER_<slug>/<arm>/item_sids.json"}`,
then poll `stage3_status` on the same path until `finished` with `test_final_stale = null`.
Per-tree checks: `python3 -m py_compile train_rqvae.py <mechanism module>`,
`CUDA_VISIBLE_DEVICES="" python3 scripts/grad_check.py`, and the targeted mechanism check used for
that tree. Per-tree audit: `diff -ru --exclude=logs --exclude=__pycache__ TIGER_CROWDING_CTRL TIGER_<slug>`.
