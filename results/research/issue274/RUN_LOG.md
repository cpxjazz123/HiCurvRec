# Issue #274 run log — curvature-selected pair ranking on the final hard q

## Issue identity

| item | value |
|---|---|
| issue | #274 |
| title | Stage2-only: curvature-selected relative pair ranking (closed vs open outer pairs) |
| body sha256 | `d258950dc4586c498329c029008acf2819e7c98a463d30a0e1c74246a2a4d541` (the body stored verbatim in `ISSUE_BODY.md`, 6673 bytes) |
| INFRA_MODE | **FROZEN** — the issue contains no sentence authorising new or repaired public infrastructure ("不重建公共资产、不改公共infra"), so every common component is only called, never edited |
| training decision | **yes**: the acceptance list (issue section 3) can only be produced by a new 3000-epoch Stage2 candidate; the frozen TIGER baseline is reused, never retrained |
| mechanism scope | one candidate tree `stage2_RQ-VAE/TIGER_pairrank_rq` copied from `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` with `ARM_NAME` as the only constant change, plus the pair-ranking loss module, the frozen pair index and the issue-specific tests and analysis |

## Frozen acceptance list (issue section 3, verbatim thresholds)

| id | criterion | threshold | source |
|---|---|---|---|
| C1 | the hold-out pair ranking accuracy: the hard-q positive-pair distance < the negative-pair distance | **strictly greater** than the same-caliber frozen TIGER value (`baselines.json`), or the pre-registered gap metric when the TIGER value is saturated | the frozen pair index's hold-out centres, the model's hard q |
| C2 | the fixed-margin satisfaction rate on the same hold-out pairs: `d_pos + m <= d_neg` | **strictly greater** than the frozen TIGER value | same |
| C3 | the SID's distinct L2 prefixes | `>= 10918` (the #265 identity, re-verified from `results/stage2_RQ-VAE/TIGER_euclid_local_curve_supervision_rq/stage2_metrics/`) | `results/research/issue271/collect_metrics.py` |
| C4 | the SID's effective rank | `>= 8` | the frozen issue-271 collector / `tools/research/stage2/sid_quality.py` |
| C5 | the SID's positive/negative prefix discrimination: the positive pair's prefix hit and the longest common prefix relative to the negative pair | `>=` the frozen TIGER value at the same caliber (the tolerance frozen in advance in `baselines.json`) | the exported SID |
| C6 | the behavioural neighbourhood fidelity at the same caliber | `>= 0.95 x` the frozen TIGER value (the relative tolerance frozen in advance) | `tools/issue252/layer_metrics.py` (the frozen implementation) |
| C7 | the global common-scale J | `<= 0.338335882630658` (the #269 re-check value, from `results/research/issue272/round3_pass_table.json`) | the candidate's own `model/global_constraint.py` |
| C8 | the 1-hop and 2-hop J | `<= 0.798295 x 1.05` and `<= 0.278106 x 1.05` (the #269 values plus the pre-registered 5 % tolerance) | the candidate's own `model/global_constraint.py` |
| C9 | the reconstruction | `<= 1.05 x` the #265 same-caliber value (the earlier reported inconsistency to be verified first) | the final training record |
| C10 | the SID's technical validity: the re-exported SID's identity matches the checkpoint and the non-collision admission passes | no collision threshold; the identity check must pass | `tools/research/stage2/export_check.py`, the exported SID's sha256 |

**Diagnostics, explicitly not criteria** (issue section 3): the curvature MAE / Spearman / sign /
within-centre ordering, the prediction variance, the slope, the collision rate, the three-layer
occupancy, the L3 refinement, the Graphlet/Orbit stratification. Stage3 is forbidden in every
outcome, so no Recall is produced or claimed.

## Round log

### Round 1 (2026-11-28) — the offline sample contract

- Read the issue; froze the C-list above; INFRA_MODE=FROZEN; the training decision (yes).
- Infra snapshot: `INFRA_SNAPSHOT_START.json` (61 public components, the same path list as the
  #272/#273 rounds). The used assets were verified against `assets/manifest.json` by sha256: the
  thirteen triplet shards, `assets/index.npz`, the three visible training graphs and
  `assets/train/item_ids.npy` all MATCH; the assets are read, never rebuilt.
- The real triplet table was surveyed (`survey_triplets.py`): 28,391,948 rows, all valid, 13
  directed structure classes; the class separates the closed triangles (`bc_dist_sq == 1`) from the
  open wedges (`bc_dist_sq == 4`) exactly.
- The class -> centre-edge direction map was derived from the visible training graph
  (`derive_class_directions.py`): 13 classes, 2-3 centre-edge direction patterns each.
- The full scan built the frozen pair index (`build_pair_index.py`): 79,185 comparability groups
  (the centre plus the unordered multiset of its two centre-edge directions), 26,583 groups
  selected (21,221 training + 5,362 hold-out) over 10,782 centres; the exclusions are recorded
  (39,905 groups without a closed candidate, 406 without an open candidate, 12,288 below the
  minimum support, 3 tied at the extremum).
- The offline diagnostics (`offline_diagnostics.py`): the curvature's within-class ordering tracks an
  independent higher-order tightness signal (the outer pair's common-neighbour count: the mean
  Spearman +0.874, the median +0.937 over 2,482 matched class groups, 98.9 % positive), so the
  curvature is **not** redundant with the closed/open label; but the raw curvature is not comparable
  across the two populations (the selected closed median +0.697 against the selected open median
  +3.802, because the open wedges carry the larger `g(b,c)^2/4` term), so the frozen rule selects
  the extremal representative **inside** each population and never compares the two populations.
- The residual correlation from shared items: 28.4 % of the training groups have an item shared
  between the positive and the negative pair; 11,122 distinct items are used, the mean item
  participation 7.63.

### Round 1 implementation (2026-11-28)

- The candidate tree `stage2_RQ-VAE/TIGER_pairrank_rq` was copied from the read-only
  `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` with `ARM_NAME` as the only constant change, plus:
  - `pair_rank_loss.py` — the production mechanism: `hard_quantized_sum` (the hard `q = sum_l c_l`
    through the differentiable embedding lookup), `pair_ranking_loss` (the hinge
    `mean max(0, d_pos - d_neg + m)`), `pair_ranking_diagnostics`, and the read-only `PairIndex`
    (the frozen groups, the deterministic sampling, the item gathering);
  - `train_rqvae.py` — the frozen constants (the pair index path, `m = 0.2529585361480713`,
    `PAIR_GROUPS = 256`, the gradient budget 0.02, the 200-step ramp), the calibration
    `_calibrate_pair_weight` (the real batches, the real training loss, the per-module gradient
    norms, the angle and the largest admissible weight), the training step (the pair term, the ramp,
    the weight), the epoch metrics (`pair_loss`, `pair_ranking_accuracy`,
    `pair_margin_satisfaction_rate`, `pair_mean_distance_gap`) and the pair calibration record;
  - `model/global_constraint.py` — the unchanged #269/#271 evaluation module, copied byte-identical
    (sha256 `0f88dc5547e03b180d26118fff6d695275ac6629234e15247794771c958c6546`) from the #272
    archive, because the common-ruler J is an evaluation and the tree needs the same module the
    #269/#272 rounds measured with.
- The tests: `tests/test_known_answers.py` (the hand-computed cases: the hinge mean, the inverted
  order, the insufficient margin, the coincident pairs with a zero gradient, the fixed margin, the
  diagnostics), `tests/test_pair_rank.py` (the reference consistency, the hard q, the gradient
  routing, the curvature invariance, the centre invariance, the invalid indices, the deterministic
  sampling), `tests/test_reference_pair_rank.py` (the independent reference: the squared-norm
  expansion and the layer-cake hinge), `tests/test_mutation.py` + `MUTATIONS.md` (7 mutations, all
  caught by named assertions) and `tests/test_repository_root.py` (the path-resolution regression).
  **17 passed** after the floor admission.
- The per-module check with the real training loss: `results/research/issue274/MODULE_CHECKS.json` —
  the encoder and all three codebooks receive gradients and move after the step; the pair term is
  finite.

**Two engineering faults found by the real entry, both fixed with a regression test**

1. *The immutable source requires the assignment-floor patch*: the first dispatch
   (`stage2_2fb1e77fa52846e0a0fe3d1dfa57fc37`) passed the smoke but the launcher refused the
   formal run ("immutable source requires an assignment-floor patch; re-admit live source") — the
   frozen baseline carries no `assignment_floor.py`. Fixed with the public tool
   `tools/research/stage2/anticollapse/patch_tree.py::apply_floor` (the tree is patched in place and
   `verify_tree` passes: 8 codewords, all used and balanced).
2. *The repository-root resolution stopped at `results/`*: the second dispatch
   (`stage2_13395322c53543f09951c15adff495d6`) died with
   `FileNotFoundError: results/stage1_GeneEmbedding/output/sentence_t5.npy`, because the baseline's
   `_repository_root()` accepts the first parent that contains a `stage2_RQ-VAE` directory, and in the
   immutable run layout that parent is `results/`. The admitted trees require **both** markers
   (`stage2_RQ-VAE/` and `tools/research/stage2/contract.py`); the candidate now matches them, and
   `tests/test_repository_root.py` reproduces the failure shape.

A third refusal was self-inflicted and is recorded for completeness: copying
`model/global_constraint.py` into the tree while an earlier dispatch's worker was still alive changed
the admitted source digest between the admission and the snapshot ("admitted source digest changed
before snapshot"); the stale workers were stopped with `stage2_stop` and the tree was left untouched
during the next dispatch.

- The formal run is dispatched as `stage2_debbfe6a8e714c12aea4dbc7d2c236e7`.

### Round 2 (2026-11-28) — the encoder channel

- Round 1's full table: `round1_pass_table.json` — C1 PASS (0.6279 > 0.5386), C4 PASS (15.61), C6
  PASS (0.08197), C7 PASS (0.2261), C9 PASS (6.70e-05), C10 PASS; C2 FAIL (0.00485), C3 FAIL
  (7,929 < 10,918), C5 FAIL (2 of 3), C8 FAIL (the 1-hop 2.81, the 2-hop 0.443). The TIGER's own
  strata are 15.158 / 2.674, so round 1 improved the 1-hop 5.4x and the 2-hop 6.0x over the TIGER
  while staying above the #269-derived thresholds.
- The round-1 diagnosis: the capacity gap is the level-1 -> level-2 co-variation (the mean L2 per L1
  30.97 against the #265's 42.65); the codebooks' effective ranks are higher than the #265's; the
  gap tail flattens (the std 0.090 against the TIGER's 0.173) and the global q scale halves.
- The probe (`probe_encoder_routing.py`, `probe_encoder_share_sweep.json`): routing the pair term
  through the encoder lifts the L2 from 7,929 to 9,058 (the 2 % share) / 9,301 (5 %) / 9,625 (20 %)
  in 100 steps, at a reconstruction ratio of 1.43-1.47 in the probe's argmin-evaluation metric.
- Round-2 candidate: `stage2_RQ-VAE/TIGER_pairrank_enc_rq` (the round-1 tree plus
  `encoder_routed_sum`, the trainer's `_pair_terms` through the encoder, and the calibration
  covering the encoder and the three codebooks). Tests: **19 passed** (including two new
  encoder-routing tests and the M8 mutation "no encoder identity"); the module checks: the encoder
  and all three codebooks receive pair gradients and move.
- The formal round-2 run is dispatched below.

### Round 2 execution and the Stage3 queue (2026-11-28, owner instruction)

- The owner instructed: for the Stage2-completed SIDs without a Stage3 run, start the Stage3
  training (a separate authorisation, which the issue's text requires: "甚至达标也等待单独授权").
- The Stage3 queue's first item, `TIGER_samecentre_delta_rq` (#273), passed the export gate
  (archive binding sha256 `4090ae9f…`) and the SID quality gate (collision 0.1417 < 0.40) and is
  training (150 epochs, the frozen protocol verified). The #269 arm is queued next (its checkpoint
  carries the run identity).
- The #274 arms are **refused by the export gate**: their checkpoints are the frozen baseline's
  format and carry no `run_id` / `source_tree` / `provenance`, and the run directories carry no
  `export_manifest.json`; the binding cannot be fabricated, so the tree must save the identity at
  the checkpoint and the framework's export check must run. The round-1 tree's repair is in place
  (`model/provenance.py`, `RQVAE.provenance()`, the trainer's identity save and `sys.path` root);
  the round-2 tree gets the same repair after its run finishes. Both then need one Stage2 re-run
  before Stage3 can consume them.
- The round-2 (encoder channel) is training; at epoch 2460 the training-pair accuracy is 0.80 and
  the margin satisfaction 0.45 at a reconstruction of 6.86e-05.

## The round-3 candidate (the queue position)

`TIGER_pairrank_cover_rq` — the round-2 tree plus the identity save, the ARM_NAME
`TIGER_pairrank_cover_rq`, the pair index `pairs_index_cover1.npz` (38,871 groups: 31,095 train,
7,776 hold-out). 19 tests pass. The dispatch waits for the Stage3 of `TIGER_samecentre_delta_rq`
(the single GPU is occupied by the #273 run; the round-3 Stage2 then takes the same recipe:
3000 epochs, the same gradient budget, the same margin).

The stage3 queue audit: of the 12 queue entries with collision < 0.4, five are the real (non
`_conformance`) exports — `TIGER_samecentre_delta_rq` (0.1417, running), `TIGER_codebook_budget_j_rq`
(0.1337), `TIGER_active_j_rq` (0.1198), `TIGER_global_j_rq` (0.1518), `TIGER_prefix_curvature_gate_rq`
(0.1635) — and the last four carry no `export_manifest`, so the frozen export gate refuses them; they
need the same identity repair and one Stage2 re-run before their Stage3 can start.

### The round-3 run (interim)

`stage2_6fe2faf7fc204330b129dfa4900ae641`. The smoke PASSED: the entry check ran one epoch and
exported the identifier (levels 256/256/256, collision 0.055843; the behaviour-graph check is
NOT_APPLICABLE). The 3000-epoch run is at epoch ~728/3000 (the A100-80G host, the shared GPU beside
the #273 Stage3). The collision trajectory: 0.0714 (epoch 50) -> 0.0924 (150) -> 0.1018 (200) ->
0.1113 (250) -> 0.1287 (500) — the same monotone rise as rounds 1-2, so the SID-capacity ceiling is
not a sampling artifact. The pair metrics at the start of training: the ranking accuracy 0.615, the
margin satisfaction 0.027, the gap 0.029, the pair weight 7.5e-06.
