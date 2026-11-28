# Issue #276 — pre-registered analysis contract (frozen before fitting)

Written 2026-11-28 13:1x, before FOLDS.json or any model fit. Changes after this point are allowed
only as documented amendments with a date and a reason; the outer evaluation is never re-run after
seeing its result.

## Unit, identity and near-duplicate groups
* Unit = one **discrete SID**: the mapping `item -> (c1, c2, c3)` (the ordered first three codes).
  Two files with the same mapping are the **same** SID (duplicate exports); files whose mappings
  differ by more than the collision suffix are different SIDs.
* Identity evidence, strongest first: (L1) R2-ledger sha256 of `item_sids.json`; (L2) computed L3
  collision equal to a stored `metrics_corrected[arm].collision`; (L3) era + directory name only.
  L3 rows are kept but flagged; the primary queue uses L1/L2 only.
* Label per SID: `recall@10` mean over records of that SID in the protocol-consistent slice
  (majority `world_size` when known, else all), with `n_records`, `spread=max-min`, and the record
  paths stored. One label per SID; repeated Stage3 runs of the same SID are never separate samples.

## Queues (separate, identity-labelled; increments only on the paired intersection)
* **Q_SID** — SID-only features from the frozen evaluator (`tools/issue252/layer_metrics.py`,
  computed on `results/stage0_build_parquet/{train,valid}.parquet` by the same `lsm` calls used in
  #275). Main 8 features, frozen:
  `l1_pos_hit, l1_neg_hit, l1_hard_er, l2_pos_hit, l2_neg_hit, l2_distinct, l3_collision, l3_refine_bits`.
  Secondary SID metrics (prefix entropy, codebook occupancy, distinct-l3 ratio, max share) are
  descriptive/robustness only, never part of the main selection.
* **Q_VEC** — Q_SID ∩ checkpoint-bound vector stats: `PR(z)`, `PR(hard q)`, `quant_error`,
  `recon_mse`, `recon_cos` (unified full-item forward, Nov-19 Linear-bias forward + the real codebook
  of the same checkpoint). Stored `vector_stats_all.json` values (`pr`,`quant_err`,`sep`,
  `distinct_ratio`) are secondary evidence with their own provenance; `PR(z)` recomputed here must
  agree with the stored `pr` within 1e-6 to be used.
  The #275 `recon` (final-epoch training batch loss) is **not** a decoder MSE and is analysed only to
  document the protocol mismatch; it never enters Q_VEC.
* **Q_GEO** — Q_SID ∩ `results/research/issue271/assets/geometry/*.json` (mean_abs_error, spearman,
  sign_accuracy; fixed scale/legal mask from those assets). Coverage is reported; unidentifiable.

## Folds (frozen rule, applied by `folds.py`, hash recorded in `FOLDS.json`)
* 5 folds over SID groups. Groups are assigned inside recall bands `<0.045`, `[0.045, 0.06]`, `>0.06`
  round-robin after a seeded shuffle (seed 42) so every band appears in every fold when it has ≥5
  groups; a band with fewer groups is still assigned by the same rule and the imbalance is reported.
* Every SID appears in exactly one test fold (OOF). Inner selection/standardisation/hyperparameters
  use the same SID-group folds inside the training part.
* No family variable exists anywhere in the model, selection, or splits (families are audit metadata).

## Models and controls (same folds, same features)
`fold_mean`, `fold_median`, `best_single` (train-fold Spearman), `ridge` (inner SID-group CV over
alpha ∈ {0.01,0.1,1,10}), `prior_l1pos` (1-D linear map on `l1_pos_hit`), `ridge6_275` (the #275
six-feature ridge refit inside each training fold), `pysr` (frozen config in `PYSR.md`:
operators `+,-,*,/ (safe),square`, `maxsize=10`, `maxdepth=4`, `niterations=100`, `populations=8`,
seed 42, `nested_constraints` default; constants/variables bounded; inner validation selects).

## Metrics and success criteria (pre-registered)
* MAE, RMSE, OOF Spearman, Kendall, reversal fraction; >0.06 precision/recall/confusion when at least
  one positive exists in the OOF table (otherwise `NOT_EVALUABLE`); direction vs the frozen TIGER
  reference recall (0.0469, `TIGER_RQ-VAE`); SID-group bootstrap (2000 reps, seed 42) CI for the
  paired MAE difference.
* **Usable formula**: OOF MAE strictly below the strongest control **and** the paired CI excludes 0
  **and** OOF Spearman ≥ the best single feature's OOF Spearman. Otherwise the result is negative.
* 5 folds, no alternative fold seeds, no post-hoc feature changes; a failed criterion is a result,
  not a bug.

## Amendment 1 — 2026-11-28 (before any model fit)

The stored `sid_utilization/vector_stats_all.json` series (`pr`, `quant_err`, `sep`) was computed with
a **bias-less** encoder forward (verified: for `r2:genrec/48` the stored `pr` 4.248628616 equals the
no-bias forward exactly, while the corrected Linear-bias forward gives PR(z) 6.729273). The issue
requires the Nov-19 corrected forward. Therefore:
* Q_VEC uses the recomputed series in `results/research/issue276/vector_forward.json`
  (`pr_z`, `pr_q`, `quant_err`, `recon_mse`, `recon_cos`, all with biases);
* stored `pr`/`quant_err` remain a secondary, flagged series (they agree with the corrected values
  only when the checkpoint's Linear biases are zero — e.g. `crowd_couser_w3` 5.993339 == 5.9933386);
* the #275 `recon` (final-epoch training batch loss) is not used anywhere in Q_VEC.
