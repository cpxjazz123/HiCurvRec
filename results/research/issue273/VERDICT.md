# Issue #273 verdict — round 1 (same-centre difference protection)

## Run

`stage2_3d8bc9a4ef5340e690b25efa20ad2c8d`, arm `TIGER_samecentre_delta_rq`, the real entry
`stage2_RQ-VAE/TIGER_samecentre_delta_rq/train_rqvae.py`, 3000 epochs, seed 42, the single visible
GPU (the frozen recipe's single-card branch), 1303.7 s of training, 0.45 s per epoch. The machine
admission before the dispatch: `entry` PASS (one epoch runs and exports the identifier, levels
256/256/256, collision 0.048278), `behavior_graph` NOT_APPLICABLE (the tree uses no behaviour graph).

Artefacts in the immutable run directory
`results/stage2_RQ-VAE/_runs/stage2_3d8bc9a4ef5340e690b25efa20ad2c8d/TIGER_samecentre_delta_rq/result/`:

| artefact | sha256 |
|---|---|
| `item_sids.json` | `f34bde311958f92b6ab1736bbfdfde8aea212e409dc95c8a8f45942d850b405b` |
| `rqvae_best.pth` | `a7f17036fdee521a680b927b59db36e1a39cc38fe65f5844cc430562347b9e29` |
| `logs/training_metrics.jsonl`, `logs/train_migrated.log` | the run's own records |

## The mechanism as implemented

`model/delta.py::same_centre_difference_loss` — the centre-balanced mean over the `m(m-1)/2`
same-centre pairs of `((p_t - p_s) - (y_t - y_s))^2`, divided by the centre's valid triplet count.
`lambda_delta` is frozen by the **per-module** gradient-norm budget against the weighted pointwise
term (`rho = 0.02`), and the term ramps in over 200 optimizer steps. The frozen `kappa_t` cache
stays the #269 candidate's read-only table; the #272 global distance term is not inherited.

## The calibration frozen on the real first batch

| quantity | value |
|---|---|
| `lambda_point` (inherited from #269) | 5.883528956474038e-06 |
| `lambda_delta` | 5.95565495586925e-07 |
| the realised global gradient share | 0.013784 (cap 0.02) |
| the pointwise / delta gradient cosine | 0.104792 (the two terms do not fight) |
| the weighted pointwise gradient norm | 6.295117e-05 |
| the delta gradient norm | 1.457005 |
| the calibration batch | 4096 items |

## Measured criteria

| criterion | value | threshold | verdict |
|---|---|---|---|
| C12 L2 distinct prefixes | 5,966 | >= 10,918 | **FAIL** |
| C15 reconstruction | 9.792938e-05 | <= 8.589e-05 | **FAIL** |
| C1, C2, C3, C4, C5, C6, C7, C9, C10, C11, C13 | NOT_RUN | -- | the owner terminated the evaluation |
| collision (recorded only) | 0.141904 (the first three codes); 0.0 for the full four-token SID | -- | recorded |
| L3 distinct triples | 21,098 | -- | recorded |
| final loss / recon | 1.375485e-04 / 9.792938e-05 | -- | recorded |

The frozen evaluation (`results/research/issue273/evaluate_candidate.py`: the common-ruler J from
the candidate's own `model/global_constraint.py`, the local and within-centre statistics from
`within_centre_metrics.py`, the SID structure from the frozen #271 collector) was **stopped by the
owner** before it produced the table. The two measured criteria were recomputed directly from the
exported SID and the final training record, so no number in this verdict is estimated or carried
over from another arm. The machine-readable record is
`results/research/issue273/round1_sid_structure.json`.

**Acceptance: not met.** Both criteria that were measured fail; the remaining eleven were not
computed. The issue is closed by the owner's instruction (2026-11-27/28) as an **incomplete
negative result**, not as a PASS. Stage3 was forbidden by the issue in every outcome and was not
run.

## What the round establishes

- The mechanism is implementable and its budget is real: the delta term reaches the encoder and the
  three codebooks, the realised gradient share stays at 1.38 % of the weighted pointwise gradient
  (below the 2 % cap), and the two gradients are nearly orthogonal (cosine 0.105), so the term
  does not fight the inherited pointwise supervision.
- The cost is visible in the two criteria that were measured: the L2 prefix capacity fell to 5,966
  (the #269 parent 6,848, the #272 round-3 7,463, the threshold 10,918) and the reconstruction rose
  to 9.7929e-05 (the threshold 8.589e-05, which the #269 met). In this configuration the
  same-centre difference protection **costs** prefix capacity and reconstruction accuracy.
- The #273-specific claim — that the signed within-centre differences (C6) are what the #272
  round-3 left missing — **remains untested**: C6 and C7 are exactly the criteria the terminated
  evaluation would have produced. The one-line measurement that would settle it (the #269 baseline
  `difference_rmse` 2.002453 against the #273 run's own `within_centre_metrics.py` output) is
  still available from the frozen artefacts.
- The first dispatch was refused by the smoke (`stage2_98834b02a3e542a7a48492034ccf76e4`): the
  calibration ran under `torch.no_grad()`, so the loss had no graph and the gradient budget could
  not be measured. The bug is fixed and covered by a regression test
  (`tests/test_samecentre_delta.py::test_delta_calibration_measures_the_gradient_budget_on_a_real_batch`).
  The candidate suite passes: **138 passed**.

## Infra

`INFRA_SNAPSHOT_END.json` equals `INFRA_SNAPSHOT_START.json` on the recorded path list
(added 0, removed 0, modified 0).

## The CLAUDE.md §63 layered metric record

**Not produced.** The frozen writer `tools/issue252/layer_metrics.py::record_layers` needs the run's
`archive_binding.json` next to the SID; the dispatched run wrote only `workspace.json` in the run
directory, so the record is marked **missing** rather than filled with zeros. The wrapper
`results/research/issue273/write_layer_metrics.py` is in place for the next dispatched run that
carries the binding.
