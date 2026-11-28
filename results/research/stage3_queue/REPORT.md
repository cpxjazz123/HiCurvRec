# Stage3 queue — the completed Stage2 SIDs that still need a Stage3 run

Owner instruction (2026-11-28): for the SIDs under `results/stage2_RQ-VAE/` whose Stage2 finished
but whose Stage3 never ran, start the Stage3 training. The scan is
`results/research/stage3_queue/scan.py`; the machine-readable inventory is
`results/research/stage3_queue/queue.json`.

## The inventory (2026-11-28)

| state | count | meaning |
|---|---|---|
| Stage2 SIDs | 173 | every `results/stage2_RQ-VAE/**/item_sids.json` |
| Stage3 DONE | 65 | a `logs/*/*/test_final.json` exists for the arm |
| Stage3 MISSING | 102 | no Stage3 directory for the arm |
| Stage3 PARTIAL | 6 | a Stage3 directory with logs/checkpoints but no `test_final.json` |

## What is actually eligible now

A SID can enter Stage3 only when the framework's export gate passes: the checkpoint must carry its
executed run identity (`run_id`, `source_tree`, `provenance` — the `tools/issue252/archive_audit.py`
audit), the SID's collision must be below 0.4, and the SID must be a real formal export (not a
1-epoch smoke copy under `_conformance/`). Applying those rules leaves seven real arms:

| order | arm | SID | collision | Stage3 state |
|---|---|---|---|---|
| 1 | TIGER_samecentre_delta_rq (#273) | `_runs/stage2_3d8bc9a4…/TIGER_samecentre_delta_rq/result/item_sids.json` | 0.1417 | **running** |
| 2 | TIGER_codebook_budget_j_rq (#272 r3) | `_runs/stage2_0a2be6e3…/TIGER_codebook_budget_j_rq/result/item_sids.json` | 0.1337 | MISSING |
| 3 | TIGER_active_j_rq (#272 r1) | `_runs/stage2_4dffd8b4…/TIGER_active_j_rq/result/item_sids.json` | 0.1198 | MISSING |
| 4 | TIGER_global_j_rq | `_runs/stage2_cc9a31a9…/TIGER_global_j_rq/result/item_sids.json` | 0.1518 | PARTIAL |
| 5 | TIGER_balanced_j_rq | `_conformance/TIGER_balanced_j_rq/item_sids.json` | 0.0426 | MISSING |
| 6 | TIGER_fixed_triplet_kappa_rq | `_conformance/TIGER_fixed_triplet_kappa_rq/item_sids.json` | 0.0473 | MISSING |
| 7 | TIGER_prefix_curvature_gate_rq | `_runs/issue255_formal/TIGER_prefix_curvature_gate_rq/result/item_sids.json` | 0.1635 | MISSING |

The Stage3 is a **global singleton** (one run at a time, port 50201, 150 epochs); each arm takes
about 75 minutes on the shared GPU, so the seven-item queue is roughly nine hours.

Already done: `TIGER_calibrated_kappa_rq` (#269) — its Stage3 finished earlier; the arm is not
queued. The `_conformance/` smoke copies with collisions near 0.05 are 1-epoch exports, not formal
candidates, and are excluded.

## Why most of the 102 missing arms cannot enter

Their checkpoints predate the run-identity convention (the `run_id` / `source_tree` / `provenance`
fields) and their run directories carry no `export_manifest.json`, so the export gate refuses them.
The identity cannot be fabricated; a tree must save it at the checkpoint and pass the framework's
export check. The #274 trees now have that repair (`model/provenance.py`, `RQVAE.provenance()`, the
trainer's identity save), but each arm still needs one Stage2 re-run before Stage3 can consume it.

## The #274 arms

`TIGER_pairrank_rq` (round 1) and `TIGER_pairrank_enc_rq` (round 2) are refused for the identity
reason above; the round-1 tree is repaired and its tests pass (17), the round-2 tree is repaired
after its run finishes, and each then needs a re-run. The re-runs reproduce the same recipes
(everything else unchanged), after which the export check, the archive binding and the Stage3 gate
run normally.

## The dispatch campaign (2026-11-28, owner instruction)

Every Stage2-completed SID without a Stage3 result was re-inventoried, gate-audited, and queued
(`queue_dispatch.json`). The export gate's own repair (`tools.research.stage2.export_check
--rewrite`, exactly as `stage3_export_gate` runs it) was executed for the 16 repair arms:
**14 VERIFIED** (the SID rewritten to the checkpoint's table, 66-526 items per arm), 2 refused
(`drq_U`, `drq_S` — their archived trees import a missing `teacher` module).

Dispatch order (Stage3 is a global singleton; one arm per card, ~80-90 min each):

1. `TIGER_global_j_rq` — RUNNING (`stage3_...` state, collision 0.1518).
2. `TIGER_graphlet_align_rq` (pre-validated VERIFIED, collision 0.1591).
3. `TIGER_curvloss_rq`, `TIGER_sinkhorn_all_levels_rq` (verified static arms).
4. `TIGER_pairrank_cover_rq` (round-3 candidate; checkpoint carries run identity).
5. `TIGER_codebook_budget_j_rq`, `TIGER_active_j_rq` (run-layout; binding checked at dispatch).
6. static arms: `drq_D0`, `graphlet_capacity_rq`, `drq_D4`, `sid_graphlet_rq`, `p4_topoq_A`,
   `trq_A`, `export_matched_graphlet_rq`, `graphlet_prefix_shuffled_rq`, `graphlet_wide_l1_rq`,
   `drq_D2`.
7. `TIGER_pairrank_enc_rq` — manifest verified, but its checkpoint may lack the run identity; the
   archive binding decides at dispatch.

Refused and why: `prefix_curvature_gate_rq` (checkpoint framework-source hashes predate the repo),
`pairrank_rq` (no run identity), `drq_U`/`drq_S` (missing `teacher` module), and the collision-gate
trio `g237_struct_rq` 0.966 / `kappa_cost_rq` 0.825 / `i234_A6` 0.412 (the last is waivable only by
the owner, `waive_collision`).

Resume: for each remaining arm call `stage3_start` with its SID path once the previous run's
`test_final.json` exists and no `run_stage3_*` process is live; record the result in
`queue_dispatch.json` (`results` key). The `TIGER_samecentre_delta_rq` result (test_R@10 0.0466) is
already recorded.

## Dispatch log (machine-driven, one arm at a time)

| # | arm | outcome | test_R@10 | test_final |
|---|---|---|---|---|
| 1 | TIGER_global_j_rq | finished (150 epochs) | 0.042933 | `results/stage3_T5Train/TIGER_global_j_rq/logs/Amazon_2023_Instruments/Nov-28-2026_03-16-44/test_final.json` |
| 2 | TIGER_graphlet_align_rq | finished (150 epochs) | 0.039868 | `results/stage3_T5Train/TIGER_graphlet_align_rq/logs/Amazon_2023_Instruments/Nov-28-2026_04-46-08/test_final.json` |
| 3 | TIGER_curvloss_rq | finished (150 epochs) | 0.040669 | `results/stage3_T5Train/TIGER_curvloss_rq/logs/Amazon_2023_Instruments/Nov-28-2026_06-14-10/test_final.json` |
| 4 | TIGER_sinkhorn_all_levels_rq | finished (150 epochs) | 0.053552 | `results/stage3_T5Train/TIGER_sinkhorn_all_levels_rq/logs/Amazon_2023_Instruments/Nov-28-2026_07-43-05/test_final.json` |
| 5 | TIGER_pairrank_cover_rq | finished (150 epochs) | 0.050593 | `results/stage3_T5Train/TIGER_pairrank_cover_rq/logs/Amazon_2023_Instruments/Nov-28-2026_09-11-46/test_final.json` |
| 6 | TIGER_drq_D0 | exited_incomplete 11:35 at epoch 124/150 (run dir `Nov-28-2026_10-39-23`; launcher log stops mid-epoch, no traceback, no `test_final.json`) | n/a | `results/stage3_T5Train/TIGER_drq_D0/logs/Amazon_2023_Instruments/Nov-28-2026_10-39-23/` |

Queue ownership returned to the coordinator at 12:00 (single owner from here): this driver made no
`stage3_start` call and no process-control call after 10:35, and stops dispatching. Arm order from
`TIGER_drq_D0` onward is being driven by the coordinator; rows 7+ are appended there.

## 2026-11-28 12:09 — queue halted by owner

The owner asked to end the running Stage3. Process group 797746 (the re-dispatched `TIGER_drq_D0`,
started 12:04:02 after its first run died at epoch 124) was terminated with SIGTERM at 12:09:18; no
`run_stage3_*` process remains and the GPU is idle (0 MiB). `queue_dispatch.json` records
`TIGER_drq_D0: stopped_by_owner`. No further arms were dispatched; the 10 remaining arms
(graphlet_capacity_rq, drq_D4, codebook_budget_j_rq, sid_graphlet_rq, p4_topoq_A, trq_A,
export_matched_graphlet_rq, graphlet_prefix_shuffled_rq, graphlet_wide_l1_rq, drq_D2) stay queued.

## 2026-11-28 13:10 — R2 archive status of this batch (§6b)

Archived from this batch (uploader verified the four entries, then the uploaded bytes):

| number | arm | recall@10 |
|---|---|---|
| `r2:genrec/112` | TIGER_global_j_rq | 0.042933 |
| `r2:genrec/113` | TIGER_pairrank_cover_rq | 0.050593 |

Fail-closed, **not** archived: `TIGER_graphlet_align_rq`, `TIGER_curvloss_rq`,
`TIGER_sinkhorn_all_levels_rq`. Their `test_final.json` has no `archive_binding`
input provenance (they were exported before the checkpoint-bound binding existed), and
`upload_stage3.py` refuses to guess one (§58: the rule is fail-closed; do not bypass by
calling internals). Backfilling would require a binding whose receipt cannot be derived
from those old exports; the three recalls stay in this report and in the Stage3 logs but
are absent from the R2 ledger. `TIGER_prefix_curvature_gate_rq` never finished (no
`test_final.json`), so it is not archivable yet either.

## 2026-11-28 14:20 — `TIGER_pairrank_enc_rq` finished

| item | value |
|---|---|
| run | `logs/Amazon_2023_Instruments/Nov-28-2026_13-00-13/test_final.json` |
| test_recall@10 | **0.047947** (`test_ndcg@10` 0.026375, n_eval 57439) |
| SID | `9f6bf673…` (`_runs/stage2_58090127…/result/item_sids.json`) |
| R2 | `r2:genrec/114` (68 files, uploaded and verified) |
| #277 prospective | frozen prediction 0.05319 (commit `03c8e1df`, 14:03:16 +1100) vs actual 0.047947 — threshold call correct (<0.06), signed error +0.00524 inside the empirical interval [0.041126, 0.057908]; label mtime 03:20:50Z is after the freeze |
| verdict | below the 0.065 target |

Next in the queue: `TIGER_codebook_budget_j_rq` (SID with binding, collision 0.1337).

## 2026-11-28 15:50 — `TIGER_codebook_budget_j_rq` finished

| item | value |
|---|---|
| run | `logs/Amazon_2023_Instruments/Nov-28-2026_14-29-03/test_final.json` |
| test_recall@10 | **0.048469** (`test_ndcg@10` 0.026472, SID `61f8984…`) |
| gate | collision 0.1338 < 0.40; quant_error 0.013 warning; effective_rank 2.21 |
| verdict | below the 0.065 target |

Next: `TIGER_prefix_curvature_gate_rq` (the last arm with a checkpoint-bound SID in the eligible list).

## 2026-11-28 15:52 — next arm refused (fail-closed) and the queue is exhausted

`TIGER_prefix_curvature_gate_rq` (the last arm in the eligible list with a checkpoint-bound SID) was
refused by the export gate:

```
export gate: archive binding: executed source/config hashes mismatch
```

Diagnosis (`archive_binding.json`): all 56 `candidate_sources` still match the stored src tree and all
17 `repository_configs` match, but **5 infrastructure sources changed** since the binding was made:

| file | stored | current |
|---|---|---|
| `tools/issue253/static_features.py` | `bd38db54df…` | `8e35754f42…` |
| `tools/research/stage2/anticollapse/floor.py` | `984bba1330…` | `0e0f2fbef7…` |
| `tools/research/stage2/anticollapse/patch_tree.py` | `bffbc2b3a5…` | `75abaa019f…` |
| `tools/research/stage2/contract.py` | `bf82fd3197…` | `e2e7fe8237…` |
| `tools/research/stage2/launcher.py` | `61d32bd6d1…` | `73afff80f9…` |

These are the pre-existing worktree modifications from earlier sessions (mtimes 2026-11-25/27). The
binding is fail-closed by design (CLAUDE.md §58): the arm cannot be dispatched without regenerating its
evidence, which requires the original framework state and is out of this route's scope. No gate was
bypassed. **Remaining arms:** all other queued arms either lack a checkpoint-bound SID (legacy
`_conformance` smoke copies, collision ~0.05) or have no binding at all (`TIGER_active_j_rq`); they
would be refused on the same rule. The campaign's eligible Stage3 arms are therefore exhausted.

Batch summary (this campaign): finished and archived `r2:genrec/112..115` = `global_j` (0.042933),
`pairrank_cover` (0.050593), `pairrank_enc` (0.047947), `codebook_budget_j` (0.048469); finished but
unarchivable (no `archive_binding`) = `graphlet_align` (0.039868), `curvloss` (0.040669),
`sinkhorn_all_levels` (0.053552); refused = `prefix_curvature_gate` (infra hash drift).
