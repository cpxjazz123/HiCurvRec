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
