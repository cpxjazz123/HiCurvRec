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
