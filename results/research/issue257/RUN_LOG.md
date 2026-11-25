# Issue 257 — frozen acceptance list and run log

Issue body sha256 (recorded at freeze time):
`f4509b74f5d0667929f47fe800bb8c21f23e5573305900bd975b796b30f4d5ad`
(re-verify with `gh issue view 257 --json body -q .body | tr -d '\n' | sha256sum`;
if it differs, re-freeze this list before judging anything).

Training: **not required by this issue** — its acceptance is a reusable pipeline plus read-only
replays of already finished runs and a controlled short four-card entry, not a new 3000/150 run
(issue §4 and the final status paragraph forbid new formal research training).

## Accepted criteria (verbatim intent, mechanically checkable)

| id | requirement | evidence |
|---|---|---|
| C1 | audit #251–#256 and build the fault ledger with class, repro, fix, scope | `FAULT_LEDGER.md` (12 rows, each with a repository evidence path) |
| C2 | timing of design/implement/diagnose/test/GPU-admit/train/eval/archive, NA where unrecoverable | `TIMING.json` (measured this round) + `RUN_LOG.md` |
| C3 | public boundary: candidate owns mechanism + tests; framework owns entry, loader/DDP, GPU/ports, logs, assets, export, eval, archive, timing, state | `tools/iteration/{identity,timeline,registry,report,archive}.py`; `tools/research/stage2/*` reused |
| C4 | candidate generated from an explicit parent; no generated adapters/checkpoints in the tree; immutable snapshots | `rounds/../` trees + `run_workspace.freeze_run`; `archive.py` verifies the binding |
| C5 | dependency-aware validation: content-hash fingerprints over the import closure, assets, config and runtime; reuse PASS only on identical dependencies | `tools/iteration/depgraph.py` + `tests/test_depgraph.py` (4 scenarios, 1 mutation) |
| C6 | three real change scenarios: mechanism math, analysis-only, shared operator; full-vs-incremental comparison | `tests/test_depgraph.py` cases; `VALIDATION_PLAN.md` |
| C7 | standard round workflow state machine with resumable, idempotent states | `tools/iteration/registry.py` + `tests/test_known_answers.py::test_registry_transitions_and_refusals` |
| C8 | automated analysis parameterised by run identity (no issue/arm hardcoding) | `tools/iteration/analysis.py`; replay of R01/R04 |
| C9 | read-only replay of ≥2 immutable historical runs reproducing their numbers | `results/research/issue257/replay/r01/REPLAY.json` (PASS, 13 checks), `.../r04/REPLAY.json` (PASS, 13 checks) |
| C10 | controlled real four-card entry through the pipeline | `results/research/issue257/short_candidate_result.json` |
| C11 | four-piece archive binding verified without re-uploading | `tools/iteration/archive.py`; `TIMING.json` → `archive_check.status = PASS` |
| C12 | docs and the issue skill updated to the real behaviour; old entry points kept compatible | CLAUDE.md §15a; `.agent/skills/complete-open-issue/SKILL.md` |

## Not run (recorded, not hidden)

- No formal 3000/150 training this round: the issue forbids it and the replays need none.
- The full-iteration ≤2 h budget cannot be *measured* without a fresh complete round (issue §6);
  only the verifiable stage costs are measured here (`TIMING.json`) and the rest is a projection,
  labelled as such.

## Log (UTC)

- 2026-11-25 — acceptance frozen; training-need decision recorded (no formal training).
- 2026-11-25 — pipeline package written; 14 tests green (`tools/iteration/tests`).
- 2026-11-25 — replay R01/R04 PASS (13 checks each, independent reduction over all 57,439 rows).
- 2026-11-25 — archive verification PASS for R04; replay cold 1.6 s / warm 1.7 s.
- 2026-11-25 — first controlled attempt (`validation_entry.py` of the archived tree) failed on a latent shape assertion of that tree's own observer (recorded in `TIMING.json.failed_attempt`); no claim was made from it.
- 2026-11-25 — canonical controlled entry passed: `checks.run_all(archived R04 tree)` -> entry PASS, behaviour_graph PASS in 47.27 s (`controlled_entry.json`).
- 2026-11-25 — real short four-card training entry passed: `steady_profile_entry.py` via `launcher.launch`, exit 0, four ranks PROFILED, 67.17 s (`controlled_four_card.json`); the run directory was moved out of the immutable R04 round into the pipeline directory.
