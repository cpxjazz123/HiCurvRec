# Issue 257 — dependency-aware validation plan

## The graph

| layer | depends on | must run every round? | why |
|---|---|---|---|
| static data and named graph | `results/stage0_build_parquet/*`, `assets/manifest.json`, `results/research/behavior_graph/` | no (reuse) | content hashes are stable; a change is a data change and invalidates everything downstream |
| shared GNN / geometry / quantisation (`tools/graph_infra/*`, `tools/issue252/*`) | their import closure | no (reuse) | a change here invalidates every candidate that reaches it — the fingerprint follows the import closure, not a path list |
| public launch / export / eval (`tools/research/stage2/*`, `tools/issue253/*`) | import closure + ports/devices | no (reuse) | framework-owned; its own tests are the receipt |
| candidate maths and gradients | the candidate tree's `model/*`, its `tests/*`, the frozen assets | **yes** | the mechanism is the only thing that changed; its reference/gradient/mutation units must run |
| real entry, four-card DDP, performance | the candidate tree + the public launcher | **once per candidate**, not per edit | an entry/DDP failure is a public or wiring fault; re-running it after an unrelated analysis edit wastes GPUs |
| final run identity (SID ↔ checkpoint ↔ Stage3) | the run's own artefacts | yes, read-only | the replay recomputes it from bytes |

## Fingerprint

`tools/issue257/depgraph.py` builds, for each validation unit, the SHA-256 of a canonical JSON of:

* **code**: every repo-local module in the unit's transitive import closure (AST-parsed, including
  relative imports inside a candidate tree via `roots=`), each with its file hash;
* **assets / configs**: declared files with their hashes;
* **tests**: the test file(s) themselves, so editing a test invalidates its own receipt;
* **runtime**: interpreter version, torch version and the sorted set of visible device names.

A PASS receipt is stored under `(test id, fingerprint)`. Reuse is allowed only when the fingerprint
matches *and* the receipt says PASS.

## Refusals (measured, not asserted)

`tests/test_depgraph.py` (14 repository tests, 6 in this file) proves:

| scenario | expected | measured |
|---|---|---|
| nothing changed | reuse | `reuse: [pipeline_units, candidate_suite, stage3_protocol]` |
| analysis-only edit (`tools/issue257/analysis.py`) | rerun only the analysis unit | `rerun: [pipeline_units]` |
| shared operator edit (`tools/graph_infra/ego_gnn.py`) | rerun every consumer | `rerun: [candidate_suite]` |
| mechanism edit (`model/mechanism.py`) | rerun the mechanism unit | `rerun: [candidate_suite]` |
| unrelated document edit (`RUN_LOG.md`) | rerun nothing | `rerun: []` |
| a FAIL receipt exists | never reuse | `reason: previous status FAIL` |
| a new import appears in a dependency | fingerprint changes | digest differs |
| mutation: the closure is not followed | the shared-operator edit is missed | the mutant test shows the missing edge |

Raw record: `results/research/issue257/VALIDATION_PLAN_DEMO.json`.

## Cost of the reusable path (measured this round)

| activity | wall time | GPU |
|---|---|---|
| read-only replay of one finished run (SID chain, 57,439-row reduction, 5 artefact hashes) | 1.6 s cold, 1.7 s warm | 0 |
| pipeline unit suite (14 tests) | 0.4 s | 0 |
| archive verification (ledger + four pieces) | 0.1 s | 0 |

The formal Stage2/Stage3 units are *not* re-run by this plan when their receipts match: that is the
intended saving, and the reason the three change scenarios above matter more than the absolute
seconds. The full-iteration ≤2 h budget from the issue cannot be measured without a fresh complete
round; only these verifiable stage costs are claimed.
