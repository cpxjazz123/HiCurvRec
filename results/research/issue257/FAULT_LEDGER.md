# Issue 257 — fault ledger of #251–#256 (evidence-backed, one row per real failure)

Classification: **public-infra** (framework owns it, every candidate pays it), **candidate**
(the mechanism's own maths), **numerical** (float/precision/order), **data**, **external**
(permissions/resources), **validator** (the checker itself was wrong). Each row names the fix,
its regression and the evidence that already exists in the repository; no row is a guess.

| # | Class | Symptom | Reproduction | Root cause | Fix / regression | Evidence |
|---|---|---|---|---|---|---|
| 1 | public-infra | DataLoader produced different orders with the same seed | `tests/test_execution.py` order assertion | loader had `sampler=None` and rebuilt the generator per epoch (`MeasuredLoader.generator`) | framework-owned sampler and generator; the reproducibility regression fails before the fix | `results/research/issue256/rounds/r02/BATCH_TRACE_RECONCILIATION.json` |
| 2 | public-infra | two runs of the same code disagreed on the static CUDA means | fresh-cache equality regression on four cards | static features were averaged with a non-deterministic kernel and a per-call cache | deterministic kernel + content-hashed cache; equality regression on all four ranks | `rounds/r02/STATIC_MEANS` regressions; `rounds/r03` `test_static_means.py` |
| 3 | numerical | `high` precision baseline depended on batch composition, `highest` did not | companion-batch gate on the real model | float32 accumulation order changed with the companion batch | production path moved to the `highest` policy; gate regression reintroduced on a fixed-parameter control | `rounds/r02/BATCH_PRECISION_DIAGNOSIS.json`, `BATCH_PRECISION_REPAIR.md` |
| 4 | public-infra | assignment floor died after the first epoch and left codes unguarded | floor lifetime test on the installed tree | the floor was installed on the candidate tree and then replaced by the framework snapshot | framework owns the floor install (`anticollapse/patch_tree.py`); lifetime regression | `rounds/r02/FLOOR_LIFETIME_*` |
| 5 | validator | auxiliary balanced-mode replay rejected a legal run | native 2×2 counterexample (`balanced=False/True`) | the verifier replayed with one fixed mode instead of the population's captured mode | verifier uses each captured population mode; counterexample regression | `rounds/r02/AUXILIARY_MODE_RED_RECEIPT.json` |
| 6 | public-infra | a generated `execution_entry.py` collided with an inherited one | real `freeze_run/load_run` ownership regression | the entry was written into live source instead of the immutable run workspace | framework-owned generation with a single owner; obsolete adapter removed from live source | `rounds/r02/FORMAL_PREWORKER_REFUSAL.json`, `.../verification_generations/pre_generated_entry/` |
| 7 | public-infra | profile collector matched a subset fingerprint and accepted a stale generation | profile identity regression | collector compared a subset of hashes instead of the full role source identity | exact before/after equality over the full closure; three red regressions | `rounds/r02/PROFILE_IDENTITY_FAILURE.json`, `PROFILE_IDENTITY_GREEN.log` |
| 8 | public-infra | MCP returned stale behaviour after a tool change | `stage2_status` after an edit | the stdio server imports code once | SIGTERM the server child, harness respawns; a real tool call is required before trusting it | CLAUDE.md §52–§58 |
| 9 | public-infra | archive/report code held absolute paths and round-specific constants | import of the module from another checkout | paths and ids were literals | repo root derived by marker; run identity resolved from artefacts (`tools/issue257/identity.py`) | `rounds/r03` `test_source_inventory.py`; CLAUDE.md §19b0 |
| 10 | public-infra | SID export not reproduced by the saved checkpoint | re-export audit | the export read live state instead of the checkpoint-bound arrays | export bound to the checkpoint; `same_checkpoint_all_items_exact` | `rounds/r03/FINAL_STAGE2.json`, `rounds/r04/FINAL_STAGE2.json` |
| 11 | validator | the layer diagnostic asserted R03 semantics on an R04 tree | `layer_validation_entry` on the level-3 tree | the check encoded "the role level must not reach the gate", true only for level 1 | exact mixture identity per level + gate row count; the prefix chain is recorded, not excluded | `rounds/r04` admission commits `446bf3d6` |
| 12 | analysis | the entropy reduction counted the export collision counter as a semantic code | recomputation of `H(L3\|L1L2)` | the fourth column is a counter, not a code | exact three-code reduction; correction recorded in `rounds/r03/ROUND_RESULT.json` | commits `6b401ea1`, `1229b1d4`; `tools/issue257/analysis.py::sid_chain` |

## What the pipeline generalises from these rows

* rows 1–10 are **public**: they now live behind `launcher`, `contract`, `run_workspace`,
  `anticollapse`, `export_policy` and the dependency graph, so a new candidate inherits the fixes
  and their regressions instead of re-deriving them;
* row 11 shows why a validation unit must be *semantic*, not copied: the dependency fingerprint
  would have reused the R03 receipt for an unchanged checker even though the tree changed meaning;
* row 12 shows why analysis must be computed from the artefacts with full precision and compared
  against stored values, not re-rounded.
