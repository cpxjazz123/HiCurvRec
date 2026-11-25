# Issue #259 audit of the existing iteration Infra

## 1. Existing public interfaces (read-only)

| interface | current behaviour | gap relative to #259 |
|---|---|---|
| `tools/iteration/depgraph.py` | hashes tests/code/assets/configs/tests, static absolute imports, one simple `from . import x` case; `Spec.fingerprint`, `Cache.record/lookup`, `plan` | no multi-level relative imports, package `__init__` chain, dynamic imports, directory/extra inputs; configs parsed as Python; missing files silently skipped; deletions not detected; env not scoped; no evidence validation |
| `tools/iteration/identity.py` | artefact-hash identity for a run | no validation-metric reader |
| `tools/iteration/analysis.py` | SID chain, role conditioning, layer record, Stage3 metrics | validation recall absent; no identity-bound saved-artifact reader |
| `tools/iteration/replay.py` | recomputes SID chain, predictions, artefact hashes | already reusable; no change needed |
| `tools/iteration/report.py` | writes the standard report set | no run/reuse counts or per-unit durations |
| `tools/iteration/timeline.py` | interval-merged phase timing | no execution-count summary |
| `tools/iteration/archive.py` | R2 ledger verification | out of scope (#259 forbids upload changes) |
| `tools/iteration/registry.py` | round state machine | out of scope |
| `tools/iteration/tests/` | known-answer and replay tests | no dependency/evidence/execution/integrity coverage |

## 2. Actual call chain in #258

1. research agent → inline python (py3.9) → `depgraph.Spec(...).fingerprint()`, `plan(specs, cache)`, `Cache.record`;
2. tests/diagnostics → raw `pytest`/`launcher.launch` with manual shell redirection (this is what failed once);
3. training → `freeze_run` + `launcher.launch` (logs built in) and Stage3 MCP;
4. analysis → `analysis.analyse`, `replay.replay`, `report.write_all`, `archive.verify`;
5. receipts under `results/research/issue258/receipts/` (7 files); logs under `.../issue258/checks/`.

## 3. Receipt inventory (#258)

| unit | evidence fields | reusable under the new contract |
|---|---|---|
| `candidate_suite` (2 files) | command, exit_code, output; one lacks `output_sha256` | no (missing hash) |
| `admission_entry` | command, exit_code, output, output_sha256 | yes |
| `behavior_graph` | command, detail, exit_code, output, output_sha256 | yes |
| `module_gradients` | command, exit_code, output, output_sha256 | yes |
| `performance` | command, exit_code, output, output_sha256, mean | yes |
| `pipeline_units` | command, exit_code, output (no hash) | no |

This is the intended fail-closed behaviour: legacy receipts missing fields are invalid, not back-filled.

## 4. Gaps found (implementation targets)

1. Relative imports only cover `from . import x` with `module is None`; `from ..pkg.mod`, multi-level and package
   `__init__` chains are missed.
2. Dynamic imports (`importlib.import_module("a.b")`, `__import__("a.b")`) are invisible; non-literal ones are
   silently ignored instead of widening validation.
3. `configs` are fed to `module_closure` and parsed as Python; non-Python/plugin/launch files have no declared slot.
4. Missing declared inputs are silently dropped from the fingerprint; a deleted asset/import can therefore look
   unchanged.
5. `plan` checks only the current closure against `changed`; a deleted dependency is not in the new closure, so it
   is missed; old receipts are not consulted.
6. `Cache.lookup` returns any file with `status=PASS` without checking log/output existence or hashes; receipts are
   overwritten per digest (no generation separation).
7. No public durable execution entry: the agent had to remember shell redirection; the 163-test rerun happened
   because the first full run had no persistent log.
8. No machine Infra-integrity snapshot/compare; the skill named the requirement but left it manual.
9. Runtime identity is all-or-nothing: a GPU count change invalidates CPU-only units.
10. No generic reader for saved validation metrics; #258 correctly reported `NOT_AVAILABLE`, but the audit found
    the reason is an R04-identity-bound evaluator, not missing data.

## 5. #258 non-training overhead decomposition (measured, not fabricated)

Merged wall of #258: **8952 s (149.2 min)**. Formal training spans: Stage2 2075 s, Stage3 2507 s.
Attributable non-training commands (from logs and command records):

| category | seconds | source |
|---|---:|---|
| candidate suite, final valid run | 400 | `checks/candidate_suite.log` |
| candidate suite, rerun because the first run had no persistent log | 383 | job output; bookkeeping |
| candidate suite, rerun after source changes | 346 | log timestamps |
| machine admission | 78 | `ADMISSION.json` evidence |
| four-card benchmark (two runs) | 199 | `PERFORMANCE.json`, rank reports |
| module/validation diagnostics (3 validation runs + 2 budget runs) | 313 | logs/rank reports |
| legal re-export | 377 | `reexport.log` |
| R2 upload | 43 | `r2_upload.log` |
| analysis/replay/report | ~15 | file timestamps |
| **unattributed remainder** (agent reasoning, evidence writing, waits, git) | **~2219** | 8952 − listed spans, marked estimated |

The 383 s no-log rerun is exactly the failure the new durable execution entry prevents. The remainder is not
claimed as machine time; it is recorded as unattributed.

## 6. What stays frozen

`assets/`, `tools/research/stage2/`, `tools/research/verification/`, Stage3 trainer, frozen baseline trees,
historical results, all research mechanisms. This issue only extends `tools/iteration/` and the two rule documents.
