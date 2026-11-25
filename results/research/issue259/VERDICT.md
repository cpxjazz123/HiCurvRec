# Issue #259 VERDICT — incremental validation, durable evidence, reusable execution

Status: **PASS** (Infra-only; formal training launches: **0**).

## 1. C1–C12 reconciliation

| id | status | evidence |
|---|---|---|
| C1 audit + contract + overhead | PASS | `AUDIT.md` (`1490a901…`), `CONTRACT.md` (`60fa2b69…`), `OVERHEAD.json` (`149b387b…`); missing timings are marked unattributed, never fabricated |
| C2 relative/package/dynamic imports | PASS | `tests/test_depgraph_imports.py`: multi-level `..`, sibling `.`, package `__init__`, literal `importlib`; non-literal → `unresolved` forces RUN |
| C3 add/delete/rename/missing/env | PASS | `tests/test_depgraph_evidence.py`: deletion, rename, missing asset, missing/empty log, config-as-data, per-unit `runtime_keys` |
| C4 stable IDs + plan reasons + no cross-candidate/no loop | PASS | `plan()` RUN/REUSE reasons; `test_same_name_different_content_does_not_reuse`; `test_touched_but_unchanged_file_does_not_loop`; `SCENARIOS.json` |
| C5 machine evidence validation + generations | PASS | `Cache.lookup_valid`: schema/unit/digest/fingerprint/PASS/command/exit=0/log non-empty/output sha; receipts `<unit>/<digest>/<generation>.json`; legacy missing fields invalid |
| C6 durable execution entry | PASS | `tools/iteration/execution.py` (`9f15a8e0…`); `tests/test_execution.py` success/failure/timeout/preflight; real suite run `SUITE_RUN.json` (`93b33f76…`) |
| C7 resume without duplicate work | PASS | `SCENARIOS.json`: no-change cross-round reuse, deleted-dependency one-shot recheck then reuse, shared change reruns only consumers |
| C8 timing/report | PASS | `report.write_all(..., executions=…)` → `EXECUTIONS.json` + REPORT counts/duration; `tests/test_report_executions.py` |
| C9 Infra integrity checkpoints | PASS / N/A | start `INFRA_BASELINE.json` (`c87e8c95…`), close `INFRA_DIFF.json` (`788e6dec…`); pre-dispatch checkpoint **N/A** because no training dispatch exists; diff contains only allowed paths |
| C10 benchmark/gradient/mutation triggers + mandatory gates | PASS | `test_performance_and_mutation_specs_follow_the_compute_path` (prose edit reuses; compute change reruns both); `Spec(cacheable=False)` mandatory gate always RUN; runtime identity cached (`lru_cache`) |
| C11 Skill/CLAUDE + analysis compatibility | PASS | `SKILL.md` and `CLAUDE.md` §95–§97 point at `execution`/`integrity`/`lookup_valid`; manual shell-redirection conflict removed; `saved_validation` returns `NOT_AVAILABLE` for the R05 gap |
| C12 TDD + regression + replay + no drift + no training | PASS | red evidence: execution/integrity import errors + 3 depgraph failures before implementation; **57 passed** via durable entry (`SUITE_RUN.json`); `SCENARIOS.json` all expected (`32fb2225…`); `REPLAY.json` status PASS; `FROZEN_CHECK.json` ok; training 0 |

## 2. Verification commands and observed results

```bash
python3.9 -m pytest tools/iteration/tests -q           # 57 passed in 3.07s
python3.9 results/research/issue259/scenario_matrix.py # 11 rows, 10 run / 4 reuse, all expected True
python3.9 - <<'PY' ... replay.replay(R05 test_final) ... PY  # status PASS, 13/13 checks
```

- R05 replay: all SID-chain, independent-reduction and artefact checks PASS; 0 failures.
- R05 saved validation: `NOT_AVAILABLE` — the only stored artefact is bound to `issue256_r04_formal`; no R05
  validation artefact exists. The test split was not substituted and no metric was fabricated.
- Frozen artefacts: 4/4 hashes unchanged (`FROZEN_CHECK.json` ok).
- Infra close diff: 9 added files, 6 changed files, 0 removed; all under `tools/iteration/`, `CLAUDE.md` or the
  Skill. Protected research/stage2/verification/manifest/Stage3 files unchanged.

## 3. Measured overhead

| item | value | source |
|---|---:|---|
| #258 merged wall | 8952 s | `OVERHEAD.json` |
| #258 no-log candidate-suite rerun (now structurally prevented) | 383 s | `OVERHEAD.json` |
| plan query, this issue (11 fixtures) | 5.7–29.5 ms | `SCENARIOS.json` |
| full iteration suite through durable entry | 3.47 s | `SUITE_RUN.json` |
| formal training launches | 0 | required |

The new entry writes `command.json` before spawn, so a missing log can no longer turn a completed run into an
unusable one; reruns now require an explicit invalidating change.

## 4. Usage

```python
from tools.iteration import depgraph, execution, integrity, analysis, report

spec = depgraph.Spec('unit', tests=[test], code=[module], runtime=True,
                     runtime_keys=('torch', 'devices'))
cache = depgraph.Cache('results/research/<issue>/receipts')
plan = depgraph.plan([spec], cache=cache, changed=[changed_path])   # RUN reasons / REUSE
execution.run_and_record(cache, spec, [python, '-m', 'pytest', test],
                         out_dir='results/research/<issue>/checks')
report.write_all(out_dir, identity=identity, plan=plan, receipts=[],
                 analysis=analysis.analyse(identity, validation_artifact=path),
                 executions=executions)
```

## 5. Limits

- No GPU work and no model training; this issue changes Infra only.
- The pre-dispatch integrity checkpoint is not applicable until a future issue actually dispatches training.
- The R05 validation metric remains `NOT_AVAILABLE`; producing it requires a run-bound validation artefact,
  which is explicitly out of scope here.
- Legacy #257/#258 receipts lacking `fingerprint`/`output_sha256` are invalid by design; they must rerun once.

## 6. Close checks (CLAUDE / Skill §6.4)

1. Production code implemented: **yes** — `depgraph.py`, `execution.py`, `integrity.py`, `analysis.py`, `report.py` are real implementations, not stubs.
2. Necessary tests actually executed: **yes** — 57 passed through `execution.run_and_record`; `SUITE_RUN.json` holds the command, log path, exit code 0 and duration; receipt is valid by `Cache.lookup_valid`.
3. Required training completed: **N/A by scope** — Infra-only issue, zero formal training launches required and performed.
4. Infra boundary respected: **yes** — BUILD authorization recorded in `RUN_LOG.md` §2 and Issue comment 1; `INFRA_DIFF.json` contains only authorized paths; frozen artefacts 4/4 unchanged.
5. Executable unfinished tasks remain: **no** — C1–C12 all have current evidence; the only `NOT_AVAILABLE` is the explicitly out-of-scope R05 validation metric.
