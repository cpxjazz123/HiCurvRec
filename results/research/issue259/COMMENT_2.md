### #259 COMPLETED — machine-enforced incremental validation and durable execution

Status: **PASS**. Formal training launches: **0**. Commit: `045a7e85afdc86e6454275db94bafe3d656f0b79` (`main`, pushed; remote SHA verified).

**Delivered**

- `depgraph.py`: absolute/multi-level relative/package `__init__`/literal dynamic import closure; non-literal dynamic imports become `unresolved` and force RUN; missing assets/configs/dirs are visible; per-unit `runtime_keys`; generation receipts; `Cache.lookup_valid` checks schema, unit, digest, fingerprint, PASS, command, exit=0, non-empty log and output `sha256`; deleted/renamed dependencies re-evaluated once and then stop looping; `Spec(cacheable=False)` for mandatory gates.
- `execution.py`: `run`/`run_and_record` write `command.json` before spawn, persist stdout/stderr, exit code, end time, duration and versions; nonzero exit is FAIL, timeout is INCOMPLETE.
- `integrity.py`: protected snapshot/compare, used at start and close.
- `analysis.py` + `report.py`: identity-bound `saved_validation`, execution counts and durations in `EXECUTIONS.json`/REPORT.
- `SKILL.md` + `CLAUDE.md` §95–§97 now point at these interfaces; the manual shell-redirection conflict is removed.

**Evidence**

- `results/research/issue259/VERDICT.md` — C1–C12 reconciliation + close checks.
- `.../SUITE_RUN.json` — 57 passed through the durable entry, exit 0, concrete log path, receipt valid.
- `.../SCENARIOS.json` — 11 fixed fixtures, 10 RUN / 4 REUSE, all expected; plan queries 5.7–29.5 ms.
- `.../REPLAY.json` — R05 read-only replay PASS, 13/13 checks.
- `.../FROZEN_CHECK.json` — 4/4 frozen artefact hashes unchanged.
- `.../INFRA_DIFF.json` — 9 added / 6 changed, 0 removed, all under authorized paths.
- `.../OVERHEAD.json` — #258 decomposition, including the 383 s no-log rerun this entry prevents.

**Not available:** the R05 validation metric remains `NOT_AVAILABLE` because the only stored artefact is bound to `issue256_r04_formal`; no R05 validation artefact exists. No test-split substitution or fabricated value.

Close checks: production code implemented yes; tests actually executed yes (exit codes/logs present); training N/A by Infra-only scope; Infra boundary respected yes; executable unfinished tasks none.
