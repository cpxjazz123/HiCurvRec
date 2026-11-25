# Issue #259 RUN_LOG — Infra-only incremental validation

## 1. Issue identity (frozen)

| field | value |
|---|---|
| number / state | 259 / OPEN |
| title | Infra only: machine-enforced incremental validation, durable evidence and reusable iteration execution |
| body bytes / sha256 | 8002 / `051b4fedf1dfe14531d4cbf6316a74d98cc3eb24929db341e5d492c3b0603b02` |
| created | 2026-11-25T13:57:26Z |

## 2. INFRA_MODE = BUILD (explicit issue authorization)

The issue states verbatim: “本 Issue 是独立的 **Infra 建设任务（INFRA_MODE=BUILD）**” and
“**明确授权**:允许新建或修复 `tools/iteration/` 内公共验证、持久化执行、证据缓存、计时和分析接口,
以及必要的公共接线、测试与文档/Skill/CLAUDE.md 同步(本 Issue 单独授权,后续研究 Issue 仍默认 FROZEN).”

Forbidden and respected: no research-mechanism change, no formal Stage2/Stage3, no TIGER or new-baseline training,
no seed runs, no graph/curvature asset rebuild, no historical result rewrite. Formal training launch count = 0.

## 3. Training decision — not required

Infra-only issue; the closure standard explicitly requires “正式训练启动数为 0”. Verification uses unit
fixtures, read-only replay of existing R05 artefacts and hash checks of frozen baselines.

## 4. Frozen acceptance list (C1..C12)

| id | issue requirement (frozen) | criterion / evidence |
|---|---|---|
| C1 | audit existing #257/#258 interfaces, tests, receipts, logs, timeline, R05 artefacts; freeze the incremental-validation contract; decompose #258 non-training overhead | `AUDIT.md`, `CONTRACT.md`, `OVERHEAD.json`; missing timing marked NOT_AVAILABLE |
| C2 | fix multi-level relative imports, package `__init__`, transitive imports; statically resolvable dynamic imports; declared non-Python/plugin deps; conservative expansion when unresolvable | `depgraph` tests + implementation; unresolved → forced RUN |
| C3 | cover added/deleted/renamed deps, missing assets, environment/config changes; runtime fingerprint scoped by validation type | scenario tests; per-unit `runtime_keys`; old-closure deletion check |
| C4 | stable unit IDs and cross-round receipt lookup; distinguish real content changes from directory/run-name changes; no same-name cross-candidate reuse; machine plan RUN/REUSE with reasons; new mechanism not auto-full; no infinite rerun from cumulative changed | `plan()` output and tests |
| C5 | machine evidence validation inside the reuse path: schema, unit ID, fingerprint, command, exit code, log exists/non-empty, output path + sha256; invalid/corrupt/FAIL/NOT_RUN never reused; production path recorded; immutable generations; legacy receipts missing fields explicitly invalid | `Cache.lookup_valid`, receipt schema v2, tests |
| C6 | durable public execution entry: preflight output path, command/cwd/versions/start/identity, full stdout/stderr, exit code, end, duration, evidence hashes; interrupted ≠ PASS; training still via existing launcher | `tools/iteration/execution.py` + tests + a real run log |
| C7 | resume from breakpoints: valid completed units not rerun; failures only affected; no duplicate formal run; no GPU-authorization change | scenario matrix proof |
| C8 | unified timing/report: run/reuse counts, per-unit duration, repeat reasons, phase wall; reuse existing timeline/report | `EXECUTIONS.json` and REPORT section |
| C9 | Infra integrity at start / before dispatch / before archive-close; protected list covers adds/deletes and excludes normal outputs | `tools/iteration/integrity.py` + snapshot comparison |
| C10 | 60-epoch benchmark, module gradients, mutation triggers encoded in the plan; mandatory real entry/behaviour graph/admission never cached; unified runtime fingerprint without repeated expensive probes | plan tests + forced-item list |
| C11 | Skill and CLAUDE.md point to actual interfaces; remove full-suite/manual-log conflicts; check analysis compatibility across run identities incl. #258 validation gap (no test-as-validation, no fabricated data) | updated docs + `saved_validation` reader |
| C12 | TDD red→green; infra regression with persisted logs; fixed fixture scenarios (first/no-change/single/shared/missing-corrupt/unknown); R05 read-only replay; frozen TIGER/historical artefacts unchanged; no formal training; commit/push/close | test logs, scenario JSON, replay JSON, hash checks, VERDICT.md |

## 5. Scope boundaries

- Modified/added only under `tools/iteration/`, plus documentation (`SKILL.md`, `CLAUDE.md`) and this evidence dir.
- `assets/`, `tools/research/stage2/`, Stage3 trainer, frozen baseline trees and all historical results are untouched.
- No model training; GPU is not used except any read-only hash/replay that requires none.

## 6. Commit preflight note (CLAUDE §91)

`python3 tools/commit_guard.py` fails at the default budget because the shared worktree contains 797 changed
paths from unrelated, pre-existing candidate/result trees. This issue's commit selects only
`tools/iteration/`, `CLAUDE.md`, the Skill and `results/research/issue259/` via `commit.draft().only(...)`.
The guard was therefore rerun with an explicit raised path/line budget and the resulting selection verified
to contain none of the unrelated trees:

```bash
python3 tools/commit_guard.py --max-paths 900 --max-lines 40000   # reason above
```
