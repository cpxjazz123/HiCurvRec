# Issue 257 verdict — reusable single-agent iteration pipeline

Status: **COMPLETED**. No formal research training was run (the issue forbids it); the acceptance is
the pipeline, its regressions, the read-only replays and a controlled real entry.

## Criteria

| id | requirement | evidence | status |
|---|---|---|---|
| C1 | audit #251–#256, fault ledger with class/repro/fix/scope | `FAULT_LEDGER.md` (12 rows, each with a repository evidence path) | PASS |
| C2 | timing, NA where unrecoverable | `TIMING.json` (replay cold/warm, pipeline suite, archive check, controlled entry) | PASS |
| C3 | public boundary; candidate owns mechanism + tests | `tools/iteration/*`, reuse of `tools/research/stage2/*` | PASS |
| C4 | candidates from explicit parents; immutable snapshots | `_archive/` trees + `run_workspace`; `archive.py` verifies the binding | PASS |
| C5 | dependency-aware validation by content-hash fingerprints | `depgraph.py`; `tests/test_depgraph.py` (4 scenarios + closure mutation) | PASS |
| C6 | three real change scenarios; minimal rerun set | `VALIDATION_PLAN_DEMO.json`: analysis-only, shared-operator, mechanism, unrelated-doc | PASS |
| C7 | standard round state machine, resumable and idempotent | `registry.py`; `tests/test_known_answers.py::test_registry_transitions_and_refusals` | PASS |
| C8 | automated analysis parameterised by run identity | `analysis.py`; no issue/arm literal | PASS |
| C9 | read-only replay of ≥2 immutable runs reproduces their numbers | `replay/r01/REPLAY.json`, `replay/r04/REPLAY.json` — 13 checks each, PASS | PASS |
| C10 | controlled real four-card entry | `controlled_entry.json` (admission gate PASS, 47.27 s) and `controlled_four_card.json` (real 4-rank training entry: launcher exit 0, 4/4 `PROFILED`, 67.17 s) | PASS |
| C11 | four-piece archive verified without re-upload | `archive.verify` PASS for R04 (ledger 105, SID prefix, four pieces) | PASS |
| C12 | docs and skill match the real behaviour | `CLAUDE.md` §95–§99, `.agent/skills/complete-open-issue/SKILL.md` §8 | PASS |

## Recompute (not retyped)

* R01: `test_R@10 = 0.05266456588728913` recomputed from all 57 439 persisted predictions and from
  the stored layer record; five artefact hashes re-read (`replay/r01/REPLAY.json`).
* R04: `test_R@10 = 0.05151552081338463`; `distinct_triples = 22995`; `l3_collision = 0.0647496644…`;
  `H(L3|L1L2) = 1.9957073319719747` (`replay/r04/REPLAY.json`).
* Replay cost: 1.6 s cold, 1.7 s warm, 0 GPU.

## Failed attempt (recorded, not hidden)

The archived tree's own `validation_entry.py` (its 1-epoch diagnostic) failed with
`gate weights do not cover the three experts`: that observer expects `last_weights` shaped `(N,3)` while
the mechanism stores the gate tensor `(N,2,3)`. The admission path uses `check_entry`, so the assertion
was never exercised; the failure is preserved in `TIMING.json.failed_attempt` and the log under
`short_candidate/`. The canonical entry gate then passed.

## §6.4 five answers

1. All production code implemented? **Yes** — eight modules, no stubs (a stub would fail the 14 tests).
2. All necessary tests executed? **Yes** — `python3.9 -m pytest tools/iteration/tests -q` → 14 passed.
3. All prescribed training completed? **Yes by absence** — the issue requires none; the controlled
   1-epoch real entry ran and passed.
4. Infra assets from `assets/` without duplication? **Yes** — no asset was created or recomputed; the
   pipeline reads existing files by hash.
5. Any executable unfinished task left? **No.**

## Limits

* The ≤2 h full-iteration budget cannot be measured without a fresh complete round; only the
  verifiable stage costs are claimed.
* `test_derived` tables and private raw user data stay out of Git; evidence lives under
  `results/research/issue257/`.

## Evidence hashes (path:sha256)

| path | sha256 |
|---|---|
| `tools/iteration/depgraph.py` | `463d42b05b9d85ecdac937f1106c0fda684979e95f22be43ddaf220334be0bc7` |
| `tools/iteration/identity.py` | `39cee2abefd8af9ead1efcc47ab91f49e13a4995dd738742539226d3dfcc48a1` |
| `tools/iteration/analysis.py` | `e1b646d5f2910d68c41c0f8cba592122cf3e12937beef42032630c5219188fdc` |
| `tools/iteration/replay.py` | `01d18138cad4be17352f7bde973749ee36c10a3ce3e30ddb77340f762acaf55c` |
| `tools/iteration/registry.py` | `08beb6074780fe4aeb37b67cab4761cba734c0f0398e63d814998ab637a09447` |
| `tools/iteration/timeline.py` | `1df45d4bc279bd67e6e9e715bff4f84497dea3f0ecc304e2b63a2803e4ca4f6c` |
| `tools/iteration/report.py` | `f74767b241d44ce68fd88e493401460ece1ac4296cd9d3cef7bb6b90e1ead0ff` |
| `tools/iteration/archive.py` | `767ef46efa8cb08d14abf044d36f76f69a6729f2a6df4c0267ec33cb82b74cd8` |
| `tools/iteration/tests/test_depgraph.py` | `a2fd0e3b345030151f9f6296af299ff86a14d5a27415c93796750b2f57bbdf15` |
| `tools/iteration/tests/test_known_answers.py` | `9d2c403cbec90bd08aa93cff4f0275a2149e43fe5aa5d22c79738cb255788793` |
| `tools/iteration/tests/test_replay.py` | `cef62c58162ff7c14bf08e93777801fa728f0e3390b98c53545c2719217081eb` |
| `results/research/issue257/FAULT_LEDGER.md` | `003428df83798be7c10159b2fcf9215411215aa637f27968d4063d5dd21b57d7` |
| `results/research/issue257/RUN_LOG.md` | `137307b91d3a5eef1bf0d96473f35cde124edae118edf1460ea8092ff4736584` |
| `results/research/issue257/VALIDATION_PLAN.md` | `fa5e3f032915ab279c877e27a3135b4f40badefedcfd58fb0ccae496a6a69287` |
| `results/research/issue257/VALIDATION_PLAN_DEMO.json` | `4a66721422014f38a5caab37179d8e54c2b4e39083cf70f2da1d109bb4fc20be` |
| `results/research/issue257/TIMING.json` | `f183404465d9c03f9df3df980f1eeed5fb3eebca361d412eb8274c1dd07bd7e4` |
| `results/research/issue257/replay/r01/REPLAY.json` | `6d9164a91dd3ab62227c0bd3af643575bcf2855e7ab06caa6c6d871ea6db1bc0` |
| `results/research/issue257/replay/r04/REPLAY.json` | `abf7e99e938a7bbcbdf4c17f1f8c49b5c3433f8676f3452eacb7c2e52c86a0f2` |
| `results/research/issue257/controlled_entry.json` | `b5c4e66e7454c0b26ea0e5ba83210038189c39feef9d157b38e5f598338f4e26` |
| `results/research/issue257/controlled_four_card.json` | `66cd6f7b159f9725e559cf6e65ef0f16827ec1fafb7bb665e921dfe40909b617` |
