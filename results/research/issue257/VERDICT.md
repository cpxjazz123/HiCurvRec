# Issue 257 verdict — reusable single-agent iteration pipeline

Status: **COMPLETED**. No formal research training was run (the issue forbids it); the acceptance is
the pipeline, its regressions, the read-only replays and a controlled real entry.

## Criteria

| id | requirement | evidence | status |
|---|---|---|---|
| C1 | audit #251–#256, fault ledger with class/repro/fix/scope | `FAULT_LEDGER.md` (12 rows, each with a repository evidence path) | PASS |
| C2 | timing, NA where unrecoverable | `TIMING.json` (replay cold/warm, pipeline suite, archive check, controlled entry) | PASS |
| C3 | public boundary; candidate owns mechanism + tests | `tools/issue257/*`, reuse of `tools/research/stage2/*` | PASS |
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
2. All necessary tests executed? **Yes** — `python3.9 -m pytest tools/issue257/tests -q` → 14 passed.
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
| `tools/issue257/depgraph.py` | `463d42b05b9d85ecdac937f1106c0fda684979e95f22be43ddaf220334be0bc7` |
| `tools/issue257/identity.py` | `39cee2abefd8af9ead1efcc47ab91f49e13a4995dd738742539226d3dfcc48a1` |
| `tools/issue257/analysis.py` | `e1b646d5f2910d68c41c0f8cba592122cf3e12937beef42032630c5219188fdc` |
| `tools/issue257/replay.py` | `5d6453a1e038fd7a89af6283384545d32531cc572a94ea4316dc8e5dc7a1c73c` |
| `tools/issue257/registry.py` | `08beb6074780fe4aeb37b67cab4761cba734c0f0398e63d814998ab637a09447` |
| `tools/issue257/timeline.py` | `1df45d4bc279bd67e6e9e715bff4f84497dea3f0ecc304e2b63a2803e4ca4f6c` |
| `tools/issue257/report.py` | `f74767b241d44ce68fd88e493401460ece1ac4296cd9d3cef7bb6b90e1ead0ff` |
| `tools/issue257/archive.py` | `767ef46efa8cb08d14abf044d36f76f69a6729f2a6df4c0267ec33cb82b74cd8` |
| `tools/issue257/tests/test_depgraph.py` | `4b26b87e52001d993b066f606aefb05dd82fe685ecb21f09ba110b39c97652bf` |
| `tools/issue257/tests/test_known_answers.py` | `a221a5ee0cbe0478c46913e277c530c6022f476bd7c2c4cb093b906bfa01f052` |
| `tools/issue257/tests/test_replay.py` | `015114f0dc63e67c5e7b922fe25e109009fa37309261e58870e2abbeb55503a7` |
| `results/research/issue257/FAULT_LEDGER.md` | `1a39b05d744e7459e807f8668654c0ba849f24262c9c1db3ed48e492df774095` |
| `results/research/issue257/RUN_LOG.md` | `b027f03c87a6acb2e153034b8ebb3115d6c95c90993d54e5d16d86e92db84025` |
| `results/research/issue257/VALIDATION_PLAN.md` | `61fdccea1e491e44df19be98f774ffbeaf0715e89d4c9d93c1481c485272a835` |
| `results/research/issue257/VALIDATION_PLAN_DEMO.json` | `d2622f78cbc312fbe111526451880575f37efda4242e30b5bd5afe1a0575e9c2` |
| `results/research/issue257/TIMING.json` | `cf6a38d12d07fe189db873778706d27bed853ebd736710b00dbae16f57e7cdb2` |
| `results/research/issue257/replay/r01/REPLAY.json` | `6d9164a91dd3ab62227c0bd3af643575bcf2855e7ab06caa6c6d871ea6db1bc0` |
| `results/research/issue257/replay/r04/REPLAY.json` | `abf7e99e938a7bbcbdf4c17f1f8c49b5c3433f8676f3452eacb7c2e52c86a0f2` |
| `results/research/issue257/controlled_entry.json` | `b5c4e66e7454c0b26ea0e5ba83210038189c39feef9d157b38e5f598338f4e26` |
| `results/research/issue257/controlled_four_card.json` | `73444ddae9ea64f72d5bfcc4825f68110cabf5a464c4b61345b3b6b9b5f350bf` |
