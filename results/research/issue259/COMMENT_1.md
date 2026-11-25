### #259 checklist frozen — BUILD mode, audit complete

**Issue body**: 8002 bytes, sha256 `051b4fedf1dfe14531d4cbf6316a74d98cc3eb24929db341e5d492c3b0603b02`.

**INFRA_MODE = BUILD** — the issue authorizes `tools/iteration/` public validation, durable execution, evidence
cache, timing and analysis interfaces plus necessary wiring/tests/docs; later research issues remain FROZEN.
Training is **not required**: this is Infra-only and the close standard requires zero formal training launches.

**Frozen acceptance list C1–C12** (full text in `results/research/issue259/RUN_LOG.md` §4): audit/contract/overhead,
relative+package+dynamic imports with conservative expansion, added/deleted/renamed deps + missing assets + scoped
environment, stable IDs and RUN/REUSE plans, machine evidence validation, durable execution, resume, timing/report,
three Infra checkpoints, benchmark/gradient/mutation triggers + non-cacheable admission, Skill/CLAUDE sync + saved
validation compatibility, TDD/regression/scenario matrix/R05 replay/no-drift/no-training/commit-close.

**Audit** (`results/research/issue259/AUDIT.md`): the current `depgraph.py` misses multi-level relative imports,
package `__init__` chains, dynamic imports and declared non-Python/plugin files; configs are parsed as Python;
missing inputs are silently dropped; deletions are invisible to `plan`; receipts are not evidence-checked and are
overwritten per digest; there is no durable execution wrapper or integrity snapshot. #258 receipt inventory: 7
receipts, of which 2 lack `output_sha256` and must be invalid under the new contract.

**#258 overhead** (`OVERHEAD.json`): merged wall 8952 s; Stage2 2075 s + Stage3 2507 s; attributable
non-training commands include 383 s for the no-log candidate-suite rerun and 346 s for the source-change rerun;
~2219 s remain explicitly unattributed (estimated, not fabricated).

**Infra baseline**: 223 protected `path:sha256` entries in `INFRA_BASELINE.json`
(sha256 `c87e8c95925bab6f1b86c18f6c10a0eb77324474ff437127fd25691d32868889`); frozen R05/TIGER hashes in
`FROZEN_ARTIFACTS.json`. No asset, research mechanism, trainer or historical result will be modified.

Next: TDD (failing tests first), then the depgraph evidence/closure implementation.
