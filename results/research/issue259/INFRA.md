# Issue #259 INFRA — BUILD scope, reuse only

Mode: `BUILD`, authorized by the issue's “INFRA_MODE=BUILD” sentence and its explicit list of allowed
`tools/iteration/` interfaces. No research mechanism, asset, trainer or historical result is modified.

## 1. Reused inputs (read-only, hash-pinned)

| input | sha256 | use |
|---|---|---|
| `assets/manifest.json` | `cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b` | registered data assets; not rebuilt |
| R05 SID `.../issue256_r05_formal/.../item_sids.json` | `dc206dba7fbdadc28bbdedb582c7c755e412d70bdfec930b380a2bd5ea9d5ba6` | read-only replay fixture |
| R05 `test_final.json` | `ed2d61223dd58af4fd77166a6ecdd1ec2c4aa6129ef534c1d60be6866db97cb9` | read-only replay fixture |
| frozen TIGER checkpoint | `af8efc29a8bcbbe5394cc5e358acf7b14c0e78ae99d8a8b9e5faf03fdddd97a8` | drift check only |
| frozen TIGER SID | `025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059` | drift check only |

`FROZEN_ARTIFACTS.json` records these four; `AUDIT.md` records the #258 receipt/log inventory.

## 2. Public component baseline

`INFRA_BASELINE.json` (sha256 `c87e8c95925bab6f1b86c18f6c10a0eb77324474ff437127fd25691d32868889`) records
223 `path:sha256` entries covering `tools/iteration/`, `tools/research/stage2/`,
`tools/research/verification/`, `tools/graph_infra/`, `tools/issue252/`, `tools/issue253/`, `CLAUDE.md`,
the skill, `assets/manifest.json` and the Stage3 trainer. The final snapshot must match except for the files
this issue is authorized to add/modify under `tools/iteration/` and the two rule documents.

## 3. Hardware and interpreters

| item | value |
|---|---|
| GPUs | 4 × L40S 46 GB (visible); **not used** by this issue |
| Stage2 interpreter | `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` |
| Stage3 / docs interpreter | `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10` |
| system python | `/usr/bin/python3` (stdlib-only MCP) |
| formal training launches | **0** (required by the close standard) |

## 4. What will be created/extended

- `tools/iteration/depgraph.py` — relative/package/dynamic import closure, missing/unresolved handling,
  scoped runtime keys, machine evidence validation, generation-aware receipts, RUN/REUSE reasons.
- `tools/iteration/execution.py` — durable public execution entry (new).
- `tools/iteration/integrity.py` — protected snapshot/compare (new).
- `tools/iteration/report.py` — run/reuse counts and per-unit durations.
- `tools/iteration/analysis.py` — identity-bound saved-validation reader; missing artefact → `NOT_AVAILABLE`.
- `tools/iteration/tests/` — deterministic fixtures for all of the above.
- `.agent/skills/complete-open-issue/SKILL.md`, `CLAUDE.md` — point to the actual interfaces.

No file under `assets/`, `tools/research/`, `stage2_RQ-VAE/`, `stage3_T5Train/` or any historical result path
will be modified.
