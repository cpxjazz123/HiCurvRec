# Issue #258 INFRA — reuse only (FROZEN)

Mode: `FROZEN`. No asset, tool, cache, evaluator, launcher, controller or report generator is created by this
round. Everything below is either read-only reuse or a normal run output produced by an existing interface.

## 1. Preconditions checked

| check | command | result | verdict |
|---|---|---|---|
| old mechanisms archived | `ls stage2_RQ-VAE/` | `_archive`, `baseline` only | PASS (§1.0a satisfied) |
| R05 not already dispatched | `ls results/stage2_RQ-VAE/_runs/` | no `issue256_r05_formal` | PASS |
| R05 candidate tree absent | `find . -maxdepth 4 -name '*role_conditioned_l2*'` | empty | PASS |
| preregistration status | `cat results/research/issue256/rounds/r05/STATUS.json` | `REGISTERED_NOT_DISPATCHED`, no worker, no GPU budget consumed | PASS |
| GPUs | `nvidia-smi` | 0–3, NVIDIA L40S 46068 MiB, all idle | PASS (four-card recipe) |
| ports | `ss -ltn | grep 5020[12]` | free | PASS |
| Stage2 interpreter | `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` | exists | PASS |
| Stage3 interpreter | `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10` | exists | PASS |
| repository HEAD at registration | `git rev-parse HEAD` | `597b1c8a04a13c89f8884ad443cc04a4e1bc3488` | PASS |

## 2. Reused Infra assets (`path:sha256`, immutable)

| asset | sha256 | use |
|---|---|---|
| `assets/manifest.json` | `cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b` | registered asset inventory (109 entries); referenced, not rebuilt |
| `assets/index.npz`, `assets/item_summary.npz`, `assets/distance_histogram.npz`, `assets/train/`, `assets/shards/`, `assets/v10_direction_ego/` | per manifest entries | behaviour/graph/geometry caches used by the inherited Stage2 tree |
| `results/stage0_build_parquet/items.parquet` | `6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3` | item catalogue and split source |
| `results/stage0_build_parquet/train.parquet` | `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815` | training interactions |
| `stage1_GeneEmbedding/output/sentence_t5.npy` | `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` | 768-d item embeddings |
| `results/research/graphlet_roles/` (frozen orbit instances) | hashed in the asset manifest / role table provenance | role labels (training-only) |
| `stage2_RQ-VAE/_archive/TIGER_role_conditioned_l3_rq/role_table.json` | `32f986dac9bf8e382c11aedd62cd54e7defbf1853ee0b930b6fe51d4f334d3eb` | the same frozen role table R04 used; copied byte-identically into the R05 tree, never regenerated |

No binary under `assets/` is overwritten or rewritten.

## 3. Reused public code (read-only)

| component | role in this round |
|---|---|
| `tools/iteration/{identity,depgraph,registry,timeline,analysis,replay,report,archive}.py` | run identity, dependency-aware validation, round state, wall-clock, analysis, recomputation, reports, archive verification (§8 of the skill) |
| `tools/research/stage2/contract.py`, `launcher.py`, `checks.py`, `graph_contract.py`, `sid_quality.py`, `anticollapse/` | workspace identity, four-card launch, machine admission (real entry + behaviour graph), SID quality gate |
| `tools/issue252/run_workspace.py` (`freeze_run`, `load_run`), `archive_audit.py`, `host_transfer.py` | immutable source snapshot and entry adapter for the preregistered run id |
| `tools/graph_infra/ego_gnn.py` | ego-graph store consumed by the mechanism |
| `tools/issue253/cuda_math.py` | compiled geometry kernels used by `model/geometry.py` |
| Stage3 MCP `stage3_start` / `stage3_status` | frozen 150-epoch protocol and collision gate |
| `tools/issue139/upload_stage3.py` + `r2_ledger.json` | exact four-piece R2 archive and ledger |

## 4. Candidate boundary

- New tree: `stage2_RQ-VAE/TIGER_role_conditioned_l2_rq/`, copied from the frozen R04 tree
  `stage2_RQ-VAE/_archive/TIGER_role_conditioned_l3_rq/` (the parent mechanism), then changed only in:
  `train_rqvae.py` (`ARM_NAME`), `model/layer_gate.py` (`LEVELS` and the docstring),
  `model/mechanism.py` (`level_weights` routing and docstring), `model/role_mix.py` (docstring),
  and the mechanism-specific tests (`tests/test_role_mixture.py` plus a new level-routing reference test and a
  new mutation). `role_table.json` is copied byte-identically. `config/train_config.json` is unchanged.
- Run identity: `run_id = issue256_r05_formal`, `arm = TIGER_role_conditioned_l2_rq`, created through
  `tools.issue252.run_workspace.freeze_run` and launched through `tools.research.stage2.launcher.launch`
  (the same entry R04 used). No MCP run-id substitute, no second run.

## 5. Public-component hash baseline

`INFRA_BASELINE.json` records `path -> sha256` for the public components listed in the skill (iteration
package, Stage2 research package, verification package, MCP servers, `assets/manifest.json`, `CLAUDE.md`,
the skill, `tools/graph_infra/`) plus the mechanism-imported `tools/issue252` and `tools/issue253` files.
`INFRA_FINAL.json` at the end of the round is produced by the same command; the round is only compliant if the
two dictionaries are identical. The generator is a one-shot inline command (`python3 - <<'PY'`) that calls
`tools.iteration.depgraph.file_sha`; it creates no module and no tool.

Result: `INFRA_FINAL.json` sha256 `e75ea1f5f54db13e57fa52a97b30b3fac5ac4ea890a21bf433b1180f2c1b4f6b`,
220 files, **changed `[]`, missing `[]`** — the public Infra is byte-identical to the baseline recorded before
the candidate was created.
