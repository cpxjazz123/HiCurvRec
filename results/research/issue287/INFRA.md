# Issue #287 — INFRA (FROZEN)

## Reused assets (read-only, by hash)

| item | path | sha256 |
|---|---|---|
| triplet/curvature table (13 shards) | `assets/shards/shard_000..012.npz` | per `assets/manifest.json` (28,391,948 rows) |
| visible directed graph | `assets/train/visible_forward.npz` | `ac7155d1424a6a346da9b8ed6e9413b81d43223a377308af0c8ba35b6b773029` |
| visible reverse graph | `assets/train/visible_reverse.npz` | manifest |
| visible undirected projection | `assets/train/visible_undirected.npz` | manifest |
| #274 frozen pair index (the mechanism's sample set) | `results/research/issue274/pairs_index.npz` | manifest of #274 (`pairs_manifest.json`) |
| #277 evaluator (P/N) | `tools/issue252/layer_metrics.py` | recorded in `EVAL_287.json` |
| SID loader | `tools/issue252/final_evaluation.py::load_sid` | manifest |
| validation rows | `results/stage0_build_parquet/valid.parquet` | recorded in `EVAL_287.json` |
| frozen TIGER baseline results | `results/research/issue274/baselines_tiger.json` | #274 |

Nothing was rebuilt: the triplet table, the graphs, the pair index and the evaluator are read as they
are; the new mechanism only re-uses the frozen #274 index and margin.

## New candidate tree

`stage2_RQ-VAE/TIGER_l1_pairrank_rq` — created from `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` with
`ARM_NAME` changed (the #274 pair-ranking implementation is reused as the issue instructs, with the
single main change of this round: the supervised distance is the **selected L1 codeword**). The
per-file changes against the baseline are listed in `REPORT.md`.

## §1.0a archiving

Before the new candidate was created, the three remaining top-level mechanism trees
(`TIGER_pairrank_rq`, `TIGER_pairrank_cover_rq`, `TIGER_pairrank_enc_rq`) were moved with `git mv` to
`stage2_RQ-VAE/_archive/`; `baseline/` and `_archive/` stayed in place and no historical evidence was
rewritten. No training was running and the GPU was free at the time.

## Environment

- Interpreter: the Stage2 environment `genrec_env_v2/bin/python3.9` (CLAUDE.md §6).
- Hardware: one visible A100-80GB → the single-card branch with the frozen global batch 4096.
- Public components were only called: `tools/research/stage2/checks.py`, `tools/iteration/*`,
  `tools/issue252/*`, `stage2_start`. No public file was modified; the protected-component snapshot
  (start vs close) shows added/removed/changed = 0.
