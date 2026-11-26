# Issue #266 contract

The canonical frozen score and measurement definitions are in `SCORE_CONTRACT.md`;
this file records the issue-wide boundaries and acceptance rules. Both were written
before any endpoint computation. No model or public-infra file is changed.

## Scope

- Read-only analysis of the final checkpoints in R2 bundles 108 (`#263`) and 109 (`#265`).
- Reuse the #262 endpoint-gradient measurement definition exactly where the issue names it.
- Recompute the frozen eight-arm score from the existing #262 C/G1 values and append 108/109.
- No Stage2/Stage3/TIGER/GNN training, optimizer step, parameter update, SID rewrite,
  new training arm or seed.

## Frozen boundaries

- Public infrastructure, `stage2_RQ-VAE/baseline/`, assets, Stage3 entry and the Skill are read-only.
- Inputs: `results/stage0_build_parquet/train.parquet`,
  `stage1_GeneEmbedding/output/sentence_t5.npy`, `assets/manifest.json`,
  `assets/v8_runtime_layout/`, the two R2 bundles and their `test_final.json`/`recall.json`.
- The only outputs are new files under `results/research/issue266/`.

## Required evidence

`SCORE_CONTRACT.md` freezes C, G1, the normalization, S, population and all ranking
definitions. `ENDPOINT_108.json` and `ENDPOINT_109.json` retain raw C/G1, hashes,
gradient shapes, nonzero checks and the zero-optimizer-step proof. `SCORES.json/CSV`
retain the ten-arm table and statistics. `REPORT.md` and `VERDICT.md` separate
prediction continuation, native-loss scale comparability, gradient sensitivity and
actual optimization benefit. `tests/` contains an independent hand-computed reference
for the score and ranking edge cases.

## Acceptance

A negative monotonicity result is a valid COMPLETED outcome. Missing fields are never
filled with zero; if a field is genuinely unavailable the exact attempted path is
recorded.
