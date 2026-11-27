# Issue #269 infrastructure record (`FROZEN`)

## Assets reused (hash-referenced, never rebuilt)

| asset | `path:sha256` |
|---|---|
| training graph | `assets/train/visible_undirected.npz:097a719a9f00292477a1434c776de1ff37fb60adb7a1adc9c517a98bc7147818` |
| packed targets | `assets/v8_runtime_layout/targets.npy:8c0ff9f8c25b98ed31802cdb62cf921df6d3cf8bbf35ee2d3de9817d6cd0807a` |
| packed offsets | `assets/v8_runtime_layout/target_offsets.npy:33f30b2979311d74cf996545bec011856ffa7acbc9cb9b995c829dbc9f87937c` |
| node moments | `assets/node_moments.npz` (read by `FrozenCurvature`; unchanged) |
| item summary | `assets/item_summary.npz` (target scale; unchanged) |
| query index | `assets/index.npz` (shard names; unchanged) |
| curved-triple shards | `assets/shards/shard_000..012.npz` (target blocks; unchanged) |
| frozen reference | `results/research/issue268/artifacts/z_ref.npy:a2643a9bc09bbedffe4c249fea7d07d54153c9c8fd6aa5e476fe4f7b63ae4a66` |
| semantic embeddings | `stage1_GeneEmbedding/output/sentence_t5.npy` (frozen 768-d input) |
| frozen TIGER weights | `results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth` (loaded read-only by the mechanism tests) |

No `assets/` file was created, modified or copied. The new κ cache is a candidate-local
result under `results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284/`, not an
asset.

## Public components (hash baseline)

`INFRA_SNAPSHOT.json` records `path:sha256` for 61 files under `tools/iteration/`,
`tools/research/stage2/`, `tools/research/verification/` plus `assets/manifest.json`,
`stage3_T5Train/train_HG-Rec.py`, `CLAUDE.md` and this skill. No public file was edited in this
round; the closing snapshot will be compared against it.

## Interpreters, hardware, ports

- Stage0/1/3: `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10`.
- Stage2 and candidate tests: `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9`.
- MCP: system `python3`.
- Host: one visible `NVIDIA A100-SXM4-40GB` (single-card branch, owner authorisation recorded in
  `results/research/issue268/PROGRESS.json` and the issue comment of 2026-11-27). Ports 50202 /
  50201 are free.

## Tree state and archive (skill §1.0a)

- Active tree: `stage2_RQ-VAE/TIGER_calibrated_kappa_rq`.
- `stage2_RQ-VAE/TIGER_relative_curvature_ranking_rq` → `_archive/TIGER_relative_curvature_ranking_rq_issue267`
  (`git mv`).
- `stage2_RQ-VAE/TIGER_fixed_triplet_kappa_rq` → `_archive/TIGER_fixed_triplet_kappa_rq_issue268`
  (`mv`, untracked tree). The #268 result directory and its frozen cache stay at
  `results/stage2_RQ-VAE/TIGER_fixed_triplet_kappa_rq/` so the #268 evidence paths remain valid.
- The calibrated tree was copied from the #268 tree; only `ARM_NAME`, the cache directory name,
  the cache schema, the coordinate-scale wiring and the matching tests differ. The #268 evidence
  and cache are untouched.

## Infra faults

None. The #268 NaN/floor stall was a candidate-mechanism bug (`model/stereographic.py`), fixed in
the calibrated tree with regression tests; it was not a public-infra defect.
