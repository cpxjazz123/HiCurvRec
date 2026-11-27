# Issue #268 infrastructure record

`INFRA_MODE=FROZEN`, from the issue text: "单 Agent；INFRA_MODE=FROZEN，复用 #259/#264 已有流程及
训练图/距离/曲率资产；允许候选专属离线几何拟合、缓存导出、诊断和测试，不修改公共 Infra 或原资产。"
No public `tools/`, `assets/`, launcher, admission, evaluator or archive source was edited.

## Reused assets (read-only, hash-referenced)

| asset | sha256 |
|---|---|
| `assets/manifest.json` | see `INFRA_SNAPSHOT.json` |
| `assets/train/visible_undirected.npz` | `097a719a9f00292477a1434c776de1ff37fb60adb7a1adc9c517a98bc7147818` |
| `assets/v8_runtime_layout/targets.npy` | `8c0ff9f8c25b98ed31802cdb62cf921df6d3cf8bbf35ee2d3de9817d6cd0807a` |
| `assets/v8_runtime_layout/target_offsets.npy` | see `INFRA_SNAPSHOT.json` |
| `assets/node_moments.npz`, `assets/item_summary.npz`, `assets/index.npz` | see `INFRA_SNAPSHOT.json` |
| `stage1_GeneEmbedding/output/sentence_t5.npy` | `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` |
| #265 final checkpoint `…/stage2_4891…/result/rqvae_best.pth` | `1f8bfecba0c74bb3cbf9da11fe6925983cbae012753100401a7168e246da7bef` |

No asset was rebuilt, copied, renamed or rewritten. No new public asset was
created.

## Frozen representations and cache

- `Z_ref`: `results/research/issue268/artifacts/z_ref.npy`, sha256
  `a2643a9bc09bbedffe4c249fea7d07d54153c9c8fd6aa5e476fe4f7b63ae4a66`, 24587×32
  float32, from the #265 final encoder (identity in `z_ref_manifest.json`).
- κ cache: `results/stage2_RQ-VAE/TIGER_fixed_triplet_kappa_rq/kappa_cache/`
  with `kappa.npy` (28,391,948 float32), `j_zero.npy`, `j_fit.npy` and
  `manifest.json`; hashes in the manifest and rechecked by
  `analysis/CACHE_VERIFICATION.json`.
- The binary cache is not committed (CLAUDE.md §92); the manifest, scripts and
  diagnostics are.

## Public component baseline

`INFRA_SNAPSHOT.json` stores `path:sha256` for
`tools/iteration/` (21 files), `tools/research/stage2/` (30), 
`tools/research/verification/` (6), `assets/manifest.json`, the Stage3 entry,
`CLAUDE.md`, the skill, the reused assets above and the frozen parent
checkpoint. It was written after all work and no public file was modified during
this issue.

## Host state

The session started with four A100-class devices visible. While this issue was
running the host changed to a single `A100-SXM4-80GB`. The frozen launcher
refuses a formal single-card Stage2:

```
tools/research/stage2/launcher.py: assert_four_cards(...)
"Stage2 needs 4 visible cards for four-rank DDP and this host shows 1;
 a single-card run is not an accepted Stage2 regime"
```

This did not block the delivered work (the geometry gate stopped the mechanism
before any dispatch), and no formal Stage2/Stage3 was attempted. It is recorded
so a future round on the same host is not misread as a mechanism failure.

## Costs

- Full fit: `111.04 s` on one A100 (201 grid values, 28.4 M rows, 256
  references/component); cache 341 MB (`kappa`, `j_zero`, `j_fit`).
- Independent validation: 23.5 s; scale control: 7.4 s; cache verification:
  4.5 s; probes: 34-61 s each.
