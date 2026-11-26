# Issue #266 infrastructure freeze

- Issue body sha256: `09c266af34eef4d0ffdb1a70d1dcdc6254b9bcb6acb3b15ba4f400ee4d0c5952`
- Mode: **FROZEN**
- Snapshot start: `INFRA_SNAPSHOT_START.json` (61 protected files, 0 missing)
- Public-infra files changed by this issue: none.

## Reused assets and bundles

R2 bundles 108/109 were copied from `r2:genrec/108` and `r2:genrec/109` into the
issue-local `_r2/` directory with `rclone copy --checksum`; every file listed in
`results/research/issue139/r2_ledger.json` was SHA-256 checked. 108: 112/112 files,
109: 113/113 files, 0 missing, 0 mismatch.

The four pieces per bundle are present: the source directory, `item_sids.json`,
`rqvae_best.pth`, and `recall.json`. The recall block is identity-bound to the local
`test_final.json` and to the same Stage2 checkpoint/SID SHA-256.

## Frozen assets used read-only

- `assets/manifest.json` — `cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b`
- `assets/v8_runtime_layout/targets.npy` and `target_offsets.npy` — read through the
  model's `FrozenCurvature` path; no regeneration.
- `assets/v6_attention_tokens/token_schema.json` — read-only token schema.
- `results/research/behavior_graph/` and `behavior_graph_directed/` four-file contracts
  were not modified; the frozen structural sampler reads the packaged v8 target table.

## Baseline/identity table

| item | value |
|---|---|
| 108 arm | `TIGER_frozen_topology_gnn_rq` |
| 108 test R@10 | `0.05189853583801946` |
| 108 checkpoint sha256 | `46ad65fb148665257e7beddd3ee1b075946eb685743437978422a7913437178e` |
| 108 SID sha256 | `a4d7241a12d4e8e52710414bb47baf5441cd63456d4774c1e820eb177eba5184` |
| 109 arm | `TIGER_euclid_local_curve_supervision_rq` |
| 109 test R@10 | `0.052594926791900976` |
| 109 checkpoint sha256 | `1f8bfecba0c74bb3cbf9da11fe6925983cbae012753100401a7168e246da7bef` |
| 109 SID sha256 | `8961e4a2062ee8925d9b4f78494fd5dbf1be091a627abd8e527b1353c5506313` |

The local `test_final.json` for each arm points at the same checkpoint/SID paths;
their test recall is reproduced from the archived `recall.json` without retraining.

## Interpreter and hardware

- Analysis interpreter: `genrec_env_v2/bin/python3.9` for candidate imports/autograd.
- Static statistics: same interpreter or system Python; no GPU training is used.
- No port, launcher or distributed environment is touched.

## End-of-issue comparison

`INFRA_SNAPSHOT_END.json` will be compared with `tools.iteration.integrity.compare`;
the expected diff is empty.


## Final identity and endpoint evidence

- `verify_bundle.py` / `IDENTITY.json`: 108 196 checks, 109 198 checks, **0 failures**.
  All 112/113 R2 objects match the ledger; local source/result/checkpoint/SID,
  token cache, directed graph, curvature target and archived recall match.
  Each bundle has 28 R2-only workspace/metadata files (e.g. `tools/`, workspace
  `results/`, source tarball); these are retained in the R2 bundle and are not
  substitutes for the local runtime source.
- 108 endpoint: `C=0.7775872945785522`, `G1=12.757243390234594`.
- 109 endpoint: `C=1.5810964107513428`, `G1=7.112086922093524`.
- Both endpoint JSONs show identical fixed-batch and structural-sample hashes,
  unchanged parameter checksums, and `optimizer_steps=0`.
- End snapshot was compared with `tools.iteration.integrity.compare`:
  `{"added": [], "removed": [], "changed": [], "ok": true}` (`INFRA_DIFF.json`).
