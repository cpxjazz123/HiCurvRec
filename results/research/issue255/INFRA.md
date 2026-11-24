# Issue255 infrastructure receipt

The asset lookup preceded implementation (COMMENT_01/02 and RUN_LOG). Existing assets were reused, not rebuilt. Paths below are repository-relative. Full per-file sizes, dtypes, shapes, origins and SHA-256 values are in `assets/manifest.json` (`cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b`) and the final checkpoint's pre-training provenance.

## Reused inputs

| Input | SHA-256 |
|---|---|
| `stage1_GeneEmbedding/output/sentence_t5.npy` | `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` |
| `results/stage0_build_parquet/train.parquet` | `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815` |
| `results/stage0_build_parquet/valid.parquet` | `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9` |
| `results/stage0_build_parquet/test.parquet` | `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc` |
| `assets/train/visible_forward.npz` | `ac7155d1424a6a346da9b8ed6e9413b81d43223a377308af0c8ba35b6b773029` |

Reused by immutable manifest entry: global item IDs and named training-visible graph; complete v8 one/two-hop runtime layouts and target table; v9 in/out ego operators; v5 six-dimensional static features and frozen normalization; all curvature shards, component moments and summaries. Sampling, sparse adjoints and Triton computation reuse `tools/issue252/` and `tools/issue253/`. Held-out banks are evaluation-only and never model inputs. Test-derived rows remain under the evaluation area.

## New asset, frozen before the formal run

Only the missing direction populations were built, in `assets/v10_direction_ego/{in_only,out_only}/`. Each population has 24,587 tokens, 285,719 node positions and 695,377 real induced directed edges. Nine arrays per population encode nodes, offsets, counts and both CSR directions. CSR pointers have one entry per node position plus one; item offsets have 24,588 entries. No existing asset was copied or overwritten.

- Generation command: Stage2 Python `-m tools.issue255.build_direction_ego`; observed exit 0 before formal dispatch. Do not rerun it over the frozen version.
- Independent all-item/all-edge audit: both variants PASS, mismatch 0; `assets/v10_direction_ego/audit.json` SHA-256 `fff152cd88f18c71d05f06faff68bef2c6e43b311c62d5d7d91c94c47ca253dd`.
- Metadata SHA-256: `7fc4f408bc910e3800b18842992726a86e645ec49c651bdbe8b5e7ed8dfa85b5`.
- Reload/hash and known-answer tests: `tools/issue255/tests/test_direction_ego.py` (11 tests in the observed tools-suite run).
- Coverage correction to COMMENT_02 wording: 116 graph-isolated items; 218 in-empty and 292 out-empty root-only tokens. In/out populations overlap at 3,539 centres and 11,800 neighbour incidences. Induced reciprocal pairs: 80,063 / 83,645. They are not disjoint populations.
- Arrays remain in the persistent shared asset tree, as with v8/v9. Metadata, audit, coverage and manifest are tracked; binary arrays are ignored by the asset-local `.gitignore`.

## Source and runtime identity

Before new-arm implementation, `TIGER_three_expert_attention_rq` was moved intact into `stage2_RQ-VAE/_archive/`; baseline and the archive container were not moved. The new arm is an independent copy of the #253 lineage explicitly authorized by issue255; 768/32, three 256-word levels, three experts and the actual assignment protocol remain frozen. No baseline retraining was performed.

- Stage2 interpreter: `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9`.
- Stage3 interpreter: `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10`.
- Intake hardware: four NVIDIA L40S; the real benchmark/profile and formal run used four ranks via the sanctioned launcher, port50202/global4096/seed42. Stage3 uses its separate frozen protocol and port50201.
- Invalid first run: `issue255_formal`, immutable `results/stage2_RQ-VAE/_runs/issue255_formal/TIGER_prefix_curvature_gate_rq/{src,result}`. All3000 epochs completed, launcher exit0, but the calibration numerator used reconstruction-only rather than the frozen full original RQ-VAE loss. `INVALID_IMPLEMENTATION.json` excludes it from scientific results. Its snapshot and artifacts remain intact.
- Frozen mechanism contract SHA-256: `a47efec99b0dff2e03adcfdbb113af925693927736382630e5a1d496de7eb7d5`.
- Frozen acceptance JSON SHA-256: `002484f5acf1427b36f95b8cc27f7b435150e8febfa50fa8b7aa17604412932c`.
- Full pre-dispatch test evidence was not saved for that invalid run. COMMENT_04 records that procedural deviation; post-formal checks are not backdated. COMMENT_07 and `OWNER_APPROVAL.json` authorize exactly one corrected generation, `issue255_formal_r1`, only after current-source test, admission, actual-loss four-rank and performance credentials pass. Active-tree repairs never modify the invalid snapshot.

## Historical comparison configuration

Appending v10 changes the shared registry bytes, not the old assets. The #253 checkpoint pins the previous manifest SHA-256 `8c58d34521181925df2e66dbe01f64b7150fb56a1eab028f59e8b4832c381ae6`. `tools/issue255/historical.py` retrieves those exact small configuration bytes from immutable revision `6d96efe092547277173906b152f2d5b3e9172ced`; every old runtime leaf, executed source, SID, export and checkpoint must still match its original hash. It neither reconstructs assets nor permits changed runtime inputs. The actual #253 epoch3000 checkpoint and completed Stage3 population57,439 passed this strict read-only compatibility check before corrected dispatch.

