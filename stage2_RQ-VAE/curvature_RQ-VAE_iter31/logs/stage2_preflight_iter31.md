# Iter31 Stage2 immediate prelaunch record

`CAPTURED_AT=2026-09-27T19:05:19+10:00`
`PRELAUNCH_CONDITIONS_PASS=YES`
`STAGE2_RUNS_BEFORE_LAUNCH=0`
`STAGE3_AUTHORIZED=NO`

## Deliberation gate

`COMMAND=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py`
`CWD=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31`
`EXIT_STATUS=0`
`RESULT=DELIBERATION_GATE_PASS`

```text
DELIBERATION_GATE_PASS
iter=31
phase=PRE_STAGE2
S00_SOURCE_TRUTH: round_1 MERGE_AB -> source_snapshot_iter31.md
S01_PROTOCOL_LOCK: round_2 REPAIR_PASS -> protocol_manifest_iter31.md
S02_HYPOTHESIS: round_1 MERGE_AB -> hypothesis_iter31.md
S03_PROVENANCE: round_1 MERGE_AB -> mechanism_manifest_iter31.md
S04_CONTRACT: round_1 MERGE_AB -> mechanism_contract_iter31.json
S05_ONE_FACTOR: round_2 REPAIR_PASS -> one_factor_diff_iter31.md
S06_IMPLEMENTATION: round_1 MERGE_AB -> implementation_plan_iter31.md
S07_PREFLIGHT: round_4 REPAIR_PASS -> preflight_contract_iter31.log
S08_MVG: round_5 REPAIR_PASS -> mvg_check_iter31.log
S09_STAGE2_EXECUTION: round_2 REPAIR_PASS -> stage2_execution_plan_iter31.md
```

A prior gate attempt failed at S09 because the round_1 operational record lacked `CHECKS_PASS=YES` and a gate-valid repair Judge verdict. No run started. The same-iteration S09 round_2 operational repair records that exact failure, preserves the original round_1 A/B/MERGE_AB decision, and has `VERDICT=REPAIR_PASS`. The required gate was rerun after that repair and passed as above; the shared checker was not changed.

## Upstream inputs: fresh identity and consumer checks

All paths resolved to the pinned `/fs04` paths. Each observed size, mtime, and SHA-256 matches the frozen S09 source packet and execution plan.

| Input | Resolved path | Bytes | mtime_ns | SHA-256 |
|---|---|---:|---:|---|
| Stage0 train | `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet` | 11,152,937 | 1790172532000000000 | `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815` |
| Stage0 valid | `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/valid.parquet` | 2,317,087 | 1790172532000000000 | `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9` |
| Stage0 test | `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/test.parquet` | 2,546,378 | 1790172532000000000 | `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc` |
| Stage0 items | `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/items.parquet` | 10,968,789 | 1790172531000000000 | `6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3` |
| Stage1 embedding | `/fs04/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy` | 75,531,392 | 1790173394000000000 | `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` |
| Stage1 item-ID sidecar | `/fs04/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json` | 161,001 | 1790173394000000000 | `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30` |
| Iter8 warm start | `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth` | 13,780,725 | 1790266017000000000 | `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b` |

Consumer audit: `curvature_config.py` defines these Stage0/Stage1 routes. `curvature_RQ-VAE.py:87-90` imports the embedding, item-ID sidecar, and train parquet; `TransitionDataset` at lines 151-174 loads the embedding and sidecar, reads train `history`/`target`, and validates each source/target ID. Valid/test are pinned input identities but are not read by the live Stage2 training loader; `items.parquet` is used here as an independent item-order check.

Fresh CPU input audit: embedding is `float32`, shape `[24587, 768]`, all finite; sidecar is exactly dense ordered IDs `0..24586`; `items.parquet.item_id` has 24,587 rows in that same dense order. Train parquet has 396,958 rows and 339,519 non-empty-history transitions; every target and every `history[-1]` source is in `0..24586` (zero invalid IDs). No input substitution.

## Current source/config identity union

All 19 current SHA-256 values equal the corresponding S09-plan identity. This includes the original execution-plan table and its ten added active HRA/FCCR/model/data/contract dependencies.

| File | SHA-256 |
|---|---|
| `curvature_config.py` | `6f2f645202417118baec3ef28af879c14bda66723b42950d5b67056c2ed735bb` |
| `curvature_RQ-VAE.py` | `b641d8b5c9897f520482d0bce5526bdcd687ae5845901f8f0be238772f49d1dd` |
| `modules/rqvae.py` | `cf878675aa4550f96b4b13480a1f9d03928549ebd0a720675cbd6f1713e28e21` |
| `modules/quantize.py` | `2970ad3e61de79c6a2fd17194258d7d5ed4b9cbbf836e5dde506a8635c2553f8` |
| `modules/step_checks.py` | `03afd7f17600043c9e577e214659890571847e4de389add70eb75a3ee16caf40` |
| `modules/sid_quality.py` | `3387d7ad2211a923b5376b5606b66411d8fbc89469baa1274f906244a5923511` |
| `scripts/mvg_check.py` | `abf4ad518890e345c05e8b0907349067faa94cd0ef39d6dab6384bf20792318d` |
| `logs/mvg_check_iter31.log` | `49318b59588ce9339e2173956733bf7b699fac4d5d113b3921bb77e7e8b59031` |
| `logs/preflight_contract_iter31.log` | `e52df0e6db569c3d747ba98f093487c1a755bd991486868c397d74d2b54c7e58` |
| `modules/hyperbolic.py` | `ca7d4fdb5b7f6d4f91cc6e5083462fa9d610386ce4bda65605d3672907e22ae6` |
| `modules/loss.py` | `c5d0106b731ee54b9e96ef40db745f7ad74725e2f76ed9c6b763e77e11108cd1` |
| `modules/encoder.py` | `139a2cfa9fad4ecf9b795d435a640a927dd01a2d52911cea7c58a8de53c0a475` |
| `modules/normalize.py` | `e063b3cc49e1d9571c6388251212054e269de43e2794e28238021fe9ccca67f6` |
| `data/schemas.py` | `01054d7e928b9392391f85a5e5a2feea78de3fc2c7054c871715d0658c1e4938` |
| `init/kmeans.py` | `7109304a3c7408d49872a079763d9c0b05820e8475f6db4d6ded0ad2e24e5e15` |
| `scripts/compute_closed_form_curvature.py` | `5c72150f2e5484c4a1e546ba1df157e7151a4f5421f52fe45fcf10402e6d8767` |
| `scripts/computed_behavior_branching.json` | `cceb416b158f2f9d2c3a12dc92ed1d0c08f2d02f6db51d365f3ac7a0bfafcedc` |
| `logs/mechanism_contract_iter31.json` | `0ea83b1f55916d02fa3eed97e02d4c39f46d7229a0afa76d5fe1262d96459b32` |
| `logs/hra_step6_contract_iter31.json` | `3e7394e91ed3f31ec4d2753e6d634c6e99590e18e5e4da98d975df0c2ae5cf17` |

## Warm start and gradient path

S08's pinned one-checkpoint/one-batch `MVG PASS` and root `CLAUDE.md` §6 gradient checks remain valid; they were not repeated. The matching `logs/mvg_check_iter31.log` records the exact Iter8 checkpoint identity, 640 ordered pairs at seed 42, `warm_start=PASS`, the three expected missing `_fixed_c` buffers only and no unexpected keys, identical FCCR values across recorded snapshots, `total_loss=5.195184230804443`, nonzero total/component gradients through encoder, decoder, and all three codebooks, fixed curvature excluded from the optimizer, and `optimizer_step_performed=False`.

## Output routes, overwrite safety, and capacity

`curvature_config.py:10-34` confirms `MECHANISM_NAME=iter31_hra_step6_common_reference` and the short Iter31 results root. `RQVAE_OUT_DIR` is `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments`; `RQVAE_CKPT_PATH` and `RAW_SIDS_NPY` are below it. `SIDS_NPY` is `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/dataset/Instruments/sids_for_hgrec.npy`; `ITEM_SIDS_JSON` is `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json`. The required `RQVAE_OUT_DIR` grep check returned the correct `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31` route.

Current `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31` is absent, so `OUT_DIR`, both external exports, and all Step0 deletion targets are absent; there is no user/result file to overwrite. `logs/train_run.log` and `logs/train_migrated.log` are absent. The source-iteration tree contains no `.pth`, `.npy`, or `item_sids.json` products. The trainer has no references to shared `SAVE_DIR_ROOT`, `CONFIG_PATH`, or `BEST_CKPT_PATH` and does not write to those shared decoder routes. Stage3 Iter31 results root is also absent and is not authorized by this record.

Step0 cleanup patterns checked in `modules/step_checks.py`: `rqvae_step*.pt`, `rqvae_best.pt`, `rqvae_best.pth`, `rqvae_final.pt`, `sids_step*.npy`, `sids_for_hgrec*.npy`, `sids_raw.npy`, `sids_final.npy`, `item_sids.json`, `quality_step*.json`, `quality_best.json`; they apply only within `RQVAE_OUT_DIR`. The trainer does not delete the whole directory. `/fs04` has 94,042,596 KiB available (83% capacity used).

## Runtime and conflict check

- Prescribed Python path exists/executable; direct version check: Python 3.9.25.
- Prescribed torchrun path exists/executable.
- Probe through the prescribed interpreter exited 0: torch `2.8.0+cu128`, CUDA available, four visible NVIDIA L40S devices.
- `nvidia-smi` reported all four 46,068 MiB GPUs at 0 MiB used and 0% utilization.
- No `curvature_RQ-VAE.py` or port-50200 torchrun process matched the process query (no output, exit 1); `ss -ltnp 'sport = :50200'` returned no listener.

## Prelaunch decision

`ALL_CANONICAL_IMMEDIATE_CHECKS=PASS`
`CONDITIONAL_S09_SINGLE_STAGE2_RUN=ELIGIBLE`
`LAUNCH_COMMAND=nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 curvature_RQ-VAE.py > logs/train_run.log 2>&1 &`
`LAUNCH_CWD=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31`

This record activates only S09's single conditional Stage2 run, with no CLI arguments, no environment overrides, no alternate route, no retry, and no SID-quality gate. It does not authorize Stage3.
