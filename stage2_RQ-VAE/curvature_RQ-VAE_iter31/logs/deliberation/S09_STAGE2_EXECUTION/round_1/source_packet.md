# Iter31 S09 Stage2 Execution Plan — Frozen Source Packet

```text
STAGE_ID=S09_STAGE2_EXECUTION
ROUND=1
ITERATION=31
ROUND_TYPE=SCIENTIFIC_EXECUTION_PLAN
STATUS=FROZEN_FOR_INDEPENDENT_A_B_REVIEW
CANONICAL_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/stage2_execution_plan_iter31.md
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
USER_INPUT_REQUIRED=NO
PARALLEL_EXECUTION=YES
NO_STAGE2_AUTHORIZATION=TRUE
NO_STAGE3_AUTHORIZATION=TRUE
```

## Decision requested

Independently audit whether exactly one Iter31 Stage2 run can proceed under the already-registered HRA-STEP6-1 / FCCR-1 contract and root project rules. Review the exact hard-coded launch route, effective inputs, all output destinations, warm-start, full-step/no-gate policy, current S00–S08 results, and root §6 gradient evidence. Identify any concrete blocker or contradiction before recommending a Judge decision. This round decides launch-plan validity only; it does not itself authorize or execute training. The Judge may approve one Stage2 execution only after all facts and pending immediate prelaunch rechecks below are accepted.

No source/config/checker edit, CLI/environment override, baseline rerun, proxy metric gate, alternate seed, extra gradient/MVG run, Stage2/Stage3 run, GPU action, performance computation, or source-tree product creation is authorized in this deliberation.

## Upstream gate state

The current shared gate `check_stage` was run for S00–S08 after the authorized repairs. All pass:

| Stage | Current round | Verdict | Canonical artifact |
|---|---:|---|---|
| S00_SOURCE_TRUTH | 1 | MERGE_AB | `logs/source_snapshot_iter31.md` |
| S01_PROTOCOL_LOCK | 2 | REPAIR_PASS | `logs/protocol_manifest_iter31.md` |
| S02_HYPOTHESIS | 1 | MERGE_AB | `logs/hypothesis_iter31.md` |
| S03_PROVENANCE | 1 | MERGE_AB | `logs/mechanism_manifest_iter31.md` |
| S04_CONTRACT | 1 | MERGE_AB | `logs/mechanism_contract_iter31.json` |
| S05_ONE_FACTOR | 2 | REPAIR_PASS | `logs/one_factor_diff_iter31.md` |
| S06_IMPLEMENTATION | 1 | MERGE_AB | `logs/implementation_plan_iter31.md` |
| S07_PREFLIGHT | 4 | REPAIR_PASS | `logs/preflight_contract_iter31.log` |
| S08_MVG | 5 | REPAIR_PASS | `logs/mvg_check_iter31.log` |

S00–S08 PASS does not independently authorize Stage2. S09 Judge must decide the exact execution plan.

## Locked protocol and Stage2 execution invariants

- Iteration/protocol: Iter31; parent and sole comparator Iter29; `PROTOCOL_ID=ITER31_HRA_SINGLE_SEED42_INSTRUMENTS_2026-09REF_VS_ITER29`; dataset Amazon_2023_Instruments / 2026-09 Refactor. S01's comparator result is historical only (`test_R@10=0.05921064085377531`, `n_eval=57439`); no baseline rerun is permitted.
- Registered mechanism: HRA-STEP6-1 changes only the Step6 common-reference aggregation under inherited FCCR-1. No Stage2 mechanism or protocol changes are part of S09.
- Hard-coded settings in current source: `SEED=42`; `MAX_GLOBAL_STEPS=100000`; `BATCH_SIZE=640` per GPU; 4 ranks imply total batch 2560; 3 layers × 256 codebook entries; `CKPT_EVERY=10000`; `COMPILE=False`. Verify the source identity/settings again immediately before launch. The current launcher is hard-coded: 4-rank `torchrun --standalone`, `--master_port=50200`, script path from `__file__`; no CLI args or environment overrides.
- Required project launch from the Iter31 source directory, after S09 Judge approval and immediate preflight: `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 curvature_RQ-VAE.py` under the root-prescribed `nohup ... > logs/train_run.log 2>&1 &` route. The source internally launches the four-rank torchrun and writes `logs/train_migrated.log`. Do not substitute another interpreter, entry point, port, topology, CLI flag, or environment override. Runtime supervision must retain the exact command, CWD, launcher status, and both logs.
- Stage2 must run once to all `100000` global steps. Current `modules/sid_quality.py::should_early_stop` always returns `(False, "")`; Gini, collisions, unique-SID count, layer utilization, entropy, HR@50, and all other SID statistics are descriptive only. Do not add outer/inner stop logic or quality gates. The current FCCR preflight reports fixed `c_l=[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`, non-trainable, time-invariant, with no cyclic schedule or curvature regularization.
- Stage2 source path is `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`; product roots are results-only. The current Stage2 results root did not exist at this audit, and a glob of the source tree found no `.pth`, `.npy`, or `item_sids.json`. Recheck immediately before run.

## Current source/config route audit

`curvature_config.py` currently declares:

```text
_CONFIG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31
MECHANISM_NAME=iter31_hra_step6_common_reference
RQVAE_OUT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments
RQVAE_CKPT_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments/rqvae_best.pth
RAW_SIDS_NPY=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments/sids_raw.npy
SIDS_NPY=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/dataset/Instruments/sids_for_hgrec.npy
ITEM_SIDS_JSON=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json
```

`RQVAE_OUT_DIR` was checked against the required short root and `<N>=31`. Both Stage3 SID exports also point to the Iter31 results root; they are not inside the source iteration subtree. No route correction is indicated by the current file. Before the actual launch, repeat the exact required `grep RQVAE_OUT_DIR` check and confirm every output resolves below `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`, with no descriptive suffix in the directory name. Current `check_step0_clean_output` deletes only matching generated checkpoint patterns inside `RQVAE_OUT_DIR`; the whole Iter31 result root is currently absent, but inventory that exact output directory before execution so no existing user product is overwritten.

The main script consumes `ITEM_EMB_NPY`, `ITEM_IDS_JSON`, and `TRAIN_PARQUET` via `TransitionDataset`; it asserts dense ordered Stage1 item IDs and checks every training transition ID against that map. It loads the exact Iter8 checkpoint at Step4, skips only legacy `c_layer_scale` and `_fixed_c` tensors, and reports the transferred tensor count. Before launch, require the checkpoint to exist with the pinned hash; during the run require its exact path and a positive transferred tensor count. A missing checkpoint path would cause the script to warn and train from random initialization; that is not acceptable for this locked protocol.

## Contemporaneous current input identities

These resolved paths, sizes, mtimes and SHA-256 values were hashed from the live files for this S09 packet and match S01's historical SHA-256 declarations. `/home/...` config paths resolve through the same filesystem to the `/fs04/...` real paths below. Rehash size/mtime/SHA-256 immediately before launch; verify every configured/runtime consumer still resolves to the intended file. Any missing/mismatched identity blocks Stage2.

| Input | Resolved path | Size bytes | mtime_ns | SHA-256 |
|---|---|---:|---:|---|
| Stage0 train | `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet` | 11152937 | 1790172532000000000 | `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815` |
| Stage0 valid | `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/valid.parquet` | 2317087 | 1790172532000000000 | `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9` |
| Stage0 test | `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/test.parquet` | 2546378 | 1790172532000000000 | `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc` |
| Stage0 items | `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/items.parquet` | 10968789 | 1790172531000000000 | `6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3` |
| Stage1 embedding | `/fs04/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy` | 75531392 | 1790173394000000000 | `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` |
| Stage1 item-ID sidecar | `/fs04/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json` | 161001 | 1790173394000000000 | `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30` |
| Iter8 warm-start checkpoint | `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth` | 13780725 | 1790266017000000000 | `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b` |

### Current input/alignment audit

- Stage1 embedding: shape `[24587, 768]`, `float32`, all values finite. Stage1 item-ID sidecar: 24,587 entries, exactly dense ordered IDs `0..24586`.
- Stage0 `items.parquet`: 24,587 rows and 24,587 unique IDs; `item_id` minimum/maximum `0..24586`; parquet item order exactly matches the Stage1 sidecar.
- Stage0 interactions: train 396,958 rows / 2,698,387 history IDs; valid 57,439 rows / 396,958 history IDs; test 57,439 rows / 454,397 history IDs. Across each file, history/target contain no null IDs, have range within `0..24586`, and have zero out-of-range IDs.
- Re-run these identity/alignment checks immediately before Stage2 if any input file identity changes; do not rely only on historical manifest values.

## Root `CLAUDE.md` §6 gradient gate — already evidenced by S08

S08 `MVG PASS` was run on one checkpoint and one batch, seed 42, 640 ordered selected pairs, with no optimizer step. It directly reports the exact Iter8 warm-start path/hash, current fixed curvatures, and same-quantized-embedding HRA-vs-Euclidean effect: finite decoder inputs `[640,32]`, max absolute difference `0.0719804987`, L2 difference `2.01821685`, relative L2 `0.08106279`. No unregistered minimum effect threshold was applied. Curvature stayed invariant in train/eval snapshots and was excluded from AdamW parameters.

The unchanged S08 checker source explicitly requires `total_loss.requires_grad` and non-null `grad_fn`, finite total loss, and for each actual component (`reconstruction`, `quantizer`, `behavior_loss`) non-detached finite loss plus `autograd.grad(..., retain_graph=True, allow_unused=True)` with at least one relevant nonzero model-parameter gradient; it then runs `loss.backward()` and verifies nonzero encoder and codebook gradients. Current logged total loss is `5.1951842308`; nonzero gradients cover 11 encoder/codebook/decoder parameters, with nonzero component gradients for reconstruction, quantizer, and behavior loss. `hra_specific_auxiliary_loss=False`, so no untested HRA-specific loss exists. These are the root §6 checks; do not rerun GPU/MVG or add an unrelated gradient script absent a concrete defect.

## Required single-run evidence and post-run checks

If and only if the S09 Judge approves, execute the exact root-mandated launch once. Preserve and report:

1. Immediate prelaunch SHA-256/size/mtime/path checks for four Stage0 parquets, both Stage1 inputs and warm-start; item universe/order; Stage2 output-root/source-tree inventory; `grep RQVAE_OUT_DIR`; current code/config identity; actual interpreter/torch/CUDA availability.
2. Exact no-argument launch command, CWD, outer `logs/train_run.log`, inner `logs/train_migrated.log`, supervisor exit status, 4-rank topology, effective per-GPU/global batch, `SEED=42`, 100000-step target, positive exact checkpoint transfer, and successful terminal `global_step=100000`. Do not run a duplicate/retry under this plan; a failure needs a later authorized repair decision.
3. Confirm `rqvae_best.pth`, `sids_raw.npy`, `sids_for_hgrec.npy`, and `item_sids.json` (plus other actual final products) are present only beneath `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`, structurally valid, consistent with 24,587 Stage1 items, and not present in the source iteration tree. Record all descriptive SID metrics without gating or stopping.
4. Do not launch Stage3 from S09; Stage3 requires its separate S11 route/evaluation adjudication.

## Independent A/B scope

Each candidate independently reviews this frozen packet and primary sources. Verify current configuration and all named consumers, input hashes/alignment, root §6 S08 evidence, exact launch command and internal DDP launcher, no-override rule, output paths/cleanup scope, warm-start failure mode, full-step/no-gate policy, and run-once evidence requirements. State concrete blockers, source citations, confidence, and whether the plan is ready for Judge approval. Do not read the other candidate. Do not edit source/artifacts other than your own candidate file. Do not run GPU checks, training, preflight commands, Stage2, or Stage3.

The Judge must adjudicate A/B and produce `logs/stage2_execution_plan_iter31.md` plus `round_1/judge.md`, stating exactly whether one Stage2 run is authorized. A `REPAIR_AND_RERUN`, `REJECT_BOTH`, or `ABORT_ITERATION` verdict means no launch until separately resolved. S09 approval, if granted, authorizes only the one Stage2 run and no Stage3 work.