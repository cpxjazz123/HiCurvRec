ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S01_PROTOCOL_LOCK/round_1/source_packet.md
STAGE_ID=S01_PROTOCOL_LOCK

# Iter30 S01 protocol-manifest proposal and rationale

## Proposed protocol lock (prospective; not execution authorization)

```text
PROTOCOL_ID=FCCR-1_iter30_matched_seed_iter26_vs_iter29_mapping_replication
STATUS=PROPOSAL_PENDING_JUDGE_AND_CANONICAL_GATES
EXPERIMENT_TYPE=single_factor
DATASET_VERSION=Amazon_2023_Instruments / 2026-09 Refactor
PARENT_ITER=iter29
PARENT_COMMIT=ecd01e4712a1badd38a0338255f4b2ec7b030aff
PARENT_ROOT_COMMIT=ecd01e4712a1badd38a0338255f4b2ec7b030aff
DIRECT_CONTROL_ITER=iter26_mapping_replicated_at_each_new_seed
CANONICAL_BASELINE_ITER=iter26
CANONICAL_BASELINE_TEST_FINAL=results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json
CANONICAL_BASELINE_TEST_R@10=0.057017009349048554
HISTORICAL_REFERENCE_ITER=iter29
HISTORICAL_REFERENCE_CLASSIFICATION=HISTORICAL_CONTEXT_ONLY; not a replacement for within-seed direct control
USER_SUCCESS_CRITERION=test_recall@10 >= 0.065

STAGE1_EMBEDDING_PATH=stage1_GeneEmbedding/output/sentence_t5.npy
STAGE1_EMBEDDING_SHA256_AT_S00_S01_CHECK=6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb
STAGE1_ITEM_ID_SIDECAR_PATH=stage1_GeneEmbedding/output/item_ids.json
STAGE1_ITEM_ID_SIDECAR_SHA256_AT_S00_S01_CHECK=3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30
STAGE0_TRAIN_PATH=results/stage0_build_parquet/train.parquet
STAGE0_TRAIN_SHA256_AT_S00_S01_CHECK=80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815
STAGE0_VALID_PATH=results/stage0_build_parquet/valid.parquet
STAGE0_VALID_SHA256_AT_S00_S01_CHECK=48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9
STAGE0_TEST_PATH=results/stage0_build_parquet/test.parquet
STAGE0_TEST_SHA256_AT_S00_S01_CHECK=5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc
STAGE0_ITEMS_PATH=results/stage0_build_parquet/items.parquet
STAGE0_ITEMS_SHA256_AT_S00_S01_CHECK=6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3
STAGE2_WARMSTART_PATH=results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth
STAGE2_WARMSTART_SHA256_AT_S00_S01_CHECK=189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b

STAGE2_SOURCE_ITER29_SHA256_AT_PACKET=2469b193bee1c4352ba180540c305c9ea837043afafd1b1585ea7b8aff805f4d
STAGE2_CONFIG_ITER29_SHA256_AT_PACKET=b41bf1fff2b481abf846d71a43c30c7ba34d0d25bd93bc04f70df486e3163f7a
STAGE2_SOURCE_ITER26_SHA256_AT_PACKET=ed629824d5a0308f4018b6dbaaa24303690a3597def7d1bcbf9abd38c3aca10f
STAGE2_CONFIG_ITER26_SHA256_AT_PACKET=4d6b3e06aa2f408a00664d971d64ed83d0e88ef5f88b852be97302a0e2786086
STAGE2_SEEDS=43,44,45
STAGE2_MAX_GLOBAL_STEPS=100000
RQ_LAYERS=3
CODEBOOK_SIZE_PER_LAYER=256
STAGE2_INPUT_DIM=768
STAGE2_HIDDEN_DIMS=[512,256,128]
STAGE2_EMBED_DIM=32
STAGE2_BATCH_SIZE_PER_GPU=640
STAGE2_WORLD_SIZE=4
STAGE2_COMMITMENT_WEIGHT=1.0
STAGE2_SINKHORN_EPS=0.05
STAGE2_SINKHORN_ITERATIONS=3
STAGE2_OPTIMIZER=AdamW(lr=1e-3,weight_decay=1e-4)
STAGE2_WARMSTART_REQUIRED=iter8 checkpoint above; fail closed; positive compatible loaded-tensor count required
STAGE2_QUALITY_METRICS=DESCRIPTIVE_ONLY; no proxy gate or early stop
STAGE2_GRADIENT_CHECK=REQUIRED_SEPARATELY_BEFORE_EACH_OF_SIX_ACTUAL_RUNS
STAGE2_FIXED_CY/PARAMETER/LOSS_POLICY=FCCR-1 only; unchanged except registered mapping; fixed non-trainable c
STAGE2_ENTRY=from iter30 directory, approved absolute genrec_env_v2 Python 3.9 + `curvature_RQ-VAE.py` with no arguments; code's internal hard-coded torchrun launcher
STAGE2_NPROC=4
STAGE2_TORCHRUN_PORT=50200 (serial; no concurrent runs)

STAGE3_ROOT_CODE_COMMIT=ecd01e4712a1badd38a0338255f4b2ec7b030aff
STAGE3_CODE_COMMIT=ecd01e4712a1badd38a0338255f4b2ec7b030aff
STAGE3_TRAINER_SHA256_AT_PACKET=9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb
STAGE3_ITER29_WRAPPER_SHA256_AT_PACKET=1cab47ff035e69aecf3aa5a2e1cfe7777c114a60b753b5e935453d90271b2ec7
STAGE3_SEEDS=43,44,45
STAGE3_EPOCHS=150
STAGE3_BEAM_SIZE=20
STAGE3_N_EVAL=57439 (locked expected test-set cardinality; record actual output)
STAGE3_TOPK=[5,10]
STAGE3_NO_EVAL=True
STAGE3_SKIP_TEST=False
STAGE3_BATCH_SIZE=4096
STAGE3_INFER_BATCH_SIZE=1024
STAGE3_LAUNCHER_NPROC=4
STAGE3_TORCHRUN_PORT=50201 (serial; no concurrent runs)
STAGE3_METADATA_CAVEAT=iter29 wrapper's intended variant was logged as unknown_variant; preserve/disclose; no trainer modification

SEED42_POOLED_ANALYSIS=EXCLUDE; historical pair does not meet fully evidenced prospective identity/protocol equivalence requirements
RAW_RESIDUAL_PROVENANCE=MEDIUM_CONFIDENCE_HISTORICAL_METHOD/VALUE_ONLY; pending S03 hard gate; not checkpoint reproduction or historical byte identity
NORMALIZED_LAYER_SCALES=[0.001,0.932889,1.0]; NOT raw residual; forbidden substitution
```

The user's inclusive `>= 0.065` criterion controls; root `CLAUDE.md` and skill also contain lower-priority strict `> 0.065` wording. Preserve the discrepancy in reports, but classify each completed protocol-valid run against the inclusive user criterion. The historical iter26 baseline result is context and fixed reference only. In this replication, the direct causal comparator for each iter29-mapping run is the newly run iter26-mapping arm at the same pair seed, not the old seed-42 score.

## Registered arms and exact six prospective runs

Use only the two already approved FCCR-1 mappings with the approved L0/L1/L2 inputs, conditional on S03 accepting those inputs:

- `iter26_mapping`: `s_l=log1p(B_l)/log1p(m_l_raw/min(m_raw))`; `z_l=(s_l-mean(s))/(std(s)+1e-12)`; `c_l=clip(0.5*exp(0.2*z_l),0.05,1.5)`. Fixed vector `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`.
- `iter29_mapping`: `c_l=0.05+1.45*0.5*(B_l/(B_l+2.0)+m_l_raw/(m_l_raw+0.1))`. Fixed vector `[1.3660953164241916,0.7347829661951981,0.6439958072706683]`.
- Inputs shared by both: `B=[19.324911558712664,1.4605688962651735,1.0148104414712726]`; `m_l_raw=[1.0,0.10941,0.09331]`. These are prospective constants under S14, not permission to reinterpret or replace them. S03 must adjudicate source/semantics and preserve the historical evidence ceiling.

Pair order is serial by seed, iter26 mapping then iter29 mapping, then next seed. Exact immutable run labels and artifact roots:

| Pair | Run label | Stage2 root | Stage3 root | Stage2 seed | Stage3 seed |
|---|---|---|---|---:|---:|
| 43 | `iter26_mapping_seed43` | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` | 43 | 43 |
| 43 | `iter29_mapping_seed43` | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` | 43 | 43 |
| 44 | `iter26_mapping_seed44` | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` | 44 | 44 |
| 44 | `iter29_mapping_seed44` | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` | 44 | 44 |
| 45 | `iter26_mapping_seed45` | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` | 45 | 45 |
| 45 | `iter29_mapping_seed45` | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` | 45 | 45 |

Within each row pair, both Stage2 and Stage3 seed values match. The only conceptual within-pair difference is the fixed mapping. Do not compare the old iter26/iter29 seed-42 single observations as substitutes for any row.

For each Stage2 root, place normal Stage2 files beneath the run root (for example `out/rqvae/instruments/rqvae_best.pth`, `sids_raw.npy`, `item_sids.json`, plus the existing Stage2 SID export path under that run root). For each Stage3 root, use `logs/` and `ckpt/` beneath the run root; the unchanged trainer will make its dataset/time-stamped evaluation subdirectory and `test_final.json` under that run's `logs/`. These are future paths, not existing outputs. Keep all generated `.pth`, `.npy`, `item_sids.json` outside the iter30 source tree. Retain full `MECHANISM_NAME=iter30_<mapping-and-seed-description>` as needed for Stage3 metadata, but the result-root component remains exactly `curvature_RQ-VAE_iter30/<run_label>`.

## Parent, direct control, and historic baseline semantics

The implementation parent is iter29 at the packet/S00-recorded current parent root commit `ecd01e4712a1badd38a0338255f4b2ec7b030aff`; it supplies the current implementation lineage, shared fixed FCCR-1 baseline path, and Stage3 code lineage. This does not make its mapping the experimental baseline. The active two-arm direct control is the iter26 mapping, run prospectively at each of seeds 43/44/45. The canonical historical baseline remains iter26's exact seed-42 `test_final.json` and `R@10=0.057017009349048554`, retained for historical reporting and target context. Iter29's seed-42 result (`0.05921064085377531`) is historical evidence for the second mapping, not canonical baseline. Iter18 remains `HISTORICAL_NONCOMPARABLE` and is not ranked or pooled.

## Input identities, actual use, and recheck boundary

At S00/early S01 packet time, reported fresh hashes matched the iter29-locked values listed above. They establish file-byte identities at that check only—not future identity, resolved runtime paths, nor actual use. The iter29 configuration declares Stage1 embedding and item-ID paths and Stage0 train/valid/test/items paths. Iter29 Stage2's `TransitionDataset(EMB_NPY, ITEM_IDS_JSON, TRAIN_PARQUET)` directly consumes embedding, item-ID sidecar, and training parquet; evaluation/data pipeline consumes the validation/test/item parquet paths through configured data construction (must verify exact consumer path on iter30). Stage3 consumes each run's own Stage2 SID file via its wrapper's `CODE_PATH`; unchanged Stage3 code performs training/evaluation with Stage0 split data. Before launch, rehash every locked file, trace resolved paths in the selected hardcoded configuration/wrapper and consumers, and verify the exact Stage2 SID file produced by each arm is the Stage3 input. Record runtime-resolved paths and successful load/use evidence per run; declarations and current hashes alone are insufficient.

Warm start is the iter8 `rqvae_best.pth` checkpoint with packet-reported SHA256 `189e0aff...56c3b`. Iter29 code loads its model tensors, skips legacy `c_layer_scale` and `_fixed_c` buffers, and keeps current fixed curvature. But current code explicitly warns and continues random initialization when the file is absent. Therefore all six runs require a fail-closed preflight and runtime assertion: exact file/hash before each run, no missing/mismatch fallback, reject malformed/incompatible state, and require a positive, recorded count of actually loaded compatible tensors on every run before training proceeds. This is a prerequisite implementation correction to the existing behavior, not a warm-start or mechanism change. Do not replace or rewrite the checkpoint.

The S14 provenance ceiling is binding. Iter29's branching values trace to iter8 raw SID plus Stage0 data, but the cited iter8 raw SID file is absent in current inventory. Raw-residual medians trace to historical iter1 calibration log/checkpoint, but historical checkpoint/raw input byte identity and independent replay are unavailable. Thus current evidence supports historical method/value provenance only at medium confidence, not a reproduced raw residual measurement. S03 must independently accept this semantic/source/value evidence or fail closed before MVG/Stage2. If the active skill's hard provenance gate cannot accept the evidence at that ceiling, report `PROVENANCE_INVALID` / abort path; do not replace raw residual with normalized scales, recompute from a different source, or narrow the six-run design.

## Seed semantics and fixed settings

Stage2 iter29 source currently hardcodes `SEED=42`; in `main`, each rank seeds Torch and NumPy with `SEED + rank`, and `DistributedSampler` uses `seed=SEED` (with epoch set from global step). Thus the planned Stage2 value 43/44/45 must be a hardcoded source value for the corresponding serial run, and both arms at each pair must receive the same value. Config's `PYTHONHASHSEED=42` and CUBLAS workspace are deterministic-runtime settings, not substitutes for the training seed; keep runtime config identical across all six runs. Stage2's four-rank torchrun launcher currently hardcodes port 50200 and `logs/train_migrated.log`; it is safe only with serial execution and per-run log paths. The script also clears stale files in its configured output directory at startup, so every run must have its own unique empty root and never point Step0 cleanup at a prior run's root.

Stage3 trainer currently hardcodes `SEED=42`; its `set_seed(seed)` seeds Python `random`, NumPy, and Torch, then Stage3 training consumes `SEED` (including distributed sampling/trainer configuration). For S14, hardcode 43/44/45 into the selected per-run wrapper's trainer seed before the wrapper invokes the existing launcher; both arms in a pair use equal Stage3 seed. Do not use CLI arguments or environment overrides. The wrapper must also bind the exact run's SID input, `LOG_PATH`, `SAVE_PATH`, and launcher log to that run's Stage3 root. Stage3 port 50201 likewise requires serialized jobs. Preserve 150 epochs, beam 20, `TOPK_LIST=[5,10]`, `NO_EVAL=True`, `SKIP_TEST=False`, 4-rank launcher, evaluation set and metric. Record actual `n_eval`; expected 57439 is not permission to assert a result in advance. Disclose existing `unknown_variant` metric metadata caveat without changing Stage3 trainer.

The no-CLI/no-environment mechanism is explicit hardcoded per-run source/config/wrapper selection, not parameterized runtime selection: for each row, install/select a reviewed hardcoded iter30 configuration with its fixed mapping, Stage2 seed, output roots and unique train log; the matching hardcoded Stage3 wrapper fixes its Stage3 seed, SID input, run-specific log/checkpoint roots and launcher log. Invoke serially from the iter30 source directory using the root-mandated absolute genrec_env_v2 Python 3.9 executable and bare `curvature_RQ-VAE.py` entry (no args); let its internal four-rank torchrun launcher run. Invoke the Stage3 wrapper with the root-mandated genrec_env Python 3.10 executable, no args; its internal hardcoded four-rank launcher runs. Do not choose runs through environment variables, CLI flags, or user input. Before every run verify no run-root artifact exists (fail, do not overwrite), resolved short roots match CLAUDE.md, `RQVAE_OUT_DIR` points under the exact Stage2 run path, Stage3 `LOG_PATH`/`SAVE_PATH` under exact Stage3 run path, and the fixed port is free. If necessary to avoid port collision, wait for the serial predecessor to exit; do not change port as a runtime override. Keep launcher logs unique per run. This operational prescription depends on a later judged hardcoded implementation/configuration and is not itself permission to make it now.

After every canonical S00–S09 gate, and only after source/config contract and MVG gates pass, run the root-required one-checkpoint/one-batch nonzero gradient pathway check immediately before each of the six Stage2 runs: `total_loss.requires_grad`, non-null grad_fn, finite nonzero gradients for intended model parameters and applicable losses, and no silent detach. Fixed curvature itself must not require gradients. A failed check blocks that run; it does not authorize skipping a pair or altering the mapping.

## Seed-42 pooling decision

Exclude the historical seed-42 pair from the prospective paired estimate. Although iter26/iter29 records have the same `n_eval` and their S01 establishes a historical direct-control relationship, seed-42 source/config hashes differ from the prospective iter30 implementation, and S00 explicitly says current hashes do not prove runtime path/use. The available evidence does not establish all compatibility necessities for pooling: exact Stage2 runtime/config identity and all input consumption across historical and new runs, reproducible seed semantics at each stage, exact Stage3 wrapper/runtime and evaluation identity, and complete per-run output provenance. Similar reported settings and equal evaluation cardinality do not prove full protocol equivalence. Keep those observations as contextual historical results only; no seed-42 weighted/pooled estimate and no substitution for pairs 43–45.

## Prospective analysis, stopping, and abort rule

For each completed valid pair `s ∈ {43,44,45}`, record both arms' actual R@5, R@10, NDCG@5, NDCG@10, `n_eval`, checkpoint/SID paths and hashes, and compute the signed within-pair primary difference:

`d_s = R@10(iter29_mapping_seed_s) - R@10(iter26_mapping_seed_s)`.

Report the three paired values, arithmetic mean paired difference, median and range; report each arm's mean and sample standard deviation across its three runs and the six individual scores against `>=0.065`. The paired estimate is the arithmetic mean of the three `d_s`; show all pairs, not only the mean. Also report each run's absolute delta from historical canonical baseline `0.057017009349048554` as descriptive context, never as a substitute for pairing. Three pairs are modest; do not claim precise uncertainty, statistical power, guaranteed target attainment, or elimination of mapping-by-seed interaction. Do not pool seed 42.

The block is complete only after all six valid Stage2→Stage3 pipelines finish and all six actual test records are audited. Do not stop early for a score crossing/failing target, a paired trend, proxy metric, or expected futility; Stage2 SID/Gini/collision/entropy statistics are descriptive only, and all non-aborted valid arms must reach unchanged Stage3. The inclusive user target classifies each valid completed run; report arm-level means separately and do not invent a target rule such as the mean replacing per-run outcomes. No interim score authorizes dropping an arm or pair.

Stop before any full run if canonical deliberation, contract/preflight, S03 provenance, warm-start fail-closed, gradient, path/identity or other applicable hard gate fails. A run failure, missing/mismatched warm start, invalid export, nonfinite training, contract violation or overwrite risk blocks progression and requires the skill's adjudicated invalid/abort handling; preserve evidence and do not rerun into or overwrite a claimed path. The six-run S14 block may not be narrowed, replaced with seed 42, or repaired by changing mapping, raw inputs, Stage1/Stage3, or other scientific factors. Any genuine infeasibility requiring such a change is a blocker/abort of this registered design, not permission for a reduced comparison. Do not report absent Stage3 outputs as if they exist.

## Genuine remaining gates and execution status

This proposal resolves seed routing operationally using serial, hardcoded per-run configuration and run roots; the existing Stage2/Stage3 entrypoints alone are hardcoded for seed 42 and one output root and cannot execute all six safely unchanged. The prescribed hardcoded routing/warm-start fail-closed changes must be reviewed and canonically authorized in their later stages; no CLI or environment selector is acceptable. Current source's absent-warmstart random-init branch must be made fail-closed before execution. Current hashes are packet-reported check-time identities, not launch-time or runtime-use proof. Most importantly, raw-residual evidence remains medium-confidence historical provenance and its acceptance under skill §7 is unresolved until S03. If S03 deems it insufficient, S14's fixed raw inputs and FCCR-1 gate leave no permitted substitute; block/abort rather than proceed with normalized scales or a reduced experiment.

No Stage2 or Stage3 training, GPU work, code/config edit, or output creation is authorized by this S01 candidate. Stage2 and Stage3 `test_final.json` paths above are planned only. Execution requires Judge C's canonical S01 lock and every subsequent required S00–S09 A/B/Judge, contract preflight, MVG, provenance, per-run gradient, and deliberation gates to pass.

## Evidence anchors

- Active skill §§2.1, 2.7, 3–4, 7, 9–10, 12, 14, 17–19; repository `CLAUDE.md` §§0–3, 6, 10–11.
- Iter30 S01 source packet: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S01_PROTOCOL_LOCK/round_1/source_packet.md`, especially locked S14 design, input/hash limits, launch facts, root constraints and questions.
- Canonical S00 snapshot: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/source_snapshot_iter30.md`.
- Iter29 canonical protocol: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md`; source facts from `curvature_RQ-VAE_iter29/curvature_RQ-VAE.py` and `curvature_config.py`; Stage3 facts from `stage3_T5Train/train_HG-Rec.py` and `curvature_RQ-VAE_iter29/scripts/run_stage3_iter29.py`.
- S14 mapping authorization, paired seeds and target: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/global_review_after_iter29.md` and canonical S00 snapshot.
