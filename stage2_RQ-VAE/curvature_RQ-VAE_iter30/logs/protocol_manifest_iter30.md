# Iter30 Protocol Manifest — S01 canonical lock

```text
PROTOCOL_ID=FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45
STATUS=CANONICAL_S01_LOCK_PENDING_LATER_GATES
EXECUTION_AUTHORIZATION=NO
DATASET_VERSION=Amazon_2023_Instruments / 2026-09 Refactor
EXPERIMENT_TYPE=single_factor; prospective matched-seed replication of two existing mappings

PARENT_ITER=iter29 (implementation/source-lineage parent; not the paired control)
PARENT_COMMIT=ecd01e4712a1badd38a0338255f4b2ec7b030aff (current S00-synchronized root/source revision)
PARENT_DELIVERABLE_COMMIT=57b4a594d92435fd74fa4bba7c7c1439a33eb102 (iter29 S13 code/results closure)
PARENT_S14_SYNC_COMMIT=fa7801c4d0f9a7e1d96da968cc7e0be891b84bd6
HISTORICAL_ITER29_S01_CODE_ROOT_COMMIT=612a5a41dfe524205b6afa46370ad0d0ce377882 (historical; not the current Stage3 execution revision)

DIRECT_PAIRED_CONTROL=iter26_mapping, freshly run at each seed 43, 44, 45
CANDIDATE_ARM=iter29_mapping, freshly run at each seed 43, 44, 45
CANONICAL_BASELINE_ITER=iter26
CANONICAL_BASELINE_TEST_FINAL=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json
CANONICAL_BASELINE_TEST_R@10=0.057017009349048554 (n_eval=57439; historical seed-42 result, not a prospective paired observation)
USER_SUCCESS_CRITERION=test_recall@10 >= 0.065 (inclusive)
HISTORICAL_REFERENCE_ITER=iter18
HISTORICAL_REFERENCE_CLASSIFICATION=HISTORICAL_NONCOMPARABLE; excluded from direct ranking and pooling

STAGE1_EMBEDDING_PATH=/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy
STAGE1_EMBEDDING_SHA256_AT_S01_CHECK=6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb
STAGE1_ITEM_ID_SIDECAR_PATH=/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json
STAGE1_ITEM_ID_SIDECAR_SHA256_AT_S01_CHECK=3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30
STAGE0_TRAIN_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet
STAGE0_TRAIN_SHA256_AT_S01_CHECK=80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815
STAGE0_VALID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/valid.parquet
STAGE0_VALID_SHA256_AT_S01_CHECK=48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9
STAGE0_TEST_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/test.parquet
STAGE0_TEST_SHA256_AT_S01_CHECK=5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc
STAGE0_ITEMS_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/items.parquet
STAGE0_ITEMS_SHA256_AT_S01_CHECK=6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3
STAGE2_WARMSTART_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth
STAGE2_WARMSTART_SHA256_AT_S01_CHECK=189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b
INPUT_HASH_SCOPE=The listed SHA256 values were directly rechecked at S01 and are check-time identities only; rehash and verify resolved consumer paths/use immediately before every applicable Stage2 and Stage3 invocation.

STAGE2_SEEDS=43,44,45; paired arms use the same seed within each pair
STAGE2_MAX_GLOBAL_STEPS=100000
RQ_LAYERS=3
CODEBOOK_SIZE_PER_LAYER=256
STAGE2_SOURCE_PARENT_SHA256_AT_S01=2469b193bee1c4352ba180540c305c9ea837043afafd1b1585ea7b8aff805f4d (iter29 source; reference identity, not a future iter30 run-profile hash)
STAGE2_CONFIG_PARENT_SHA256_AT_S01=b41bf1fff2b481abf846d71a43c30c7ba34d0d25bd93bc04f70df486e3163f7a (iter29 config; reference identity, not a future iter30 run-profile hash)
STAGE2_CONTROL_ITER26_SOURCE_SHA256_AT_S01=ed629824d5a0308f4018b6dbaaa24303690a3597def7d1bcbf9abd38c3aca10f
STAGE2_CONTROL_ITER26_CONFIG_SHA256_AT_S01=4d6b3e06aa2f408a00664d971d64ed83d0e88ef5f88b852be97302a0e2786086

STAGE3_CODE_COMMIT=ecd01e4712a1badd38a0338255f4b2ec7b030aff (current source revision for all prospective Stage3 runs)
STAGE3_TRAINER_PATH=/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train/train_HG-Rec.py
STAGE3_TRAINER_SHA256_AT_S01=9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb
STAGE3_SEEDS=43,44,45; paired arms use the same seed within each pair
STAGE3_EPOCHS=150 configured maximum; preserve the current trainer's unchanged NO_EVAL=True / EARLY_STOP=10 train-loss patience behavior; final test remains required
STAGE3_BEAM_SIZE=20
STAGE3_N_EVAL=57439 expected from the locked test split; verify actual n_eval in every future test_final.json
STAGE3_TOPK=[5,10]
STAGE3_NO_EVAL=True
STAGE3_SKIP_TEST=False
STAGE3_BATCH_SIZE=4096
STAGE3_INFER_BATCH_SIZE=1024
STAGE3_LAUNCHER_NPROC=4
STAGE3_TORCHRUN_PORT=50201; serial launch only
STAGE3_ITER29_WRAPPER_SHA256_AT_S01=1cab47ff035e69aecf3aa5a2e1cfe7777c114a60b753b5e935453d90271b2ec7 (historical wrapper identity)
STAGE3_METADATA_CAVEAT=The iter29 wrapper assigned its intended RQVAE_VARIANT but training metrics recorded unknown_variant; disclose and preserve this nonfatal caveat, without modifying the Stage3 trainer.

SEED42_POOLING=EXCLUDE_FROM_PRIMARY_PAIRED_ESTIMATOR; report historical pair separately only
RAW_RESIDUAL_PROVENANCE=UNRESOLVED_S01_HARD_GATE; historical method/value evidence is medium confidence, not checkpoint-level reproduction or historic byte identity; S03 must independently adjudicate
NORMALIZED_LAYER_SCALES=[0.001,0.932889,1.0]; distinct from raw_residual_medians and forbidden as a substitute
WARMSTART_POLICY=FAIL_CLOSED_REQUIRED_BEFORE_EXECUTION; all six runs must verify the locked checkpoint and log a positive count of compatible transferred tensors
STAGE2_QUALITY_METRICS=DESCRIPTIVE_ONLY; no quality-proxy admission gate or early stop
```

## Historical results and comparison roles

The exact primary result files were read and agree with the S00 packet and historical protocol records:

| Historical run | Role | test_final.json | R@5 | R@10 | NDCG@5 | NDCG@10 | n_eval |
|---|---|---|---:|---:|---:|---:|---:|
| iter26, seed 42 | canonical historical baseline; same mapping as prospective direct-control arm | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json` | 0.03788366789115409 | 0.057017009349048554 | 0.025169911931406087 | 0.03133359761524377 | 57439 |
| iter29, seed 42 | historical candidate-mapping context only | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json` | 0.03953759640662268 | 0.05921064085377531 | 0.026252776900288842 | 0.03257647953179143 | 57439 |

The historical iter29-minus-iter26 R@10 point difference is `+0.002193631504726755`. It is a single historical pair, not the prospective paired estimate, and does not establish causal benefit or run-to-run variance. Iter29's canonical S01 names iter26 its sole historical direct control and classifies iter18 `HISTORICAL_NONCOMPARABLE`; iter18 is not ranked here.

The prospective causal comparator is a **new iter26-mapping run at the same seed**, not the historical seed-42 score. The prospective comparison has two arms only: the iter26 mapping is the registered control/baseline arm; the iter29 mapping is the candidate arm. These are two fixed FCCR-1 mapping settings in one controlled comparison, not an added mechanism or mechanism stack.

## S14-approved mappings and common registered inputs

All vectors are in `[L0,L1,L2]` order. S14 authorizes only these two existing FCCR-1 mappings and new matched seeds 43, 44, and 45:

```text
B_l = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
m_l_raw = [1.0, 0.10941, 0.09331]

CONTROL_ARM=iter26_mapping
s_l = log1p(B_l) / log1p(m_l_raw / min(m_raw))
z_l = (s_l - mean(s)) / (std(s) + 1e-12)
c_l = clip(0.5 * exp(0.2 * z_l), 0.05, 1.5)
FIXED_C=[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]

CANDIDATE_ARM=iter29_mapping
c_l = 0.05 + 1.45 * 0.5 * (B_l/(B_l+2.0) + m_l_raw/(m_l_raw+0.1))
FIXED_C=[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

Both mappings remain closed-form, computed before Stage2, fixed/non-trainable/time-invariant, with no cyclic schedule, curvature regularization, new curvature-conditioned optimizer, or new curvature-conditioned auxiliary loss. Preserve the common existing behavior-loss weight `0.20` and temperature `0.07`, curvature regularization `0`, configured Sinkhorn `sk_eps=0.05` / 3 iterations, and all other Stage2 mechanisms/settings. Existing effective-epsilon dependence on fixed curvature is an inherited, mapping-mediated computation; it is not a new Sinkhorn rule. Do not change Stage1, Stage3 trainer, the warm-start, or any non-mapping conceptual factor.

The raw-residual vector is the explicit historical `raw_residual_medians` value, not the normalized vector `[0.001,0.932889,1.0]`. Historical iter29 S03 supports method/value provenance at medium confidence only: the old calibration checkpoint and iter8 raw-SID evidence needed for historic byte identity/replay are unavailable. Iter26's preserved consumer used the ambiguous legacy `residual_norm` JSON key, although the same JSON also contains `raw_residual_medians` with equal numbers; that alias does not itself prove raw-residual semantics. Future iter30 consumers must use the explicit `raw_residual_medians` key only. This protocol lock does not clear S03.

## Parent, source lineage, and protocol compatibility

`PARENT_ITER=iter29` identifies the implementation/source-lineage parent; it does not make iter29 the direct experimental control. The Stage2 initialization for every future arm is instead the same immutable iter8 checkpoint listed above. The newly rerun iter26 mapping is the direct within-seed control. `CANONICAL_BASELINE_ITER=iter26` records the exact historical baseline result required by the protocol; that seed-42 observation cannot replace any new pair.

The current source packet/S00 root execution revision is `ecd01e4712a1badd38a0338255f4b2ec7b030aff`. The current Stage3 trainer byte identity is `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`. `612a5a41dfe524205b6afa46370ad0d0ce377882` is the historical iter29 S01 code-root record only; it is **not** the source revision to record for the prospective Stage3 executions. The iter29 closure and S14 synchronization commits are separately identified in the lock header above.

The seed-42 pair is excluded from the primary estimator. Although both historical results have `n_eval=57439` and iter29 S01 identifies iter26 as the direct control, their historical Stage2 source/config identities differ from the future iter30 per-run profiles, and the available evidence does not bind all historical invocations to complete runtime, resolved-input-use, and per-run-output identities at the level required for prospective pooling. Same dataset label, settings summaries, and evaluation cardinality alone do not prove full compatibility. Keep the historical scores in the separate table above; do not append their difference as a fourth observation, pool them into an arm mean, or use them to replace a failed predeclared pair.

## Input identities and actual consumers

Fresh SHA256 measurements at this S01 check match every value in the lock header. They attest only current byte identity at check time. Before **each** applicable Stage2 and Stage3 invocation, rehash the locked Stage1/Stage0 files and warm-start, verify the hardcoded absolute paths, and preserve the resolved-path/hash record. Also verify actual runtime load/use against the source call chain; a configuration declaration or matching pre-launch hash does not prove that a consumer opened that file. For Stage3, record the exact Stage2 SID file path and digest and verify it is the resolved file actually loaded.

Direct inspection of iter29 Stage2 source/config establishes the current call chain:

- Stage2's `TransitionDataset` loads the Stage1 embedding with `np.load`, reads the Stage1 item-ID sidecar, and reads Stage0 `train.parquet` histories/targets. SID corpus generation also consumes the Stage1 embedding. These are actual Stage2 inputs.
- Stage2 does not read Stage0 `valid.parquet`, `test.parquet`, or `items.parquet` on this training/export path. They remain locked dataset-identity files; do not report them as Stage2 runtime consumers. `ITEM_JSON_PATH` is declared in the config but is not imported by this trainer path.
- Stage3 consumes Stage0 `train.parquet` for training and Stage0 `test.parquet` for final evaluation. With `NO_EVAL=True`, it does not construct/load the validation dataset; Stage0 `valid.parquet` is therefore not consumed by this configured Stage3 path. Stage0 `items.parquet` and Stage1 embedding/sidecar are not Stage3 inputs.
- The Stage3 wrapper supplies the matching run's absolute Stage2 `item_sids.json` as `CODE_PATH`. `_resolve_code_file` tries that absolute path first if present; later verification must fail closed unless the resolved file is exactly that run's file, so a missing path cannot fall through to a stale SID file.
- `SKIP_TEST=False` causes the final test split to be loaded and evaluated. Record actual `n_eval`, expected to be `57439`, in each future `test_final.json`.

## Locked Stage2 and Stage3 settings / seed semantics

### Stage2

The inspected parent trainer currently hardcodes `SEED=42`; for this protocol the corresponding run profile must hardcode exactly 43, 44, or 45 in source. Its `main()` seeds Torch and NumPy with `SEED + rank`; `DistributedSampler` uses `seed=SEED`. The data transition choices use NumPy and loader workers follow the PyTorch loader seed. The paired arms use the identical seed, with no claim of bitwise reproducibility. `PYTHONHASHSEED="42"` is set by config after interpreter startup and is not a substitute for the training seed; preserve the same deterministic/runtime settings across all six profiles.

Keep Stage2 input dimension 768; hidden dimensions `[512,256,128]`; embedding dimension 32; 3 quantizer layers × 256 codebook entries; commitment weight 1.0; batch 640/GPU, four DDP ranks (global batch 2560); `MAX_GLOBAL_STEPS=100000`; checkpoint cadence 10000; Sinkhorn `sk_eps=0.05`, 3 iterations; AdamW `lr=1e-3`, `weight_decay=1e-4`; existing loss/data-order/runtime behavior and four visible GPUs. Its no-argument entrypoint uses the internal hardcoded four-process launcher on port 50200. Run serially. Gini, collision, unique-SID, utilization, entropy, and other Stage2 quality metrics are descriptive only and do not gate training or Stage3.

Current iter29 warm-start handling is incompatible with the locked protocol: it checks file existence, but if absent prints a warning and continues from random initialization; it has no locked SHA check and does not establish complete compatible transfer merely by reporting a count. S06/S07 must make each run fail closed on absent or mismatched file, wrong path, unreadable/malformed/incompatible state, unexpected or otherwise unexplained missing model tensors, or zero transferred compatible tensors. Skip only the explicitly identified legacy curvature entries (`layers.{0,1,2}.c_layer_scale` and `layers.{0,1,2}._fixed_c`) as appropriate to the current fixed-curvature model. Each of the six actual invocations must verify the locked digest, positive loaded-tensor count, and successful required-tensor coverage before training continues. Do not accept random initialization, replace the checkpoint, or silently recover by another warm-start.

Before **each of the six actual Stage2 trainings**, separately run the root-mandated one-checkpoint/one-batch gradient-path check for that run's hardcoded config: `total_loss.requires_grad`, non-null `grad_fn`, finite nonzero gradients on intended trainable model parameters and applicable loss paths, with detach paths checked. Fixed curvature must remain non-trainable and must not receive gradients. This per-run check supplements later contract preflight/MVG and cannot be replaced by a single block-level check.

### Stage3

The prospective runs use current unchanged `stage3_T5Train/train_HG-Rec.py` at root revision `ecd01e4712a1badd38a0338255f4b2ec7b030aff`, with the exact SHA256 above. The trainer's source constant currently defaults to seed 42; its `set_seed(seed)` seeds Python `random`, NumPy, Torch, and all CUDA devices, and its `DistributedSampler` uses that seed. Each hardcoded no-argument wrapper must set the pair's matching Stage3 seed 43/44/45, the exact Stage2 SID path, variant label, and unique output paths in memory before invoking the existing launcher.

Keep the unchanged Stage3 protocol: 150 configured maximum epochs; `TOPK_LIST=[5,10]`; beam size 20; `NO_EVAL=True`; `SKIP_TEST=False`; batch size 4096; inference batch 1024; existing model/optimizer/scheduler/BF16/determinism configuration; four ranks; port 50201; and root-required NCCL settings (`NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_TIMEOUT=3600`, `TORCH_NCCL_BLOCKING_WAIT=1`). Preserve the existing `NO_EVAL=True` training-loss patience (`EARLY_STOP=10`) unchanged; this is not permission to stop, select, or drop any run based on test R@10, the inclusive target, paired trends, or Stage2 proxy metrics. Every valid non-aborted arm must reach its unchanged trainer's normal completion and execute its final test (`SKIP_TEST=False`). The two historical iter26/iter29 runs both reached epoch 150 under this configuration.

The current Stage3 trainer also fixes `FAST=True`, `BF16=True`, `COMPILE=False`; 4 encoder/decoder layers, `d_model=128`, `d_ff=1024`, 6 heads, `d_kv=64`, dropout `0.1`, ReLU activation/feed-forward projection; AdamW configuration is the trainer's existing optimizer with `lr=0.003` and `weight_decay=0.05`; cosine LR scheduler is enabled with 10% warmup and minimum factor `0.20`. Preserve all these settings unchanged. Determinism/runtime constants remain as currently sourced (`PYTHONHASHSEED=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True, warn_only=True)`, cuDNN deterministic true/benchmark false, float32 matmul precision `high`); root-mandated NCCL values are applied by the existing launcher. The exact trainer SHA256 above binds the prospective Stage3 execution source; run-specific wrapper edits may only bind seed/SID/variant/output/launcher paths.
Retain the remaining current Stage3 data/evaluation settings as well: `CODEBOOK_SIZE=[256,256,256,1]`, `MAX_LEN=20`, `N_USER_TOKENS=1`, `NUM_WORKERS=4`, `PREFETCH_FACTOR=8`, `EXCLUDE_HISTORY=True`, and `PIN_MEMORY=True`. `SCREEN_BASELINE_LOG=""` remains empty (screen disabled); keep `SCREEN_SKIP_TEST_ON_FAIL=True` unchanged but inactive, so it cannot suppress the required final test.

## Six-run design and exact future path convention

All six future runs are planned; none of the paths below claims an output currently exists. Process the runs serially, pairing the two arms at each seed; do not run concurrent launchers sharing ports or logs.

| Pair | Arm / run label | Stage2 seed | Stage3 seed | Planned Stage2 root | Planned Stage3 root |
|---:|---|---:|---:|---|---|
| 43 | iter26 control — `iter26_mapping_seed43` | 43 | 43 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` |
| 43 | iter29 candidate — `iter29_mapping_seed43` | 43 | 43 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` |
| 44 | iter26 control — `iter26_mapping_seed44` | 44 | 44 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` |
| 44 | iter29 candidate — `iter29_mapping_seed44` | 44 | 44 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` |
| 45 | iter26 control — `iter26_mapping_seed45` | 45 | 45 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` |
| 45 | iter29 candidate — `iter29_mapping_seed45` | 45 | 45 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` |

For each `<label>` the hardcoded Stage2 paths must be:

```text
RQVAE_OUT_DIR=<Stage2 root>/out/rqvae/instruments/
RQVAE_CKPT_PATH=<RQVAE_OUT_DIR>/rqvae_best.pth
RAW_SIDS_NPY=<RQVAE_OUT_DIR>/sids_raw.npy
SIDS_NPY=<Stage2 root>/dataset/Instruments/sids_for_hgrec.npy
ITEM_SIDS_JSON=<Stage2 root>/item_sids.json
MECHANISM_NAME=iter30_<arm>_seed<seed>   # full descriptive name, not the short result root
```

For each matching Stage3 wrapper:

```text
CODE_PATH=<matching Stage2 root>/item_sids.json
RQVAE_VARIANT=iter30_<arm>_seed<seed>
LOG_PATH=<Stage3 root>/logs/
SAVE_PATH=<Stage3 root>/ckpt/
_STAGE3_LAUNCHER_LOG=<Stage3 root>/logs/_stage3_launcher.log
```

The planned Stage3 result/checkpoint convention is:

```text
<Stage3 root>/logs/Amazon_2023_Instruments/<runtime timestamp>/test_final.json
<Stage3 root>/ckpt/Amazon_2023_Instruments/<same runtime timestamp>/HG_Rec_best.pth
<Stage3 root>/logs/_stage3_launcher.log
```

The current timestamp format is `%b-%d-%Y_%H-%M-%S`; future timestamps and outputs are not known yet. Keep all generated `.pth`, `.npy`, and `item_sids.json` outside the iter30 source subtree. Use a distinct Stage2 outer stdout log and internal launcher log for every label (e.g. `logs/train_run_<label>.log` and `logs/train_migrated_<label>.log` under the iter30 source `logs/`), plus a distinct Stage3 wrapper stdout log beneath each Stage3 run root. Record each run-profile source/config/wrapper SHA256 before invocation; future run-profile hashes are not available at S01.

The hardcoded serial method is the only allowed no-CLI/no-environment selector: each reviewed run profile must bake in exactly one arm, seed, full mechanism label, input paths, output paths, and log paths; invoke the project entrypoint/wrapper with no arguments, no CLI flags, no shell-variable injection, and no environment-based run selection. Use the existing root-mandated Python launch environments and internal four-rank launchers. Before every launch verify the baked-in values and hash the profile. Inventory the full run destination—including external SID JSON/NPY paths—before launch; it must be absent/empty or have an unambiguous recorded pre-run inventory. Stage2 Step0 cleanup only covers its own `RQVAE_OUT_DIR`, not external SID JSON/NPY destinations. Do not accept stale exports/checkpoints or overwrite any of the six run roots. These routing changes require later implementation/preflight approval; this S01 lock does not authorize them.

## Prospective paired analysis and complete-block rule

The primary estimand is the difference between mappings within each predeclared new seed pair:

```text
R26_s = test_recall@10 from iter26_mapping_seed<s>
R29_s = test_recall@10 from iter29_mapping_seed<s>
d_s = R29_s - R26_s, for s ∈ {43,44,45}
mean_delta = (d_43 + d_44 + d_45) / 3
sample_sd = sqrt(sum((d_s - mean_delta)^2) / 2)
SE = sample_sd / sqrt(3)
paired_range = [min(d_43,d_44,d_45), max(d_43,d_44,d_45)]
```

Report all six actual run rows with arm, seed, final result path, `n_eval`, R@5, R@10, NDCG@5, NDCG@10, and associated Stage2 SID/checkpoint paths and hashes. Report all three signed `d_s`, their mean, sample SD, SE, and range; optionally include a df=2 paired-t interval only as highly imprecise descriptive uncertainty. Separately report each arm's three-run mean/spread and count of individual runs meeting the inclusive target. Show the historical seed-42 pair and its exact metrics in the separate context table above, never in the primary estimate. The mapping contrast is the paired delta; target attainment is not an effect-size threshold.

Classify target attainment for each completed, protocol-valid Stage3 result using the user's inclusive `test_recall@10 >= 0.065`; equality at `0.065` meets the target. The user's criterion overrides lower-priority strict `>0.065` wording in repository/skill text. Do not promise attainment, treat a below-target score as mechanism failure by itself, or stop early because a run/arm crosses or misses the target. Do not stop on Stage2 SID quality metrics or a favorable/unfavorable paired trend. A primary paired estimate is reportable only after all three predeclared pairs and all six valid Stage2→Stage3 pipelines have completed their locked protocols and all six final test records are audited. No seed, arm, or pair may be silently omitted, substituted, or replaced with seed 42. A failed/invalid run is not a score and cannot be dropped to form a partial primary mean; preserve its evidence and use later adjudicated invalid/abort handling rather than an automatic restart or overwrite.

## Blocking gates and execution boundary

This manifest locks the experimental design, not permission to edit code or start work. No Stage2 or Stage3 launch is authorized here. The following remain blocking:

1. `S02_HYPOTHESIS` must canonically register the exact two S14-approved mappings and shared inputs without inventing or retuning a map.
2. `S03_PROVENANCE` must independently adjudicate the medium-confidence historical raw-residual evidence under skill §7. If it fails, stop before MVG/Stage2; do not use the normalized vector, another checkpoint, or a newly inferred replacement.
3. `S04_CONTRACT`'s primary FCCR-1 candidate contract covers the iter29 mapping. `S07_PREFLIGHT` and `S08_MVG` must additionally verify the iter26 control configuration/vector and fixed behavior for both arms, without creating a second conceptual mechanism or introducing a second contract mechanism.
4. `S05_ONE_FACTOR` must verify the mapping-only contrast. `S06_IMPLEMENTATION` must implement approved hardcoded per-run routing and make warm-start loading fail closed. `S07_PREFLIGHT` must pass the actual source/config checks; `S08_MVG` must verify both fixed vectors, immutability/time invariance, model gradient health, and measurable counterfactual activation. No PASS is claimed by this manifest.
5. The required S00–S09 A/B/Judge deliberation chain and no-argument deliberation gate must pass. S09 must separately authorize the Stage2 wiring/launch. Each of the six Stage2 runs needs its own immediately-prior input/warm-start/path/output checks and gradient-path verification. Each Stage3 run additionally requires the valid Stage2 result and later canonical S10/S11 authorization.
6. Before every invocation, rehash relevant locked inputs and verify the exact resolved paths and actual consumers/use; verify positive compatible warm-start tensor loading for every Stage2 run; verify each Stage3 wrapper resolves/loads its own arm's SID file; verify source/config/profile hashes and unique short-root destinations.
7. Preserve root `CLAUDE.md` no-CLI/no-environment-override rules, short `curvature_RQ-VAE_iter30` Stage2/Stage3 result roots, no-overwrite routing, no Stage1/Stage3 modifications, and complete six-run/block analysis requirements.

`AUTONOMOUS_NEXT_ACTION=Advance to S02_HYPOTHESIS using this judge-approved protocol lock. Keep both S14 mappings and all registered inputs/seed pairs unchanged, do not launch Stage2/Stage3, and do not treat any later gate as passed until separately adjudicated with direct evidence.`
