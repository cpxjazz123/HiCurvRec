ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S01_PROTOCOL_LOCK/round_1/source_packet.md
STAGE_ID=S01_PROTOCOL_LOCK

# Iter30 S01 protocol-lock proposal — Agent A

## Decision and authorization boundary

**Proposal:** register only the S14-authorized prospective comparison of the existing iter26 and iter29 FCCR-1 fixed mappings, using three new matched pairs (seeds 43, 44, 45), for six complete Stage2/Stage3 pipelines if every upstream hard gate passes. No third mapping, mechanism change, seed substitution, shortened run, or Stage1/Stage3 change is proposed. This is a protocol proposal, not an authorization to edit implementation, run MVG, launch Stage2/Stage3, or write the canonical protocol manifest. Canonical-only propagation remains subject to Judge C and later stages.

The registered block is operationally designable under the project's hardcoded-parameter rules, with run-specific hardcoded configuration and serial launches. **The current iter29 code is not ready to execute this block as-is**: it hardcodes one seed/one mapping/one output path and warns then uses random initialization if the warm-start file is absent. Iter30 implementation/preflight must create the six specified hardcoded profiles/values, unique output destinations, and fail-closed warm-start verification without changing the scientific design. Those are prerequisites, not permission to launch. I found no direct evidence that implementing those routing/guard changes requires narrowing or changing the registered experiment.

`raw_residual_medians` remains a hard gate pending iter30 S03. Prior S03 accepted its historical method/value provenance at medium confidence, but the old calibration checkpoint and iter8 raw-SID file are absent and historic byte identity/replay are unproven. S03 must independently decide whether that bounded evidence satisfies the active §7 gate. If it fails, stop before MVG/Stage2 and close/abort under the skill; do not substitute normalized scales or infer values anew.

## Proposed manifest — field-complete S01 record

```text
PROTOCOL_ID=FCCR-1_iter30_matched_iter26_vs_iter29_seed43-45
STATUS=PROPOSED_S01_LOCK_PENDING_JUDGE_AND_LATER_GATES
DATASET_VERSION=Amazon_2023_Instruments / 2026-09 Refactor
EXPERIMENT_TYPE=single_factor matched-seed replication; three predeclared pairs
PARENT_ITER=iter29
PARENT_COMMIT=ecd01e4712a1badd38a0338255f4b2ec7b030aff (S14-synchronized current root commit recorded in S00)
PARENT_ITER_DELIVERABLE_COMMIT=57b4a594d92435fd74fa4bba7c7c1439a33eb102 (iter29 code/results closure commit)
DIRECT_PAIRED_CONTROL=iter26 mapping arm, newly run at each matched seed 43, 44, 45
CANDIDATE_ARM=iter29 mapping, newly run at each matched seed 43, 44, 45
CANONICAL_BASELINE_ITER=iter26 (sole historical direct control / same mapping as prospective paired-control arm)
CANONICAL_BASELINE_TEST_FINAL=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json
CANONICAL_BASELINE_TEST_R@10=0.057017009349048554 (n_eval=57439; historical, not a substitute for new within-seed control outcomes)
HISTORICAL_REFERENCE_ITER=iter18
HISTORICAL_REFERENCE_CLASSIFICATION=HISTORICAL_NONCOMPARABLE; excluded from ranking
USER_SUCCESS_CRITERION=test_recall@10 >= 0.065 (inclusive; user instruction controls over lower-priority strict > 0.065 text)
STAGE1_EMBEDDING_PATH=/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy
STAGE1_EMBEDDING_SHA256=6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb
STAGE1_ITEM_ID_SIDECAR_PATH=/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json
STAGE1_ITEM_ID_SIDECAR_SHA256=3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30
STAGE0_TRAIN_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet
STAGE0_TRAIN_SHA256=80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815
STAGE0_VALID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/valid.parquet
STAGE0_VALID_SHA256=48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9
STAGE0_TEST_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/test.parquet
STAGE0_TEST_SHA256=5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc
STAGE0_ITEMS_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/items.parquet
STAGE0_ITEMS_SHA256=6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3
STAGE2_SEED=43,44,45; within each pair both arms use the same seed
STAGE2_MAX_GLOBAL_STEPS=100000
RQ_LAYERS=3
CODEBOOK_SIZE_PER_LAYER=256
STAGE2_WARMSTART_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth
STAGE2_WARMSTART_SHA256=189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b
STAGE3_CODE_COMMIT=612a5a41dfe524205b6afa46370ad0d0ce377882 (iter29 S01's last functional Stage3 code commit; tracked subtree continuity to current root is recorded; current train_HG-Rec.py SHA256 below)
STAGE3_TRAINER_PATH=/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train/train_HG-Rec.py
STAGE3_TRAINER_SHA256=9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb
STAGE3_SEED=43,44,45; within each pair both arms use the same seed
STAGE3_EPOCHS=150
STAGE3_BEAM_SIZE=20
STAGE3_N_EVAL=57439 expected from the locked test split; verify actual test_final.json on each run
SEED42_POOLING=EXCLUDE from the primary prospective paired estimator; report the old matched seed-42 result pair separately as historical context only
```

### Historical baseline semantics and lineage

Iter29 is the implementation/source parent for iter30; it is **not** the new paired control. The new within-pair direct control is a fresh iter26-mapping run at seed 43, 44, or 45. The canonical historical baseline record remains iter26's exact Stage3 result above. Iter29's historical candidate record is `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`, with `n_eval=57439` and `test_recall@10=0.05921064085377531`; its old point difference from the iter26 record is `+0.002193631504726755`, not a prospective paired observation and not evidence of a reproducible effect. Iter29 S01 identifies iter26 as its sole protocol-valid direct control and labels iter18 `HISTORICAL_NONCOMPARABLE`; iter18 must not be pooled or directly ranked.

The S00 packet records the S14-synchronized current root commit `ecd01e4712a1badd38a0338255f4b2ec7b030aff` and the iter29 Stage2/result closure commit `57b4a594d92435fd74fa4bba7c7c1439a33eb102`. Current historical source byte identities rechecked for this audit are iter26 `curvature_RQ-VAE.py` SHA256 `ed629824d5a0308f4018b6dbaaa24303690a3597def7d1bcbf9abd38c3aca10f`, `curvature_config.py` SHA256 `4d6b3e06aa2f408a00664d971d64ed83d0e88ef5f88b852be97302a0e2786086`; iter29 `curvature_RQ-VAE.py` SHA256 `2469b193bee1c4352ba180540c305c9ea837043afafd1b1585ea7b8aff805f4d`, `curvature_config.py` SHA256 `b41bf1fff2b481abf846d71a43c30c7ba34d0d25bd93bc04f70df486e3163f7a`, Stage3 wrapper SHA256 `1cab47ff035e69aecf3aa5a2e1cfe7777c114a60b753b5e935453d90271b2ec7`, and Stage3 trainer SHA256 `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`. These are evidence about inspected historical/current source, **not hashes for not-yet-materialized iter30 profiles**. S06/S09 must capture each actual iter30 per-run source/config/wrapper hash and the Stage3 trainer hash before launch.

### Two exact S14-registered mapping arms and shared formula inputs

Layer order is `[L0,L1,L2]`. Use only these existing mappings, exactly, with the same shared historical inputs in both arms:

```text
B_l = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
m_l_raw = [1.0, 0.10941, 0.09331]

ARM=iter26_mapping
s_l = log1p(B_l) / log1p(m_l_raw / min(m_raw))
z_l = (s_l - mean(s)) / (std(s) + 1e-12)
c_l = clip(0.5 * exp(0.2 * z_l), 0.05, 1.5)
FIXED_C=[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]

ARM=iter29_mapping
c_l = 0.05 + 1.45 * 0.5 * (B_l/(B_l+2.0) + m_l_raw/(m_l_raw+0.1))
FIXED_C=[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

Both remain FCCR-1: closed-form, computed before Stage2, fixed/non-trainable/time-invariant, no schedule, no curvature regularization, no new optimizer or auxiliary loss. Existing behavior loss (weight `0.20`, temperature `0.07`), inherited curvature-consuming quantization/Sinkhorn behavior, and configured `sk_eps=0.05` / 3 iterations stay unchanged; any effective-epsilon difference induced by the different fixed `c` is a mediated effect of the mapping, not a new Sinkhorn rule. No stacking, Stage1/Stage3 modification, or other mechanism is part of this protocol.

The raw-residual vector is not the normalized layer-scale vector `[0.001,0.932889,1.0]`. Iter29's registered JSON attributes the raw vector to iter10 `calibrate_residual_scales.py` and its log, with measurement “per-layer, per-item residual-vector L2 norm over embedding dimensions; median over items,” `source_label=iter1`, step 100000. The referenced historical checkpoint and iter8 raw SID file are currently absent; no replay or historic hash exists. The iter29 S03 Judge approved historical method/value provenance at medium confidence only, not checkpoint-level reproduction. Iter30 S03 must independently re-establish semantic/source/value evidence under the same S14 evidence ceiling and either pass the skill's hard gate or block/abort. Do not invoke the old producer that overwrites the raw key; do not replace the raw vector with normalized scales, a fallback, or a new calibration.

## Input identity, data consumers, and recheck boundary

I independently re-ran SHA256 over the current Stage1 embedding, item-ID sidecar, Stage0 train/valid/test/items parquet files, and iter8 warm-start file. All seven measured hashes exactly match the locked values listed above. This proves present byte identity at this protocol-preparation check only. It does **not** prove a later launch will resolve the same files, that the historical raw-residual data had those bytes, or that a future process actually consumed a path. Immediately before each Stage2 invocation, rehash the relevant files, log/verify resolved absolute paths against the run's hardcoded config, and preserve that evidence; do not substitute on mismatch.

Direct call-chain review of iter29 Stage2 source/config shows:

- `curvature_config.py` declares Stage1 `ITEM_EMB_NPY`, `ITEM_IDS_JSON`, Stage0 parquet paths, and the per-run output paths. The training entry imports the embedding, sidecar and `TRAIN_PARQUET`; `TransitionDataset` loads the embedding with `np.load`, loads the JSON sidecar and requires dense IDs exactly `0..N-1`, and reads `train.parquet` histories/targets. At checkpoints/final export, corpus SID generation again consumes the Stage1 embedding. Thus Stage1 embedding and sidecar and Stage0 train are actual Stage2 inputs.
- Stage2's present trainer does not read Stage0 `valid.parquet`, `test.parquet`, or `items.parquet` in this code path; they remain locked dataset identity inputs rather than asserted Stage2 runtime consumers.
- Stage3 uses the locked Stage0 directory and `train.parquet` plus `test.parquet`; the Stage3 wrapper supplies the exact run's absolute Stage2 `item_sids.json`, and `_resolve_code_file` accepts that exact existing path first. With `NO_EVAL=True`, validation data is not loaded; `SKIP_TEST=False` and the final test path is loaded/evaluated. Stage3 does not consume Stage1 embeddings/sidecar or Stage0 `items.parquet` in this path. `n_eval=57439` is expected from the locked test set and must be checked in every actual result.
- The Stage2 config's `ITEM_JSON_PATH` declaration is not among the values imported by this trainer path; do not describe it as a consumed training input.

The iter29 wrapper/reporting caveat remains disclosed: prior metrics recorded `unknown_variant` even though the wrapper assigned its intended RQVAE variant. This optional metadata discrepancy is not evidence of a different SID input or model setting; preserve it and do not edit Stage3 trainer to “fix” it. Stage3 run provenance must be bound by its exact hardcoded wrapper `CODE_PATH`, Stage2 SID digest, output path, Stage3 source digest, seed, and launcher logs.

## Seed semantics and fixed settings

### Stage2

The inspected trainer hardcodes `SEED=42` in iter29. `main()` seeds `torch.manual_seed(SEED + rank)` and `np.random.seed(SEED + rank)` for each DDP rank; `DistributedSampler` receives `seed=SEED`. The data transition sampler's random choices use NumPy, and workers derive from the PyTorch loader seed. For each future profile the hardcoded `SEED` must be exactly 43, 44, or 45, and both mappings for that pair must have the same seed. Do not use a CLI argument or environment variable to choose it. `curvature_config.py` sets the same CUBLAS workspace and `PYTHONHASHSEED="42"` for every run; the latter is assigned after interpreter startup and is not a per-run RNG mechanism. Do not claim bitwise determinism from matched seeds.

Keep Stage2 settings fixed: input dimension 768; hidden `[512,256,128]`; embedding dimension 32; three quantizer layers; codebook size 256/layer; commitment weight 1.0; per-GPU batch 640, four DDP ranks/total batch 2560; `MAX_GLOBAL_STEPS=100000`; checkpoint cadence 10000; `sk_eps=0.05`, `sk_iters=3`; AdamW `lr=1e-3`, weight decay `1e-4`; existing loss and data order; same four visible GPUs and Python/torchrun environment. Internal Stage2 launcher is the existing hardcoded four-process torchrun launcher on port 50200. Stage2 Gini/collision/unique/entropy/utilization statistics are descriptive only; they are never a stop or admission gate.

### Stage3

The unmodified trainer hardcodes `SEED=42`, `set_seed(seed)` seeds Python `random`, NumPy, PyTorch and all CUDA devices, and its `DistributedSampler` uses the supplied seed. The run-specific no-argument wrapper must set in memory the same pair seed 43/44/45 before training. Keep Stage3 source/trainer and runtime unchanged: 150 epochs; `TOPK_LIST=[5,10]`; `BEAM_SIZE=20`; `NO_EVAL=True`; `SKIP_TEST=False`; batch 4096; inference batch 1024; existing model/optimizer/scheduler/BF16/determinism settings. Use existing internal 4-rank launcher, port 50201, visible devices `0,1,2,3`, and the root-mandated NCCL values. It produces the final `test_final.json`; record R@5, R@10, NDCG@5, NDCG@10 and `n_eval` from that file. No evaluation/selection by a different metric is allowed.

## Exact run labels, roots, and no-overwrite routing

The arm token is `iter26_mapping` or `iter29_mapping`. These are the exact six future labels, with Stage2 and Stage3 seed equal to the suffix:

| Order | Run label | Seed | Stage2 output root | Stage3 output root |
|---:|---|---:|---|---|
| 1 | `iter26_mapping_seed43` | 43 | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` |
| 2 | `iter29_mapping_seed43` | 43 | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` |
| 3 | `iter29_mapping_seed44` | 44 | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` |
| 4 | `iter26_mapping_seed44` | 44 | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` |
| 5 | `iter26_mapping_seed45` | 45 | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` |
| 6 | `iter29_mapping_seed45` | 45 | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` |

Within each Stage2 root, hardcode:

```text
RQVAE_OUT_DIR=<Stage2 root>/out/rqvae/instruments
RQVAE_CKPT_PATH=<RQVAE_OUT_DIR>/rqvae_best.pth
RAW_SIDS_NPY=<RQVAE_OUT_DIR>/sids_raw.npy
SIDS_NPY=<Stage2 root>/dataset/Instruments/sids_for_hgrec.npy
ITEM_SIDS_JSON=<Stage2 root>/item_sids.json
MECHANISM_NAME=iter30_<arm>_seed<seed>
```

Full mechanism names carry `iter30_`; all model/data artifacts remain outside the source subtree. For each Stage3 root, the run-specific wrapper hardcodes `CODE_PATH=<matching Stage2 root>/item_sids.json`, `SEED=<matching seed>`, and `RQVAE_VARIANT=iter30_<arm>_seed<seed>`, with `LOG_PATH=<Stage3 root>/logs/` and `SAVE_PATH=<Stage3 root>/ckpt/`. The actual result locations are therefore:

```text
<Stage3 root>/logs/Amazon_2023_Instruments/<runtime timestamp>/test_final.json
<Stage3 root>/ckpt/Amazon_2023_Instruments/<same run timestamp>/HG_Rec_best.pth
<Stage3 root>/logs/_stage3_launcher.log
```

The current trainer timestamp format is `%b-%d-%Y_%H-%M-%S`; the timestamp is necessarily generated at execution, so no future timestamp/result file is claimed to exist. The outer wrapper stdout log also receives a run-specific path under that run's Stage3 `logs/` or iter30 source `logs/`; never redirect two active processes to a common file.

For every run, first verify the hardcoded `RQVAE_OUT_DIR` line resolves beneath the mandated short Stage2 `curvature_RQ-VAE_iter30` root and has the matching label. Check the entire run destination, including external SID JSON/NPY destinations, is absent/empty or has an unambiguous recorded pre-run inventory before launch. Stage2 Step0 deletes its allowlisted stale products **only inside `RQVAE_OUT_DIR`**; it does not clear the external per-run `item_sids.json` or `sids_for_hgrec.npy`, hence those must be unique and separately checked. Keep the six output roots disjoint. Before a launch, bind its config/code hash to the label/seed/mapping/path; never let a stale SID/checkpoint be accepted. Capture output sizes, mtimes, SHA256 and full item/SID coverage after completion. No `.pth`, `.npy`, or `item_sids.json` may be written under `stage2_RQ-VAE/curvature_RQ-VAE_iter30/`.

### Hardcoded serial launch procedure (future, gated)

After all required canonical approvals, the single iter30 Stage2 source/config is prepared for exactly one label at a time using source literals only: hardcoded seed, mapping, full mechanism label and unique output paths; no argparse/sys.argv selector, CLI flags, shell variable injection or environment override. Preserve hashes of that run's code/config. Do not modify an active process's source/config. Run the six Stage2 jobs serially (no concurrent 50200 launchers); before each, verify the baked-in label, map, seed and `RQVAE_OUT_DIR`, hash inputs/warmstart, inventory destinations, and complete the mandatory per-run gradient gate. Use the root-required no-argument launch from the iter30 source directory, one run at a time: `nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 curvature_RQ-VAE.py > logs/train_run.log 2>&1 &`. Monitor each launched process through terminal exit; do not start the next until it has exited successfully or the failed attempt has been declared invalid. Preserve/rename both `logs/train_run.log` and the trainer's hardcoded `logs/train_migrated.log` to label-specific log paths after every process exits and before reusing either filename. The source-internal four-rank torchrun/50200 launch is unchanged. Never run two Stage2 jobs concurrently or rely on Step0 to protect a previous run.

Stage3 has no CLI/env selector either. Only after the applicable S10/S11 decisions authorize it and the exact Stage2 export is verified, the run-specific hardcoded no-argument wrapper follows the existing iter29 pattern: load unchanged `stage3_T5Train/train_HG-Rec.py`, set only its in-memory SID path, seed, variant label and unique `LOG_PATH`/`SAVE_PATH`, point its internal launcher at that wrapper, then use the existing four-rank launcher on port 50201 and root-required environment. Run wrappers serially; preserve each wrapper hash, Stage3 outer log and `_stage3_launcher.log`. This changes run routing/seed, not Stage3 trainer behavior or the S14-matched protocol. Do not launch the Stage3 wrapper until later canonical authorization.

## Warm-start fail-closed requirement and gradient checks

All six arms must use the single immutable iter8 checkpoint path and locked SHA256 above. The current iter29 implementation checks only `os.path.isfile`; if missing, it prints a warning and proceeds from random initialization. That is incompatible with S14's fixed warm start. Before each run, require file existence and exact SHA256. The approved iter30 implementation must raise/exit nonzero on absence, hash mismatch, unreadable/malformed state, wrong path, zero transferred tensors, unexpected keys, or missing model tensors other than the explicitly skipped legacy `layers.{0,1,2}.c_layer_scale` and `layers.{0,1,2}._fixed_c` entries. Log the verified digest and positive loaded tensor count. Never accept random initialization as fallback and never replace/re-export the warm start. The SHA check plus loaded-state log demonstrates intended artifact identity and use; retain exact `torch.load` source path and transfer evidence per invocation.

Before **each of the six actual Stage2 trainings**, perform the root §6 one-checkpoint/one-batch gradient-path check against that run's model/map/seed config. Require `total_loss.requires_grad`, non-null `grad_fn`, finite nonzero gradients on intended trainable model parameters and all applicable loss paths; fixed curvature itself must remain non-trainable and receive no gradient. Check detach paths do not silently sever backward. Any failed check blocks that invocation. It supplements (does not replace) later FCCR-1 preflight/MVG, and it is repeated for each Stage2 training, not just once for the block.

## Seed-42 pooling decision

`SEED42_POOLING=EXCLUDE` from the primary prospective three-pair estimate. Iter29 S01 and the iter29 one-factor audit do identify iter26 as its sole protocol-valid direct control and document shared historical settings/inputs; those old scores remain useful context. However, the historical arms have different Stage2 source/config identities and run/output profiles, and the available records do not establish every relevant historical runtime/input-consumption identity to the same per-invocation standard proposed for iter30. A cross-run claim of complete compatibility would exceed the evidence required by S14. Report the old iter26/iter29 pair separately, with its exact scores and known limitations; do not append its difference as a fourth observation, use it to replace any seed 43–45 pair, or label it a current seed-matched replication result. This conservative exclusion is not a claim the historical comparison was invalid (iter29 S01 did validate it as the direct historical comparison); it only declines to pool it into this prospectively defined estimate absent complete additional proof.

## Paired analysis, target and complete-block rule

For each predeclared seed `s ∈ {43,44,45}`, pair the two actual valid results and calculate:

```text
R26_s = iter26_mapping_seed<s> test_recall@10
R29_s = iter29_mapping_seed<s> test_recall@10
d_s = R29_s - R26_s
mean_delta = (d_43 + d_44 + d_45) / 3
sample_sd = sqrt(sum_s((d_s - mean_delta)^2) / 2)
range = [min(d_s), max(d_s)]
SE = sample_sd / sqrt(3)
```

Report all six rows with `n_eval`, R@5, R@10, NDCG@5, NDCG@10, arm and seed; report all three signed paired deltas, the paired mean, sample SD, range and SE (optionally the descriptive df=2 paired-t interval, explicitly labeled highly imprecise). Separately report each arm's mean/spread and count of seed results meeting target; show the historical iter26/iter29 results in a separate context table, not in the primary mean. The mapping contrast is the paired delta; the project's attainment target is not an effect-size threshold.

Apply the user's inclusive target exactly: an observed `test_recall@10 == 0.065` meets the target; report target status per completed run/arm and do not replace it with lower-priority `> 0.065` wording. Do not promise target attainment or call a below-target score mechanism failure by itself. A valid complete six-pipeline block is required before presenting the prospective paired estimate or interpreting its spread. There is no performance-based early stop: run every authorized Stage2 arm to 100000 global steps and every valid non-aborted Stage3 arm to 150 epochs/final test even if early results are above or below 0.065. SID geometry/quality metrics are descriptive and cannot stop or gate Stage3.

Abort/block conditions are predeclared: S03 provenance FAIL; input/warm-start identity or actual-consumer path mismatch; warm-start not verified/fully transferred; per-run gradient/MVG/preflight/deliberation gate FAIL; wrong seed/mapping/config/output route; failed 100000-step training, NaN/Inf, contract violation, corrupt/missing SID export, or failed/wrong Stage3 test path/count. Preserve logs/outputs and report the attempt invalid; do not silently reduce to available pairs, substitute seeds, count partial outputs, use stale data, or automatically restart/overwrite a run. A failure is not an observed score and no incomplete paired mean is the registered result. Only a formally adjudicated implementation repair that preserves the entire locked mechanism/protocol may be considered under the skill; it cannot silently alter the six-run scope. If one of these hard gates fails, stop the iteration/block under the skill's autonomous abort rules rather than altering the design.

## Remaining prerequisites and self-rejection conditions

Before any Stage2 launch, the canonical chain must contain the Judge-approved iter30 S00–S09 artifacts and passes: S01 lock; S02 exact mapping/hypothesis registration (both arms remain the already fixed S14 equations); S03 provenance gate; S04 mechanism contract; S05 one-factor audit; S06 minimal implementation; S07 contract preflight; S08 FCCR-1 MVG and measurable counterfactual activation; S09 launch/wiring authorization; per-run input/output/warm-start checks and gradient verification; and the no-argument deliberation gate. Each Stage3 job separately requires completed valid Stage2 analysis and canonical S10/S11 authorization. Stage2 proxies are not gates. No future `test_final.json` or results are claimed present.

I would reject this proposal if the Judge establishes that S14 did not authorize one of these two exact mappings/seeds; the provenance gate cannot pass at the historical evidence ceiling; full source audit finds a non-mapping conceptual difference that cannot be isolated; the immutable checkpoint cannot be loaded fail-closed; any run requires an environment/CLI override, Stage1/Stage3 mechanism change, output outside the mandated short roots, a stale-output collision, or a reduced/retuned block. If provenance fails, that is a hard gate, not a license to substitute normalized residuals or another data source. If direct primary evidence instead proves the block cannot be executed without changing its registered design, report that as a blocker and do not narrow scope.

## Primary evidence audited

- Current `CLAUDE.md`, active skill §§2–4, 6–10, 12–18, iter30 S00 canonical snapshot, and the exact shared S01 source packet.
- Iter29 S14 canonical Judge/report and iter29 S01 protocol manifest; iter29 S05 one-factor record and S03 Judge; iter26 protocol manifest and both exact historical `test_final.json` files.
- Iter29 `curvature_RQ-VAE.py`, `curvature_config.py`, `modules/step_checks.py`, `scripts/run_stage3_iter29.py`, and `scripts/computed_behavior_branching.json`; current Stage3 `train_HG-Rec.py`, `data/dataset.py`, and `model/utils.py` for seeds, loaders, final-test path/count, launcher and timestamp behavior.
- Direct SHA256 measurements in this audit confirm the seven current Stage1/Stage0/warm-start identities in the packet and the historical Stage2/Stage3 source hashes listed above. These are point-in-time identity measurements, not a substitute for prelaunch rechecks or execution evidence.
