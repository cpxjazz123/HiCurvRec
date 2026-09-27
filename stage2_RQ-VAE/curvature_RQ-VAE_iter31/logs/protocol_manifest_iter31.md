# Iter31 Protocol Manifest — S01 canonical lock

```text
STAGE_ID=S01_PROTOCOL_LOCK
ROUND=1
VERDICT=MERGE_AB
PROTOCOL_ID=ITER31_HRA_SINGLE_SEED42_INSTRUMENTS_2026-09REF_VS_ITER29
STATUS=CANONICAL_S01_LOCK
DATASET_VERSION=Amazon_2023_Instruments / 2026-09 Refactor

PARENT_ITER=iter29
PARENT_CONDITION=latest completed active Stage2/Stage3 lineage selected for Iter31
PARENT_ROOT_COMMIT=21e0488dcad39b3fd277b071a12daec59b56ba2f (reported by S00/cancellation archive; not freshly queried)
ITER29_HISTORICAL_SOURCE_ROOT_COMMIT=612a5a41dfe524205b6afa46370ad0d0ce377882
ITER29_CODE_RESULTS_CLOSURE_COMMIT=57b4a594d92435fd74fa4bba7c7c1439a33eb102
ITER29_LATER_S14_COMMIT=fa7801c4d0f9a7e1d96da968cc7e0be891b84bd6
PARENT_COMMIT_NOTE=The current Iter31 parent root identity is distinct from Iter29's historical source/run root, closure commit, and later review commit. These are reported historical identities; this Judge performed no GitHub query.

CANONICAL_BASELINE_ITER=iter29
DIRECT_COMPARATOR_ITER=iter29
COMPARATOR_TEST_FINAL=results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json
COMPARATOR_METRIC=test_recall@10 (project name: test_R@10)
COMPARATOR_R@10=0.05921064085377531
COMPARATOR_N_EVAL=57439
COMPARATOR_TEST_R@5=0.03953759640662268
COMPARATOR_TEST_NDCG@5=0.026252776900288842
COMPARATOR_TEST_NDCG@10=0.03257647953179143
ADOPTION_TARGET=test_R@10 > 0.065 (strict)
ITER29_SHORTFALL_TO_TARGET=0.005789359146224693

ITER26_VERIFIED_ALTERNATIVE_NOT_SELECTED=results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json
ITER26_R@10=0.057017009349048554
ITER26_N_EVAL=57439
ITER29_MINUS_ITER26_OBSERVED_DELTA=+0.002193631504726755
ITER26_SELECTION_NOTE=Iter26 is Iter29's historical direct control, not Iter31's selected comparator. Iter29 is the newer completed active condition. The point delta is one run per condition; the cited ~0.003 historical noise reference is approximate and unestimated, not a causal or variance estimate.
ITER18_CLASSIFICATION=HISTORICAL_NONCOMPARABLE; not a direct comparator.
ITER30_CLASSIFICATION=ITERATION_CANCELLED_BY_USER_DIRECTION_CHANGE; SCIENTIFIC_RESULT=NONE; no Stage2, Stage3, or GPU training; not a parent result or comparator.

EXPERIMENT_TYPE=single_factor (CONDITIONAL; S05 must adjudicate final mechanism boundary)
HRA_STATUS=direction label only; no HRA equation, mechanism semantics, or FCCR-1 contract transition selected by S01
STAGE1_EMBEDDING_PATH=stage1_GeneEmbedding/output/sentence_t5.npy
STAGE1_EMBEDDING_SHA256_HISTORICAL=6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb
STAGE1_ITEM_ID_SIDECAR_PATH=stage1_GeneEmbedding/output/item_ids.json
STAGE1_ITEM_ID_SIDECAR_SHA256_HISTORICAL=3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30

STAGE2_SEED=42
STAGE2_MAX_GLOBAL_STEPS=100000
RQ_LAYERS=3
CODEBOOK_SIZE_PER_LAYER=256
STAGE2_WARMSTART_PATH=results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth
STAGE2_WARMSTART_SHA256_HISTORICAL=189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b
STAGE2_METRIC_POLICY=DESCRIPTIVE_ONLY; no Stage2 hard quality gate or quality-metric early stop
STAGE2_EXECUTION_PROTOCOL=Use the root-required hard-coded no-argument Stage2 entry/launcher only after all later canonical gates and root CLAUDE.md §6 gradient checks pass; no CLI or environment overrides.
ITER29_STAGE2_HISTORICAL_EXECUTION=4-rank DDP; recorded port 50200; world_size=4; per-GPU batch=640; total batch=2560; 24587 items; 339519 train transitions; 100000 global steps completed; 11 warm-start tensors loaded. Source: Iter29 stage2_execution_plan_iter29.md and stage2_output_integrity_iter29.log.
ITER31_STAGE2_TOPOLOGY_NOTE=The Iter29 port/batch values above are historical run facts, not independently locked Iter31 source/runtime settings. Later canonical source/execution audit must establish actual Iter31 effective settings, confirm protocol compatibility and root-mandated no-argument route, and must not introduce ad hoc overrides.

STAGE3_CODE_COMMIT=21e0488dcad39b3fd277b071a12daec59b56ba2f (reported current Iter31 parent/root commit identity; not a fresh Git query; pin current trainer/runner/runtime hashes and effective configuration in later adjudicated gates)
STAGE3_SEED=42
STAGE3_MAX_EPOCHS=150
STAGE3_EARLY_STOP=10 (preserve existing train-loss patience semantics)
STAGE3_NO_EVAL=True
STAGE3_SKIP_TEST=False
STAGE3_BEAM_SIZE=20
STAGE3_TOPK_LIST=[5,10]
STAGE3_N_EVAL=57439
STAGE3_EVALUATION_DATA=Amazon_2023_Instruments test split
STAGE3_FIXED_PROTOCOL_NOTE=Preserve the Iter29 model, data, evaluator and training configuration without Stage3 model/evaluation changes. Iter29's evaluation plan states NO_EVAL=True and SKIP_TEST=False; its train-loss patience may stop before the 150-epoch maximum, while Iter29's actual outcome completed epoch 150. Lock maximum 150 and unchanged patience/checkpoint semantics; do not guarantee Iter31 reaches exactly 150. Its plan additionally records BATCH_SIZE=4096, INFER_SIZE=1024, FAST=True, BF16=True, COMPILE=False, LR=0.003, WEIGHT_DECAY=0.05, scheduler enabled, warmup=0.10, minimum LR factor=0.20; model 4 encoder/decoder layers, D_MODEL=128, D_FF=1024, 6 heads, D_KV=64, dropout=0.1, ReLU. Pin current Iter31 source/config/runner and effective evaluation identity at later gates.

ITER31_STAGE2_ARTIFACT_ROOT=results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/
ITER31_STAGE3_ARTIFACT_ROOT=results/stage3_T5Train/curvature_RQ-VAE_iter31/
ITER31_SOURCE_ARTIFACT_RULE=Source iter subtree holds code/configs/scripts/logs and allowed input copies only; no .pth, .npy, item_sids.json.
STAGE3_RESULT_ROOT_STYLE=Short directory curvature_RQ-VAE_iter31; do not include descriptive mechanism suffix in LOG_PATH/SAVE_PATH.

HISTORICAL_STAGE0_TRAIN=results/stage0_build_parquet/train.parquet; SHA256=80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815
HISTORICAL_STAGE0_VALID=results/stage0_build_parquet/valid.parquet; SHA256=48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9
HISTORICAL_STAGE0_TEST=results/stage0_build_parquet/test.parquet; SHA256=5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc
HISTORICAL_STAGE0_ITEMS=results/stage0_build_parquet/items.parquet; SHA256=6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3

REQUIRED_IDENTITY_RECHECKS_BEFORE_ANY_STAGE2_RUN=Recompute current SHA256/size/mtime and verify resolved consumption of all four Stage0 parquet inputs; both Stage1 assets; and exact immutable Iter8 warm-start checkpoint. Verify embedding/item-ID alignment and item universe. Historical hashes in Iter29 manifest do not establish current bytes or actual runtime consumption. Any missing/mismatched identity or consumer route blocks protocol-valid execution.
REQUIRED_STAGE3_RECHECKS_BEFORE_ANY_STAGE3_RUN=Pin current Iter31 trainer, runner, and relevant runtime/source identity; verify exact produced Stage2 SID path/identity and consumer resolution; verify same Stage0 test split, metric implementation, beam=20, and n_eval=57439; verify fixed short output roots. The Iter29 manifest/smoke is historical and does not prove current source continuity or runtime consumption.

CURRENT_STAGE3_ROUTE_CONFLICT=UNRESOLVED. Root CLAUDE.md §5 specifies direct launch of stage3_T5Train/train_HG-Rec.py. Current trainer defaults CODE_PATH to generic results/stage2_RQ-VAE/curvature_RQ-VAE/item_sids.json (train_HG-Rec.py lines 124-126), derives generic/variant LOG_PATH and SAVE_PATH (156-159), and sets _LAUNCHER script to __file__ with generic launcher log (1175-1181). Iter29's wrapper instead overrides SID path, descriptive variant, short per-iteration result roots, launcher script, and launcher log (run_stage3_iter29.py lines 15-30). Do not claim direct route is compliant or use the Iter29 wrapper as proof of Iter31 routing. Later canonical S06/S11 must adjudicate a resolution without changing Stage3 model/evaluation; S01 makes no source edit and authorizes no launch.

ITER29_STAGE3_CAVEATS=Training metrics reported unknown_variant although wrapper supplied iter29_bounded_rational_additive_mapping; available outcome evidence did not establish a model/data identity failure. Launcher recorded nonfatal NCCL initialization and c10d barrier device-context warnings; Stage3/test/supervisor completed exit 0. Preserve caveats; do not describe startup as warning-free.
NO_TRAINING_AUTHORIZATION=TRUE
```

## Canonical decision and evidence limits

Iter29 is selected as both the parent condition and the sole direct historical comparator because it is the latest completed active Stage2/Stage3 lineage with a primary final test record and documented protocol linkage to Iter26. Iter30's cancellation record explicitly says it has no scientific result and no Stage2/Stage3/GPU execution. Iter29's exact result is below the current strict target; it is not an Iter31 result. Iter26 is verified but not selected because it is Iter29's older control and would compare against a different completed mechanism condition. Iter18 remains `HISTORICAL_NONCOMPARABLE`.

Protocol ID is fixed above to one normalized string. The word HRA labels the requested research direction only and does not decide the equation or contract. `single_factor` remains conditional on the later S05 one-factor adjudication. Historical declarations, hashes and launch summaries do not constitute fresh file-byte checks or proof of runtime use. No baseline rerun, replication, sweep, or root-cause iteration is authorized. Iter29 S14's matched-seed recommendation was superseded by Iter30's explicit cancellation and current skill policy forbidding replication.

S01 is a protocol lock only. It authorizes no code modification, preflight bypass, HRA equation, FCCR-1 amendment, GPU use, Stage2 run, Stage3 run, or baseline rerun. Training remains blocked pending all later canonical gates, current identity/path checks, root `CLAUDE.md` §6 Stage2 gradient-path checks, and separate Stage3 route/evaluation adjudication.