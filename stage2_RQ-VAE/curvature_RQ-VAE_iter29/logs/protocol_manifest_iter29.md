# Iter29 Protocol Manifest — S01 canonical lock

```text
PROTOCOL_ID=FCCR-1_iter29_bounded_alternative_closed_form_vs_iter26
STATUS=CANONICAL_S01_LOCK
EXPERIMENT_TYPE=single_factor
DATASET_VERSION=Amazon_2023_Instruments / 2026-09 Refactor
PARENT_ITER=iter26
PARENT_ROOT_COMMIT=612a5a41dfe524205b6afa46370ad0d0ce377882
PARENT_COMMIT=612a5a41dfe524205b6afa46370ad0d0ce377882
DIRECT_CONTROL_ITER=iter26
CANONICAL_BASELINE_ITER=iter26 (sole protocol-valid direct control)
CANONICAL_BASELINE_TEST_FINAL=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json
CANONICAL_BASELINE_TEST_R@10=0.057017009349048554
HISTORICAL_REFERENCE_ITER=iter18
HISTORICAL_CLASSIFICATION=HISTORICAL_NONCOMPARABLE
USER_SUCCESS_CRITERION=test_recall@10 >= 0.065

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

STAGE2_SEED=42
STAGE2_MAX_GLOBAL_STEPS=100000
RQ_LAYERS=3
CODEBOOK_SIZE_PER_LAYER=256
STAGE2_CONTROL_PROTOCOL=iter26; all conditions held fixed except the separately adjudicated FCCR-1 mapping
STAGE2_QUALITY_METRICS=DESCRIPTIVE_ONLY; no Stage2 quality gate
STAGE2_WARMSTART_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth
STAGE2_WARMSTART_SHA256=189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b

STAGE3_ROOT_CODE_COMMIT=612a5a41dfe524205b6afa46370ad0d0ce377882
STAGE3_CODE_COMMIT=612a5a41dfe524205b6afa46370ad0d0ce377882
STAGE3_SUBTREE_CONTINUITY=tracked stage3_T5Train subtree unchanged from iter26 recorded Stage3 code commit 0052f4bff117e0aa16c9a1080be1014fbd417cc6 through current parent root commit
STAGE3_SEED=42
STAGE3_EPOCHS=150
STAGE3_BEAM_SIZE=20
STAGE3_N_EVAL=57439
DIRECT_CONTROL_TEST_FINAL=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json
DIRECT_CONTROL_TEST_RECALL_AT_10=0.057017009349048554
DIRECT_CONTROL_N_EVAL=57439

HISTORICAL_REFERENCE_TEST_FINAL=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json
HISTORICAL_REFERENCE_TEST_RECALL_AT_10=0.05988962203380978
HISTORICAL_REFERENCE_N_EVAL=57439
HISTORICAL_REFERENCE_PROTOCOL_MANIFEST=ABSENT at stage2_RQ-VAE/curvature_RQ-VAE_iter18/logs/protocol_manifest_iter18.md
```

## Roles and comparison rule

`DIRECT_CONTROL_ITER=iter26` is the sole direct comparator and the immediate parent for this one-factor experiment. The S14 Global Review authorizes one bounded alternative FCCR-1 closed-form mapping, using the same behavior-branching and raw-residual-median inputs, with all other protocol conditions held fixed to iter26. The mapping itself, its equation, constants, substitutions, and numerical outputs are **pending S02 and subsequent adjudication**; this manifest does not select or imply them.

`HISTORICAL_REFERENCE_ITER=iter18` retains its exact observed score and result path for historical context only. Its classification is `HISTORICAL_NONCOMPARABLE` for direct ranking: the expected iter18 protocol manifest is absent, and the primary `iteration_bridge.md` records a different cyclic/learnable-curvature regime and P18B fixed initial-curvature AdamW beta2 mechanism. Equal `n_eval`, matching recorded Stage3 settings, or Stage3 subtree continuity do not establish equivalent Stage2 representations or full protocol identity. Do not report an iter29-versus-iter18 direct delta or rank it as the control.

## Locked paths, identity checks, and execution obligations

The Stage1 and Stage0 paths and byte hashes above were independently checked against the present files and match the S00 packet. The iter8 warm-start path and SHA-256 likewise match the locked immutable checkpoint. Before any Stage2 launch, recheck that the actual Stage1/Stage0 files and warm-start checkpoint have these exact identities and that resolved iter29 paths/configuration actually consume them. Any mismatch blocks a protocol-valid direct comparison; declarations alone do not prove runtime use. Do not substitute or rewrite the warm-start checkpoint. Preserve iter26 seed policy (42), 100,000 Stage2 steps, 3×256 quantization, and the listed Stage3 settings. Stage2 quality metrics remain descriptive only, consistent with root project policy.

The current parent root commit and locked Stage3 root code commit are both `612a5a41dfe524205b6afa46370ad0d0ce377882`. The tracked Stage3 subtree is continuous from iter26's recorded Stage3 code commit `0052f4bff117e0aa16c9a1080be1014fbd417cc6` through the current parent. This establishes tracked Stage3 source continuity only, not full Stage2 identity, untracked/runtime configuration identity, or deterministic run equivalence.

The iter26 exact `test_final.json` records `test_recall@10=0.057017009349048554` and `n_eval=57439`. The iter18 exact `test_final.json` records `test_recall@10=0.05988962203380978` and `n_eval=57439`; it remains historical-only as specified above. User success is inclusive: `test_recall@10 >= 0.065`; only a completed, protocol-valid Stage3 result can establish success. Neither the target nor a positive improvement is guaranteed.

## FCCR-1 and remaining provenance limits

FCCR-1 remains controlling: fixed, closed-form, precomputed, non-trainable, time-invariant curvature derived from behavior branching and raw residual median. No cyclic/learnable curvature, optimizer or Sinkhorn change, new auxiliary loss, Stage1/Stage3 change, or mechanism stacking is authorized. The alternative mapping's equation and numeric outputs are not part of S01 and remain pending S02+ adjudication.

The contract records `raw_residual_median` as an input, but its provenance is **not independently certified by this protocol lock**. Historical calibration logs report medians `[1.0, 0.10941, 0.09331]`, while the cited baseline checkpoint is not currently present and the iter26 branching artifact aggregates sources. S03 MUST independently audit the raw-residual-median provenance and approve the specific input evidence before the mapping is approved or propagated. Do not treat the historical calibration values or contract declaration alone as a verified reproduction.

Results are stochastic single-run observations. Matching seeds and locked settings do not eliminate runtime/randomness differences, estimate run-to-run variance, or prove a causal mechanism effect; Stage3 subtree continuity does not establish complete protocol equality. Iter26's score is below the user target and provides no guarantee that iter29 will meet it. Report deviations and limits rather than inferring success or causality.

## Primary records

This lock adjudicates the independent Agent A and Agent B proposals against `logs/deliberation/S01_PROTOCOL_LOCK/round_1/source_packet.md`, `logs/source_snapshot_iter29.md`, the S00 Judge decision, `stage2_RQ-VAE/curvature_RQ-VAE_iter28/logs/global_review_after_iter28.md` and its S14 Judge decision, iter26's `protocol_manifest_iter26.md`, `mechanism_contract_iter26.json`, and exact Stage3 `test_final.json`, plus iter18's `iteration_bridge.md` and exact Stage3 `test_final.json`. The exact measured file hashes listed above are the immutable pre-run identity obligations; the parent/code-commit and run settings are locked from those primary records.