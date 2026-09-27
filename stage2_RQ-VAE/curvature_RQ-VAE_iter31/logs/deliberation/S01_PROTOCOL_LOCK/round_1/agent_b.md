ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S01_PROTOCOL_LOCK/round_1/source_packet.md
STAGE_ID=S01_PROTOCOL_LOCK
ROUND=1

## Recommendation

- `RECOMMENDATION=Iter31 parent: current root revision 21e0488dcad39b3fd277b071a12daec59b56ba2f; sole historical direct comparator: Iter29; one fixed seed-42 run; no replication, sweep, or root-cause study.`
- `PROPOSED_PROTOCOL_ID=ITER31_HRA_SINGLE_SEED42_INSTRUMENTS_2026-09REF_VS_ITER29`
- `PARENT_ITER=Iter31 fresh direction; not Iter30 continuation`
- `PARENT_ROOT_COMMIT=21e0488dcad39b3fd277b071a12daec59b56ba2f`
- `COMPARATOR=Iter29 completed Stage3 result`
- `COMPARATOR_TEST_FINAL=results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`
- `COMPARATOR_METRIC=test_recall@10 (the project’s test_R@10)`
- `COMPARATOR_R@10=0.05921064085377531`
- `COMPARATOR_N_EVAL=57439`
- `NO_TRAINING_AUTHORIZATION=TRUE`

The protocol ID is a proposed label only, not an on-disk canonical manifest or permission to proceed. Judge C must produce the canonical S01 artifact before propagation.

## Primary evidence and parent identity

The frozen packet’s S01 scope says to choose the parent and one comparator from primary protocol/outcome evidence, while explicitly prohibiting HRA-equation choice, code changes, training authorization, replication, sweep, and root-cause work (`stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S01_PROTOCOL_LOCK/round_1/source_packet.md`, lines 14–18, 42–46). S00 records Iter30 as user-cancelled with no scientific result and identifies the next work as fresh Iter31 HRA deliberation (`stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/iteration_cancel_iter30.md`, lines 1–18, 26–30; `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/source_snapshot_iter31.md`, lines 19–23).

Use the current parent root commit separately from Iter29’s historical source/run revisions. The frozen packet and S00 candidate evidence report that the Iter30 cancellation archive was pushed and locally/remotely verified at `21e0488dcad39b3fd277b071a12daec59b56ba2f`; this is prior, packet-reported verification, not a fresh Git or GitHub check (`stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S00_SOURCE_TRUTH/round_1/agent_b.md`, lines 15–17, 79–83). This assignment performed no Git operation. Do not substitute the older Iter29 manifest root `612a5a41dfe524205b6afa46370ad0d0ce377882` for the Iter31 parent. Nor should the later Iter29 commits be called its parent: its code/results closure was `57b4a594d92435fd74fa4bba7c7c1439a33eb102`, followed by distinct S14 commit `fa7801c4d0f9a7e1d96da968cc7e0be891b84bd6` (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/git_closure_iter29.md`, sections “Post-commit and remote verification” and “S14 Global Review synchronization”; Iter29 protocol manifest lines 41–43).

## Why Iter29 is the direct comparator, not Iter26

Iter29’s exact primary `test_final.json` records `test_recall@10=0.05921064085377531` and `n_eval=57439`; Iter26’s exact file records `test_recall@10=0.057017009349048554` and the same `n_eval=57439` (`results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`, lines 3–7; Iter26 counterpart at `.../curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`, lines 3–7). Iter29’s recorded point difference over Iter26 is +0.002193631504726755, which its outcome characterizes as near-parity/neutral within an approximate, unestimated historical noise reference around 0.003; one run per condition proves neither causal gain nor run variance (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/stage3_outcome_iter29.md`, lines 11–18, 24–28; `logs/global_review_after_iter29.md`, “Protocol-compatible evidence only”).

I evaluated, rather than automatically inheriting, Iter29’s own Iter26 control. Iter26 was recorded as using the same dataset/version, seed 42, 100,000 Stage2 steps, 3 layers × 256 codes, Stage3 seed 42, 150 epochs, beam 20, and `n_eval=57439` (`stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/protocol_manifest_iter26.md`; Iter29 protocol manifest lines 7–17, 32–49). Iter29’s S01 calls Iter26 its sole direct control and states its other protocol conditions were held to Iter26 apart from its separately adjudicated fixed-curvature mapping (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md`, lines 4–17, 32–50). Iter29 therefore supplies the most recent completed, active, well-documented Stage2/Stage3 result in this sequence and matches the immediate fixed-protocol family and evaluation identity. It is preferable as the one historical point of reference for the new forward candidate; Iter26 remains useful context but should not become a second comparator.

This is protocol compatibility for one historical comparison, not a claim of exact causal identification: Iter29 and Iter31 have different Stage2 mechanisms, only one observation each, and possible runtime/stochastic differences remain. Iter29’s manifest records root/source revision `612a5a...`; its closure and subsequent S14 review have distinct commit identities, so this is not a claim that Iter29 ran at the current Iter31 root revision. The current Stage3 source/runner and runtime identity must be freshly pinned for Iter31.

Iter18 is not eligible as an alternative direct baseline: Iter29’s manifest says the expected Iter18 protocol manifest is absent and classifies it `HISTORICAL_NONCOMPARABLE`; its primary bridge describes cyclic curvature and a distinct P18B fixed-initial-curvature AdamW beta2 mechanism, rather than the fixed-curvature setup at issue (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md`, “HISTORICAL_REFERENCE” and “FCCR-1 and remaining provenance limits”; `stage2_RQ-VAE/curvature_RQ-VAE_iter18/logs/iteration_bridge.md`, “Post iter18” sections). Equal test cardinality or matching Stage3 settings cannot repair a Stage2 protocol mismatch.

## Locked protocol proposal

### Data and model inputs

- Dataset/version: `Amazon_2023_Instruments / 2026-09 Refactor`.
- Preserve the Iter29/Iter26 protocol’s Stage2 input model, optimizer, loss, Sinkhorn, training data, and non-mechanism configuration; the HRA proposal must not be treated as a settled operation or FCCR-1 transition by this S01 decision. Any later difference is subject to S02–S08 adjudication.
- Keep Stage2 and Stage3 configuration/paths hard-coded in source. No CLI arguments, flags, or environment overrides (root `CLAUDE.md` §1; Iter31 packet lines 42–46).

### Stage2

- `SEED=42`.
- `MAX_GLOBAL_STEPS=100000`; complete all steps if a later authorized run remains valid. There is no SID/Stage2-quality hard gate or early stop: Gini, collision, unique-SID, entropy and other SID statistics are descriptive only (root `CLAUDE.md` §§2, 6; Iter31 packet lines 18, 46).
- `N_LAYERS=3`, `CODEBOOK_SIZE_PER_LAYER=256`.
- Four-rank DDP; Iter29’s recorded launch used its project launcher on port `50200`; its completion log records world size 4, per-GPU batch 640 and total batch 2560 (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/stage2_execution_plan_iter29.md`, “Locked settings” and launch specification; `logs/stage2_output_integrity_iter29.log`, `TRAINING_COMPLETION`). Preserve the corresponding Iter31 project-mandated no-argument launcher/protocol, subject to a later canonical run plan; no ad hoc CLI/environment override.
- Immutable warm start: `results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`, declared SHA-256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b` (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md`, lines 32–39; execution-plan “Warm-start use”). Recheck the actual file and prove the actual resolved Stage2 input path and checkpoint consumption immediately before and during any later authorized run.
- Training must not begin until all required deliberation gates are canonical and the project-mandated single-checkpoint/single-batch nonzero gradient-path checks pass (root `CLAUDE.md` §6). S01 does not certify that check for Iter31.

### Stage3

Preserve the comparator’s fixed recommendation model/evaluation protocol: Stage3 `SEED=42`, `NUM_EPOCHS=150`, `BEAM_SIZE=20`, `n_eval=57439`, Amazon Instruments test split, `TOPK_LIST=[5,10]`; no Stage3 model, data, or evaluation change is part of this request (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md`, lines 41–47; `logs/stage3_evaluation_plan_iter29.md`, “Fixed Stage3 protocol”). The Iter29 evaluation plan further records `BATCH_SIZE=4096`, `INFER_SIZE=1024`, `EARLY_STOP=10`, `FAST=True`, `BF16=True`, `COMPILE=False`, LR `0.003`, weight decay `0.05`, scheduler enabled with warmup `0.10` and minimum LR factor `0.20`; preserve the model configuration (4 encoder/decoder layers, `D_MODEL=128`, `D_FF=1024`, 6 heads, `D_KV=64`, dropout `0.1`, ReLU) and current checkpoint-selection behavior. `NO_EVAL=True`, `SKIP_TEST=False`, `EXCLUDE_HISTORY=True`; validation is skipped by this recorded config, while final beam test remains enabled (`logs/stage3_evaluation_plan_iter29.md`, “Fixed Stage3 protocol”). Lock current Stage3 source/runner hashes and resolved runtime config before any later launch; the old smoke is not a runtime code hash for the actual Iter29 execution.

### Acceptance criterion

The only adoption criterion is completed, protocol-valid `test_R@10 > 0.065` (strict). Iter29’s historical manifest/outcome used the older inclusive `>=0.065`; that wording is superseded by root `CLAUDE.md` §2 and the Iter31 S01 packet lines 14 and 46. Iter29 is below the strict target by 0.005789359146224693; this is its historical result, not an Iter31 result (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/stage3_outcome_iter29.md`, lines 24–28; exact test JSON lines 3–7).

## Input identity hashes and provenance obligations

These are historical Iter29 manifest/prelaunch identities, not claims that this Agent freshly recomputed or verified current file bytes. Before any later Stage2 or Stage3 run, rehash the files and verify that the actual resolved consumers use them; mismatch blocks a protocol-valid comparison. The source packet explicitly warns that prior declarations do not prove runtime consumption (`stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S01_PROTOCOL_LOCK/round_1/source_packet.md`, lines 26–29, 42).

| Input/asset | Historical identity SHA-256 | Provenance / obligation |
|---|---|---|
| Stage0 `train.parquet` | `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815` | Rehash, resolve and confirm training loader uses this exact file. |
| Stage0 `valid.parquet` | `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9` | Rehash; preserve its role/config even if validation is not consumed in the locked Stage3 mode. |
| Stage0 `test.parquet` | `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc` | Rehash and resolve exact test consumer; confirm 57,439 evaluation examples. |
| Stage0 `items.parquet` | `6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3` | Rehash and confirm item-universe/order identity. |
| Stage1 `sentence_t5.npy` | `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` | Rehash and prove exact Stage2 embedding input use. |
| Stage1 `item_ids.json` | `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30` | Rehash; prove row/item alignment to embeddings and actual resolver use. |
| Immutable Iter8 warm-start | `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b` | Rehash before launch; during run require log evidence of loading this exact checkpoint with positive transferred tensor count. |

Source: Iter29 protocol manifest lines 19–39. Exact locations are declared there under `stage1_GeneEmbedding/output/` and `results/stage0_build_parquet/`; do not substitute copied/stale inputs.

### Iter29 comparator output identity/provenance

- Stage2 `item_sids.json`: `58665e08a97e122f47a77ed3616b67efdeb8eadba8253771a1498cb5e57487ff`.
- Stage2 `out/rqvae/instruments/rqvae_best.pth`: `3a464a8de3ba3cec8dab0223b0705abb6e485567f19ddc1bf02db70a49df35bb`.
- Actual Stage3 SID array at `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments/sids_for_hgrec.npy`: `a8efe6e1315166486aff162e1bc6a5722b5d5d9ac1b537b83bc003ec0ea7cde1`.
- Stage3 `_stage3_launcher.log`: `fe9a49a8d7a4f46ffc0a91d20cf6a4768696bc54b778774714231f029f283806`; exact final `test_final.json`: `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`; `training_metrics.jsonl`: `412860c203e20498e17ffd9c7a87d5e39836fcee563f1835f503e1df65a1b88f`; best Stage3 checkpoint: `70facbde7de7961bbe71e23be6f4f951f30e696b959d04be66cdf7d32dab7e94` (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/stage2_output_integrity_iter29.log`, `OUTPUT_ARTIFACTS`; `logs/git_closure_iter29.md`, “Required deliverables and primary checks”). These identify Iter29 assets only and are not Iter31 expected outputs.

For Iter31, all Stage2 deliverables must be written under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`, Stage3 deliverables under `results/stage3_T5Train/curvature_RQ-VAE_iter31/`; source iter subtree may contain code/configs/scripts/logs and allowed input copies but no `.pth`, `.npy`, or `item_sids.json`. Stage3 `LOG_PATH`/`SAVE_PATH` use the short `curvature_RQ-VAE_iter31` root, while any variant keeps its full descriptive name (root `CLAUDE.md` §§0, 10–11; Iter31 packet lines 44–46). No result artifacts were written by this report.

## Caveats and risks

1. **Current versus historical source revision.** Iter31 parent root is `21e0488...` per packet-reported archive evidence, while Iter29’s protocol source root was `612a5a...`, its later closure commit `57b4a...`, and its separate S14 review commit `fa7801...`. The Iter29 Stage3 smoke recorded the final patched `train_HG-Rec.py` digest `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb` and wrapper digest `1cab47ff035e69aecf3aa5a2e1cfe7777c114a60b753b5e935453d90271b2ec7`; the smoke itself launched no real torchrun child (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/stage3_launcher_smoke_iter29.log`, `FINAL_PATCHED_SOURCE_SHA256` and `TORCHRUN_CHILD_STARTED=False`). Pin current Iter31 trainer, wrapper, and active settings before execution. Do not imply the smoke hash establishes the runtime file bytes used by Iter29 training.
2. **Historical input provenance is bounded.** Iter29 S03 accepted historical method/value provenance at medium confidence, not checkpoint-level replay, independent recomputation, or byte-identity proof for branching/raw residual inputs. Treat Stage0/Stage1/warm-start file digests as recorded historical identities, and independently establish present bytes plus runtime consumption. Iter29 manifest states the raw-residual provenance limit explicitly (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md`, “FCCR-1 and remaining provenance limits”); Iter31 S01 does not resolve HRA input semantics.
3. **Iter29 metadata mismatch.** Its wrapper passed `RQVAE_VARIANT=iter29_bounded_rational_additive_mapping`, but training metrics report `unknown_variant` because the logger reads `config.get('variant')`. The outcome judged this optional-metadata discrepancy not evidence of model/data identity failure; report it without erasing the caveat (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/stage3_outcome_iter29.md`, lines 30–32).
4. **Warnings.** Iter29 launcher logs show nonfatal NCCL initialization warnings and a c10d barrier device-context warning; process/evaluation/supervisor completed exit 0. It is inaccurate to describe startup as warning-free (`stage3_outcome_iter29.md`, line 32; `git_closure_iter29.md`, Stage3 artifact section).
5. **Single-run interpretation.** The Iter29–Iter26 score difference is an observed point delta, not causal evidence; the cited ~0.003 historical noise reference is approximate and unestimated. Neither baseline score reaches current strict target.
6. **No stale protocol inheritance.** Iter29’s S14 recommended a three-pair replication, but Iter30 was explicitly cancelled and the active skill forbids replication; S00 says not to revive that recommendation (`stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/source_snapshot_iter31.md`, lines 19–23; Iter31 source packet lines 14–18). Iter30 has no score and is not a comparator.
7. **Mechanism/contract boundary.** This protocol decision does not choose HRA’s equation, assert its validity, or modify FCCR-1. HRA’s status and contract transition remain later-stage questions; subsequent gates must decide these before implementation or training. No Stage2/Stage3 authorization is granted.

## Explicit pre-run identity rechecks

Before any subsequent run, after the appropriate canonical S01–S09/S11 approvals:

1. Rehash the four Stage0 parquet inputs, both Stage1 files, and immutable Iter8 checkpoint against the recorded digests above; record sizes, mtimes, resolved paths, and current SHA-256. A mismatch or missing file blocks execution.
2. Confirm the Stage2 config/runtime resolves to those exact files and records the actual data and embedding fingerprints; verify Stage1 item ordering/IDs align with embeddings and Stage0 item universe. Prove the real run loads the locked warm-start by exact path and positive tensor count—do not rely on manifest claims alone.
3. Lock the current Iter31 Stage2 source/config and Stage3 code/runner/runtime identities separately from parent commit and historical run/closure commits. Recheck Stage3 test split, cardinality `n_eval=57439`, metric implementation, beam 20, and fixed short result roots. Exact Stage3 source continuity is not inferred solely from the Iter29 manifest.
4. Verify all settings remain hard-coded, with no CLI/environment overrides. Check the Stage2 `RQVAE_OUT_DIR` resolves to the Iter31 results short root and Stage3 paths likewise resolve beneath the Iter31 short root. Keep generated model/SID artifacts out of source subtree.
5. For the actual Stage2 authorization gate, complete the required one-checkpoint/one-batch nonzero gradient tests and all active contract/MVG/preflight gates. Stage2 quality statistics remain descriptive. Stage3 must use its later independent adjudication and single authorized run.

## Final decision

`PARENT_ROOT_COMMIT=21e0488dcad39b3fd277b071a12daec59b56ba2f` (reported as the Iter30 cancellation archive/current Iter31 root; not freshly queried here).

`DIRECT_HISTORICAL_COMPARATOR=Iter29`

`COMPARATOR_TEST_FINAL=results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`

`COMPARATOR_METRIC=test_recall@10 / test_R@10`

`COMPARATOR_R@10=0.05921064085377531`

`COMPARATOR_N_EVAL=57439`

`ADOPTION_TARGET=test_R@10 > 0.065` (strict)

`NO_TRAINING_AUTHORIZATION=TRUE` — S01 protocol recommendation only. No code, result, or repo artifact was written; no tests, formatter, linter, training, or project-wide commands were run.