ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S01_PROTOCOL_LOCK/round_1/source_packet.md
STAGE_ID=S01_PROTOCOL_LOCK
ROUND=1

## Recommendation

- `PARENT_ITER=iter29`
- `PARENT_ROOT_COMMIT=21e0488dcad39b3fd277b071a12daec59b56ba2f` (the current Iter31 parent/root identity given in the frozen packet; not Iter29's historical run/source commit).
- `CANONICAL_BASELINE_ITER=iter29`
- `DIRECT_COMPARATOR_ITER=iter29`
- `PROPOSED_PROTOCOL_ID=HRA_iter31_vs_iter29_AmazonInstruments_2026-09_seed42`
- `EXPERIMENT_TYPE=single_factor` (conditional on later adjudication of the HRA contract and one-factor definition).
- `NO_TRAINING_AUTHORIZATION=TRUE`

The parent choice identifies the latest completed active Stage2/Stage3 lineage suitable to inherit. The comparator is the latest completed protocol-matched result, not an automatic continuation of Iter29's own historical control selection. Iter30 is a user cancellation with no result, not a parent experiment; its archived cancellation is not evidence that the Iter29 parent artifacts or results were superseded. The S00 snapshot records Iter30 as canceled with `SCIENTIFIC_RESULT=NONE` and no training, and explicitly bars reviving Iter29 S14's replication suggestion. (S00 snapshot, “Authority, current policy, and iteration boundary”; source packet lines 15–18.)

## Exact comparator and target

Use the primary final record:

`results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`

Its direct JSON values are `n_eval=57439`, `test_recall@10=0.05921064085377531`, `test_recall@5=0.03953759640662268`, `test_ndcg@5=0.026252776900288842`, and `test_ndcg@10=0.03257647953179143`. It is below the current strict adoption criterion `test_R@10 > 0.065` by `0.005789359146224693`. The strict criterion comes from root `CLAUDE.md` §2 and current skill §§0, 12; Iter29's older inclusive `>=0.065` wording does not carry forward. (Iter29 `test_final.json`; Iter29 `stage3_outcome_iter29.md`; frozen packet lines 14, 28; `CLAUDE.md` §2.)

## Why Iter29, after explicitly checking Iter26

Iter29 and Iter26 are not equivalent conditions: each is a different fixed-curvature mapping. However, Iter29's own protocol manifest and S14 records make Iter26 its prior direct control, with the unchanged protocol settings fixed around that mapping comparison. Their exact primary outcomes both use `n_eval=57439`: Iter26 R@10 `0.057017009349048554` in `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`; Iter29 R@10 `0.05921064085377531` in Iter29's exact `test_final.json`. The point difference is `+0.002193631504726755`. The historical review calls that within an approximate `~0.003` noise reference which was not independently estimated; there is one run per mapping. Therefore Iter29 is protocol-compatible as the later, completed comparison condition, but its advantage is not a causal estimate and neither establishes run variance. (Iter29 `protocol_manifest_iter29.md`, initial protocol fields and “Roles and comparison rule”; Iter26 `protocol_manifest_iter26.md`; both exact JSON files; Iter29 `global_review_after_iter29.md`, “Protocol-compatible evidence only”; frozen packet lines 26–40.)

This resolves rather than inherits Iter29's choice: Iter26 was appropriate to Iter29's historical mapping comparison, but Iter31's requested HRA-only candidate should be compared to its immediate completed parent condition, Iter29. Iter26 is older and would compare Iter31 to a different mapping, introducing an avoidable inherited-mechanism difference. Iter29's S14 matched-seed recommendation is not followed: Iter30's cancellation superseded that proposed direction, and current skill §2.9 forbids replication as an iteration purpose. No replication or second baseline run is proposed. (S00 snapshot; Iter29 S14 Judge `CANONICAL_DECISION` and `AUTONOMOUS_NEXT_ACTION`; skill §§2.9, 2.11.)

Iter18 is excluded: Iter29's manifest explicitly marks it `HISTORICAL_NONCOMPARABLE` and gives no Iter18 protocol manifest; its bridge documents a distinct cyclic/learnable-curvature and optimizer mechanism. Matching evaluation count or some Stage3 settings does not restore full protocol compatibility. (Iter29 manifest, `HISTORICAL_REFERENCE_*` and comparison rule; Iter18 `iteration_bridge.md`; frozen packet lines 34–40.)

## Locked protocol proposal

Dataset and evaluation identity: Amazon 2023 Instruments / 2026-09 Refactor, same Stage0 split/data identity, item ordering and Stage1 assets as Iter29, evaluated on the same Instruments test protocol with `n_eval=57439`. Rerun no baseline. Existing and candidate Stage3 comparison must use `test_R@10` and current strict threshold `>0.065`; Stage2 SID/Gini/collision/entropy statistics are descriptive only and must not gate or stop the run. (Iter29 manifest; root `CLAUDE.md` §§1–2; frozen packet lines 42–45.)

**Stage2** (inherited protocol settings, not an HRA specification): seed `42`; `MAX_GLOBAL_STEPS=100000`; `N_LAYERS=3`; codebook size `256` per layer; immutable warm-start path `results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`, expected SHA-256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`. Iter29's recorded execution was four-rank DDP, world size 4, per-GPU batch 640, total batch 2560, 24,587 items, 339,519 train transitions; preserve the actual configured values and launch protocol after confirming Iter31's source. The recorded run reached global step 100,000 and output integrity passed. No proxy threshold/early stopping. (Iter29 `protocol_manifest_iter29.md`; `stage2_output_integrity_iter29.log`, `TRAINING_COMPLETION`, `FINAL_LOG_METRICS`, `OUTPUT_ARTIFACTS`, `OUTPUT_INTEGRITY`; root `CLAUDE.md` §§2–3.)

Before any later Stage2 run, root `CLAUDE.md` §6 requires its one-checkpoint/one-batch gradient-path checks; that obligation is prospective and is not fulfilled by this S01 decision. All launch arguments and paths must be hard-coded; no CLI or environment overrides. Stage2 results must land only under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`, and the Iter31 source subtree must not contain `.pth`, `.npy`, or `item_sids.json`. (Root `CLAUDE.md` §§0–3, 6, 10; source packet lines 42–45.)

**Stage3**: hold the model/evaluation protocol unchanged from Iter29: seed `42`, `150` epochs, beam size `20`, `n_eval=57439`, same Instruments test set/metric implementation and same four-GPU torchrun topology. The recorded Iter29 launcher used `--nproc_per_node=4`, `--master_port=50201`, `CUDA_VISIBLE_DEVICES=0,1,2,3`; child environment recorded `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_TIMEOUT=3600`, and `TORCH_NCCL_BLOCKING_WAIT=1`. Its smoke also recorded BF16 autocast and the Instruments evaluation shape (24,587 items, vocabulary 784, four digits, beam 20); those are historical observations to compare against the actual locked source/config, not inferred guarantees. Before a later run, record current Stage3 source/runner identities and confirm these settings/evaluation identity remain unchanged; source-tree continuity alone is insufficient. Stage3 output must resolve under `results/stage3_T5Train/curvature_RQ-VAE_iter31/`, using short `curvature_RQ-VAE_iter31` paths rather than a mechanism-suffixed directory. (Iter29 manifest; `stage3_launcher_smoke_iter29.log`; root `CLAUDE.md` §§1, 5, 11; source packet lines 42–45.)

## Input identities to preserve and recheck

Iter29's protocol manifest records these identities; they are obligations for a subsequent protocol-compatible run, not proof that Iter31's runtime will consume them:

- Stage0 `train.parquet`: `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`.
- Stage0 `valid.parquet`: `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9`.
- Stage0 `test.parquet`: `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc`.
- Stage0 `items.parquet`: `6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3`.
- Stage1 `stage1_GeneEmbedding/output/sentence_t5.npy`: `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`.
- Stage1 `stage1_GeneEmbedding/output/item_ids.json`: `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30`.
- Stage2 warm-start checkpoint above: `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`.

The Iter29 Stage2 outputs establish historical linkage/assets: checkpoint SHA-256 `3a464a8de3ba3cec8dab0223b0705abb6e485567f19ddc1bf02db70a49df35bb`; `item_sids.json` `58665e08a97e122f47a77ed3616b67efdeb8eadba8253771a1498cb5e57487ff`; and the actual Stage3 input is `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments/sids_for_hgrec.npy`, SHA-256 `a8efe6e1315166486aff162e1bc6a5722b5d5d9ac1b537b83bc003ec0ea7cde1`. The integrity log notes a packet path typo (`out/sids_for_hgrec.npy`); do not reproduce that false path or create an alias. These are Iter29 artifacts, not Iter31 inputs to substitute for the candidate's new Stage2 output. (Iter29 `protocol_manifest_iter29.md`; `stage2_output_integrity_iter29.log`; `git_closure_iter29.md`.)

**Mandatory rechecks before any later run:** recompute/verify the four Stage0, two Stage1, and warm-start identities; verify item ordering and actual resolved paths; inspect the source/config and runtime wiring to establish that those exact assets are consumed. A historical manifest or hash list alone does not prove runtime consumption. Any mismatch, missing identity, or incompatible current Stage3 source/config blocks direct protocol ranking until resolved by the required later gates. Do not silently substitute inputs. (Frozen packet lines 42–45; Iter29 protocol manifest, “Locked paths, identity checks, and execution obligations.”)

## Provenance, metadata, and caveats

1. **Parent commit distinction:** record current Iter31 parent/root `21e0488dcad39b3fd277b071a12daec59b56ba2f` separately from Iter29 historical manifest/source root `612a5a41dfe524205b6afa46370ad0d0ce377882`, Iter29 Stage2/results closure commit `57b4a594d92435fd74fa4bba7c7c1439a33eb102`, and subsequent Iter29 S14 commit `fa7801c4d0f9a7e1d96da968cc7e0be891b84bd6`. Do not relabel any of the historical Iter29 commits as Iter31's parent root or imply the Iter29 run occurred at the later root. The packet describes `21e...` as the cancellation archive's previously verified GitHub commit, not as a fresh network verification performed here. (S00 snapshot; frozen packet user-direction boundary; Iter29 manifest and closure audit.)
2. **Stage3 source/runtime identity:** Iter29's manifest records Stage3 root/source commit `612a5a...`; its smoke records final trainer SHA-256 `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb` and runner SHA-256 `1cab47ff035e69aecf3aa5a2e1cfe7777c114a60b753b5e935453d90271b2ec7`. The smoke asserted launch routing/settings but did not start the child. These hashes are historic evidence, not a recorded runtime hash for the completed run. Lock current Iter31 Stage3 code/config/runner and effective inputs before launch. (Iter29 manifest; `stage3_launcher_smoke_iter29.log`.)
3. **Optional variant metadata:** the wrapper supplied `RQVAE_VARIANT=iter29_bounded_rational_additive_mapping`; the training metrics record `unknown_variant`, because its logger read `config.get('variant')`. Available evidence identifies no model/data identity failure from this discrepancy, but do not call the metadata fully consistent. (Iter29 `stage3_outcome_iter29.md`; frozen packet line 31.)
4. **Runtime warnings:** Iter29's Stage3 launcher reports nonfatal NCCL initialization warnings and a c10d barrier device-context warning; the Stage3 outcome records completion and exit 0. Preserve the warning caveat; it is not evidence by itself of invalid metrics. (Iter29 `logs/_stage3_launcher.log`; `stage3_outcome_iter29.md`; frozen packet line 31.)
5. **Uncertainty:** Iter29 and Iter26 are single observations; the approximate historical `~0.003` noise reference was not independently estimated. Do not claim causal benefit, variance estimation, or guaranteed target attainment. Iter31 cannot claim promotion absent its own completed, protocol-valid result strictly above `0.065`. (Iter29 S14 Judge; `global_review_after_iter29.md`; root `CLAUDE.md` §2.)

## Risks and compatibility limits

This is a defensible **protocol** baseline recommendation, not a claim that the proposed HRA implementation is mathematically compatible with active FCCR-1 or is a validated one-factor mechanism. The S00 snapshot says FCCR-1 remains active and HRA contract transition/equation are unresolved. S01 must not choose HRA math or bypass later S02/S03/S04/S05/S07/S08 gates. A contract/preflight incompatibility or any second conceptual change blocks training. Current Stage2 parent/source settings and runtime identity must be audited; preserve the requested HRA-only scope and unchanged Stage1/Stage3 in subsequent deliberation. (S00 snapshot “Authority…” and “Items intentionally unresolved”; skill §§3–8; source packet lines 15–18.)

The Iter29-to-Iter26 compatibility argument supports Iter29 as a later direct comparator because their protocol manifests/outcomes align on dataset, evaluation size, seed and Stage3 settings, and Iter29's manifest explicitly treats Iter26 as its direct control. That does not make their mechanisms identical, prove effect, or make Iter26 the controlling Iter31 comparator. Iter18 is not restored by equal `n_eval`; its Stage2 regime is different and lacks a protocol manifest. (Iter29 and Iter26 manifests; Iter29 global review; Iter18 bridge.)

## No execution authorization

`NO_TRAINING_AUTHORIZATION=TRUE`. This S01 candidate does not approve any HRA equation, contract transition, implementation, preflight bypass, GPU use, Stage2 training, Stage3 run, or test execution. No source artifacts are written by this report. Before any later training, the subsequent canonical 2+1 gates, explicit mechanism/contract and provenance resolution, gradient-path check, identity and path checks, and launch approvals are all still required. No replication, sweep, or root-cause iteration is recommended. Do not perform one unless a later valid independent adjudication under controlling rules permits it.

## Citation index (paths are repository-relative)

- Strict target, no CLI/environment overrides, Stage2 descriptives, gradients and path rules: `CLAUDE.md` §§0–3, 6, 10–11.
- Active FCCR-1, no replication, canonical A/B/Judge stages, hard gates: `skill://curvature-rqvae-iter` §§1–5, 2.9, 3–4, 9–12.
- Iter31 S00 facts and authority: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/source_snapshot_iter31.md`, “Authority, current policy, and iteration boundary”; S01 evidence constraints: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S01_PROTOCOL_LOCK/round_1/source_packet.md` lines 14–50.
- Iter29 protocol/assets/settings: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md` lines 1–68.
- Iter26 protocol: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/protocol_manifest_iter26.md` lines 1–end.
- Exact comparator records: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`; `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`.
- Stage2 identity and outcome: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/stage2_output_integrity_iter29.log`; closure artifact hashes and commit distinction: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/git_closure_iter29.md`.
- Stage3 protocol/outcome/metadata: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/stage3_outcome_iter29.md`; launcher settings/smoke boundaries and code hashes: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/stage3_launcher_smoke_iter29.log`; raw runtime warning evidence: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/_stage3_launcher.log`.
- Iter29-vs-Iter26 interpretation and superseded S14 action: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/global_review_after_iter29.md`; `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S14_GLOBAL_REVIEW/round_1/judge.md`.
- Iter18 exclusion: `stage2_RQ-VAE/curvature_RQ-VAE_iter18/logs/iteration_bridge.md`; Iter29 manifest “Roles and comparison rule.”
