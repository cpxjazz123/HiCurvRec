# Iter30 S01 Protocol Packet — Round 1

STAGE_ID=S01_PROTOCOL_LOCK
ROUND=1
ITERATION=30

## Objective and gate

Produce a canonical protocol manifest for the S14-authorized finite three-pair FCCR-1 replication. Do not implement code or launch training. Independently determine whether the registered experiment is feasible without violating one-factor isolation, no-CLI/no-environment-override rules, source/result path constraints, or raw-input provenance requirements. This design has two already-registered mappings and three new matched seed pairs (43,44,45), hence six complete Stage2/Stage3 pipelines if all gates pass. No silent reduction, overwrite, uncontrolled restart, or user-direction state is allowed.

## Controlling canonical inputs

- Use only S00's canonical `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/source_snapshot_iter30.md` and Judge, current repository `CLAUDE.md`, active `curvature-rqvae-iter` skill, iter29 S14 canonical Judge/report, and direct primary source/config/result records below.
- User-level target is inclusive `test_recall@10 >= 0.065`; repository and skill strict `> 0.065` wording is lower priority. Report the two separately if discussing source wording.
- FCCR-1 stays fixed, closed-form, non-trainable, and time-invariant. No learned/scheduled curvature, regularizer, optimizer/loss/Sinkhorn/behavior change, mechanism stacking, or Stage1/Stage3 change. S14 authorizes only the existing iter26 and iter29 mapping arms.

## S14-locked scientific comparison

- Seed pairs are exactly 43, 44, 45. Within each pair, both mapping arms must use the same Stage2 seed and Stage3 seed. Other Stage1/Stage0 identities and ordering, warm start, Stage2/Stage3 code and settings, evaluation set, runtime configuration, and metric computation are held identical. Only mapping differs conceptually within a pair.
- Iter26 mapping: `s_l=log1p(B_l)/log1p(m_l_raw/min(m_raw))`; `z_l=(s_l-mean(s))/(std(s)+1e-12)`; `c_l=clip(0.5*exp(0.2*z_l),0.05,1.5)`. Fixed vector `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`.
- Iter29 mapping: `c_l=0.05+1.45*0.5*(B_l/(B_l+2.0)+m_l_raw/(m_l_raw+0.1))`. Fixed vector `[1.3660953164241916,0.7347829661951981,0.6439958072706683]`.
- Shared L0/L1/L2 input vectors approved in S14: branching `[19.324911558712664,1.4605688962651735,1.0148104414712726]`; historical raw-residual medians `[1.0,0.10941,0.09331]`. Re-establish semantics/source/value under S03 before propagation. Provenance ceiling remains medium-confidence historical method/value evidence, not checkpoint reproduction or historical byte identity; do not substitute normalized scales `[0.001,0.932889,1.0]`.
- S14 allows seed-42 prior evidence in pooled analysis only if S01 proves required protocol/identity compatibility; otherwise exclude it from the prospective paired estimate. The three new pairs remain mandatory and cannot be replaced.

## Exact historical result and baseline records

- Iter26 direct-control result: `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`, `n_eval=57439`, `test_recall@10=0.057017009349048554`.
- Iter29 mapping result: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`, `n_eval=57439`, `test_recall@10=0.05921064085377531`.
- Iter29's canonical protocol manifest identifies iter26 as its sole direct control and iter18 `HISTORICAL_NONCOMPARABLE`; do not use iter18 for direct comparison. S14 classifies iter29 `ACTIVE_NEUTRAL + PROMOTION_FAIL`; one observation per mapping does not establish variance, causal benefit, or family failure.
- Iter29 S01 baseline fields: Stage1 path/hashes and Stage0 paths/hashes are recorded at `logs/protocol_manifest_iter29.md:19-30`; warm start path/hash at lines 38-39; Stage2 seed 42, max 100000, 3 layers × 256 at lines 32-39; Stage3 code commit/settings seed42/150 epochs/beam20/n_eval57439 at lines 41-50. Iter26 S01 corroborates the historical settings and exact baseline record.

## Directly inspected implementation facts and launch constraints

1. Iter29 Stage2 `curvature_RQ-VAE.py` currently hardcodes `SEED=42`; it seeds `torch` and NumPy with `SEED + rank` and `DistributedSampler(seed=SEED)`. The internal launcher uses four ranks, port 50200, and `logs/train_migrated.log` (source lines 99-111, 543-575, 908-934). `curvature_config.py` hardcodes `PYTHONHASHSEED=42`, CUBLAS workspace, GPU visibility, and absolute input/output paths. No CLI arguments are supported.
2. Stage2 shape/settings in canonical iter29 S05/source: input 768, hidden `[512,256,128]`, embedding 32, 3 quantizer layers, 256 codes/layer, batch 640/GPU, 4 ranks, 100000 global steps, commitment weight 1.0, Sinkhorn `sk_eps=0.05`/3 iterations, AdamW `lr=1e-3`/weight decay `1e-4`, no new loss/optimizer; mapping-mediated curvature use only. Stage2 quality metrics are descriptive-only.
3. Stage2 imports `curvature_config.py` by module name and currently validates one iter29 mapping against fixed iter29 input/contract paths. A six-run design must hard-code each run's seed, map arm, unique output paths, and per-run logs without CLI/environment selection and without stale artifact reuse. The existing DDP port/log path and Step0 cleanup must not collide across the six serial runs.
4. Stage2 warm start is hardcoded to `results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`. The current code loads it and skips legacy `c_layer_scale`/`_fixed_c` buffers, but if the file is absent it only warns and continues from random initialization (iter29 source lines 626-665). The path exists; fresh SHA256 matches locked `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`. Protocol and implementation must fail closed on absent/mismatched checkpoint and verify positive loaded tensor count for every run; do not accept random-init fallback.
5. Stage2 source hashes at packet creation: iter26 `curvature_RQ-VAE.py` `ed629824d5a0308f4018b6dbaaa24303690a3597def7d1bcbf9abd38c3aca10f`, `curvature_config.py` `4d6b3e06aa2f408a00664d971d64ed83d0e88ef5f88b852be97302a0e2786086`; iter29 code `2469b193bee1c4352ba180540c305c9ea837043afafd1b1585ea7b8aff805f4d`, config `b41bf1fff2b481abf846d71a43c30c7ba34d0d25bd93bc04f70df486e3163f7a`.
6. Stage3 `train_HG-Rec.py` hardcodes `SEED=42`, 150 epochs, `TOPK_LIST=[5,10]`, `BEAM_SIZE=20`, `NO_EVAL=True`, `SKIP_TEST=False`, batch 4096, infer batch 1024. Its `set_seed()` applies the seed to Python/NumPy/Torch. The four-rank launcher is hardcoded by root rules at port 50201; no Stage3 source/config changes are authorized. Iter29 Stage3 wrapper `scripts/run_stage3_iter29.py` sets SID path, `RQVAE_VARIANT`, LOG_PATH/SAVE_PATH under the short iter29 result root, and launcher log. Stage3 source hash is `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`; iter29 wrapper hash `1cab47ff035e69aecf3aa5a2e1cfe7777c114a60b753b5e935453d90271b2ec7`.
7. Stage3 has an existing optional metadata caveat: iter29's wrapper set the intended variant but training metrics recorded `unknown_variant`; S14/S12 classified this as nonfatal metadata mismatch. Preserve as a disclosed limitation; do not modify Stage3 trainer to fix it. The paired runs must use unchanged Stage3 code/settings and exact Stage2 SID input for their own arm.
8. Iter29 `scripts/computed_behavior_branching.json` stores approved input vectors and explicit limitations. Branching provenance cites iter8 raw SIDs plus Stage0 train data; the recorded iter8 raw SID `.npy` is absent in the current result inventory. Raw medians cite historical iter1 calibration log/checkpoint path; historical checkpoint/raw input byte identity and independent replay are unavailable. S03 must re-adjudicate this medium-confidence source/value evidence under the existing S14 authorization; if it fails the active skill's hard provenance gate, stop before MVG/Stage2 and close/abort according to skill, not by substituting new values.

Fresh `sha256sum` checks at S00/early S01 matched the recorded current identities:
- Stage1 `sentence_t5.npy`: `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`.
- Stage1 `item_ids.json`: `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30`.
- Stage0 `train.parquet`: `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`.
- `valid.parquet`: `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9`.
- `test.parquet`: `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc`.
- `items.parquet`: `6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3`.
- Iter8 warm-start: `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`.
These establish file identity at check time only, not the resolved runtime path/use at execution.

## Required protocol-manifest decisions

The canonical S01 manifest must include all skill §4 fields and additionally make the multi-run design auditable: the three exact seed pairs, control/candidate map labels, each Stage2/Stage3 seed, pairwise-only factor difference, input/data/warmstart identity hashes, Stage2/Stage3 source/config hashes, complete settings, baseline-vs-paired-control semantics, output/log/checkpoint path convention, non-overwrite execution order, no-CLI launcher method, per-run Stage2 gradient check obligation, no early stopping on SID proxies, prospective paired outcome calculations and target rule, and seed-42 pooled-analysis decision. Do not claim future test_final paths or run outcomes already exist.

Expected output roots (nested run-specific directories are necessary, but exact layout must be adjudicated):
- Stage2: `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/<arm>_seed<43|44|45>/...`.
- Stage3: `results/stage3_T5Train/curvature_RQ-VAE_iter30/<arm>_seed<43|44|45>/{logs,ckpt}/...`.
- Source: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/` may contain only source/config/scripts/logs/input copies; no generated `.pth`, `.npy`, or `item_sids.json`. Preserve full `MECHANISM_NAME=iter30_<description>` and short output roots.
Root `CLAUDE.md` §13 commit/push rules apply to the final complete block; any `.gitignore` exception must be checked and iter30-specific.

## Questions Agent A/B must resolve independently

1. Is S14's six-run block feasible with a clean no-CLI/no-env hardcoded design under both short roots, and how are source/config, per-run seeds, output paths, launchers/logs, and serial execution isolated without changing model/runtime factors?
2. Which iteration/code commit is the implementation parent versus direct paired control/baseline? Resolve `PARENT_ITER`, `PARENT_COMMIT`, `CANONICAL_BASELINE_ITER`, and historical known baseline file/metric without conflating the seed-42 baseline with new within-seed controls.
3. Can the existing seed-42 iter26/iter29 pair be pooled with seeds 43-45 after auditing all relevant protocol/source/output differences? If not fully established, exclude it from prospective paired estimates.
4. Is the optional random-init fallback compatible with S14's fixed warm-start requirement? Specify the fail-closed implementation/preflight requirement while preserving no same-iteration retuning.
5. Are Stage1/Stage0/warmstart current hashes and actual consumers sufficiently recorded, and what remains as a prelaunch recheck? Preserve the raw-residual medium-confidence limitation and determine whether S03 can satisfy the active contract without changing the S14-locked inputs.
6. State the exact paired analysis, user-target interpretation, block completion/abort rules, and no-overwrite/output capture requirements. Do not promise target attainment or statistical power.

## Candidate instructions

Agent A and Agent B independently produce a complete `protocol_manifest_iter30.md` proposal and rationale from the same packet and primary evidence. Each writes only `logs/deliberation/S01_PROTOCOL_LOCK/round_1/agent_a.md` or `agent_b.md`, begins with the exact skill metadata headers, and must not read the other draft. Do not edit code or the canonical manifest. Flag any infeasibility or hard-gate failure with direct evidence; if the S14 design cannot be implemented without changing it, do not silently narrow it. No execution is authorized by this stage.
