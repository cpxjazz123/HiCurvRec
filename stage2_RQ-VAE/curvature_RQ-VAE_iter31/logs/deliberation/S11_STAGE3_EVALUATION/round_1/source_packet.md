# Iter31 S11 Stage3 evaluation — frozen source packet

```text
STAGE_ID=S11_STAGE3_EVALUATION
ROUND=1
ITERATION=31
PROTOCOL_ID=ITER31_HRA_SINGLE_SEED42_INSTRUMENTS_2026-09REF_VS_ITER29
PARENT_ITER=iter29
CANONICAL_BASELINE_ITER=iter29
S10_VERDICT=MERGE_AB
STAGE3_AUTHORIZATION=NOT_YET_GRANTED
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
USER_INPUT_REQUIRED=NO
```

## Objective and scope

Independently audit whether the completed Iter31 Stage2 candidate is wired to the unchanged, protocol-compatible Stage3 consumer/evaluator, whether all Stage3 outputs can stay beneath the required short Iter31 results root, and whether one Stage3 run is safe to authorize. Agent A and Agent B must use this same frozen packet independently and in parallel, may inspect the cited primary files directly, and must write only their own candidate artifact in this round. Do not edit source, run a Stage3 model/train/test, launch GPU work, rerun Stage2/S08, perform a baseline rerun, or classify/promo­te the candidate. Judge C starts only after both candidates complete and writes the canonical `logs/stage3_evaluation_plan_iter31.md` plus this round's `judge.md`. A Judge PASS may authorize exactly one Stage3 execution; it does not itself establish an outcome or promotion.

Execution DAG:

```text
PARALLEL_GROUP_1=Agent A and Agent B independently audit this identical packet and primary sources; no GPU or shared source writes.
PARALLEL_GROUP_2=Judge C verifies both completed audits against primary repository/output evidence and writes the canonical Stage3 plan.
SERIAL_DEPENDENCIES=Judge only after A/B; a single Stage3 run only after canonical S11 authorization and fresh immediate prelaunch checks; S12 only after that run completes.
```

## Canonical prior decisions and Stage2 handoff

- S01 canonical protocol: `logs/protocol_manifest_iter31.md`, SHA-256 `212698d114b2be0d70e45bcd0e22973a0644535257f32d276930bf5881aecc19`. It locks Iter29 as the sole direct comparator, test metric `test_recall@10` (project `test_R@10`), strict adoption target `>0.065`, Instruments test split, seed 42, maximum 150 epochs, patience 10 with existing train-loss semantics, `NO_EVAL=True`, `SKIP_TEST=False`, beam 20, top-k `[5,10]`, and `n_eval=57439`. It says Stage3 model/data/evaluator/configuration remain unchanged, notes training may early-stop before the 150-epoch maximum, and requires current source/runtime/SID/test/path rechecks.
- S06 canonical direct-route plan: `logs/implementation_plan_iter31.md`, SHA-256 `8eb8ae94282465ffb7baeb4043dc0e74f7550dcf1a39b44236e2863d612f8620`, especially §“Stage3 direct route; root §5/§11 path reconciliation”. It chooses the existing `stage3_T5Train/train_HG-Rec.py` directly, no wrapper/no arguments; hard-codes the Iter31 SID, descriptive metadata identity, short `LOG_PATH` and `SAVE_PATH`, and internal launcher log under the Iter31 short results root. The outer nohup redirection is narrowly resolved to `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_stage3_run.log`, retaining the prescribed source CWD, absolute Python 3.10 interpreter, direct trainer entry and effective protocol. S06 explicitly leaves actual source-consumer/output placement and Stage3 authorization to S11.
- S09 Stage2 plan: `logs/stage2_execution_plan_iter31.md`, SHA-256 `b34cc06956809487c46e7b4372a9d472bdcaf88b52bf3da2ac9539d4fa7ad099`; it authorizes only Stage2 and explicitly leaves Stage3 for its separate gate. Completion record: `logs/stage2_completion_iter31.md`, SHA-256 `7face65d5498946019503ac59e526d2e029fe12c74065e29495d896ae540fecf`.
- S10 Judge: `logs/deliberation/S10_STAGE2_ANALYSIS/round_1/judge.md`, SHA-256 `3018f374743fbfd77e619338d95597c4f9390f0b404cb2d360ea915f6450db84`, verdict `MERGE_AB`, both hard gates PASS. Its canonical report is `logs/sid_geometry_iter31.md`, SHA-256 `4bd7d18b10ea88ed896c7f77dce234bda555b5e7374375dd56c0e9e953685da2`. S10 supports a completed contract-valid non-aborted Stage2 run, fixed FCCR-1 vector, a finite registered same-batch HRA-vs-Euclidean direct difference, and valid final SID wiring. All SID statistics are descriptive only; S10 makes no Stage3 claim or promotion decision and expressly leaves Stage3 unauthorized pending S11.
- Stage2 completion reports 100,000 steps, four ranks, seed 42, successful supervisor, and `SID_WIRING_PASS`. Iter31 `item_sids.json` is 1,258,304 bytes, SHA-256 `d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7`; it contains 24,587 ordered dense keys `0..24586`, width-four integer rows. The raw SID and four-token export parity, JSON/NPY equality, and uniqueness of all 24,587 exported rows were CPU-verified. Collision and other SID quality statistics are not S11 gates.

## Stage3 primary source identity and static configuration

Current source hashes, pinned for immediate prelaunch equality checks:

| Active source | SHA-256 |
|---|---|
| `stage3_T5Train/train_HG-Rec.py` | `210fd25a7c2a4fba7c6097e937f57f77bfd9340c3d0ce39eea823a280eadfc8b` |
| `stage3_T5Train/data/dataset.py` | `8899f1348e740996c572cd95439365f4caf2fe791acc4a0e22ed7d837937030e` |
| `stage3_T5Train/data/dataloader.py` | `d490b89c82a8ce3e1d8fa7577040038ef3411d030ce327fad95a999334f98af4` |
| `stage3_T5Train/model/hg_rec.py` | `f2876a95ee25b3f25d417792867c560da3053f8e5880120eda89affd9fd5cad1` |
| `stage3_T5Train/model/utils.py` | `a471b26da34ab30372f2cb2cc7baf05a6747be8830db4f178eac3b228364623c` |

Trainer imports the listed dataset, dataloader, model and utility modules. No Iter31 Stage3 wrapper is part of the approved route. The direct trainer currently pins:

- `DATASET_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/`; train/valid/test filenames are `train.parquet`, `valid.parquet`, `test.parquet`.
- `CODE_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json`; `RQVAE_VARIANT=iter31_hra_step6_common_reference`.
- `LOG_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/`; `SAVE_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/ckpt/`.
- `_LAUNCHER["torchrun"]=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun`, `_LAUNCHER["script"]=os.path.abspath(__file__)`, `nproc=4`, `master_port=50201`, `visible_dev=0,1,2,3`, internal launcher output `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/_stage3_launcher.log`.
- The trainer child environment sets `CUDA_VISIBLE_DEVICES=0,1,2,3`, `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_TIMEOUT=3600`, and `TORCH_NCCL_BLOCKING_WAIT=1`; retain the existing launcher behavior and other source defaults.
- Protocol values currently in source: seed 42; max epochs 150; train-loss patience 10 when `NO_EVAL=True`; `NO_EVAL=True`, `SKIP_TEST=False`; train batch 4096, infer batch 1024; `FAST=True`, BF16=True, COMPILE=False; beam size 20, top-k `[5,10]`, `EXCLUDE_HISTORY=True`; existing scheduler/model/optimizer settings are not to be changed. The final test path builds an evaluation dataset from `test.parquet`, evaluates once on rank 0 after training, loads the selected best checkpoint when present, and writes `test_final.json` and test metrics. Do not guarantee exactly 150 epochs: preserve the locked maximum and existing early-stop/checkpoint semantics.
- With these roots, expected trainer outputs are below `logs/Amazon_2023_Instruments/<timestamp>/` (including `test_final.json`, `training_metrics.jsonl`, `HG_Rec.log`) and `ckpt/Amazon_2023_Instruments/<timestamp>/HG_Rec_best.pth`; internal and outer launcher logs are the two paths above. Verify every actual write consumer in source. Root `CLAUDE.md` §§5 and 11 require Stage3 outputs to remain beneath this short Iter31 result root.

S06's path reconciliation has one launch-order detail: at packet time the whole Iter31 Stage3 result root and its `logs/` directory do not exist. The shell opens the outer `> .../logs/_stage3_run.log` redirection before starting Python, so an authorized launch must first create exactly the new Iter31 `logs/` parent (`mkdir -p .../curvature_RQ-VAE_iter31/logs`) and then issue the unchanged direct no-argument nohup command from `stage3_T5Train`. This is only output-directory preparation; it must occur only after S11 Judge authorization. No existing destination is to be overwritten.

## SID consumer and test data identity checks

- Iter31 Stage0 test split: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/test.parquet`, 2,546,378 bytes, SHA-256 `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc`. S01 records the same identity.
- Read-only CPU audit of this exact split against the exact Iter31 `item_sids.json`: `n_test=57439`; 24,587 SID keys; all target IDs, history IDs, and full seen-history IDs have SID keys (`missing_targets=0`, `missing_history=0`, `missing_seen=0`). Distinct target IDs=16,462; distinct history/seen IDs=24,584; SID key range `0..24586`; target range `1..24585`.
- `data/dataset.py` accepts JSON with dense ordered numeric-string keys and non-empty, fixed-width, non-negative integer SID rows; maps each raw SID component to token `raw+1`; rejects duplicate token tuples; ignores `codebook_size` for the shared token namespace; derives vocabulary from the actual maximum SID token. During dataset preparation it raises on any test target/history/seen item missing from the SID map. `train_HG-Rec.py` derives `vocab_size`, EOS and SID width from the actual training dataset when constants are zero. Iter31 extension tokens in raw range `[768,778]` therefore use the existing dynamically derived namespace; no Stage3 token/config change is authorized.
- `evaluate()` reports lowercase `recall@K` / `ndcg@K`, divides by the exact `n_eval`, excludes seen history, and uses the existing full item-level beam mapping/fallback semantics. Final JSON keys are `test_recall@5`, `test_recall@10`, `test_ndcg@5`, `test_ndcg@10`, plus `n_eval` and the selected checkpoint path. Preserve this evaluator unchanged.

## Canonical comparator and target

- Iter29 exact direct comparator: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`, 351 bytes, SHA-256 `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`:
  - `n_eval=57439`
  - `test_recall@5=0.03953759640662268`
  - `test_recall@10=0.05921064085377531`
  - `test_ndcg@5=0.026252776900288842`
  - `test_ndcg@10=0.03257647953179143`
- Iter29 `training_metrics.jsonl` SHA-256 `412860c203e20498e17ffd9c7a87d5e39836fcee563f1835f503e1df65a1b88f`; it records `num_epochs=150`, `no_eval=true`, train batch 4096, 4 ranks, the Iter29 SID/result roots, and actual completion at epoch 150 before its final test. Iter29 run records `variant=unknown_variant` despite S01's known wrapper-routing caveat; this metadata discrepancy alone is not evidence of a model/data identity failure. Historical launcher log: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/_stage3_launcher.log`, SHA-256 `fe9a49a8d7a4f46ffc0a91d20cf6a4768696bc54b778774714231f029f283806`; it records 4-rank training, BF16, beam=20/top-k=[5,10], and nonfatal NCCL/c10d warnings. Do not call the historical run warning-free.
- Strict goal: `test_R@10` / `test_recall@10 > 0.065`; Iter29 shortfall is `0.005789359146224693`. A Stage3 outcome is protocol-comparable only after S11 verifies actual Iter31 route, source/runtime, test split, and unchanged evaluator. No outcome or causal/promotion conclusion is known at S11.

## Runtime and destination status at packet freeze

- Prescribed executable `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10`: observed PyTorch `2.13.0+cu130`, CUDA runtime `13.0`, CUDA available, four `NVIDIA L40S` devices, BF16 supported. The fixed `torchrun` executable exists and is executable. A no-training import smoke through that interpreter successfully imported torch, NumPy, pandas, transformers, tqdm, scikit-learn, and all five listed Stage3 Python modules. One earlier identical import attempt timed out at 90 seconds; the bounded repeat completed successfully, so no training failure occurred.
- At packet-time host check: `nvidia-smi` showed four L40S GPUs at 0 MiB / 0% compute, `pgrep -af train_HG-Rec.py` had no matches, and port 50201 had no listener (`ss` empty; loopback connect returned refused). These transient checks must be repeated immediately before launch.
- `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/` and its outer-log parent are absent at packet freeze. No Iter31 Stage3 output exists or was launched.

## Required decision and checks

Each candidate must independently verify from primary sources and report:

1. Exact active SID consumer path/hash, SID keys/width/token range/uniqueness and compatibility with every Stage3 test target, history and seen-history item; no stale or alternate SID path.
2. Exact dataset/test split, metric/evaluation semantics and `n_eval`; unchanged direct comparison to Iter29 and strict target; preservation of the registered model/training/evaluation configuration without an unregistered Stage3 change.
3. Every write path from current trainer/launcher (outer launcher output, internal launcher log, per-run training/evaluation logs, `test_final.json`, metrics and checkpoint) resolves below the short Iter31 results root; the missing outer-log directory can be created safely after authorization; no overwrite of pre-existing Iter31 results.
4. Current source/runtime/launcher identities match the packet; prescribed no-argument direct launch from the prescribed CWD/interpreter; no wrapper, no arguments, no port/device/environment overrides. Specify exact immediate prelaunch checks and the exact single-run command, including creating only the missing `logs/` parent after Judge approval.
5. State uncertainty/risks honestly (including the known Iter29 `unknown_variant` metadata and nonfatal launcher warnings) without turning SID proxies into gates or claiming a Stage3 effect.

Judge C must issue a filled S11 deliberation record with both hard gates, a concrete canonical decision, confidence, `USER_INPUT_REQUIRED=NO`, and an autonomous next action. If route or protocol fails a repairable operational check, stop before Stage3 and use a scoped same-iteration operational repair; do not launch via a wrapper or change the model/evaluator. If all hard gates pass, the canonical `logs/stage3_evaluation_plan_iter31.md` must authorize exactly one direct Stage3 run and preserve all route/runtime/protocol checks. S11 must not classify promotion or weaken the strict target. No Stage3 launch is authorized by this source packet alone.
