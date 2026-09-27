# Iter31 Stage3 evaluation plan — S11 canonical

STAGE_ID=S11_STAGE3_EVALUATION
ROUND=1
AUTHORIZATION=CONDITIONAL_AUTHORIZE_EXACTLY_ONE_DIRECT_RUN_AFTER_IMMEDIATE_PRELAUNCH_CHECKS
STAGE3_LAUNCHED=NO
USER_INPUT_REQUIRED=NO
PROTOCOL_ID=ITER31_HRA_SINGLE_SEED42_INSTRUMENTS_2026-09REF_VS_ITER29
PARENT_ITER=iter29
CANONICAL_BASELINE_ITER=iter29

## Canonical S11 decision

Both independent S11 audits pass the hard gates. The direct Stage3 consumer is the existing `stage3_T5Train/train_HG-Rec.py`, with no wrapper or arguments. It consumes the exact Iter31 SID JSON and Stage0 Instruments split; current trainer, dataset, dataloader, model, and utility source hashes match the frozen packet. The runtime derives SID width/vocabulary from the loaded dataset, so the four-token Iter31 namespace and extension IDs are handled without a Stage3-specific token change. The unchanged Stage3 training and test evaluator emit test recall@5/10, NDCG@5/10, and n_eval under existing checkpoint/seen-history semantics. All known file-write sinks resolve beneath the short Iter31 result root.

This plan authorizes exactly one future run, and only if every immediate prelaunch check below passes. This plan does not itself execute or initiate the run. The Stage3 run remains unlaunched; no training/evaluation or GPU workload is part of S11 adjudication.

## Locked evaluation and comparator

- Dataset: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/`; training file `train.parquet`; final test file `test.parquet`.
- Test split SHA-256: `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc`; locked `n_eval=57439`.
- SID input: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json`; adjudication-time SHA-256 `d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7`.
- Stage3 protocol stays unchanged: seed 42; maximum 150 epochs and existing train-loss patience 10 semantics; `NO_EVAL=True`; `SKIP_TEST=False`; train batch 4096; inference batch 1024; FAST/BF16 enabled; COMPILE disabled; beam size 20; top-k `[5,10]`; full seen-history exclusion; preserve existing optimizer, scheduler, model, checkpoint and one-rank final test behavior. Do not guarantee exactly 150 epochs.
- Compare only to the canonical Iter29 `test_final.json`: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`, SHA-256 `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`; `n_eval=57439`, test recall@5 `0.03953759640662268`, test recall@10 `0.05921064085377531`, test NDCG@5 `0.026252776900288842`, test NDCG@10 `0.03257647953179143`.
- Strict target: `test_R@10` / `test_recall@10 > 0.065`. Do not weaken the target. S11 makes no performance, mechanism-status, causal, or promotion conclusion.

## Metadata limitation

`RQVAE_VARIANT` resolves from the exact absolute `CODE_PATH` to `iter31_hra_step6_common_reference`, but that value is not added to the `config` dictionary. `_record` uses `config.get("variant", "unknown_variant")`, so `training_metrics.jsonl` is expected to contain `variant=unknown_variant`. Treat this as a disclosed trace-metadata limitation, not a launch blocker: `code_path` and the active source constant specify the exact hashed Iter31 SID file; the variant label does not select the SID, model, evaluation path, or result directory. Do not edit Stage3 source to change the label in this gate. The result and route identity must still be verified independently from actual artifacts after the run.

## Output route and write sinks

Short result root (no descriptive suffix): `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/`.

- Outer nohup output: `.../logs/_stage3_run.log`.
- Internal torchrun child output: `.../_stage3_launcher.log`.
- Per-run `HG_Rec.log`, `training_metrics.jsonl`, optional screen summary, final `test_final.json`, and DDP signaling files: `.../logs/` (run logs under `Amazon_2023_Instruments/<timestamp>/`; DDP signals under `_ddp_sync/`).
- Best checkpoint: `.../ckpt/Amazon_2023_Instruments/<timestamp>/HG_Rec_best.pth`.
- The source fixes `LOG_PATH` and `SAVE_PATH` to this short root, and the launcher log/script/rank/port/device constants match the approved S06 route. No sink may be redirected elsewhere.
- The Iter31 result root was absent when S11 adjudicated. Do not overwrite, clean, or reuse any existing destination. Because the shell opens redirection before Python starts, after authorization and successful prelaunch checks only, create the missing outer-log parent with the exact `mkdir -p` shown below. That command creates only `.../curvature_RQ-VAE_iter31/logs` (and its missing parent chain); do not pre-create `ckpt`, run-specific directories, or other destination paths.

## Required immediate prelaunch checks

Perform these checks immediately before the sole run; the candidate audits' prior runtime, GPU, process, port, and destination observations are transient and are not substitutes. Any failed, changed, or ambiguous check blocks execution before directory creation. Do not try an alternate route.

1. Recompute SHA-256 for all five active source files and require exact packet identity:
   - `stage3_T5Train/train_HG-Rec.py`: `210fd25a7c2a4fba7c6097e937f57f77bfd9340c3d0ce39eea823a280eadfc8b`
   - `stage3_T5Train/data/dataset.py`: `8899f1348e740996c572cd95439365f4caf2fe791acc4a0e22ed7d837937030e`
   - `stage3_T5Train/data/dataloader.py`: `d490b89c82a8ce3e1d8fa7577040038ef3411d030ce327fad95a999334f98af4`
   - `stage3_T5Train/model/hg_rec.py`: `f2876a95ee25b3f25d417792867c560da3053f8e5880120eda89affd9fd5cad1`
   - `stage3_T5Train/model/utils.py`: `a471b26da34ab30372f2cb2cc7baf05a6747be8830db4f178eac3b228364623c`
   Verify root `CLAUDE.md` §§5/11 and canonical S01, S06, S10 records are unchanged. Confirm direct trainer constants, source-derived `RQVAE_VARIANT`, exact `CODE_PATH`, `LOG_PATH`, `SAVE_PATH`, launcher script/log, four-rank route and no intervening source/config change. Keep the `unknown_variant` limitation disclosed.
2. Recompute and require exact SID JSON hash above; verify the active resolved code file is exactly that absolute Iter31 file (no fallback/alternate SID), and that the Stage2 completion/S10 parity and schema evidence still describes this handoff: dense item IDs `0..24586`, 24,587 unique width-four integer rows. Confirm the locked test file exists at the exact Stage0 path and its hash is exact; its dataset construction must preserve `n_eval=57439`, all targets/history/complete seen-history IDs covered, and runtime-derived SID width four/vocabulary. No path fallback or data substitution is acceptable.
3. Confirm unchanged source values for `NO_EVAL=True`, `SKIP_TEST=False`, seed, batch sizes, epoch maximum/patience behavior, model/optimizer/scheduler, beam 20, top-k `[5,10]`, full seen-history exclusion, evaluator/metric definitions and one final rank-0 test. Confirm every output consumer remains inside the short Iter31 root and the launcher retains prescribed port/devices/NCCL behavior. Do not edit model, data, evaluator, metrics, or protocol.
4. Confirm the entire Iter31 Stage3 result root is still absent; if it exists, stop without overwriting or deleting anything. Confirm the prescribed interpreter `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10` and its `torchrun` are executable; recheck Python 3.10 / PyTorch-CUDA runtime, required imports, CUDA availability, four intended devices and BF16 support. Recheck no competing `train_HG-Rec.py` process and port 50201 free; ensure all four devices are available and suitable. The existing launcher must inject `CUDA_VISIBLE_DEVICES=0,1,2,3`, `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_TIMEOUT=3600`, and `TORCH_NCCL_BLOCKING_WAIT=1`; do not override these or add shell environment/port/device settings.
5. Confirm the working directory is `/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train`. If any check fails, do not create a directory, launch, modify source, or switch to a wrapper; record the exact operational blocker for scoped same-iteration disposition.

## Sole authorized command sequence

Only after all checks above pass, create just the missing shell-redirection parent, then issue exactly one direct no-argument launch. Do not execute it as part of S11 adjudication or before checks. No wrapper, extra arguments, environment overrides, alternate interpreter, or second run is authorized.

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs
nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 train_HG-Rec.py > /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_stage3_run.log 2>&1 &
```

The source entrypoint then launches its existing four-rank `torchrun` directly, with `--standalone`, `--nproc_per_node=4`, `--master_port=50201`, script path `os.path.abspath(__file__)`, devices `0,1,2,3`, and the mandated NCCL environment. S11 authorizes one such invocation conditionally; actual launch is downstream execution, not performed or claimed here.

## Evidence references and limits

- Frozen source packet: `local://S11_ITER31_SOURCE_PACKET.md`.
- Candidates: `logs/deliberation/S11_STAGE3_EVALUATION/round_1/agent_a.md` and `agent_b.md` (audit evidence only; this plan is canonical).
- Canonical protocol: `logs/protocol_manifest_iter31.md`; direct route: `logs/implementation_plan_iter31.md`; Stage2 handoff: `logs/stage2_completion_iter31.md`; S10 decision/report: `logs/deliberation/S10_STAGE2_ANALYSIS/round_1/judge.md` and `logs/sid_geometry_iter31.md`.
- Current source hash, exact SID hash, test split hash and Iter29 comparator hash were independently recomputed during Judge C adjudication and matched packet pins. Primary Stage3 source confirms SID parsing/+1 namespace, fail-closed coverage checks, evaluator semantics, output sinks, no-argument direct launcher, and metadata default behavior.
- CPU dataset construction cited in candidate evidence is not a Stage3 model/training/test run. No Stage3 launch, training, test evaluation, result classification, baseline rerun, Stage2/S08 rerun, or GPU workload occurred for this adjudication.
- Prior Iter29 launcher warnings are historical caveats, not evidence that the present route is warning-free; do not suppress warnings or change launcher settings under this plan.

## Round-2 corrective addendum — supersedes conflicting round-1 duration/attempt limits

This addendum is the current canonical S11 instruction, adjudicated from the frozen round-2 packet and candidates. Preserve the round-1 decision and its audit history above; do not treat its `EARLY_STOP=10`, “do not guarantee exactly 150 epochs,” or “no second run” limitation as operative. Those clauses are explicitly superseded by root `CLAUDE.md` §5.1 and this correction. `SAME_ITERATION_REPAIR=AUTHORIZED`.

The first attempt at `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_20-15-25/` is invalid and noncanonical: its `training_metrics.jsonl` records only epochs 1–21, then `event=early_stop`, `trigger=train_loss_patience_exhausted`, `patience=10`; its subsequent `test_final.json` does not make it a complete 150-epoch run. Do not use that attempt's metrics for promotion or causal classification. Preserve its timestamped metrics, test record, and checkpoint unchanged.

Authorize at most one direct, no-argument Iter31 Stage3 replacement, and only after the final prelaunch source/hash/runtime/GPU/process/port/output-isolation gates below all pass. This is an operational same-iteration replacement, not replication, a new experiment, or variance estimation. No Stage3 launch is performed by this adjudication. If any gate fails or is ambiguous, do not launch, change routes, or attempt another run.

### Locked science and protocol

Keep unchanged: seed 42; Stage0 Instruments train/test data and exact test split; Iter31 SID input and identity; model, optimizer, scheduler, checkpoint-selection semantics; `NUM_EPOCHS=150`; `NO_EVAL=True`; `SKIP_TEST=False`; `SCREEN_BASELINE_LOG=""`; train/inference batch sizes 4096/1024; FAST/BF16 enabled and COMPILE disabled; beam size 20; top-k `[5,10]`; full seen-history exclusion; evaluator and metric definitions; four-rank prescribed launcher/port/devices/NCCL settings; and the short Iter31 output root. Do not change any other scientific or evaluation factor.

### Final prelaunch gate — complete immediately before launch

Record checks and inspected source hashes in `logs/stage3_preflight_iter31.md` before any invocation:

1. Recompute and require the packet-pinned hashes for `stage3_T5Train/train_HG-Rec.py` (`638b8bb61403f0497d080c792e63c6d07dab32bfaa6fe8fc903a67c158c51d80`), `data/dataset.py` (`8899f1348e740996c572cd95439365f4caf2fe791acc4a0e22ed7d837937030e`), `data/dataloader.py` (`d490b89c82a8ce3e1d8fa7577040038ef3411d030ce327fad95a999334f98af4`), `model/hg_rec.py` (`f2876a95ee25b3f25d417792867c560da3053f8e5880120eda89affd9fd5cad1`), and `model/utils.py` (`a471b26da34ab30372f2cb2cc7baf05a6747be8830db4f178eac3b228364623c`); verify root `CLAUDE.md` §§5/5.1 and the canonical S01/S06/S10 records remain applicable. Inspect the actual direct trainer and launcher path and rescan all active Stage3 Python/wrapper/launcher paths for early-stop, patience, metric/loss exits and screen-triggered test skips. Require no metric-triggered early termination, fixed 150-epoch loop, and the unchanged values above. The generic `skip_test` branch is acceptable only while the active source constant remains `False`.
2. Recompute exact Iter31 SID hash `d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7`, exact Stage0 `test.parquet` hash `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc`, and exact Iter29 comparator hash `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`. Require the exact absolute Iter31 `CODE_PATH` without fallback/substitution, unchanged data coverage and expected `n_eval=57439`, and unchanged Iter29 comparator identity/values.
3. The Iter31 result root already exists. Do not remove, clean, or reuse it. Before launch, preserve the existing fixed root `_stage3_launcher.log` and `logs/_stage3_run.log` (including the currently empty outer log) with recorded original/destination paths and SHA-256. Move the stale `logs/_ddp_sync/shutdown_signal.txt` (`test_done`) into the first invalid attempt's audit area, record its source/destination and SHA-256, and ensure the active sentinel path is absent before launch. Leave the first attempt's timestamped metrics, test record, and checkpoint untouched. Confirm the replacement timestamp and prospective per-run log/checkpoint directories are new and unused; every sink stays beneath `results/stage3_T5Train/curvature_RQ-VAE_iter31/`.
4. Immediately recheck the prescribed `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10` and its `torchrun`, Python/PyTorch-CUDA runtime/imports, CUDA availability, four intended devices and BF16 support, device suitability, absence of a competing `train_HG-Rec.py` process, and port 50201 availability. Confirm the required launcher-injected `CUDA_VISIBLE_DEVICES=0,1,2,3` and NCCL settings; do not add overrides. Require working directory `/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train`.

### Sole authorized invocation and validity

After all four gates pass, make exactly one direct no-argument invocation; no wrapper, extra arguments, alternate interpreter, environment override, or second run:

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train
nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 train_HG-Rec.py > /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_stage3_run.log 2>&1 &
```

The replacement is valid for downstream S12 only when its own isolated run records training epochs 1–150 in order, has no early-stop event, produces a complete final `test_final.json` with `n_eval=57439`, and all ranks/launcher exit cleanly. Require the result and checkpoint paths to identify the new timestamp subtree under the unchanged short Iter31 root. A test event labeled epoch 150 without 150 actual train records is insufficient. The existing 30-minute nonzero-rank shutdown wait makes clean all-rank exit an observed post-run requirement, not an assumption.

Round-2 Judge C artifact: `logs/deliberation/S11_STAGE3_EVALUATION/round_2/judge.md`. The adoption target and Iter29 comparator remain those recorded above; S11 makes no performance, causal, or promotion conclusion.
