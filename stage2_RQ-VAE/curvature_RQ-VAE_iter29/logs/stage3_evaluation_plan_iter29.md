# Iter29 Stage3 evaluation plan — S11 canonical (ACCEPT_A)

## Authorization boundary

This is a Judge-approved method and one-run authorization, not evidence of a Stage3 run. S10 passed and verified the Stage2 output/SID export; it did not authorize a Stage3 launch or establish efficacy. Stage3 has not been launched for iter29, and this plan makes no Stage3 result, recall, promotion, or efficacy claim. Launch is permitted only after the one source patch below, the no-GPU/no-result-path smoke, and every prelaunch gate have passed and been recorded. A missing, stale, mismatched, ambiguous, or failed gate blocks launch.

S11 verdict is `ACCEPT_A`; the candidate B outer-log path was corrected here. Every iter29 Stage3 artifact, including both outer and launcher logs, must be inside `results/stage3_T5Train/curvature_RQ-VAE_iter29/`.

## Stage2 input and Stage3 route

The exact Stage3 SID input is:

- `CODE_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json`
- S10 SHA-256: `58665e08a97e122f47a77ed3616b67efdeb8eadba8253771a1498cb5e57487ff`; size recorded by S10: `1260651` bytes.
- Corresponding exported 4-token NPY: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments/sids_for_hgrec.npy`, S10 SHA-256 `a8efe6e1315166486aff162e1bc6a5722b5d5d9ac1b537b83bc003ec0ea7cde1`, recorded size `786912` bytes.

S10 checked 24,587 consecutive item keys `0..24586`, exact JSON/NPY row alignment, unique 4-token tuples, and 3-token collision extensions `768..780`. Stage3's `item2code()` ignores its `codebook_size` argument; it sorts the numeric JSON keys, requires complete `0..N-1` coverage and fixed-width nonnegative integer rows, shifts every SID component by `+1` into one shared semantic namespace, and rejects duplicate complete tuples. The largest raw extension `780` therefore becomes semantic token `781`; dynamic sizing gives `semantic_vocab_size=781`, user token `782`, EOS `783`, PAD/decoder-start `0`, and runtime vocabulary size `784`. Stage3 does not create collision extensions; the Stage2 fourth token makes the full tuples unique. Retain this data contract unchanged.

The sole runner is `stage2_RQ-VAE/curvature_RQ-VAE_iter29/scripts/run_stage3_iter29.py`, invoked with no arguments. It imports `stage3_T5Train/train_HG-Rec.py`, assigns the exact `CODE_PATH`, `RQVAE_VARIANT="iter29_bounded_rational_additive_mapping"`, and explicit short-name paths:

- `LOG_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/`
- `SAVE_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter29/ckpt/`
- `_LAUNCHER["script"]` is the absolute iter29 wrapper path.
- `_LAUNCHER["log"]=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/_stage3_launcher.log`.

The wrapper must be used because the global trainer defaults/variant map do not select iter29. With the exact JSON present, `_resolve_code_file()` selects that absolute first candidate; preflight must fail rather than allow a fallback path. Timestamped training logs, `training_metrics.jsonl`, `test_final.json`, checkpoints, and `_ddp_sync` files all remain below the same iter29 short-name result root.

## Fixed Stage3 protocol — no model/data/evaluation changes

The only conceptual input difference from the sole protocol-valid direct control is the registered iter29 SID representation. Preserve the trainer, wrapper, dataset, model and evaluator behavior and all their constants. In particular:

- Dataset: `Amazon_2023_Instruments`; `DATASET_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/`; files `train.parquet`, `valid.parquet`, `test.parquet`; `MAX_LEN=20`; `N_USER_TOKENS=1`; `CODEBOOK_SIZE=[256,256,256,1]`; dynamic `VOCAB_SIZE=0`; no changes to history/target/SID mapping or split generation.
- Training: `SEED=42`, `BATCH_SIZE=4096`, `INFER_SIZE=1024`, `NUM_EPOCHS=150`, `EARLY_STOP=10`, `FAST=True`, `BF16=True`, `COMPILE=False`; preserve scheduler settings `USE_LR_SCHEDULER=True`, warmup `0.10`, minimum LR factor `0.20`, and optimizer `LR=0.003`, `WEIGHT_DECAY=0.05`.
- Model: 4 encoder and 4 decoder layers, `D_MODEL=128`, `D_FF=1024`, 6 heads, `D_KV=64`, dropout `0.1`, ReLU activation and feed-forward projection. Do not change any model config or checkpoint selection behavior.
- Evaluation: `TOPK_LIST=[5,10]`, `BEAM_SIZE=20`, `NO_EVAL=True`, `SKIP_TEST=False`, empty `SCREEN_BASELINE_LOG`, `EXCLUDE_HISTORY=True`; preserve the current loader and evaluator settings. Validation remains skipped under `NO_EVAL`; rank 0 selects/saves a best checkpoint by training loss using the existing patience rule (not a validation-best checkpoint). Because the screen curve is empty, its screen-failure test-skip branch is inactive; `SKIP_TEST=False` leaves the final test enabled. Rank 0 then builds the full test dataset, reloads the selected best checkpoint when present, evaluates the test split, and writes `test_final.json`. Record the actual completed epoch because the existing train-loss patience may stop before the 150-epoch maximum.

Do not introduce a Stage2 metric gate or change any setting to improve throughput, avoid a failure, or influence the observed score. The only permitted source change is the mandatory child-environment patch below.

## Sole authorized source patch and retained operational risk

In `stage3_T5Train/train_HG-Rec.py:_launch_via_torchrun()`, retain `env = os.environ.copy()` and the existing hard-coded `CUDA_VISIBLE_DEVICES`, but replace only the five mandated `setdefault` calls with unconditional child-env assignments:

```python
env["NCCL_IB_DISABLE"] = "1"
env["NCCL_P2P_DISABLE"] = "1"
env["NCCL_SHM_DISABLE"] = "1"
env["NCCL_TIMEOUT"] = "3600"
env["TORCH_NCCL_BLOCKING_WAIT"] = "1"
```

The values are mandatory under root `CLAUDE.md` §5; unconditional assignment is required because inherited parent values currently override `setdefault`, and the current IB/SHM defaults (`0`) contradict the mandate. Amend the adjacent comment to make clear that this repository rule overrides the old recommendation; explicitly preserve the warning that disabling IB, P2P and SHM can close available transports and has previously been associated with cross-NUMA all-reduce deadlock/connectivity or performance risk. Monitor actual DDP startup and completion. Do not revert the mandated values or silently substitute another transport configuration. Leave `NCCL_NET_GDR_LEVEL`, `NCCL_SOCKET_IFNAME`, `NCCL_DEBUG`, command construction, process handling, wrapper, and every model/data/evaluation setting unchanged. This is a launch-environment compliance repair, not a scientific Stage3 parameter change.

## Required no-GPU / no-result-path smoke (after the patch)

Use one disposable smoke harness outside the repository result tree, with the prescribed Python 3.10 environment. It must not call `trainer.main()`, initialize CUDA/DDP, invoke a real torchrun child, or create/write anywhere below `results/stage3_T5Train/`.

1. **Wrapper routing without side effects:** execute the wrapper with `RANK` unset while stubbing its `importlib` module-loading boundary to return a fake trainer with a fresh `_LAUNCHER` dict. Fake `main()` and `_launch_via_torchrun()` only record calls. Assert the wrapper sets the exact iter29 `CODE_PATH`, variant, short `LOG_PATH`/`SAVE_PATH`, absolute wrapper script, and `_stage3_launcher.log` path, and selects `_launch_via_torchrun()` rather than `main()`. The fake trainer prevents directory creation or result writes.
2. **Actual launcher child environment without a child process:** add the Stage3 source directory to `sys.path` as the wrapper does, then import the actual trainer module without calling `main()`. Set its in-memory `_LAUNCHER` to the prescribed torchrun executable, four ranks, port `50201`, visible devices `0,1,2,3`, and a fresh `/tmp` log path. Seed the parent environment with conflicting sentinel values for all five mandated variables; monkeypatch `_subprocess.call` to capture argv/env and return `0`. Invoke `_launch_via_torchrun()` and catch its expected `SystemExit(0)`. Assert exact argv:

   `[/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun, --standalone, --nproc_per_node=4, --master_port=50201, /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter29/scripts/run_stage3_iter29.py]`

   Assert the captured child env contains all five exact mandatory values despite the parent sentinels, keeps `CUDA_VISIBLE_DEVICES=0,1,2,3`, and leaves the parent environment and Stage3 model/data/evaluation constants unchanged. The mock must not execute a process; only the disposable `/tmp` log may be written. Verify the result tree was not created or modified by this smoke, restore any temporary parent environment changes, and remove only the smoke's own `/tmp` files. Record assertions as smoke evidence, not as a training/GPU result. If any assertion fails, do not launch Stage3.

## Exact prelaunch gates — all required immediately before the one launch

Record check output, SHA-256, file size, mtime, resolved path and pass/fail for each relevant item. A failure blocks launch; declarations in this plan or the earlier S10 record do not replace current rechecks.

1. **Adjudication and source scope.** Confirm this S11 `ACCEPT_A` decision and canonical plan are present. Apply the patch once and verify the source diff contains only the five unconditional child assignments and the accurate adjacent risk comment. No model/data/eval or wrapper change is allowed. Record the post-patch SHA-256 of `train_HG-Rec.py` and the wrapper; verify the wrapper's runtime overrides and all locked constants. Run the smoke above successfully after the patch; ensure the trainer file has not changed since that smoke.
2. **Stage2 consumer freshness, exact hash and cardinality.** Recheck the exact iter29 JSON path and SHA-256 `58665e08a97e122f47a7ed3616b67efdeb8eadba8253771a1498cb5e57487ff`, plus the matching 4-token NPY path and SHA-256 `a8efe6e1315166486aff162e1bc6a5722b5d5d9ac1b537b83bc003ec0ea7cde1`. Record current stat data and confirm identity/freshness against S10's verified outputs (S10 recorded these output mtimes after the Stage2 run). Verify exactly 24,587 numeric-string keys covering `0..24586`, four nonnegative integer tokens in every row, exact row equality with the 4-token NPY, all complete tuples unique, and the expected max raw ID `780`. Confirm wrapper `CODE_PATH` resolves to this exact existing file, so `_resolve_code_file()` cannot choose a fallback. Verify every item id referenced by the Stage0 train/test histories, `seen_history`, and targets is covered by the JSON. Any changed/missing/stale/misaligned artifact or uncovered id blocks launch; do not substitute an alternate SID file.
3. **Stage0 exact input identity and full-test cardinality.** Recompute and require all S01 hashes, at the exact Stage0 root paths:

   - `train.parquet`: `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`
   - `valid.parquet`: `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9`
   - `test.parquet`: `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc`
   - `items.parquet`: `6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3`

   Verify the trainer's dataset root/file names still match; `train` and `test` are consumed, `valid` is intentionally not loaded because `NO_EVAL=True`, and `items.parquet` is identity/provenance evidence rather than a direct trainer input. The split loader emits one dataset record per existing parquet row. Read-only cardinality verification must show the complete current test dataset has exactly `57,439` rows, matching S01 `STAGE3_N_EVAL=57439` and the baseline; record the actual train row count and confirm it equals the train parquet row count. Do not generate replacement windows or use another split. Any hash, route, SID-coverage or test-cardinality mismatch blocks launch.
4. **No stale Stage3 outputs / destination freshness.** Immediately before launch, inventory the entire `results/stage3_T5Train/curvature_RQ-VAE_iter29/` tree. It must have no stale timestamped run directories, `HG_Rec_best.pth`, `test_final.json`, `training_metrics.jsonl`, `HG_Rec.log`, `_stage3_launcher.log`, `_stage3_run.log`, or `_ddp_sync` signal files (`early_stop_signal.txt`, `shutdown_signal.txt`) from a prior attempt. S10 observed only the pre-existing empty `logs/` directory; the current audit likewise observed that `logs/` empty, but neither observation replaces this immediate prelaunch recheck. Record absence/stat evidence; confirm the short `logs/` and `ckpt/` roots are writable and the absolute outer-log parent exists. Do not clean or overwrite unclassified stale artifacts; any ambiguous prior output blocks launch.
5. **Runtime/hardware and rendezvous.** Verify `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10` and its `torchrun` executable exist and are executable, the prescribed Python imports the actual PyTorch/CUDA stack, CUDA is available, exactly four selected devices `0,1,2,3` are visible/usable and each supports BF16, and fixed rendezvous port `50201` is free. Confirm sufficient result storage and the single foreground supervisor can write both logs. `BF16=True` requires CUDA/BF16; CPU fallback is invalid. Do not treat the parent shell's inherited NCCL values as compliance evidence; the approved smoke must prove the child's five values.
6. **Single-run supervision ready.** Prepare exactly one foreground process identity with no restart policy, no shell backgrounding, no `nohup`, no retry wrapper, and no automatic restart. Record supervisor name/identity and start time. The supervisor must own the Python process to terminal exit; follow both logs and capture its exit status. Do not interpret a startup/readiness warning as permission to relaunch.

## Exactly one supervised no-argument invocation

Only after all above gates pass, from the iter29 source directory, launch once using the project process supervisor (`hub start`):

```text
name=iter29-stage3
cwd=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter29
application=/bin/bash
args=["-c", "exec /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 scripts/run_stage3_iter29.py > /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/_stage3_run.log 2>&1"]
pty=false
restart=no
```

The Python entry receives only the wrapper path—no CLI arguments or environment overrides. The wrapper's one torchrun child is configured as `--standalone --nproc_per_node=4 --master_port=50201 <absolute iter29 wrapper>` with `CUDA_VISIBLE_DEVICES=0,1,2,3`; the five mandatory child environment values are applied by the approved patch. The supervisor stdout/stderr log is the absolute `.../curvature_RQ-VAE_iter29/logs/_stage3_run.log` (corrected from the rejected candidate B path). The wrapper/torchrun log is `.../curvature_RQ-VAE_iter29/logs/_stage3_launcher.log`. Both are inside the required iter29 short-name subtree.

This is one run, not one run per candidate. If start fails, readiness times out, logs are delayed/partial, a rank fails, the process exits nonzero, or results are invalid/missing, preserve all available evidence and classify the attempt; never restart or invoke the wrapper a second time for this iteration.

## Postrun evidence and evaluation decision

After the sole supervised process reaches terminal exit, preserve and verify:

- Supervisor identity, start/end time, terminal exit status and the two fixed logs. `_stage3_launcher.log` must show actual torchrun/DDP startup and all four ranks, not merely a supervisor spawn; inspect for errors, hangs, missing ranks, non-finite/invalid execution and clean completion.
- Fresh timestamped `logs/Amazon_2023_Instruments/<timestamp>/HG_Rec.log` and `training_metrics.jsonl`, plus the corresponding `ckpt/Amazon_2023_Instruments/<timestamp>/HG_Rec_best.pth`, all below `results/stage3_T5Train/curvature_RQ-VAE_iter29/`. Require their mtimes after the supervisor start. Verify the best-checkpoint path exists and the evaluator reloads that same checkpoint.
- Exactly one fresh `logs/Amazon_2023_Instruments/<timestamp>/test_final.json`, also newer than run start. Verify no screen-failure/test-skip marker and no `skip_test` marker; require the final test event, `n_eval=57439`, finite `test_recall@5`, `test_recall@10`, `test_ndcg@5`, and `test_ndcg@10`, and the correct short-root log/checkpoint paths. Missing, stale, duplicate, partial, malformed, wrong-path or skipped-test output invalidates the Stage3 run; preserve it and do not retry.

The sole direct comparator is the exact iter26 file `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`, independently read for this audit. It records `n_eval=57439`, `test_recall@5=0.03788366789115409`, `test_recall@10=0.057017009349048554`, `test_ndcg@5=0.025169911931406087`, and `test_ndcg@10=0.03133359761524377`. Iter18 remains `HISTORICAL_NONCOMPARABLE`; do not rank against it.

Only after a valid completed final test, calculate the actual `test_recall@10` delta against iter26's exact `0.057017009349048554` and report whether the inclusive user criterion `test_recall@10 >= 0.065` is met. Record all four test metrics and `n_eval`; a single stochastic run does not estimate run-to-run variance or alone establish a robust causal effect. Until those postrun artifacts exist, make no Stage3 score, efficacy, or promotion claim. Any downstream result classification proceeds through S12 using the collected primary evidence.