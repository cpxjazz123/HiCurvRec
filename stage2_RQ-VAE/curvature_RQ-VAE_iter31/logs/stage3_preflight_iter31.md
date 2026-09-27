# Iter31 Stage3 immediate prelaunch preflight — S11

```text
STAGE_ID=S11_STAGE3_EVALUATION
S11_JUDGE_VERDICT=MERGE_AB
STAGE3_AUTHORIZATION=CONDITIONAL_ONE_RUN_AFTER_ALL_CHECKS_PASS
PREFLIGHT_TIME=2026-09-27T20:12:33+10:00
PREFLIGHT_RESULT=PASS
STAGE3_LAUNCHED=NO_AT_RECORD_TIME
```

## Canonical authorization

The S11 Judge record `logs/deliberation/S11_STAGE3_EVALUATION/round_1/judge.md` returns `MERGE_AB`, both candidate hard gates PASS, and conditionally authorizes exactly one direct Stage3 run after the checks in `logs/stage3_evaluation_plan_iter31.md` pass. Judge explicitly classifies the `RQVAE_VARIANT`/`variant=unknown_variant` behavior as a disclosed non-blocking metadata limitation and prohibits a source change in this gate. The canonical plan is the only run authorization. No run or result is implied by this preflight record.

## Immediate source and input identity checks

Recomputed SHA-256 values match the canonical S11 plan for every pinned active Stage3 source, SID/test input, protocol/evidence record, and Iter29 comparator:

| Path | SHA-256 | Result |
|---|---|---|
| `stage3_T5Train/train_HG-Rec.py` | `210fd25a7c2a4fba7c6097e937f57f77bfd9340c3d0ce39eea823a280eadfc8b` | PASS |
| `stage3_T5Train/data/dataset.py` | `8899f1348e740996c572cd95439365f4caf2fe791acc4a0e22ed7d837937030e` | PASS |
| `stage3_T5Train/data/dataloader.py` | `d490b89c82a8ce3e1d8fa7577040038ef3411d030ce327fad95a999334f98af4` | PASS |
| `stage3_T5Train/model/hg_rec.py` | `f2876a95ee25b3f25d417792867c560da3053f8e5880120eda89affd9fd5cad1` | PASS |
| `stage3_T5Train/model/utils.py` | `a471b26da34ab30372f2cb2cc7baf05a6747be8830db4f178eac3b228364623c` | PASS |
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json` | `d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7` | PASS |
| `results/stage0_build_parquet/test.parquet` | `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc` | PASS |
| Iter29 canonical `test_final.json` | `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662` | PASS |
| `logs/protocol_manifest_iter31.md` | `212698d114b2be0d70e45bcd0e22973a0644535257f32d276930bf5881aecc19` | PASS |
| `logs/implementation_plan_iter31.md` | `8eb8ae94282465ffb7baeb4043dc0e74f7550dcf1a39b44236e2863d612f8620` | PASS |
| `logs/sid_geometry_iter31.md` | `4bd7d18b10ea88ed896c7f77dce234bda555b5e7374375dd56c0e9e953685da2` | PASS |
| `logs/deliberation/S10_STAGE2_ANALYSIS/round_1/judge.md` | `3018f374743fbfd77e619338d95597c4f9390f0b404cb2d360ea915f6450db84` | PASS |
| `logs/stage2_completion_iter31.md` | `7face65d5498946019503ac59e526d2e029fe12c74065e29495d896ae540fecf` | PASS |

Resolved trainer constants read by importing the current trainer without entering `__main__`: `CODE_PATH` is the exact Iter31 SID JSON; `RQVAE_VARIANT=iter31_hra_step6_common_reference`; `LOG_PATH`/`SAVE_PATH` use the short Iter31 Stage3 root; direct script, torchrun, four ranks, port 50201 and visible devices `0,1,2,3` match the canonical plan. The logical working directory check returned `/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train`. `readlink -f` resolves both `/home/.../train_HG-Rec.py` and the launcher’s `/fs04/.../train_HG-Rec.py` to the same physical script path; this is the same source file, not another checkout or worktree.

The actual `GenRecDataset` CPU smoke under the prescribed Python 3.10, with the exact Iter31 SID JSON and Stage0 test split in evaluation mode, passed: `rows=57439`, `num_items=24587`, `n_digit=4`, `max_sid_token=779`, `vocab_size=782`, `max_token_seq_len=82`; missing target/history/seen IDs = 0. No model, final test evaluation, training, or GPU workload occurred.

## Runtime and live-resource checks

The no-training runtime/import check completed successfully under `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10`: Python 3.10 environment, PyTorch `2.13.0+cu130`, CUDA runtime `13.0`, CUDA available, four NVIDIA L40S devices, BF16 supported; `torchrun` exists and is executable; required NumPy/pandas/transformers/tqdm/scikit-learn and Stage3 source modules import. The source-derived dataset and route fields are recorded above. An earlier import attempt timed out at 90 seconds; the repeated complete import passed, followed by this direct no-training trainer/dataset inspection. No training failure occurred.

Fresh live checks after the no-training inspection:

- `nvidia-smi` at 2026-09-27 20:11:35: all four L40S devices showed 0 MiB used, 0% compute, and no running GPU processes.
- `pgrep -af train_HG-Rec.py`: no matches (exit 1, expected absent).
- `ss -ltn sport = :50201`: empty; no port listener.
- Stage3 root `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/` and its `logs/` parent remain absent; no destination exists to overwrite.

These checks passed. The GPU/process/port/destination observations are transient; the canonical S11 plan requires one final live recheck immediately before the command. If any changes, stop without creating a directory or launching.

## Sole authorized command sequence (not yet executed)

After the final live recheck and only under the canonical S11 authorization:

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs
nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 train_HG-Rec.py > /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_stage3_run.log 2>&1 &
```

The `mkdir -p` creates only the missing results-side parent required because shell redirection opens before Python starts. The existing direct trainer then launches its existing four-rank torchrun and writes all results beneath the short Iter31 results root. No wrapper, CLI arguments, environment override, source/config edit, alternative interpreter, second run, or Stage3 classification is authorized here.

## Final immediate live recheck before invocation

At 2026-09-27 20:14:14, after the no-training runtime/actual-dataset inspection and immediately before creating the results-side log parent:

- `nvidia-smi`: all four L40S devices still at 0 MiB / 0% compute; no GPU processes.
- `pgrep -af train_HG-Rec.py`: no matches (expected exit 1).
- `ss -ltn sport = :50201`: no listener.
- Result root and `logs/` parent: still absent.
- The pinned source/input hashes and route values had just matched the canonical S11 plan; no source/config changes occurred after those checks.

Final live checks PASS. Proceed only with the one canonical direct command, creating only the missing results-side `logs/` parent first.

## Corrected no-early-stop preflight — user-authorized replacement

```text
STAGE_ID=S11_STAGE3_EVALUATION
ROUND=2
S11_JUDGE_VERDICT=REPAIR_AND_RERUN
SAME_ITERATION_REPAIR=AUTHORIZED
PREFLIGHT_TIME=2026-09-27T21:23:47+10:00
PREFLIGHT_RESULT=PASS
STAGE3_LAUNCHED=NO_AT_RECORD_TIME
```

The user directed removal of Stage3 early stopping and authorized one corrected Iter31 rerun. Round-2 Judge C (`logs/deliberation/S11_STAGE3_EVALUATION/round_2/judge.md`) conditionally authorized that replacement. This record supersedes the round-1 patience-10 preflight for the replacement; the first attempt remains invalid and preserved as historical evidence.

### No-early-stop source gate

- Root `CLAUDE.md` §5.1 hash: `5915dd53c810fe650b85131f437f171bd79ee10c8a2fd96d30bc46c1d39a58f2`; it requires all configured epochs and a final test, and requires this scan before launch.
- Active trainer `stage3_T5Train/train_HG-Rec.py` hash: `638b8bb61403f0497d080c792e63c6d07dab32bfaa6fe8fc903a67c158c51d80`.
- Search `EARLY_STOP|early_stop|patience|screen_skip_test_on_fail|screen_failed_skip_test` across `stage3_T5Train/**/*.py`: no matches.
- Python AST smoke: parse PASS; training loop is `range(1, config['num_epochs'] + 1)`; no `break` or `return` in its body; no early-stop tokens remain. Resolved hardcoded settings: `NUM_EPOCHS=150`, `NO_EVAL=True`, `SKIP_TEST=False`, `SCREEN_BASELINE_LOG=""`.
- The only remaining test-skip branch is explicitly config-controlled and inactive under `SKIP_TEST=False`; screen-failure cannot stop training or skip the final test.

### Source and input identity

| Path | SHA-256 | Result |
|---|---|---|
| `stage3_T5Train/data/dataset.py` | `8899f1348e740996c572cd95439365f4caf2fe791acc4a0e22ed7d837937030e` | PASS |
| `stage3_T5Train/data/dataloader.py` | `d490b89c82a8ce3e1d8fa7577040038ef3411d030ce327fad95a999334f98af4` | PASS |
| `stage3_T5Train/model/hg_rec.py` | `f2876a95ee25b3f25d417792867c560da3053f8e5880120eda89affd9fd5cad1` | PASS |
| `stage3_T5Train/model/utils.py` | `a471b26da34ab30372f2cb2cc7baf05a6747be8830db4f178eac3b228364623c` | PASS |
| Iter31 `item_sids.json` | `d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7` | PASS |
| Stage0 `test.parquet` | `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc` | PASS |
| Iter29 comparator `test_final.json` | `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662` | PASS |

The prescribed Python 3.10.20 imported the actual trainer without entering `__main__`: PyTorch `2.13.0+cu130`, CUDA 13.0, CUDA available, four devices, BF16 supported, and `torchrun` executable. Resolved route values: seed 42; train/inference batch 4096/1024; beam 20; top-k `[5,10]`; full seen-history exclusion; exact absolute Iter31 `CODE_PATH` exists and maps to `iter31_hra_step6_common_reference`; fixed short Iter31 `LOG_PATH`/`SAVE_PATH`. At 21:19:38, `get_local_time()` yielded unused log/checkpoint timestamp `Sep-27-2026_21-19-38`; both prospective directories were absent. The process working directory was requested as `/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train`; its canonical physical path is the same checkout under `/fs04/ar57/wenyu/GeneRec/stage3_T5Train`.

### Previous-run evidence preservation and immediate live checks

Before launch, preserved these first-attempt root files under `logs/Amazon_2023_Instruments/Sep-27-2026_20-15-25/`; pre/post-move hashes matched:

| Original path | Preserved path | SHA-256 |
|---|---|---|
| `results/stage3_T5Train/curvature_RQ-VAE_iter31/_stage3_launcher.log` | `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_20-15-25/_stage3_launcher_initial_invalid.log` | `292fadd279c1a9ed7636fd74d917418b6fd3906059eca0a85c9bc235822a9995` |
| `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_stage3_run.log` | `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_20-15-25/_stage3_run_initial_empty.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_ddp_sync/shutdown_signal.txt` | `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_20-15-25/_ddp_shutdown_signal_initial_test_done.txt` | `979fc65999a298b412c1eb4a9769b4aeedec816b0f50c415b5598ccb6204fd72` |

At 21:23:47: all four L40S devices showed 0 MiB / 0% utilization; port 50201 had no listener; `pgrep -af '[t]rain_HG-Rec.py'` returned no match. The stale shutdown sentinel and both fixed root logs are absent from active paths; all three preserved destinations exist. The first attempt's timestamped `HG_Rec.log`, `training_metrics.jsonl`, `test_final.json`, and checkpoint were not moved or altered. Stage3 root exists and is expected; the earlier attempt remains isolated in its 20:15:25 timestamp directory.

### Authorized invocation

After this PASS record, launch exactly one direct no-argument Stage3 process from `/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train` with the prescribed Python 3.10 and existing four-rank launcher. No wrapper/CLI arguments/environment overrides. The process must produce a new timestamp subtree. Post-run acceptance still requires actual train events for epochs 1–150, no early-stop event, final `test_final.json` with `n_eval=57439`, fresh `test_done`, and clean rank/launcher exit.

The preliminary runtime probe that imported an unneeded `recbole` helper failed with `ModuleNotFoundError`; the corrected probe used the trainer's actual `model.utils.get_local_time` import and the full trainer import passed. The initial process scan also matched its own command line; the corrected `[t]rain_HG-Rec.py` scan returned no trainer process. These were preflight-harness corrections, not application failures.

## Launch observation (2026-09-27 21:29:09 +10:00)

The authorized no-argument Iter31 Stage3 replacement was launched via Hub process `Iter31Stage3Rerun`, using the prescribed Python 3.10 executable from `/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train`; outputs remain under the Iter31 short-name results root. The Hub `ready.port=50201` probe timed out after 120 seconds, but the application launch is confirmed independently: a fresh run directory `Sep-27-2026_21-27-08` exists; its launcher log records completed epochs 1 and 2 of 150; the `train_HG-Rec.py` parent, torchrun and four DDP ranks were observed alive; all four GPUs were at 28,379 MiB and 99–100% utilization. `training_metrics.jsonl` was actively growing. This is a readiness-probe timeout, not a failed training launch; no termination or second launch was attempted. Current observed progress: 2/150 epochs.

## Post-run acceptance (2026-09-27 22:09:10 +10:00)

```text
STAGE3_LAUNCHED=YES
RUN_DIR=results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_21-27-08
HUB_PROCESS=Iter31Stage3Rerun
HUB_EXIT_CODE=0
HUB_UPTIME=38m32s
TRAIN_EPOCHS=1..150_exactly_once
EARLY_STOP_EVENTS=0
FINAL_TEST_EVENTS=1
FINAL_TEST_EPOCH=150
N_EVAL=57439
DDP_SHUTDOWN_SIGNAL=test_done
RANK_PROCESSES_REMAINING=0
POSTRUN_ACCEPTANCE=PASS
```

Deterministic parse of `training_metrics.jsonl` reported `train_start=1`, `train=150`, `ckpt_saved=122`, `test=1`, `train_end=1`; ordered training epochs equal exactly `1..150`; no `early_stop` event; one final test event at epoch 150; one `train_end`. The test record and `test_final.json` agree:

```text
R@5=0.03809258517731855
R@10=0.05659917477671965
NDCG@5=0.025285381077977752
NDCG@10=0.031247955601888432
n_eval=57439
```

Observed completion evidence: final `test_done` sentinel; supervised Hub process exited 0; at 22:09:10 no `train_HG-Rec.py` process remained and all four GPUs were at 0 MiB / 0%. The Hub's initial TCP port readiness probe timed out, but training was independently confirmed active during startup from the timestamped run, epoch logs/metrics, live ranks and GPU utilization; it was not a training failure. Exactly one corrected run was invoked.

Post-run artifact hashes:

```text
_stage3_launcher.log                              b8abdf6036f782de7a32e3104d6dfafa915d72ccb618ac9f061a8214941449cd
logs/_stage3_run.log                              e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
logs/_ddp_sync/shutdown_signal.txt                979fc65999a298b412c1eb4a9769b4aeedec816b0f50c415b5598ccb6204fd72
logs/Amazon_2023_Instruments/Sep-27-2026_21-27-08/training_metrics.jsonl 0f706eb1c7daefd1bf9eac04a64d26424ded1c8b6f237bf06c3cb4009af499dd
logs/Amazon_2023_Instruments/Sep-27-2026_21-27-08/test_final.json d446c6b622e51c140cfaab1be5a76e3cb6ee56ec318ecb0dd5c4b3ca8c8f3957
logs/Amazon_2023_Instruments/Sep-27-2026_21-27-08/HG_Rec.log 96f99a05d2f5ff994b28c5ea8fc0e61793c95e11ec572ec2d9de9a02b58d718b
ckpt/Amazon_2023_Instruments/Sep-27-2026_21-27-08/HG_Rec_best.pth f7ab9ea713434559becea8f5827390849f845260dce3a7b908c16b661ba6796d
```

Protocol-compatible Iter29 comparator was independently reparsed: exactly 150 ordered train events, no early-stop event, one final test with `n_eval=57439`; its `test_final.json` hash and metrics match the locked S11 comparator. Performance classification is deferred to S12.
