# S11 round-2 operational repair record

ROUND_TYPE=OPERATIONAL_REPAIR
LOCKED_SCIENCE_CHANGED=NO
USER_DIRECTED_STAGE3_DURATION_CORRECTION=150_FIXED_EPOCHS_NO_EARLY_STOP
SAME_ITERATION_REPAIR=AUTHORIZED
S11_JUDGE_VERDICT=REPAIR_AND_RERUN
PREFLIGHT_RESULT=PASS
STAGE3_LAUNCHED=NO_AT_RECORD_TIME
PARALLEL_EXECUTION=YES

## Repair scope

The user explicitly instructed that Stage3 has no early stopping, that all active early-stop mechanisms must be checked and removed before Stage3, and that Iter31 Stage3 be rerun. This higher-priority direction supersedes the erroneous patience-10 wording in S01/S11 round 1. No Stage2 mechanism, Stage3 model, optimizer, scheduler, seed, data, evaluator, metric, beam, checkpoint-selection rule, or output root changed.

Files changed:

- `CLAUDE.md` §5.1 (lines 27–33): added mandatory pre-Stage3 source audit/removal gate, fixed configured epoch budget, and post-run validity criteria.
- `stage3_T5Train/train_HG-Rec.py` (constant/config and epoch-loop sections around lines 60–61, 164–171, 531–585, 814–1040): removed `EARLY_STOP`, train-loss and validation patience state/counters, `_early_stop_file` signaling, metric-triggered loop break, and screen-failure early test skip. Best-checkpoint selection remains. `SKIP_TEST=False` and the explicit false-only configuration branch remain unchanged.

## Commands and deterministic checks

1. Active Stage3 Python scan: `EARLY_STOP|early_stop|patience|screen_skip_test_on_fail|screen_failed_skip_test` across `stage3_T5Train/**/*.py` → no matches.
2. Python AST smoke on `train_HG-Rec.py` → syntax PASS; the `main()` epoch loop is `range(1, config['num_epochs'] + 1)`; no `break`/`return` in the epoch loop; no early-stop tokens; constants are `NUM_EPOCHS=150`, `NO_EVAL=True`, `SKIP_TEST=False`.
3. SHA-256 verification (all matched S11 round-2 packet pins):

```text
CLAUDE.md                                      5915dd53c810fe650b85131f437f171bd79ee10c8a2fd96d30bc46c1d39a58f2
stage3_T5Train/train_HG-Rec.py                 638b8bb61403f0497d080c792e63c6d07dab32bfaa6fe8fc903a67c158c51d80
stage3_T5Train/data/dataset.py                8899f1348e740996c572cd95439365f4caf2fe791acc4a0e22ed7d837937030e
stage3_T5Train/data/dataloader.py             d490b89c82a8ce3e1d8fa7577040038ef3411d030ce327fad95a999334f98af4
stage3_T5Train/model/hg_rec.py                 f2876a95ee25b3f25d417792867c560da3053f8e5880120eda89affd9fd5cad1
stage3_T5Train/model/utils.py                  a471b26da34ab30372f2cb2cc7baf05a6747be8830db4f178eac3b228364623c
Iter31 item_sids.json                          d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7
Stage0 test.parquet                             5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc
Iter29 comparator test_final.json              b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662
```

4. Prescribed runtime/trainer import under `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10` → Python 3.10.20, PyTorch 2.13.0+cu130, CUDA 13.0, CUDA available, 4 devices, BF16 supported, torchrun executable. Imported trainer constants: seed 42, batch/inference batch 4096/1024, beam 20, top-k `[5,10]`, seen-history exclusion true, exact absolute Iter31 `CODE_PATH` exists and resolves to the same physical `/fs04` alias of the `/home` checkout, variant `iter31_hra_step6_common_reference`, `LOG_PATH`/`SAVE_PATH` use short Iter31 result root.
5. Live state at 2026-09-27T21:23:47+10:00: 4 NVIDIA L40S GPUs each 0 MiB / 0%; port 50201 has no listener; `pgrep -af '[t]rain_HG-Rec.py'` no match.
6. Prospective timestamp probe at 21:19:38 returned `Sep-27-2026_21-19-38`; its log and checkpoint directories were absent. Existing 20:15:25 run and checkpoint remain untouched; the Iter31 root is expected to exist.

The preliminary runtime probe imported an unused `recbole` helper and returned `ModuleNotFoundError`; it was corrected to use the trainer's actual `model.utils.get_local_time` import and passed. The first process probe matched its own command line; corrected `[t]rain_HG-Rec.py` returned no match. These were harness corrections, not application failures.

## Preserved prior fixed-path files

The first invalid attempt's fixed root logs and stale DDP sentinel were moved under its existing 20:15:25 audit directory. Pre/post hashes match; no timestamped test/metrics/checkpoint was changed.

| Original | Preserved destination | SHA-256 |
|---|---|---|
| `results/stage3_T5Train/curvature_RQ-VAE_iter31/_stage3_launcher.log` | `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_20-15-25/_stage3_launcher_initial_invalid.log` | `292fadd279c1a9ed7636fd74d917418b6fd3906059eca0a85c9bc235822a9995` |
| `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_stage3_run.log` | `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_20-15-25/_stage3_run_initial_empty.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_ddp_sync/shutdown_signal.txt` | `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_20-15-25/_ddp_shutdown_signal_initial_test_done.txt` | `979fc65999a298b412c1eb4a9769b4aeedec816b0f50c415b5598ccb6204fd72` |

At the recorded final check the active sentinel, root launcher-log, and outer-log paths were absent; all preserved destinations existed. The corrected direct no-argument run is authorized once by the round-2 Judge after the recorded preflight PASS. Post-run validity remains conditional on actual epochs 1–150, no early-stop event, full final test with `n_eval=57439`, fresh shutdown signal, and clean rank/launcher exit.

## Run-launch observation (2026-09-27 21:29:09 +10:00)

The single round-2-authorized corrected run was started as Hub process `Iter31Stage3Rerun` with the prescribed no-argument Python 3.10 command and Iter31 result root. Hub's TCP-port readiness condition timed out after 120 seconds. Independent application evidence confirms successful training startup: new run `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_21-27-08/`; launcher log reports epochs 1 and 2 completed out of 150; trainer parent, torchrun and four DDP ranks are alive; all four GPUs show 28,379 MiB and 99–100% utilization; metrics file is growing. Therefore the readiness-probe timeout is not a Stage3 launch failure. No termination or additional launch occurred. Progress at this observation: 2/150 epochs; full-run validity and final-test requirements remain pending.

## Post-run repair verification (2026-09-27 22:09:10 +10:00)

ROUND_TYPE=OPERATIONAL_REPAIR
LOCKED_SCIENCE_CHANGED=NO
ALL_AUTHORIZED_CHECKS_PASS=YES
STAGE3_LAUNCHED=YES

Exactly one corrected no-argument Stage3 run completed in `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_21-27-08/`. Hub process `Iter31Stage3Rerun` exited 0 after 38m32s. A deterministic parse found events `train_start=1`, `train=150`, `ckpt_saved=122`, `test=1`, `train_end=1`; train epochs were exactly 1–150 in order, with zero `early_stop` events. `test_final.json` matches the single epoch-150 test row (`n_eval=57439`): R@5 `0.03809258517731855`, R@10 `0.05659917477671965`, NDCG@5 `0.025285381077977752`, NDCG@10 `0.031247955601888432`. Checkpoint is under the same new timestamp subtree. DDP sentinel is `test_done`; no trainer process remained and four GPUs were free at final inspection.

The independent Hub TCP-port readiness probe timed out after 120 seconds; it did not indicate an application failure. During that probe window the timestamped launcher log recorded completed epochs 1 and 2, the trainer/torchrun/four ranks were alive, the metrics file grew, and all four GPUs were occupied. No termination or second run occurred. The Hub wrapper ultimately exited 0.

Post-run artifact SHA-256:

```text
_stage3_launcher.log                              b8abdf6036f782de7a32e3104d6dfafa915d72ccb618ac9f061a8214941449cd
logs/_stage3_run.log                              e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
logs/_ddp_sync/shutdown_signal.txt                979fc65999a298b412c1eb4a9769b4aeedec816b0f50c415b5598ccb6204fd72
training_metrics.jsonl                            0f706eb1c7daefd1bf9eac04a64d26424ded1c8b6f237bf06c3cb4009af499dd
test_final.json                                   d446c6b622e51c140cfaab1be5a76e3cb6ee56ec318ecb0dd5c4b3ca8c8f3957
HG_Rec.log                                       96f99a05d2f5ff994b28c5ea8fc0e61793c95e11ec572ec2d9de9a02b58d718b
HG_Rec_best.pth                                  f7ab9ea713434559becea8f5827390849f845260dce3a7b908c16b661ba6796d
```

The protocol-compatible Iter29 comparator was reparsed from its primary metrics: epochs 1–150 exactly once; zero early-stop events; one epoch-150 test, `n_eval=57439`; test metrics agree with the pinned comparator JSON. All authorized operational checks now pass. S12 independently adjudicates mechanism effect and promotion.
