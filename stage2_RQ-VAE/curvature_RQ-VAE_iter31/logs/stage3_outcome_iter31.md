# Iter31 Stage3 outcome — execution facts for S12

STAGE_ID=S11_STAGE3_EVALUATION
RUN_VALIDITY=EXECUTION_AND_EVALUATION_GATES_PASS
MECHANISM_CLASSIFICATION=DEFERRED_TO_S12
PROMOTION_DECISION=DEFERRED_TO_S12
USER_INPUT_REQUIRED=NO

## Authorized execution and route

- Corrected run: `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_21-27-08/`.
- Direct no-argument launch used the prescribed Python 3.10.20 executable from `stage3_T5Train`; the existing launcher used four DDP ranks, port 50201, devices 0–3 and configured NCCL settings. Supervised Hub process `Iter31Stage3Rerun` exited 0 (uptime 38m32s).
- Active trainer SHA-256: `638b8bb61403f0497d080c792e63c6d07dab32bfaa6fe8fc903a67c158c51d80`. Root `CLAUDE.md` §5.1 SHA-256: `5915dd53c810fe650b85131f437f171bd79ee10c8a2fd96d30bc46c1d39a58f2`.
- Locked route values: seed 42; 150 epochs; `NO_EVAL=True`; train/inference batch 4096/1024; beam 20; top-k `[5,10]`; full seen-history exclusion; final test on Stage0 Instruments split; Iter31 SID JSON SHA-256 `d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7`; test split SHA-256 `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc`.
- The `train_start` event records `num_epochs=150`, `no_eval=true`, `batch_size=4096`, the exact Iter31 SID `code_path`, the new timestamped log/checkpoint paths, `rank=0`, `world_size=4`. `variant=unknown_variant` is the previously disclosed logging-label limitation in the canonical S11 plan; the SID route is identified by its exact absolute path and hash. This metadata label does not change the evaluation route.

## Completion evidence

Deterministic parsing of the run's `training_metrics.jsonl`:

```text
train_start=1
train=150
ckpt_saved=122
test=1
train_end=1
training_epochs=1..150 exactly once
early_stop_events=0
final_test_epoch=150
n_eval=57439
```

The DDP shutdown sentinel is `test_done`. The Hub process exited 0; final resource inspection found no `train_HG-Rec.py` process and all four GPUs at 0 MiB / 0% utilization. Hub's initial TCP port readiness probe timed out after 120 seconds, but the application was independently observed training during that interval from epoch logs/metrics, live ranks and GPU utilization; it was not terminated and no second invocation occurred.

## Final test and comparator

Candidate `test_final.json` (SHA-256 `d446c6b622e51c140cfaab1be5a76e3cb6ee56ec318ecb0dd5c4b3ca8c8f3957`):

| Metric | Iter31 corrected run | Canonical Iter29 | Delta (Iter31 − Iter29) |
|---|---:|---:|---:|
| n_eval | 57,439 | 57,439 | 0 |
| R@5 | 0.03809258517731855 | 0.03953759640662268 | -0.0014450112293041342 |
| R@10 | 0.05659917477671965 | 0.05921064085377531 | -0.0026114660770556603 |
| NDCG@5 | 0.025285381077977752 | 0.026252776900288842 | -0.0009673958223110901 |
| NDCG@10 | 0.031247955601888432 | 0.03257647953179143 | -0.0013285239299029965 |

The locked strict target is R@10 `> 0.065`; Iter31 is 0.008400825223280353 below it. Iter29 is the canonical comparator (`test_final.json` SHA-256 `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`). Its primary training metrics independently confirm exactly 150 ordered epochs, zero early-stop events, one epoch-150 test, and `n_eval=57439`; the comparator is protocol-compatible with the corrected Iter31 execution.

The initial Iter31 run at `Sep-27-2026_20-15-25` is invalid and noncanonical because its metrics record only epochs 1–21 followed by a patience-10 early-stop event. It is preserved as historical evidence and is excluded from S12 comparison.

Stage2 contract/output wiring and direct HRA Step6 effect are documented in canonical `logs/sid_geometry_iter31.md`; its descriptive SID statistics are not performance gates. S11 established route/preflight authorization and no-early-stop execution, not a mechanism-effect or promotion conclusion. S12 must independently classify mechanism status and promotion status separately, using this outcome plus primary run/comparator artifacts.

## Artifact hashes

```text
_stage3_launcher.log                              b8abdf6036f782de7a32e3104d6dfafa915d72ccb618ac9f061a8214941449cd
logs/_stage3_run.log                              e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
logs/_ddp_sync/shutdown_signal.txt                979fc65999a298b412c1eb4a9769b4aeedec816b0f50c415b5598ccb6204fd72
training_metrics.jsonl                            0f706eb1c7daefd1bf9eac04a64d26424ded1c8b6f237bf06c3cb4009af499dd
test_final.json                                   d446c6b622e51c140cfaab1be5a76e3cb6ee56ec318ecb0dd5c4b3ca8c8f3957
HG_Rec.log                                       96f99a05d2f5ff994b28c5ea8fc0e61793c95e11ec572ec2d9de9a02b58d718b
HG_Rec_best.pth                                  f7ab9ea713434559becea8f5827390849f845260dce3a7b908c16b661ba6796d
Iter29 training_metrics.jsonl                    412860c203e20498e17ffd9c7a87d5e39836fcee563f1835f503e1df65a1b88f
```
