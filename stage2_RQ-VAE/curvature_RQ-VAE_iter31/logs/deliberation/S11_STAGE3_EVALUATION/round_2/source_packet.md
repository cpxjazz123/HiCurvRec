# Iter31 S11 corrected Stage3 route — frozen source packet

STAGE_ID=S11_STAGE3_EVALUATION
ROUND=2
ROUND_TYPE=OPERATIONAL_REPAIR
STATUS=FROZEN_SOURCE_PACKET
SOURCE_PACKET=logs/deliberation/S11_STAGE3_EVALUATION/round_2/source_packet.md

## User instruction

The user directly stated:

- “帮我把这个earlystop去掉，stage3没有早停机制。”
- “修改 规则，在进入stage3之前先检查有没有任何earlystop机制如果有必须移除。”
- “然后重新运行一下iter31的stage3”

This instruction supersedes the erroneous S11 round-1 plan that preserved `EARLY_STOP=10`. Do not ask the user for further direction.

## Why this repair is required

The original `logs/stage3_evaluation_plan_iter31.md` and `logs/protocol_manifest_iter31.md` incorrectly specified train-loss patience 10 and allowed less than 150 epochs. The first Stage3 attempt stopped at epoch 21 due to `train_loss_patience_exhausted`; it is not canonical evidence under the user-required full-epoch Stage3 protocol. Its final test is noncanonical and must not be used for causal classification or promotion.

Primary first-attempt evidence:

- Run directory: `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_20-15-25/`.
- `training_metrics.jsonl` SHA-256: `07d743a7e9943c9d73e762a84e7ade679808d50f44e33738f71e9389be327868`; epoch 21 train record is followed by `event=early_stop`, `trigger=train_loss_patience_exhausted`, `patience=10`, then test.
- `test_final.json` SHA-256: `cb254e64c71029c78c3e4ed6d2249c545f0ab8f62fc9f76936f43fbb9c815b08`; `n_eval=57439`, R@5 `0.013422935636066087`, R@10 `0.021170284997997876`, NDCG@5 `0.006919332448219404`, NDCG@10 `0.00943575468903065`. Treat all these metrics as invalid for promotion.
- The test event's `epoch=150` is a configured-value label, not evidence that 150 training epochs ran; the explicit epoch-21 early-stop record proves otherwise.
- The launcher log shows ranks progressing into epoch 22 while rank 0 had taken the early-stop/test path; the supervisor was later terminated with SIGTERM. Root `_ddp_sync/shutdown_signal.txt` currently contains stale `test_done` from that attempt.
- Existing root `_stage3_launcher.log` SHA-256: `292fadd279c1a9ed7636fd74d917418b6fd3906059eca0a85c9bc235822a9995`. Preserve it before the new launcher overwrites that fixed path.

## Corrected rule and implementation evidence

Project root `CLAUDE.md` now has §5.1 `Stage3 no-early-stop gate`, SHA-256 `5915dd53c810fe650b85131f437f171bd79ee10c8a2fd96d30bc46c1d39a58f2`. It requires auditing the active trainer/launcher before each Stage3 launch, removing metric-triggered early exits, completing configured `NUM_EPOCHS` (currently 150), and rejecting an incomplete run.

Active trainer: `stage3_T5Train/train_HG-Rec.py`, SHA-256 `638b8bb61403f0497d080c792e63c6d07dab32bfaa6fe8fc903a67c158c51d80`.

The repaired source removes the `EARLY_STOP` constant/config, train-loss and validation patience counters/signals, the `_early_stop_file` path, the screen-failure training break, and screen-failure test skip. Best-checkpoint selection remains but cannot terminate training. `EARLY_EVAL_EPOCHS` remains diagnostic only. `SCREEN_BASELINE_LOG` is empty; `SKIP_TEST=False`.

Concurrent static checks passed before this packet was frozen:

- Search of active `stage3_T5Train/**/*.py` for `EARLY_STOP|early_stop|patience|screen_skip_test_on_fail|screen_failed_skip_test`: no matches.
- Python AST parse: PASS.
- The `main()` epoch loop is `range(1, config['num_epochs'] + 1)` and contains no `break` or `return`.
- Hardcoded values: `NUM_EPOCHS=150`, `NO_EVAL=True`, `SKIP_TEST=False`.
- Current runtime constants: seed 42, train batch 4096, inference batch 1024, BF16 and FAST enabled, compile disabled, beam 20, top-k `[5,10]`, full seen-history exclusion.

## Locked data and comparator

- Stage3 SID input: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json`, SHA-256 `d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7`.
- Stage0 test split: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/test.parquet`, SHA-256 `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc`.
- Expected `n_eval=57439`.
- Canonical comparator remains Iter29 `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`, SHA-256 `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`: R@5 `0.03953759640662268`, R@10 `0.05921064085377531`, NDCG@5 `0.026252776900288842`, NDCG@10 `0.03257647953179143`.
- Strict adoption target remains `test_R@10 > 0.065`.

## Rerun scope and output preservation

This is one corrected replacement execution after the invalid early-stop attempt, explicitly authorized by the user. It is not a seed replication or variance study. Preserve the Iter31 Stage2 SID, Stage0 data split, model, optimizer, scheduler, seed, beam, metrics, evaluator, checkpoint-selection semantics, and output root. The only training-duration correction is full configured 150 epochs with no metric-triggered stop.

- Direct no-argument entrypoint: `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 train_HG-Rec.py`, from `/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train`.
- Existing launcher contract: `torchrun`, 4 ranks, port 50201, `CUDA_VISIBLE_DEVICES=0,1,2,3`, mandated NCCL settings in the trainer; do not add CLI arguments or environment overrides.
- Fixed short output root: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/`.
- The root already contains the invalid attempt. Do not clean or overwrite its timestamped run/checkpoint. The corrected invocation must create a new timestamped run directory.
- Before launch, preserve the existing root launcher log, empty outer run log, and stale `_ddp_sync/shutdown_signal.txt` under the first invalid run directory; this prevents log loss and ensures nonzero ranks wait on a fresh shutdown signal. Record each move and hash in the repair record.
- Before launch, repeat the immediate runtime/GPU/process/port checks from S11 and the new no-early-stop source scan/hash gate. Do not launch if any check fails.

## Parallel execution DAG

PARALLEL_GROUP_1=Static trainer/rule scan; input/output/baseline identity reads; current runtime and live-resource checks.
PARALLEL_GROUP_2=Agent A and Agent B independently audit this same frozen packet and corrected route in parallel.
SERIAL_DEPENDENCIES=Judge C only after A/B complete; final immediate preflight after Judge; one Stage3 run after preflight; S12 classification after completed outputs.

No Stage3 run has been launched for the corrected source at packet-freeze time.
