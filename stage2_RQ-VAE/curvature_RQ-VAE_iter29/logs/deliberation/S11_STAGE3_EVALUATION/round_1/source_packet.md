# S11_STAGE3_EVALUATION — canonical source packet

STAGE_ID=S11_STAGE3_EVALUATION
ITER=29
ROUND=1

## Objective

Independently audit Stage3 data wiring, fixed evaluation protocol, output routing, and one-run launch method using the adjudicated S10 result. Produce a complete method/patch plan only. Do not change source, launch Stage3, or use a Stage2 metric as an evaluation gate. Judge C must authorize any source change and the single Stage3 execution.

## Controlling rules and canonical result

- Root `CLAUDE.md` §§1, 5, 11, 13: project scripts take no CLI args or environment overrides; Stage3 is supervised using the prescribed Python 3.10 environment and 4-rank `_launch_via_torchrun`; all Stage3 outputs belong under `results/stage3_T5Train/curvature_RQ-VAE_iter29/`; the Stage3 `LOG_PATH`/`SAVE_PATH` resolve to this short directory, not the descriptive mechanism suffix; iteration closure requires commit/push of code, Stage2 and Stage3 artifacts.
- `logs/protocol_manifest_iter29.md:13-17,41-50,70`: iter26 is the sole protocol-valid direct control (`test_recall@10=0.057017009349048554`, `n_eval=57439`); Stage3 seed 42, epochs 150, beam 20, expected `n_eval=57439`; the registered user success criterion is inclusive `test_recall@10 >= 0.065`.
- S10 is adjudicated `MERGE_AB`: `logs/deliberation/S10_STAGE2_ANALYSIS/round_1/judge.md` and canonical `logs/sid_geometry_iter29.md`. Stage2 output is complete and verified; no Stage3 process/result exists yet.

## Stage2 Stage3-input evidence

- Exact consumer JSON: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json`, SHA-256 `58665e08a97e122f47a7ed3616b67efdeb8eadba8253771a1498cb5e57487ff` (`logs/stage2_output_integrity_iter29.log:24-28`). It contains 24587 index-keyed, unique 4-token SIDs; every JSON row matches the 4-token NPY, which preserves the raw 3-token prefix. The extension IDs are 768–780. See `logs/sid_geometry_iter29.md:35-46` and `logs/stage2_output_integrity_iter29.log`.
- Stage2 source/config/logs and the output-integrity record establish one completed run at 100000 steps. Stage2 descriptive statistics and S08 MVG do not establish Stage3 efficacy.

## Stage3 route and model/data contract to verify

- Sole runner: `scripts/run_stage3_iter29.py`. It imports `stage3_T5Train/train_HG-Rec.py`, then explicitly assigns `trainer.CODE_PATH` to the exact iter29 `item_sids.json`; `trainer.RQVAE_VARIANT="iter29_bounded_rational_additive_mapping"`; `trainer.LOG_PATH` and `trainer.SAVE_PATH` to `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/` and `/ckpt/`; `_LAUNCHER["script"]` to this runner; `_LAUNCHER["log"]` to the iter29 `_stage3_launcher.log`; and calls `trainer.main()` under torchrun ranks or `_launch_via_torchrun()` otherwise. The wrapper is no-argument. The global Stage3 module's default `CODE_PATH`/variant table does not contain iter29; the wrapper override must be verified as the actual rank-worker route.
- `train_HG-Rec.py:104-128,526-682,1110-1160`: Stage0 dataset root; `CODEBOOK_SIZE=[256,256,256,1]`; token vocabulary derives at runtime (`VOCAB_SIZE=0`). `data/dataset.py:80-156,183-204` accepts numeric JSON item keys, shifts SID components by +1, checks full SID uniqueness, derives vocabulary from maximum observed token, and constructs user/EOS ranges. Audit how the fourth-token IDs 768–780 route through this contract.
- Locked training/eval constants in `train_HG-Rec.py:55-65,92-101,160-172`: 150 epochs, seed 42, BF16, top-k `[5,10]`, beam 20, `NO_EVAL=True`, `SKIP_TEST=False`, empty `SCREEN_BASELINE_LOG`. With no screen curve and `skip_test=False`, final test evaluation is not skipped. `NO_EVAL=True` means no validation set/validation-based checkpoint selection: training-loss best checkpoint and train-loss patience are used; rank 0 then reloads that checkpoint and evaluates test. The final JSON stores `test_recall@10`, other test metrics, `n_eval=len(test_dataset)`, and checkpoint path under the timestamped dataset subdirectory of the short `LOG_PATH`/`SAVE_PATH` roots.
- Direct comparison is to the exact iter26 result path/value in the protocol manifest. Record actual final `test_recall@10` and `n_eval`; do not infer them from training/validation metrics.

## Mandatory launch-environment discrepancy

The controlling root rule requires the child training process to receive these exact hard-coded environment values: `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_TIMEOUT=3600`, `TORCH_NCCL_BLOCKING_WAIT=1`.

Current `stage3_T5Train/train_HG-Rec.py:1175-1213` uses `setdefault`; it defaults IB to `0`, P2P to `1`, SHM to `0`, timeout to `3600`, and blocking wait to `1`. Its comments at lines 1199–1205 explicitly describe a contrary IB/SHM configuration and warn that disabling IB/P2P/SHM previously caused transport/deadlock problems. The current shell inspection found none of the five variables set in the inspected shell; that is not proof of the Hub process environment. A compliant launch must not rely on inherited environment variables to satisfy the hard rule. Independently assess the smallest safe code change that guarantees the five mandated values for this run, preserves every Stage3 model/data/evaluation constant, updates any now-inaccurate comment, and has a no-GPU smoke verification plan. Explicitly retain the transport/deadlock risk from the existing code comment; do not silently choose the conflicting values.

## Single-run supervision method to assess

After a later S11 Judge decision and any Judge-authorized minimal patch/smoke:

- Run the wrapper once from the iter29 source directory with `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 scripts/run_stage3_iter29.py` and no args.
- Use the project process supervisor as a no-restart foreground process; preserve `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/_stage3_run.log` and wrapper-created `.../logs/_stage3_launcher.log`. Verify actual DDP rank startup from the launcher log, not only Hub spawn status. Do not repeat on a readiness warning or failed attempt.
- Before that sole launch, propose exact preflight checks: Stage2 JSON path/hash/freshness against S10; Stage0 train/valid/test input hashes against S01 (or explain which are consumed/skipped by `NO_EVAL=True`); output tree contains no stale run/ckpt/test artifacts; Python/Torch/CUDA and 4 GPUs available; Stage3 path assignments resolve to the iter29 short output root; the JSON produces 24587 items, 4-token unique SIDs and expected vocabulary; full-test cardinality equals locked 57439; exact child NCCL env values; one-run/supervisor identity.
- After exit, inspect terminal status, DDP/launcher output, `training_metrics.jsonl`, `HG_Rec_best.pth`, and exactly one fresh timestamped `test_final.json`. Verify `test_recall@10`, `n_eval`, all requested metrics, correct checkpoint/result paths, and no test-skip marker. Stage3 is run only once.

## Required independent candidate output

Each candidate must state direct source facts, identify the minimal compliant patch or explain why none is needed, map the Stage3 inputs and outputs, audit full final-test behavior and protocol compatibility, name the environment/deadlock risk, and propose the complete no-argument one-run plan plus no-GPU patch smoke. Do not make edits, launch, or claim a recall result. Skip formatters, linters, and project-wide tests.
