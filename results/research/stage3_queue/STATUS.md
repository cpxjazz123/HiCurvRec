# Stage3 队列状态（自动更新：每完成一臂由 runner 写入 queue_report.json；本文件为人工快照）

| 顺序 | 臂 | Stage3 | test_R@10 | R2 |
|---|---|---|---|---|
| 1 | TIGER_factorized_joint_codebook | done | 0.0513588 | 15 |
| 2 | TIGER_af3sign_capacity | done | 0.0545100 | 16 |
| 3 | TIGER_M1_learned_kappa | running | - | - |

队列：33 臂，按 Stage2 SID 完成时间倒序（`results/research/stage3_queue.json`）。
runner：`results/research/stage3_queue/run_queue.py`（PID 1622507），日志 `run.log`。
R2 账本：`results/research/issue139/r2_ledger.json`（当前最大 16）。

## 2026-11-09 16:30 - the remaining arms keep their historical recall

Three of the eleven retrained trees were re-run through Stage 3 and landed within
0.0004-0.0012 of their historical value on a *new* SID:

    TIGER_BEHAVIOR_R48                          0.0619266 -> 0.0631104
    TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample       0.0637372 -> 0.0633019
    TIGER_behavior_residual_geometry_transport  0.0633890 -> 0.0642247

That is single-seed noise, so the owner stopped the queue: the remaining eight
arms keep the recall already in the bucket (`51`-`56`), marked
`recall_paired: false` because it was measured on the SID that retraining
replaced. No further Stage 3 is started for this batch.
