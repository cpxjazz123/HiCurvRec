# Stage3 队列状态（自动更新：每完成一臂由 runner 写入 queue_report.json；本文件为人工快照）

| 顺序 | 臂 | Stage3 | test_R@10 | R2 |
|---|---|---|---|---|
| 1 | TIGER_factorized_joint_codebook | done | 0.0513588 | 15 |
| 2 | TIGER_af3sign_capacity | done | 0.0545100 | 16 |
| 3 | TIGER_M1_learned_kappa | running | - | - |

队列：33 臂，按 Stage2 SID 完成时间倒序（`results/research/stage3_queue.json`）。
runner：`results/research/stage3_queue/run_queue.py`（PID 1622507），日志 `run.log`。
R2 账本：`results/research/issue139/r2_ledger.json`（当前最大 16）。
