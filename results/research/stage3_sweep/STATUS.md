# Stage3 sweep 已结项


## 结项（2026-11-20，owner 指令）

owner 明确：本臂训练与评估完成后即结束，不再启动下一个臂。因此 **22/23 个 Stage2 机制完成 Stage3**，
`TIGER_sinkhorn_all_levels_rq` 按指令**不启动**（其 Stage2 SID 已存在，可随时补跑）。

- 完成 22 个臂，`test_recall@10` 范围 **0.0382–0.0572**；
- 最高 `TIGER_mixed_rq_F0` 0.0572（基线 `TIGER_RQ-VAE` 0.0555）；
- **`> 0.065` 的臂:0 个**；
- 21 个臂已按 §6b 四件套归档 R2（编号 69–90 等）；唯一未归档的是源码树缺失的 `af3_curvature_drive_rq`。

产物:`REPORT.md`、`summary.json`、`index.json`、`queue.log`、`archive.log`。
