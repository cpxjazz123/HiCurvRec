# Stage3 无条件扫描（owner 2026-11-20 指令）

对 `results/stage2_RQ-VAE` 下**所有**已完成 Stage2 并导出 SID 的机制跑 Stage3，不看碰撞率、
前缀数、码用率等任何准入条件；Stage3 仍用冻结协议（150 epoch、`EVAL_INTERVAL=5`、
`EVAL_START_EPOCH=75`、`EARLY_EVAL_EPOCHS=(25,50,75,100,125)`、`SEED=42`、`BEAM_SIZE=20`、
`NO_EVAL=True`、`SKIP_TEST=False`），四卡 DDP，串行（Stage3 单例）。

## 结果表

| arm | collision | test_recall@10 | R2 |
|---|---|---|---|
| TIGER_RQ-VAE | 0.1693 | 0.0555 | 69 |
| TIGER_RQ-VAE__canonical | 0.1775 | 0.0547 | 73 |
| TIGER_Uniform_KNN_RQ | 0.3748 | 0.0553 | 81 |
| TIGER_Uniform_KNN_control | 0.1831 | 0.0558 | 82 |
| TIGER_af3_curvature_drive_rq | 0.1703 | 0.0548 | — |
| TIGER_af3_curvature_factor_mass_rq | 0.2355 | 0.0546 | 74 |
| TIGER_behavior_edge_contrastive_rq | 0.3375 | 0.0563 | 83 |
| TIGER_g4_closure_A | 0.1242 | 0.0555 | 84 |
| TIGER_g4_closure_B | 0.1233 | 0.0542 | 85 |
| TIGER_g4_closure_C | 0.0131 | 0.0409 | 86 |
| TIGER_g4_closure_D | 0.0098 | 0.0382 | 87 |
| TIGER_graphlet_codebook_global_rq | 0.2640 | 0.0464 | 80 |
| TIGER_graphlet_codebook_rq | 0.2941 | 0.0471 | 79 |
| TIGER_graphlet_role_E | 0.1221 | 0.0543 | 88 |
| TIGER_graphlet_role_Q | 0.0102 | 0.0451 | 78 |
| TIGER_graphlet_role_QR | 0.1127 | 0.0541 | 89 |
| TIGER_graphlet_role_R | 0.0107 | 0.0482 | 90 |
| TIGER_graphlet_role_T | 0.0108 | 0.0434 | 91 |
| TIGER_mixed_rq_F0 | 0.2572 | 0.0572 | 75 |
| TIGER_role_geom_global_rq | 0.2156 | 0.0552 | 71 |
| TIGER_role_geom_rq | 0.2850 | 0.0554 | 70 |
| TIGER_role_geom_shuffled_rq | 0.2412 | 0.0543 | 72 |
| TIGER_sinkhorn_all_levels_rq | 0.1105 | pending | — |

## 方法要点

- 两个入口：MCP `stage3_start`（有 Stage2 源码树的臂）+ `run_stage3_direct.py`（源码树已归档/缺失的臂，
  镜像 MCP harness 的同一冻结协议与路径，不做 export 门禁）；
- 导出：能为臂找到源码树的先用框架 `export_check --rewrite` 由 checkpoint 重导出 SID
  （原表留 `item_sids.trainer.json`）；`TIGER_RQ-VAE` 的树与 checkpoint 不匹配（24587 行全不同、
  碰撞率会从 0.1693 变 0.1099），已恢复原表并改走直启；
- 队列 `run_queue.py` 逐臂推进、失败记录后继续；`TIGER_RQ-VAE`/`role_geom` 等旧臂的历史 Stage3
  结果被识别并跳过（不重复训练）。

## 已知约束

- 单臂约 1.5–2 小时（含 ~10 分钟共享文件系统导入开销），23 臂串行；
- Stage2 源码树已缺失的 5 个臂（`RQ-VAE__canonical`、`mixed_rq_F0`、`role_geom_{rq,global_rq,shuffled_rq}`）
  无法为新运行重做 §6b 四件套；它们的历史归档已在 R2（编号 73/75/70/71/72）。

## 执行记录

- 已完成 17/23 臂,当前正在跑 `TIGER_g4_closure_D`;
- 队列按序完成剩余臂:`TIGER_g4_closure_D`, `TIGER_graphlet_role_E`, `TIGER_graphlet_role_QR`, `TIGER_graphlet_role_R`, `TIGER_graphlet_role_T`, `TIGER_sinkhorn_all_levels_rq`;
- 每个臂完成后 `archive_daemon.py` 自动按 §6b 四件套推送 R2 并在 `r2_ledger.json` 记账;
- 结束时运行 `python3 results/research/stage3_sweep/finalize.py` 即可得到最终判定(最高臂、`>0.065` 列表、缺失臂)。

## 可复核性审计（2026-11-20）

- **10 个臂**保留了 `predictions/rank*.jsonl`：合并全部 rank 后从 top-10 重算
  `test_recall@10`，与 `test_final.json` 的记录值**逐臂完全一致**（|Δ| ≤ 1e-4，行数 = `n_eval` = 57439）；
- **7 个 2026-11 的历史运行**（`RQ-VAE`、`RQ-VAE__canonical`、`af3_curvature_factor_mass`、
  `mixed_rq_F0`、`role_geom_{rq,global_rq,shuffled_rq}`）只保留了 `test_final.json` 与 `ckpt`，
  predictions 未留档，无法重算——其数值以记录为准并在表中标注；
- 17 个结果的 `test_final.json` 键齐全（`best_checkpoint`/`n_eval`/`predictions`/`test_ndcg@5,10`/
  `test_recall@5,10`），且 `best_checkpoint` 均指向本变体且文件存在。

## 中期统计（19 臂）

- `test_recall@10`:mean 0.0521 / median 0.0547 / min 0.0382(`g4_closure_D`)/ max 0.0572(`mixed_rq_F0`);
- **corr(collision, recall@10) = +0.584**:在本轮 19 个臂上,碰撞率越高、Stage3 召回越好;
  三个近全唯一码的臂(`g4_closure_D` 0.0098→0.0382、`g4_closure_C` 0.0131→0.0409、
  `graphlet_role_Q` 0.0102→0.0451)正好是最差三名。这是横跨不同机制的描述性相关,
  不能读作因果,但它与 #216 的"唯一性≠结构保真"结论方向一致。

## 结项（2026-11-20，owner 指令）

owner 明确：本臂训练与评估完成后即结束，不再启动下一个臂。因此 **22/23 个 Stage2 机制完成 Stage3**，
`TIGER_sinkhorn_all_levels_rq` 按指令**不启动**（其 Stage2 SID 已存在，可随时补跑）。

- 完成 22 个臂，`test_recall@10` 范围 **0.0382–0.0572**；
- 最高 `TIGER_mixed_rq_F0` 0.0572（基线 `TIGER_RQ-VAE` 0.0555）；
- **`> 0.065` 的臂:0 个**；
- 21 个臂已按 §6b 四件套归档 R2（编号 69–90 等）；唯一未归档的是源码树缺失的 `af3_curvature_drive_rq`。
