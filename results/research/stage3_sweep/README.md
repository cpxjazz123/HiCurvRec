# Stage3 sweep（owner 2026-11-20 指令：无条件进入 Stage3）

对象：`results/stage2_RQ-VAE/*/item_sids.json` 下**所有已完成 Stage2 的机制**（23 个臂）。
不看准入：碰撞率、前缀数、码用率等一律不作门禁；Stage3 仍用冻结协议
（`NUM_EPOCHS=150`、`EVAL_INTERVAL=5`、`EVAL_START_EPOCH=75`、`EARLY_EVAL_EPOCHS=(25,50,75,100,125)`、
`SEED=42`、`BEAM_SIZE=20`、`NO_EVAL=True`、`SKIP_TEST=False`）。

两条启动路径：
1. **MCP `stage3_start`**（首选）——先用 `tools.research.stage2.export_check --sid <SID> --tree <树> --rewrite`
   写出 `export_manifest.json`（框架拥有导出，原表另存 `item_sids.trainer.json`），再走 MCP。
2. **`run_stage3_direct.py`**（仅用于 Stage2 源码树已不存在的臂）——镜像 MCP 自己的 harness，
   同样的冻结协议与路径，只是不做 export 门禁；owner 指令明确不看准入。

臂与 Stage2 源码对应：

| 臂 | Stage2 源码 |
|---|---|
| TIGER_graphlet_role_{E,T,R,Q,QR} | `stage2_RQ-VAE/TIGER_graphlet_role_rq` |
| TIGER_g4_closure_{A,B,C,D} | `stage2_RQ-VAE/_archive/TIGER_g4_closure_rq` |
| TIGER_sinkhorn_all_levels_rq | `stage2_RQ-VAE/TIGER_sinkhorn_all_levels_rq` |
| TIGER_Uniform_KNN_RQ / _control | `_archive/TIGER_Uniform_KNN_RQ` / `_control` |
| TIGER_af3_curvature_factor_mass_rq | `_archive/TIGER_af3_curvature_factor_mass_rq` |
| TIGER_behavior_edge_contrastive_rq | `_archive/TIGER_behavior_edge_contrastive_rq` |
| TIGER_graphlet_codebook_rq / _global_rq | `_archive/TIGER_graphlet_codebook_rq` / `_global_rq` |
| TIGER_RQ-VAE | `baseline/TIGER_RQ-VAE` |
| TIGER_RQ-VAE__canonical / TIGER_af3_curvature_drive_rq / TIGER_mixed_rq_F0 / TIGER_role_geom_{rq,global_rq,shuffled_rq} | 源码树已不在仓库 → `run_stage3_direct.py` |

结果一律写入 `results/stage3_T5Train/<同名变体>/`，`test_final.json` 的 `test_recall@10` 是唯一成功标准（>0.065）。
