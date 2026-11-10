# question35 — Stage2 已验收通过；本轮唯一目标：按冻结协议跑完 Stage3 并给出 test_recall@10

**状态：CLOSED（已验收：完整 Stage3 跑完 150 epoch；负结果，见 `struct_mixed_rq/CONCLUSION.md`）**　承接：`answer34.md`

## 0. Stage2 验收结论：**通过**

我逐项复核了 `results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/`，与 answer34 一致：

| 项 | 实测 |
|---|---|
| 事件 | `train_start` → 3000 行 `"event":"train"`（epoch 1..3000）→ `ckpt_saved(final_epoch)` → `train_end`；`training_metrics.jsonl` 共 3004 行 |
| 配方 | `epochs=3000`、`batch_size_per_rank=1024`、`total_batch_size=4096`、`seed=42`、`early_stop=disabled`、`world_size=4` |
| 墙钟 | `total_wall_time_s=1038.082`（≈17.3 min），launcher 与 4 rank 均退出 |
| epoch=3000 | `loss=1.818569580791518e-4`、`recon=7.262202416313812e-05`、`lr=0.0`、`codeword_h_max=[0.47435,0.14896,0.06408]` |
| 产物 SHA | `rqvae_best.pth 9da6aa4b…`、`item_sids.json 1839d1ce…`、`training_metrics.jsonl 9efc6e0d…`、`stage2_metrics/1839d1…json 0bbe9a16…`（与 answer34 全部匹配） |
| 分层记录 | `status=complete`、`technically_valid_sid=true`、`stage3_results_used=false`；我独立从 SID 复算：24587 商品、**L1 224/256 = 87.5%**、L2 前缀 **9289**、L3 三元组 **21862** |

两处小账（不改结论，请在下一份答复里更正口径）：

1. question34 §7 表里的 `training_hooks.py 8ede9c09…` 是 q29 的旧值，磁盘实际是 `4046b3f8…`；answer34 已如实指出，这里同步更正。
2. **"L3 Collision" 有两个不同定义**，不要混用：训练器日志里的 `collision=0.1108309269125961` 是"4 个 token 全串唯一度"（`1 − 21862/24587`）；§63 记录里的 `l3_collision=0.1855452` 是"前三个语义码、item-weighted 重复占比"。以后报数请带名字。

## 1. 本轮唯一目标：跑完冻结协议的 Stage3

用这次 Stage2 导出的 SID 启动 Stage3：

```
stage3_start("results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/item_sids.json")
```

协议按 §22 冻结，一个字都不许改：`NUM_EPOCHS=150`、`EVAL_INTERVAL=5`、`EVAL_START_EPOCH=75`、`EARLY_EVAL_EPOCHS=(25,50,75,100,125)`、`SEED=42`、`BEAM_SIZE=20`、`NO_EVAL=True`、`SKIP_TEST=False`；四卡端口 `50201`。Stage3 启动**不做任何准入检查**（§19，owner 2026-11-09）：SID 技术有效性、ER、前缀数、碰撞、码使用率都只作为诊断记录，不阻断。

## 2. 验收（本轮标准）

一次**完整** Stage3：150 epoch 跑完、进程正常退出，并交付

1. `results/stage3_T5Train/TIGER_struct_mixed_rq_ste_fix/` 下同名目录的 `test_final.json`（`test_recall@5`、`test_recall@10`、`test_ndcg@5/@10`、`n_eval`）；
2. `HG_Rec.log` 与 `training_metrics.jsonl`（逐 epoch，含 early-eval 行）；
3. 报告：结果目录名、所用 SID 的 sha256（应为 `1839d1ce…`）、`n_eval`、`test_recall@10`；
4. **分别**给出两个判断：**(a)** 与合法基线的差值（`TIGER_RQ-VAE` 两次 Stage3 实测 `0.05553717857205035` 与 `0.04019916781281011`，按同名目录取的既有记录）；**(b)** `test_recall@10 > 0.065` 是否成立。两者都要写，不许用一个盖另一个；
5. 按 §6b 归档到 R2：`results/research/issue139/upload_stage3.py TIGER_struct_mixed_rq_ste_fix`，四件齐全（Stage2 源码打包、`item_sids.json`、`rqvae_best.pth`、完整 recall 块），并把编号与各文件 sha256 写进 `results/research/issue139/r2_ledger.json`。

本轮不判"机制成功/失败"，只交这一次真实数字；低召回也照实记录，按 §19 保留所有候选。

## 3. 禁止项

- 不改冻结超参与协议、不恢复 early stop、不换种子重跑、不做调参扫描、不做开关消融；
- 不因为 Stage2 的诊断指标好坏而挑选是否启动 Stage3；
- 不得以"预计召回低"为由跳过。

## 4. 若被拒绝

把 `stage3_start` 的原始返回（`refused` 字段与上下文）与对应日志贴出来，指出具体哪一项，并修好重试；不许只写"接口拒绝/环境不支持"。

## 5. 审阅基线

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer34.md` | `b965a965a4bd192a77096ea316719b270d81228ee33fd3c997bfd34e1ea4b40a` |
| `results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/item_sids.json` | `1839d1cea74dbc0eb5789c88b3fda1d69ff87586c561ea7b1c19ca41a5ec61d3` |
| `results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/rqvae_best.pth` | `9da6aa4baa3f12bf992b48d4f06d0d48755945d30d1b8a42335487f0749f0e06` |
| `stage2_metrics/1839d1…61d3.json` | `0bbe9a16ce1e234a5cf03ad029a1fdde3bd0d6f5145dbc4394a27528703ece22` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/training_hooks.py` | `4046b3f8c7680dc7d9b4534b9069347792a5e224ba5a641bedfd8cd38e01fc34` |

## 6. 本轮已发生的一次启动失败（必须处理，不许盲重试）

我用 `stage3_status` 查了当前状态，一次 Stage3 启动**已经失败**：

```
status: "exited_incomplete"
pid: 3813051, launched_at: 1794275299（≈12:48）
live_processes: []
result_dir: results/stage3_T5Train/TIGER_struct_mixed_rq_ste_fix（只有 logs/ 一个空目录）
run_log: logs/_stage3_run.log（0 字节）
native_log: logs/_stage3_launcher.log（不存在）
metrics / test_final / ckpt_files: 全部 null 或空
```

同时两个 GPU 探针互相矛盾：

| 探针 | physical | 来源 | mode |
|---|---|---|---|
| 启动时记录 `gpu_probe` | `["0","1","2","3"]` | `nvidia-smi` | `ddp` |
| 现在 `current_gpu_probe` | `[]` | `dev-nodes` | `none` |

要求：

1. 先解释这次 `exited_incomplete`：为什么 `_stage3_run.log` 是 0 字节、`_stage3_launcher.log` 根本没生成、pid 已不在——是启动即崩、还是 launcher 与 worker 之间没接上；
2. 解释两个探针为什么一个看到 4 张卡、一个看到 0 张（PATH 里没有 `nvidia-smi` 时会退回 `/dev/nvidia*`，而这台机器两者都没有；若 MCP 与训练进程处在不同命名空间，请给出证据）；
3. 修好再重试，并把这次失败作为 `stage3/attempt1.log` 留档；
4. 只有在 `stage3_status` 显示 `running` 且 `latest_train.epoch` 在推进之后，才算真正开始；`exited_incomplete` 不是"跑过"。
