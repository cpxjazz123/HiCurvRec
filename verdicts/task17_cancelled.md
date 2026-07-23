# Task #72 — CANCELLED（用户主动取消）

> **取消日期**: 2026-07-17
> **取消原因**: 用户明确取消，方向切换为 Task #73 后续（KGAT TF-GPU 化 + 完整训练）
> **状态**: ⛔ **已取消**（Stage 2.1/2.2 产物保留；Stage 3 训练未 kill，PID 923347 后台仍在跑）

---

## 取消时已完成的阶段

| 阶段 | 状态 | 产物 |
|------|------|------|
| Stage 2.1 RQ-VAE 训练 (L=3, W=512, 15000 步) | ✅ 完成 | `GRID/logs/train/runs/2026-07-17/20-00-54/checkpoints/checkpoint_000_015000.ckpt` |
| Stage 2.2 RQ-VAE SID 推断 | ✅ 完成 | `GRID/logs/inference/runs/2026-07-17/22-29-42/pickle/merged_predictions_tensor.pt` (11924, 3) |
| Bridge → (4, 11924) int64 dedup | ✅ 完成 | `products/task72/inference/l3w512_dedup_sid.pt` |
| Stage 3 TIGER 训练 (50000 步) | ⏸️ 用户取消时在跑 | `…/22-32-51/checkpoints/` 已有 step=125/250/375 三个 ckpt（val@3 best=0.01906）|
| Stage 4 TIGER 推断 | ❌ 未启动 | — |
| R@10 评估 + Task #72 verdict | ❌ 未完成 | — |

---

## Stage 3 val 历史（取消时记录）

| val@ | global step | val/recall@5 | val/recall@10 | val/ndcg@10 | val/loss |
|---|---|---|---|---|---|
| 1 | 125 | 0.00314 | 0.00592 | 0.00282 | 17.30 |
| 2 | 250 | 0.01437 | 0.02318 | 0.01190 | 14.50 |
| 3 | 375 | **0.01906** | 0.03127 | 0.01597 | 13.00 |

训练在 step 8000/50000 (~16%) 时被取消。best ckpt = `step=000375.ckpt` (val/recall@5=0.01906)。

---

## 进程状态

- **PID 923347** Stage 3 训练**仍在后台跑**（未 kill），用户取消时已训 8000 步
- 保留原因：避免浪费已训算力；如果用户后续想 resume 可直接继续
- 如需立即释放 GPU 0：`kill 923347`
- GPU 0 当前占用 32 GB / 97% util

---

## 后续

- 用户明确取消 Task #72，方向切换为 KGAT TF-GPU 化 + 完整训练（Task #73 后续）
- 等用户登记新活跃任务到 §16

---

**cancelled at**: 2026-07-17，用户明确指令"取消 task72"
**result**: Task #72 已取消。Stage 2.1/2.2/bridge 产物保留可用；Stage 3 进程 PID 923347 后台仍在跑（如需释放 GPU 请手动 kill）。归档到 `verdicts/task72_cancelled.md`，§16 表格中 Task #72 行待移除（依赖 §16 写入权限，建议下一轮 loop tick 处理）。