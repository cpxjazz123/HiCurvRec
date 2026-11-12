# #198-A 配方一致性（对照 `TIGER_mixed_rq`）

来源：两棵树的 `train_rqvae.py` 常量 + `TIGER_sign_rq/logs/training_metrics.jsonl` 的 `train_start` 记录（`audit_a.json`）。

## 常量逐项（两树**完全相同**，`differences` 中常量差集为空）

| 常量 | 值 | 常量 | 值 |
|---|---|---|---|
| `EPOCHS` | 3000 | `CODEBOOK_NUM` | 3 |
| `BATCH_SIZE_PER_RANK` | **1024** | `CODEBOOK_SIZE` | (256, 256, 256) |
| `LR` | 1e-3 | `CODEBOOK_DIM` | 32 |
| `WEIGHT_DECAY` | 1e-4 | `BETA` | 0.25 |
| `WARMUP_EPOCHS` | 50 | `VQ_TYPE` | "vq" |
| `SEED` | 42 | `SK_EPSILON` / `SK_ITERS` | 0.003 / 50 |
| `OPTIMIZER` | "AdamW" | `PCA_DIM` | 0 |
| `GRADIENT_CLIP_NORM` | 1.0 | `EVAL_INTERVAL` | 50 |
| `NUM_WORKERS` | 0 | | |

## SIGN 实际日志（`train_start`）

`epochs=3000`、`batch_size_per_rank=1024`、**`total_batch_size=4096`**、`world_size=4`、`optimizer=AdamW`、`lr=0.001`、`weight_decay=0.0001`、`warmup_epochs=50`、`seed=42`、`codebook_num=3`、`codebook_size=[256,256,256]`、`codebook_dim=32`、`beta=0.25`、`vq_type="vq"`、`sk_epsilon=0.003`、`sk_iters=50`、`pca_dim=0`、`gradient_clip_norm=1.0`、`num_workers=0`、`eval_interval=50`。

- **全局 batch = 4096**（`world_size 4 × 1024`），**不是 #181 的 16,384**。
- `steps_per_epoch = ceil(train_items / global_batch) = ceil(24556 / 4096) = 6`，与"每 rank 6139 件 ÷ 1024 = 6 个 batch"一致。

## 基线侧的可比性限制（如实标注）

`TIGER_mixed_rq/logs/` 目录**不存在**（该臂训练时未落 `training_metrics.jsonl`），故基线的 **logged** 配方为 `NOT_AVAILABLE`。可比的是**源码常量**，而它们逐项相同。这不影响本项判定：与 SIGN 同源的 `TIGER_RQ-VAE` 一 epoch 对照实测 `lr = 2.0833333333333333e-05`（warmup 50 的 1/50），与其 1024/rank、4096 global 一致。

## 判定

**（i）配方 / batch 不一致 ⇒ `EXCLUDED`。**
