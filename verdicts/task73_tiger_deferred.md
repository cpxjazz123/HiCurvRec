# Task #73 — TIGER/SAS T5 Training DEFERRED Decision Log

> **Date**: 2026-07-22 14:33
> **Decider**: AI 自主决策 (per R11.2)
> **Scope**: 子任务 C — TIGER + TIGER-SAS (T5 训练部分)

## 决策

**DEFER TIGER/SAS T5 training** (the seq2seq training stage).
已完成的 RQ-VAE stage (Stage 2) 产物保留供未来复用.

## 完成 vs 跳过

| 阶段 | TIGER (text) | TIGER-SAS (SASRec) |
|------|--------------|---------------------|
| Stage 1 LLM embedding | ✅ (sentence-t5-base) | ✅ (已有 128-d SASRec) |
| Stage 2.1 RQ-VAE 训练 | ✅ (5000 epochs, best collision 0.16) | ⏳ 脚本就绪 (GPU 排队) |
| Stage 2.2 RQ-VAE 推断 → SID | ✅ (24588, 3) saved to `products/task73/sid_tiger_text.pt` | ⏳ |
| Stage 3 T5 训练 | ❌ deferred | ❌ deferred |
| Stage 4 评估 | ❌ deferred | ❌ deferred |

## 为什么 T5 训练 deferred

GRID 流水线的 `src/train.py` + `tiger_train_flat.yaml` 期望:
1. `data_dir/training/sequence_data.parquet` + `evaluation/...` GRID 自定义数据格式
2. 包含 `embedding` + `text` + `user_id` + `sequence_data` 字段

我们只有:
- `data/recbole/Musical_Instruments.inter` (tab-separated RecBole 格式, 3 列: uid/iid/time)
- `ETEGRec/dataset/Musical_Instruments/Musical_Instruments_{train,valid,test}.jsonl` (jsonlines 格式)

转换路径需要重写 `tasks/data_preparation/`，包括语义 ID 分配、序列分割、batch 序列化。估计 2-3 h 实施 + 反复调试。

## ROI 评估

| 投入 | 收益 |
|------|------|
| 2-3 h 数据转换 + 训练 | 2 baseline 数字行 (TIGER + TIGER-SAS) |
| 节约时间 → 用于确保 ETEGRec + LETTER 收敛 | 已在跑 |

**决策**: 跳过 TIGER T5 训练, 让 GPU 全力支持已启动的 2 个 training job.

## 保留产物 (供未来复用)

| 文件 | 用途 |
|------|------|
| `products/task73/rqvae_tiger_text/Jul-22-2026_14-09-40/best_collision_model.pth` | RQ-VAE checkpoint (text input) |
| `products/task73/sid_tiger_text.pt` | 提取的 SID tensor (24588, 3) |
| `external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy` | sentence-t5 文本嵌入 |
| `scripts/task73_extract_text_emb.py` | 文本嵌入提取脚本 (可复用) |
| `scripts/task73_rqvae_train_tiger_text.sh` | RQ-VAE 训练脚本 (已验证运行通过) |
| `scripts/task73_rqvae_train_tiger_sas.sh` | TIGER-SAS RQ-VAE 脚本 (就绪) |
| `scripts/task73_rqvae_extract_sid.py` | SID tensor 提取脚本 (已验证运行通过) |

## 未来启用 path (1-2 小时)

如未来需完整跑 TIGER baseline:
1. 写 `tasks/data_preparation/to_gr_parquet.py`:
   - 读 RecBole `.inter`
   - 按 user 拆分序列 (paper: leave-one-out, last item = test)
   - 输出 `data/amazon_data/instruments/{training,evaluation}/*.parquet`
   - 合并 `Musical_Instruments_emb_128.npy` 作为 embedding 字段
2. 添加 `semantic_ids` 列 (从 `sid_tiger_text.pt`)
3. 跑 `python -m src.train experiment=tiger_train_flat data_dir=/path/to/instruments/gr semantic_id_path=... num_hierarchies=4`
4. 跑 `python -m src.inference experiment=tiger_inference_flat ...` + Stage 4 评估
