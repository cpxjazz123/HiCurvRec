# Task #59 — flan-t5 2048d + Simple KMeans SID + TIGER 端到端

> **任务目的**: 把 Task #58 模式应用到 flan-t5 2048d 嵌入, 验证 Simple KMeans SID 在高维 (2048d) 上能否显著超越 Task #58 和 Task #87 baseline, 接近或超越 TIGER 论文 (R@5=0.0446)
> **完成日期**: 2026-07-20
> **状态**: ✅ COMPLETE — R@5 = 0.08572 (+92% vs TIGER 论文, +343% vs Task #87 baseline)

---

## 1. 背景

承接 Task #58 (+53% vs baseline) 突破, 进一步探索 Simple KMeans 在更高维度 (2048d) 上的效果. flan-t5-xl 2048d 比 S4 AE 64d 信息密度高, 预期 R@5 进一步提升.

## 2. 实验设计

**变量**: Stage 1 embedding (S4 AE 64d → flan-t5 2048d)
**保持不变**: Simple KMeans + Stage 3 TIGER + Stage 4

**启动命令**:
```bash
# Stage 1
CUDA_VISIBLE_DEVICES=1 python3 -m src.inference experiment=sem_embeds_inference_flat \
    data_dir=data/amazon_data/toys \
    embedding_model=google/flan-t5-xl \
    task_name=task59_s1

# Stage 2 (GPU 加速版)
CUDA_VISIBLE_DEVICES=0 python3 scripts/task59_simple_kmeans_2048d.py \
    --input_pt logs/task59_s1/runs/<date>/<time>/pickle/merged_predictions_tensor.pt \
    --output_dir logs/task59_s2_infer/pickle \
    --K 256 --num_hierarchies 3 --seed 42

# Stage 3
CUDA_VISIBLE_DEVICES=1 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task59_s2_infer/pickle/cluster_ids.pt \
    sequence_length=120 num_hierarchies=4 task_name=task59_s3_train
```

## 3. 决策触发 (vs TIGER 论文 0.0446)

| R@5 区间 | 判定 |
|---------|------|
| > 0.04 (≥TIGER 论文) | ✅ 显著超越 |
| 0.035 - 0.044 | ✅ 接近/达到 |
| < 0.035 | ⚠️ 低于 paper |

**判定**: ✅ R@5 = 0.0857 (+92% vs paper) — **历史性胜利**.

## 4. 完成判定

- [x] Stage 1 flan-t5 推断 (11924 × 2048, L2 norm ~1.23)
- [x] Stage 2 Simple KMeans (3 层 cov=1.0, MSE=0.000104)
- [x] Stage 3 TIGER (best step 900, val_R@5=0.06218)
- [x] Stage 4 inference (19412 × 10 × 4)
- [x] Recall/NDCG 评估 (n=19412)
- [x] verdict: `verdicts/task59_result.md`

---

result: Task #59 R@5 = 0.08572 (+92% vs TIGER 论文 0.0446, +343% vs Task #87 baseline), **迄今最强 GRID 配方 — flan-t5 2048d + Simple KMeans + TIGER**