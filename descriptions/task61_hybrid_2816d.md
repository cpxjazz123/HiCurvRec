# Task #61 — Hybrid 2816d (flan-t5 2048d + sentence-t5 768d) + Simple KMeans + TIGER

> **任务目的**: 拼接 Task #59 (flan-t5 2048d) 和 Task #60 (sentence-t5 768d) 两个 LLM embedding 为 2816d, 探索两个 LLM 的语义是否能互补, 进一步突破 R@5 (预期 R@5=0.10+)
> **执行日期**: 2026-07-20
> **状态**: 🟡 待启动

---

## 1. 背景

Task #59 (flan-t5 2048d) 和 Task #60 (sentence-t5 768d) 均达到 R@5 ≈ 0.084, 二者表达力相似. 但 flan-t5 是 encoder-decoder T5, sentence-t5 是 sentence-level 优化模型, 二者语义可能有互补性.

**假设**: 拼接 2816d 后, R@5 可能突破 0.10, 因为:
- flan-t5 提供一般语义
- sentence-t5 提供 sentence-level 语义
- 二者拼接 = 更丰富的语义空间

## 2. 实验设计

**变量**: embedding 维度 (单一 LLM → 双 LLM 拼接 2816d)
**保持不变**: Simple KMeans (K=256, 3 layer) + Stage 3 TIGER + Stage 4

**启动命令**:
```bash
# Stage 1: 拼接 2816d (复用 Task #59 + #60 embedding)
python3 scripts/task61_concat_embeddings.py \
    --input_pt_2048 logs/task59_s1/runs/2026-07-20/04-05-08/pickle/merged_predictions_tensor.pt \
    --input_pt_768 logs/task60_s1/runs/2026-07-20/05-53-06/pickle/merged_predictions_tensor.pt \
    --output_pt logs/task61_s1/merged_predictions_2816d.pt

# Stage 2: Simple KMeans (2816d 也工作)
CUDA_VISIBLE_DEVICES=0 python3 scripts/task59_simple_kmeans_2048d.py \
    --input_pt logs/task61_s1/merged_predictions_2816d.pt \
    --output_dir logs/task61_s2_infer/pickle \
    --K 256 --num_hierarchies 3 --seed 42

# Stage 3: TIGER
CUDA_VISIBLE_DEVICES=1 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task61_s2_infer/pickle/cluster_ids.pt \
    sequence_length=120 num_hierarchies=4 task_name=task61_s3_train

# Stage 4: 推断 + 评估
CUDA_VISIBLE_DEVICES=1 python3 -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task61_s2_infer/pickle/cluster_ids.pt \
    ckpt_path=/home/wlia0047/ar57/wenyu/GeneRec/logs/task61_best.ckpt \
    sequence_length=120 num_hierarchies=4 task_name=task61_s4_infer

python3 scripts/task58_recall_eval.py \
    --constrained_pt logs/task61_s4_infer/runs/<date>/<time>/pickle/merged_predictions_tensor.pt \
    --sid logs/task61_s2_infer/pickle/cluster_ids.pt \
    --ckpt /home/wlia0047/ar57/wenyu/GeneRec/logs/task61_best.ckpt \
    --out_json verdicts/task61_recall_eval.json
```

## 3. 决策触发

| R@5 区间 | 判定 |
|---------|------|
| > 0.10 | ✅ 拼接显著提升 |
| 0.0857 - 0.10 | ✅ 与单 LLM 持平或略升 |
| 0.08 - 0.0857 | ⚠️ 拼接无收益, 单 LLM 已最优 |
| < 0.08 | ❌ 拼接引入噪声 |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 拼接 | ~5s |
| Stage 2 Simple KMeans | ~5s |
| Stage 3 TIGER | ~120 min (2816d 比 768d/2048d 都大, 训练稍慢) |
| Stage 4 inference | ~2 min |
| 评估 | ~1 min |
| **总计** | **~125 min** |

## 5. 风险与缓解

**风险 1**: 2816d 可能引入噪声 → 监控 val_R@5 曲线, 如果比 Task #60 还差, 提前停训
**风险 2**: flan-t5 和 sentence-t5 维度语义空间差异大 → 均值中心化后再拼接
**风险 3**: Simple KMeans GPU OOM on 2816d → 用 chunked 处理 (脚本已支持)

## 6. 完成度跟踪

- [ ] Stage 1 拼接 2816d
- [ ] Stage 2 Simple KMeans (cov=1.0)
- [ ] Stage 3 TIGER (best ckpt)
- [ ] Stage 4 inference + 评估
- [ ] 写 verdict