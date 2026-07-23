# Task #60 — sentence-t5 768d + Simple KMeans SID + TIGER 端到端 (复现 TIGER 论文)

> **任务目的**: 用 TIGER 论文实际使用的 sentence-t5-base 768d embedding + Simple KMeans residual quantization + TIGER, 验证 Simple KMeans 配方在论文原 embedding 上的效果, 复现 TIGER 论文 Toys 配置
> **执行日期**: 2026-07-20
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #59 (+92% vs TIGER 论文) 突破. Task #59 用的是 flan-t5-xl 2048d (GRID 默认), 但 TIGER 论文 (Rajput et al. 2023) 实际使用 sentence-t5-base 768d. Task #60 用 sentence-t5-base 复现论文配置, 验证 Simple KMeans 配方在论文原 embedding 上的表现.

**预期**:
- sentence-t5-base 768d 信息密度可能比 flan-t5 2048d 低 (768d vs 2048d)
- 但 Simple KMeans 应该仍然有效 (GPU KMeans 在任意维度上 cov=1.0)
- 预测 R@5 = 0.05 ~ 0.07 (论文 0.0446 ~ Task #59 0.0857)

## 2. 实验设计

**变量**: Stage 1 embedding (flan-t5 2048d → sentence-t5 768d)
**保持不变**: Simple KMeans + Stage 3 TIGER + Stage 4 + 评估脚本

**启动命令**:
```bash
# Stage 1: sentence-t5-base 768d 推断
CUDA_VISIBLE_DEVICES=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python3 -m src.inference experiment=sem_embeds_inference_flat \
    data_dir=data/amazon_data/toys \
    embedding_model=sentence-transformers/sentence-t5-base \
    task_name=task60_s1

# Stage 2: Simple KMeans (GPU 加速, 768d 也工作)
CUDA_VISIBLE_DEVICES=0 python3 scripts/task59_simple_kmeans_2048d.py \
    --input_pt logs/task60_s1/runs/<date>/<time>/pickle/merged_predictions_tensor.pt \
    --output_dir logs/task60_s2_infer/pickle \
    --K 256 --num_hierarchies 3 --seed 42

# Stage 3: TIGER
CUDA_VISIBLE_DEVICES=1 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task60_s2_infer/pickle/cluster_ids.pt \
    sequence_length=120 num_hierarchies=4 task_name=task60_s3_train

# Stage 4: 推断 + 评估
CUDA_VISIBLE_DEVICES=1 python3 -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task60_s2_infer/pickle/cluster_ids.pt \
    ckpt_path=/home/wlia0047/ar57/wenyu/GeneRec/logs/task60_best.ckpt \
    sequence_length=120 num_hierarchies=4 task_name=task60_s4_infer

python3 scripts/task58_recall_eval.py \
    --constrained_pt logs/task60_s4_infer/runs/<date>/<time>/pickle/merged_predictions_tensor.pt \
    --sid logs/task60_s2_infer/pickle/cluster_ids.pt \
    --ckpt /home/wlia0047/ar57/wenyu/GeneRec/logs/task60_best.ckpt \
    --out_json verdicts/task60_recall_eval.json
```

## 3. 决策触发 (vs TIGER 论文 0.0446)

| R@5 区间 | 判定 |
|---------|------|
| > 0.0446 (TIGER 论文) | ✅ 复现成功 |
| 0.035 - 0.0446 | ⚠️ 略低, 需要分析 |
| < 0.035 | ❌ 失败, 调查根因 |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 sentence-t5 推断 | ~15 min (768d, 比 2048d 快) |
| Stage 2 Simple KMeans | ~5s |
| Stage 3 TIGER | ~60-80 min (768d 介于 64d 和 2048d 之间) |
| Stage 4 inference | ~2 min |
| Recall/NDCG 评估 | ~2 min |
| **总计** | **~80-100 min** |

## 5. 风险与缓解

**风险 1**: sentence-t5-base 首次下载可能慢 → 用 `huggingface-cli download` 预热
**风险 2**: 768d 与 Task #59 的 2048d 收敛行为不同 → 监控 val_R@5 曲线
**风险 3**: Sentence-T5 输出可能未 L2 norm → 检查 norm 后再 mean-center

## 6. 完成度跟踪

- [ ] Stage 1 sentence-t5 推断 launch + finish
- [ ] Stage 2 Simple KMeans launch + finish (cov=1.0 全 3 层)
- [ ] Stage 3 TIGER launch + finish (best ckpt)
- [ ] Stage 4 inference (19412 × 10 × 4)
- [ ] Recall/NDCG 评估完成
- [ ] 写 verdict + 更新 loop.md §16