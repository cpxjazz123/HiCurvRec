# Task #53 — 真实 TIGER 训练 (S4 AE + log1p)

> **任务目的**: 用 S4 AE embedding + L1 log1p 后处理, 跑完整 Stage 2 RQ-VAE + Stage 3 TIGER, 验证 Campaign 推荐的端到端 Recall 是否优于 Task #87 baseline (R@5=0.01937)
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #52 终判推荐路线: "终止 PM-RQ + 用 log1p + QMP 矩阵选 embedding". Campaign Task #51 已在 Stage 2 proxy 上证明 log1p 修复有效 (RQ recon 12.72 → 0.05), 但端到端 TIGER Recall 尚未验证. Task #53 是 Campaign 推荐路线的端到端验证.

## 2. 实验设计

**变量**: Stage 1 embedding (S4 AE) + L1 log1p 后处理
**保持不变**:
- Stage 2 RQ-VAE: snap-research `[768,256,128]→64` + Adagrad lr=1e-3 + bf16 + batch=2048 + init_buf=3072 (Task #87 v6 配置, 防止坍缩)
- Stage 3 TIGER: 4-layer T5 + Adafactor + inverse_sqrt + 2000 user bins (Task #87 对齐配置)
- seed=42 (跨 run 固定)
- num_hierarchies=3 (Stage 2) → 4 (Stage 3 推断后)

**启动命令**:
```bash
# Step 0: log1p 后处理 (Task #52 推荐路线: e' = e/||e|| · log(1+||e||))
python3 scripts/task53_log1p_preprocess.py \
    --input products/task48_s4_ae/entity_embedding.pt \
    --output products/task53_log1p_s4_ae/entity_embedding.pt

# Stage 2.1 RQ-VAE 训练 (S4 AE + log1p fused)
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 \
python3 -m src.train experiment=rqvae_train_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=products/task53_log1p_s4_ae/entity_embedding.pt \
    embedding_dim=64 num_hierarchies=3 codebook_width=256 \
    task_name=task53_s2_train

# Stage 2.2 RQ-VAE 推断
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 \
python3 -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=products/task53_log1p_s4_ae/entity_embedding.pt \
    embedding_dim=64 num_hierarchies=3 codebook_width=256 \
    task_name=task53_s2_infer

# Stage 3 TIGER 训练
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 \
python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task53_s2_infer/runs/task53_s2_infer/pickle/cluster_ids.pt \
    sequence_length=120 num_hierarchies=4 seed=42 \
    task_name=task53_s3_train

# Stage 4 TIGER 推断
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 \
python3 -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task53_s2_infer/runs/task53_s2_infer/pickle/cluster_ids.pt \
    sequence_length=120 num_hierarchies=4 \
    task_name=task53_s4_infer
```

## 3. 决策触发 (vs Task #87 baseline R@5=0.01937)

| R@5 区间 | 决策 |
|---------|------|
| > 0.025 (≥+29%) | ✅ Campaign 推荐路线全面胜出, 进入 paper §5 写推荐方案 |
| 0.0194 - 0.025 | ✅ 与 baseline 持平或小幅胜, 验证 log1p 在端到端有效但幅度有限 |
| 0.014 - 0.0194 | ⚠️ 持平 baseline (Task #87 R@5=0.01937 是上限), 需查 SID 质量 |
| < 0.014 | ❌ 低于 baseline, 检查 Stage 2 RQ-VAE 是否坍缩 (Task #87 v6 配置保障) |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 2.1 RQ-VAE 训练 | ~3.5 h |
| Stage 2.2 RQ-VAE 推断 | ~5 min |
| Stage 3 TIGER 训练 | ~3 h (含 inverse_sqrt scheduler, 比 flat baseline 快) |
| Stage 4 TIGER 推断 | ~5 min |
| Recall/NDCG 评估 | ~5 min |
| **总计** | **~6-8 h** |

## 5. 风险与缓解

**风险 1**: Stage 2 RQ-VAE 码本坍缩 → 缓解: 严格使用 Task #87 v6 配置 (Adagrad + bf16 + batch=2048 + init_buf=3072)
**风险 2**: S4 AE 64d embedding 维度太低 (vs T5 768d / MCKG 64d 同样) → 缓解: Task #51 已证 64d 足够
**风险 3**: Stage 3 100k steps 时间超预算 → 缓解: 早停 patience=10 (Task #87 配置)
**风险 4**: 4 GPU 抢占 → 缓解: 单卡独占 (cuda:0), 留 1/2/3 给 Task #54-#164 并行

## 6. 完成度跟踪

- [ ] Stage 2.1 RQ-VAE launch
- [ ] Stage 2.1 finish (loss<0.005 + L0/L1/L2 coverage>0.7)
- [ ] Stage 2.2 RQ-VAE inference (cluster_ids.pt shape (4, 11924))
- [ ] Stage 3 TIGER training launch
- [ ] Stage 3 finish (val/recall@5 plateau 或 patience 触发)
- [ ] Stage 4 TIGER inference (merged_predictions_tensor.pt)
- [ ] Recall@5/NDCG@5 评估
- [ ] 写 verdict + 更新 Task #52 推荐路线
