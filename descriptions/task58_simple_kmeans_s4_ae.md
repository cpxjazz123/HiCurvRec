# Task #58 — Simple KMeans SID + S4 AE 64d + TIGER 端到端

> **任务目的**: 用 Simple KMeans 替代 RQ-VAE 的神经编码部分, 验证 S4 AE 64d + log1p + Simple KMeans SID 能否恢复 Task #87 baseline R@5 (0.01937)
> **完成日期**: 2026-07-20
> **状态**: ✅ COMPLETE — R@5 = 0.02962 (+53% vs baseline)

---

## 1. 背景

承接 Task #53 失败结论 (Neural RQ-VAE 在 S4 AE 64d 上失败, R@5=0.002). 探索 Simple KMeans 跳过 encoder/decoder 是否能恢复 baseline.

## 2. 实验设计

**变量**: Stage 2 算法 (Neural RQ-VAE → Simple KMeans residual quantization)
**保持不变**: S4 AE 64d + log1p + Stage 3 TIGER + Stage 4 推断

**启动命令**:
```bash
# Stage 2
python3 scripts/task58_simple_kmeans_sid.py \
    --input_pt products/task53_log1p_s4_ae/entity_embedding.pt \
    --output_dir logs/task58_s2_infer/pickle \
    --K 256 --num_hierarchies 3 --seed 42

# Stage 3
CUDA_VISIBLE_DEVICES=1 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task58_s2_infer/pickle/cluster_ids.pt \
    sequence_length=120 num_hierarchies=4 task_name=task58_s3_train
```

## 3. 决策触发 (vs Task #87 baseline 0.01937)

| R@5 区间 | 判定 |
|---------|------|
| > 0.025 (≥+29%) | ✅ 胜出 |
| 0.0194 - 0.025 | ✅ 持平或小幅胜 |
| < 0.0194 | ⚠️/❌ 持平或低于 baseline |

**判定**: ✅ R@5 = 0.02962 (+53%) — **胜出**.

## 4. 完成判定

- [x] Stage 2 Simple KMeans (cov 1.0 全 3 层)
- [x] Stage 3 TIGER (best step 100, val_R@5=0.02746)
- [x] Stage 4 inference (19412 users × 10 × 4)
- [x] Recall/NDCG 评估完成
- [x] verdict: `verdicts/task58_result.md`

---

result: Task #58 R@5 = 0.02962 (+53% vs baseline 0.01937), **历史性胜利 — 验证 Simple KMeans 替代 Neural RQ-VAE 的端到端可行性**