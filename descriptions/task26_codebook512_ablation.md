# Task #26 — Toys Stage 2 RQ-VAE codebook_width=512 ablation (Stage 3 重训)

> **任务目的**: 验证 RQ-VAE codebook_width 从 256 (Task #87) 扩大到 512 是否能在 Toys flat baseline 上带来 Recall 增益. 单变量 ablation, 锁定 Stage 2 codebook 容量对 Stage 3 Recall 的影响.

> **完成日期**: ❌ 2026-07-19 取消 (用户反馈逻辑有问题: K ablation 不直接回答 Task #27 的 kNN preservation 假设)
> **状态**: ❌ CANCELLED (2026-07-19 14:05 自主决策启动)

---

## 1. 背景

- Task #87 (Toys flat baseline, codebook_width=256, seed=42) 实测 R@5=**0.01937**
- Task #25 (同 baseline, seed=123) 实测 R@5=**0.01489** → CV=18.5%
- Task #87 + Task #25 锁定 Toys baseline 区间 [0.01489, 0.01937], 均值 0.0171
- Toys 11924 items × 4 digits × K=256 → 256^4 = 4.3×10^9 理论 code 空间, 利用率 < 1% (11924/4.3×10^9)
- 核心疑问: codebook_width=256 的容量是否够用? 扩大到 512 (理论 code 空间 6.9×10^10) 是否有增益?

### 1.1 假设
- **H1**: codebook_width=512 → 减少码字冲突 → SID 质量提升 → R@5 提升 ≥ 25% (超 CV)
- **H0 (null)**: codebook_width=512 vs 256 在 Toys 11924 items 上无显著差异 (因 items 远小于 K^4)

### 1.2 单变量隔离
- **唯一变化**: Stage 2.1 RQ-VAE `codebook_width: 256 → 512`
- **保持不变**:
  - Stage 1: sentence-t5-base (Task #87 已完成)
  - Stage 2.1 其他超参: encoder [512,256,128]→32, decoder [128,256,512], Adagrad lr=0.4, max_steps=20000, batch_size=1024, fp32, init_buffer_size=1024, BetaQuantizationLoss(beta=0.25)
  - Stage 2.2: num_hierarchies=3 (推断后 +1 dedup → 4 digits)
  - Stage 3: Adafactor lr=0.01, InverseSqrtScheduler warmup=10000, num_user_bins=2000, LSH Hashing Trick, dropout=0.1, max_steps=100000, val_check_interval=2000, sequence_length=120, seed=42
  - Stage 4: sequence_length=120, top_k=10
- **R5 硬约束符合性**: ✓ 算法 RQ-VAE (R5 允许), ✓ num_hierarchies=3→4 (R5), ✓ seed=42 (R5), ✓ Toys 单数据集 (R5)

---

## 2. 实验设计

### 2.1 Stage 2.1 RQ-VAE 训练 (K=512)

**启动命令**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

CUDA_VISIBLE_DEVICES=0 nohup python -m src.train experiment=rqvae_train_tiger \
    data_dir=data/amazon_data/toys \
    embedding_path=logs/task87_s1/runs/2026-07-19/03-46-29/pickle/merged_predictions_tensor.pt \
    embedding_dim=768 \
    num_hierarchies=3 \
    codebook_width=512 \
    task_name=task26_s2_train_k512 \
    > logs/task26_s2_train_k512.log 2>&1 &
```

**注**: Stage 1 embedding 用 Task #87 已生成的 sentence-t5-base embedding (logs/task87_s1/...).

### 2.2 Stage 2.2 RQ-VAE 推断 (K=512)
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=rqvae_inference_tiger \
    data_dir=data/amazon_data/toys \
    embedding_path=logs/task87_s1/runs/2026-07-19/03-46-29/pickle/merged_predictions_tensor.pt \
    embedding_dim=768 num_hierarchies=3 codebook_width=512 \
    task_name=task26_s2_infer_k512
```

### 2.3 Stage 3 TIGER 训练 (seed=42)
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train experiment=tiger_train_tiger \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task26_s2_infer_k512/runs/<id>/pickle/cluster_ids.pt \
    sequence_length=120 \
    num_hierarchies=4 \
    seed=42 \
    task_name=task26_baseline_k512 \
    > logs/task26_s3_train_k512.log 2>&1 &
```

### 2.4 Stage 4 推断 + 评估
复用 `scripts/task23_105_pmrq_eval.py --task_id 108 --ckpt products/task26_baseline_k512/stage3_train/best.ckpt`.

---

## 3. 决策触发

| 指标条件 | R@5 | 决策 |
|----------|-----|------|
| **R@5 > 0.024** (>Task #87 +25%, 显著超 CV) | 实测 | ✅ K=512 显著优势, 锁定更大 codebook 方向 |
| **R@5 ∈ [0.019, 0.024]** (=Task #87 区间) | 实测 | ⚠️ K=512 与 K=256 无显著差异, 选 K=256 (省显存) |
| **R@5 ∈ [0.014, 0.019]** (=Task #25 区间下限) | 实测 | ⚠️ K=512 略弱于 K=256, 锁定 K=256 为最优 |
| **R@5 < 0.014** (<Task #25 区间下限) | 实测 | ❌ K=512 严重退化, 建议改 K=128 (下一轮) |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 2.1 RQ-VAE 训练 | ~50 min (cuda:0, 1024 batch × 20k steps, K=512 双倍码字显存占用略增) |
| Stage 2.2 RQ-VAE 推断 | ~5 min (cuda:0) |
| Stage 3 TIGER 训练 | ~6h (cuda:0, 100k steps, K=512 输入端无差异) |
| Stage 4 推断 | ~1 min |
| Recall/NDCG 评估 | ~5 min (CPU) |
| **总计** | **~7h** (含早停可能节省) |

---

## 5. 风险与缓解

**风险 1**: K=512 codebook collapse (利用率 < 50%) → Stage 2.1 训练监测 epoch cov
- 缓解: 看 `train/codebook/cov` 指标, 若 < 0.7 早停

**风险 2**: K=512 显存溢出 → Stage 2.1 batch_size=1024, K=256 时用 ~6GB, K=512 约 8GB (在 46GB 限制内)
- 缓解: 若 OOM, 退回 batch_size=512

**风险 3**: Stage 3 训练与 Task #87 相同超参, K=512 SID 端不影响 → Stage 3 内存占用与 Task #87 一致
- 缓解: 无需特殊处理

**风险 4**: 与其他实验抢 cuda:0 (若有)
- 缓解: 启动前 `nvidia-smi` 核对 (R7)

**风险 5**: 推断 ckpt_path 路径含 `=` (Task #87 教训)
- 缓解: symlink 到 `products/task26_baseline_k512/stage3_train/best.ckpt`

---

## 6. 完成度跟踪

- [x] Task #26 description 写盘
- [ ] Stage 2.1 启动 (cuda:0, K=512)
- [ ] Stage 2.1 codebook cov ≥ 0.7 (健康)
- [ ] Stage 2.2 推断 (5 min)
- [ ] Stage 3 启动 (cuda:0, seed=42)
- [ ] Stage 3 best ckpt 保存 (ModelCheckpoint monitor=val/recall@5)
- [ ] Stage 4 推断 + Recall/NDCG 评估
- [ ] 写 `verdicts/task26_k512_ablation_result.md`
- [ ] 与 Task #87 K=256 + Task #25 seed=123 对比 → 三方表

---

## 7. 关联

- 基准: Task #87 (`verdicts/task87_tiger_baseline_result.md`, K=256, seed=42, R@5=0.01937)
- 关联: Task #25 (`verdicts/task25_baseline_seed123_result.md`, K=256, seed=123, R@5=0.01489)
- 假设: H1 = K=512 > K=256 增益; H0 = 无显著差异
- 上游 Stage 1: Task #87 sentence-t5-base embedding (logs/task87_s1/runs/2026-07-19/03-46-29/pickle/merged_predictions_tensor.pt)
