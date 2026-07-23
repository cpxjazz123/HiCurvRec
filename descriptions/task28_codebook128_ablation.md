# Task #28 — Toys Stage 2 RQ-VAE codebook_width=128 ablation

> **任务目的**: 验证 RQ-VAE codebook_width 缩小到 128 (Task #87 baseline K=256 的一半) 是否导致 codebook collapse 或 Recall 退化. 增量数据点加入 Task #27 邻域质量相关性分析 (Plan B), 用于验证 ρ=-0.67 是否在更大 n 下仍显著.

> **完成日期**: ❌ 2026-07-19 取消 (用户反馈逻辑有问题: K ablation 不直接回答 Task #27 的 kNN preservation 假设)
> **状态**: ❌ CANCELLED (2026-07-19 自主决策启动, Task #27 Plan B 增量)

---

## 1. 背景

### 1.1 上游结论

- **Task #27** (✅ 完成 14:50): 7 tokenizer 对比, Spearman ρ = **-0.667** (Normalized Hamming vs R@5, n=6, p=0.148). 形式上达用户 ρ>0.6 阈值, 但 n=6 检验力不足, 非统计显著. **LOO 分析**: 去 Task #25 后 ρ=-0.90 → 真实相关可能更强, 但被 seed variance (CV=18.5%) 主导
- **Task #87 / Task #25** baseline 区间 [0.01489, 0.01937], CV=18.5%
- **Task #26** (在跑 cuda:0) K=512 ablation: 验证扩大 codebook 是否带来 Recall 增益

### 1.2 假设

- **H1**: K=128 vs K=256 → 显著 Recall 退化 (≥20%, 因 11924 items × 4 digits 需要 ≥ 2^log2(K^4) = 2^28 = 256M bins 才安全, K=128 仅 4^28/256^4 = 1/16 容量)
- **H0 (null)**: K=128 在 Toys 11924 items 上与 K=256 无显著差异 (因 items 远小于 K^4)
- **H-collpase**: K=128 → codebook 大量未使用 (利用率 < 50%) → 训练失败

### 1.3 单变量隔离

- **唯一变化**: Stage 2.1 RQ-VAE `codebook_width: 256 → 128`
- **保持不变**:
  - Stage 1: sentence-t5-base (Task #26 已生成, 复用同一 embedding)
  - Stage 2.1 其他超参: encoder [512,256,128]→32, decoder [128,256,512], Adagrad lr=0.4, max_steps=20000, batch_size=1024, fp32, init_buffer_size=1024, BetaQuantizationLoss(beta=0.25)
  - Stage 2.2: num_hierarchies=3 (推断后 +1 dedup → 4 digits)
  - Stage 3: Adafactor lr=0.01, InverseSqrtScheduler warmup=10000, num_user_bins=2000, LSH Hashing Trick, dropout=0.1, max_steps=100000, val_check_interval=2000, sequence_length=120, seed=42
  - Stage 4: sequence_length=120, top_k=10
- **R5 硬约束符合性**: ✓ 算法 RQ-VAE (R5 允许), ✓ num_hierarchies=3→4 (R5), ✓ seed=42 (R5), ✓ Toys 单数据集 (R5)

---

## 2. 实验设计

### 2.1 Stage 2.1 RQ-VAE 训练 (K=128)

**启动命令** (cuda:1):
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

CUDA_VISIBLE_DEVICES=1 nohup python -m src.train experiment=rqvae_train_tiger \
    data_dir=data/amazon_data/toys \
    embedding_path=logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt \
    embedding_dim=768 \
    num_hierarchies=3 \
    codebook_width=128 \
    task_name=task28_s2_train_k128 \
    > logs/task28_s2_train_k128.log 2>&1 &
```

### 2.2 Stage 2.2 RQ-VAE 推断 (K=128)
```bash
CUDA_VISIBLE_DEVICES=1 python -m src.inference experiment=rqvae_inference_tiger \
    data_dir=data/amazon_data/toys \
    embedding_path=logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt \
    embedding_dim=768 num_hierarchies=3 codebook_width=128 \
    task_name=task28_s2_infer_k128
```

### 2.3 Stage 3 TIGER 训练 (seed=42)
```bash
CUDA_VISIBLE_DEVICES=1 python -m src.train experiment=tiger_train_tiger \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task28_s2_infer_k128/runs/<id>/pickle/cluster_ids.pt \
    sequence_length=120 \
    num_hierarchies=4 \
    seed=42 \
    task_name=task28_baseline_k128 \
    > logs/task28_s3_train_k128.log 2>&1 &
```

### 2.4 Stage 4 推断 + 评估
复用 `scripts/task23_105_pmrq_eval.py --task_id 110 --ckpt products/task28_baseline_k128/stage3_train/best.ckpt`.

---

## 3. 决策触发 (vs Task #87 baseline K=256)

| 指标条件 | 预期 R@5 | 决策 |
|----------|---------|------|
| **R@5 > 0.024** (>Task #87 +25%, 显著超 CV) | 实测 | ✅ K=128 显著优势 (反直觉) |
| **R@5 ∈ [0.019, 0.024]** (=Task #87 区间) | 实测 | ⚠️ K=128 与 K=256 无显著差异, 选 K=128 (省显存) |
| **R@5 ∈ [0.014, 0.019]** (=Task #25 区间下限) | 实测 | ⚠️ K=128 略弱, 锁定 K=256 为最优 |
| **R@5 < 0.014** (<Task #25 区间下限) | 实测 | ❌ K=128 严重退化, 选 K=256 (容量不足) |
| **codebook cov < 0.5 (任一层)** | n/a | ❌ K=128 collapse, 早停 |

### 3.1 Task #27 增量影响

- 增加 1 个数据点 (n: 6→7)
- 与 Task #26 K=512 形成两侧 ablation (K=128 / K=256 / K=512)
- 重新计算 Spearman ρ, 若 ρ ≥ 0.6 仍显著 (p<0.05) → Plan B 达成, 进入论文 Discussion

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 2.1 RQ-VAE 训练 (cuda:1) | ~40 min (K=128 比 K=256 略快) |
| Stage 2.2 RQ-VAE 推断 | ~5 min (cuda:1) |
| Stage 3 TIGER 训练 (cuda:1) | ~6h |
| Stage 4 推断 + 评估 | ~15 min |
| **总计** | **~7h** (与 Task #26 cuda:0 完全并行) |

---

## 5. 风险与缓解

**风险 1**: K=128 codebook collapse (利用率 < 50%)
- 缓解: 看 train/layer_X/frac_layer_coverages_step 指标, 若 < 0.5 早停
- Task #26 实测 Layer 0 cov=0.625, Task #28 K=128 可能更低

**风险 2**: 与 Task #26 抢 cuda:1 (Stage 3 同时跑)
- Task #26 走 cuda:0 (K=512, Stage 2.1/2/3 顺序)
- Task #28 走 cuda:1 (K=128, Stage 2.1/2/3 顺序)
- 两 Stage 3 不在同一卡, 无冲突 (Stage 3 ~29GB + 同卡 Stage 2.1 6GB = 35GB < 46GB, 但避免风险分开跑)

**风险 3**: 与 Task #27 经验传递
- Task #26 修复了 setup_metadata_dir 缺 os.makedirs bug
- Task #28 启动应直接受益

---

## 6. 完成度跟踪

- [ ] Stage 1 复用 Task #26 embedding
- [ ] Stage 2.1 启动 (cuda:1, K=128)
- [ ] Stage 2.1 codebook cov ≥ 0.5 健康
- [ ] Stage 2.2 推断 (5 min)
- [ ] Stage 3 启动 (cuda:1, seed=42)
- [ ] Stage 3 best ckpt 保存
- [ ] Stage 4 推断 + Recall/NDCG 评估
- [ ] 写 `verdicts/task28_k128_ablation_result.md`
- [ ] 与 Task #87 K=256 + Task #26 K=512 对比 → 三方表
- [ ] 重跑 `scripts/task27_knn_quality_recall.py` 加入 K=128 → n=7

---

## 7. 关联

- 基准: Task #87 (`verdicts/task87_tiger_baseline_result.md`, K=256, seed=42, R@5=0.01937)
- 关联: Task #26 (K=512, 在跑 cuda:0)
- 上游 Stage 1: Task #26 sentence-t5-base embedding (logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt)
- 下游: Task #27 (邻域质量相关性) Plan B 增量
- 决策: §1 自主决策 + §5.3 强制并行 + Task #27 推荐 Plan B