# Task #30 — Toys Stage 2 RQ-VAE codebook_width=384 ablation

> **任务目的**: 验证 RQ-VAE codebook_width=384 (K=256 baseline 与 K=512 ablation 之间) 是否是 sweet spot. 增量数据点加入 Task #27 邻域质量相关性分析 (Plan B 第四增量).
>
> **完成日期**: ❌ 2026-07-19 取消 (用户反馈逻辑有问题: K ablation 不直接回答 Task #27 的 kNN preservation 假设)
> **状态**: ❌ CANCELLED Stage 2.1 (2026-07-19 15:08 启动)

---

## 1. 背景

### 1.1 上游结论

- **Task #27** (✅): 7 tokenizer, Spearman ρ=-0.667 (n=6, p=0.148). 用户反馈 "样本太小 (n=6), 结论不可靠"
- **Task #26** K=512 (在跑 cuda:0): 扩大 codebook ablation
- **Task #28** K=128 (在跑 cuda:1): 缩小 codebook ablation
- **Task #29** K=64 (在跑 cuda:2): 极限缩小 ablation
- **Task #30** K=384 (本任务, 启动 cuda:3): K=256 与 K=512 中间档 sweet spot 验证

### 1.2 假设

- **H1 (sweet spot)**: K=384 容量大于 K=256 但未饱和, 邻域质量比 K=256 更好, Recall 也更高 (若 Phase A 验证 K=256 已饱和, K=384 应与 K=256 类似)
- **H0 (saturated)**: K=256 已达 Toys 11924 items 的容量上限, K=384 vs K=256 无显著差异
- **H-collapse**: K=384 vs K=512 → 因 codebook 增长太快, 早期 cov 可能偏低

### 1.3 单变量隔离

- **唯一变化**: Stage 2.1 RQ-VAE `codebook_width: 256 → 384`
- **其他全部保持与 Task #87 baseline 一致** (per CLAUDE.md R5)
- 复用 Task #26 sentence-t5-base embedding

---

## 2. 实验设计

### 2.1 Stage 2.1 RQ-VAE 训练 (K=384)

**启动命令** (cuda:3):
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

CUDA_VISIBLE_DEVICES=3 nohup python -m src.train experiment=rqvae_train_tiger \
    data_dir=data/amazon_data/toys \
    embedding_path=logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt \
    embedding_dim=768 \
    num_hierarchies=3 \
    codebook_width=384 \
    task_name=task30_s2_train_k384 \
    > logs/task30_s2_train_k384.log 2>&1 &
```

### 2.2 后续 Stage 2.2 + Stage 3 + Stage 4
同 Task #26/#28/#29 模板, 改 codebook_width=384 + task_name 前缀 task30.

---

## 3. 决策触发 (vs Task #87 baseline K=256)

| 指标条件 | 预期 R@5 | 决策 |
|----------|---------|------|
| **R@5 > 0.024** (>Task #87 +25%, 显著超 CV) | 实测 | ✅ K=384 显著优势 (sweet spot) |
| **R@5 ∈ [0.014, 0.024]** (baseline 区间) | 实测 | ⚠️ K=384 与 K=256 无显著差异 |
| **R@5 < 0.014** | 实测 | ❌ K=384 严重退化 |
| **codebook cov < 0.5 (任一层)** | n/a | ❌ collapse, 早停 |

### 3.1 Task #27 增量影响

- 增加 1 个数据点 (n: 9→10)
- 与 K=64/128/256/512 形成五点对照
- 重新计算 Spearman ρ, n=10 → 期望 p<0.05 (若 ρ≥-0.7)

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 2.1 RQ-VAE 训练 (cuda:3) | ~3h (15:08 启动, ETA ~18:10) |
| Stage 2.2 RQ-VAE 推断 | ~5 min |
| Stage 3 TIGER 训练 (cuda:3) | ~6h |
| Stage 4 推断 + 评估 | ~15 min |
| **总计** | **~9.5h** (与 Task #26/#28/#29 完全并行) |

---

## 5. 风险与缓解

**风险 1**: K=384 codebook collapse (利用率 < 50%)
- 缓解: 看 train/layer_X/frac_layer_coverages_step 指标, 若 < 0.5 早停

**风险 2**: 与 Task #26/#28/#29 抢 cuda:3 (Stage 3 同时跑)
- Task #30 走 cuda:3, 4 张 A40 各跑独立任务, 无冲突

---

## 6. 完成度跟踪

- [x] Stage 1 复用 Task #26 embedding
- [x] Stage 2.1 启动 (cuda:3, K=384, PID 2732885, 15:08)
- [ ] Stage 2.1 codebook cov ≥ 0.5 健康
- [ ] Stage 2.2 推断 (5 min)
- [ ] Stage 3 启动 (cuda:3, seed=42)
- [ ] Stage 3 best ckpt 保存
- [ ] Stage 4 推断 + Recall/NDCG 评估
- [ ] 写 `verdicts/task30_k384_ablation_result.md`
- [ ] 与 K=64/128/256/384/512 五方对比
- [ ] 重跑 `scripts/task27_knn_quality_recall.py` 加入 K=384 → n=10

---

## 7. 关联

- 基准: Task #87 (`verdicts/task87_tiger_baseline_result.md`, K=256, R@5=0.01937)
- 关联: Task #26 K=512 (cuda:0), Task #28 K=128 (cuda:1), Task #29 K=64 (cuda:2)
- 上游 Stage 1: Task #26 sentence-t5-base embedding (logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt)
- 下游: Task #27 Plan B 第四增量 (n: 9→10)
- 决策: §1 自主决策 + §5.3 强制并行 + 用户 "样本太小" 反馈触发 Plan B 扩展