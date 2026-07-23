# Task #29 — Toys Stage 2 RQ-VAE codebook_width=64 ablation

> **任务目的**: 验证 RQ-VAE codebook_width=64 (vs K=256 baseline / K=128 ablation / K=512 ablation) 是否触发 codebook collapse 或 Recall 严重退化. 增量 Task #27 Plan B 第三数据点 (n: 6→8 with Task #26 + #28 + #29).

> **完成日期**: ❌ 2026-07-19 取消 (用户反馈逻辑有问题: K ablation 不直接回答 Task #27 的 kNN preservation 假设)
> **状态**: ❌ CANCELLED (2026-07-19 自主决策启动, Plan B 第三增量)

---

## 1. 背景

### 1.1 上游结论

- **Task #27** (✅): 7 tokenizer, Spearman ρ=-0.667 (n=6, p=0.148). Plan B 推荐扩 n≥10
- **Task #26** K=512 (在跑 cuda:0): 扩大 codebook ablation
- **Task #28** K=128 (在跑 cuda:1): 缩小 codebook ablation
- **Task #29** K=64 (本任务, 启动 cuda:2): 极限缩小 ablation

### 1.2 假设

- **H1-collapse**: K=64 在 11924 items × 4 digits 下严重不足 → 64^4 = 16.7M bins ≪ 2^32, 可能部分 collapse 但仍有冗余
- **H1-regression**: K=64 → R@5 < 0.014 (低于 baseline 区间下限)
- **H0**: K=64 触发 collapse 但 Recall 仍 ≥ baseline floor

### 1.3 单变量隔离

- **唯一变化**: Stage 2.1 RQ-VAE `codebook_width: 256 → 64`
- **其他全部保持与 Task #87 baseline 一致** (per CLAUDE.md R5)
- 复用 Task #26 的 sentence-t5-base embedding

---

## 2. 实验设计

### 2.1 Stage 2.1 RQ-VAE 训练 (K=64)

**启动命令** (cuda:2):
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

CUDA_VISIBLE_DEVICES=2 nohup python -m src.train experiment=rqvae_train_tiger \
    data_dir=data/amazon_data/toys \
    embedding_path=logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt \
    embedding_dim=768 \
    num_hierarchies=3 \
    codebook_width=64 \
    task_name=task29_s2_train_k64 \
    > logs/task29_s2_train_k64.log 2>&1 &
```

### 2.2 后续 Stage 2.2 + Stage 3 + Stage 4
同 Task #26/#28 模板, 改 codebook_width=64 + task_name 前缀 task29.

---

## 3. 决策触发 (vs Task #87 baseline K=256)

| 指标条件 | 预期 R@5 | 决策 |
|----------|---------|------|
| **R@5 > 0.024** (>Task #87 +25%) | 实测 | ✅ K=64 显著优势 (反直觉, 极不可能) |
| **R@5 ∈ [0.014, 0.024]** (baseline 区间) | 实测 | ⚠️ K=64 与 K=256 无显著差异 (容量充足) |
| **R@5 < 0.014** | 实测 | ❌ K=64 严重退化, 选 K≥256 |
| **codebook cov < 0.5 (任一层)** | n/a | ❌ collapse, 早停 |

### 3.1 Task #27 增量

- 加入 n: 6 → 8 (with Task #26 K=512, Task #28 K=128, Task #29 K=64)
- 与 Task #87 K=256 形成 4 组对照 (K=64/128/256/512)
- 重新计算 Spearman ρ, n=8 检验力 ρ>0.5 即可达 p<0.05

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 2.1 RQ-VAE (cuda:2) | ~30 min (K=64 比 K=512 略快) |
| Stage 2.2 推断 | ~5 min |
| Stage 3 TIGER (cuda:2) | ~6h |
| Stage 4 推断 + 评估 | ~15 min |
| **总计** | **~7h** (与 Task #26 cuda:0, Task #28 cuda:1 三重并行) |

---

## 5. 风险与缓解

**风险 1**: K=64 collapse 严重 (cov < 0.4) → 早停
**风险 2**: 与 Task #26/#28 Stage 3 抢 GPU → 三 Stage 3 各占独立 GPU, 不会撞
**风险 3**: 128^4 vs 64^4 信息容量差 → Task #27 已有 Task #85 m=2 trivial data point (mode collapse), 不会再有大跌

---

## 6. 完成度跟踪

- [ ] Stage 1 复用 Task #26 embedding
- [ ] Stage 2.1 启动 (cuda:2, K=64)
- [ ] Stage 2.1 codebook cov ≥ 0.5 健康
- [ ] Stage 2.2 推断 (5 min)
- [ ] Stage 3 启动 (cuda:2, seed=42)
- [ ] Stage 3 best ckpt 保存
- [ ] Stage 4 推断 + Recall/NDCG 评估
- [ ] 写 `verdicts/task29_k64_ablation_result.md`
- [ ] 与 K=64/128/256/512 四方对比 → 与 Task #27 重跑 (n=8)

---

## 7. 关联

- 基准: Task #87 (`verdicts/task87_tiger_baseline_result.md`, K=256, R@5=0.01937)
- 关联: Task #26 K=512 (cuda:0), Task #28 K=128 (cuda:1)
- 上游 Stage 1: Task #26 sentence-t5-base embedding (logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt)
- 下游: Task #27 Plan B 增量 (n: 6→8)
- 决策: §1 自主决策 + §5.3 强制并行 + Task #27 Plan B 推荐