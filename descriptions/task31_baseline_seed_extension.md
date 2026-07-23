# Task #31 — Baseline (TIGER-aligned, K=256) 多 seed 扩展 (n=4 → n=8)

> **任务目的**: 扩 Task #27 (邻域质量 vs R@5 相关性) 的 n=4 → n=8, 通过多 TIGER seed 跑同一 baseline SID (`products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt`), 隔离 seed variance 与 SID 质量贡献, 让 80% power (|ρ|=0.67) 所需 n=16 的目标达成一半.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

### 1.1 Task #27 n=4 困境
- 当前 4 个 tokenizer (Task85_m1 / Task87 / Task85_m0 / Task107) 全部在 baseline ±30% 区间
- Spearman ρ (norm_hamming vs R@5) = +0.21 (p=0.79) — 不显著
- Power (假设 |ρ|=0.67) = **0.127** → 几乎无检验力
- 80% power 需要 n=16
- Jackknife LOO 任意一个数据点都能翻转结论方向 → 样本量严重不足

### 1.2 seed variance 已知大小
- Task #87 (seed=42): R@5=0.01937
- Task #107 (seed=123): R@5=0.01489
- 同一 SID 不同 seed → R@5 差 -23% (CV=18.5%)
- **seed variance 已经主导 y 轴方差**, 1 个数据点本质上是 1 个 seed sample

### 1.3 Task #31 假设
- 如果 R@5 主要由 seed variance 决定, 那么多 seed 跑同一 SID 能给出 **R@5 的真实分布**
- 把 4 个 token × 2 seeds = 8 个数据点, 提供 R@5 variance 的估计
- 对 Task #27 邻域假设提供更稳定的相关性估计

---

## 2. 实验设计

**变量**: TIGER 训练 seed (5 个新 seed)
**保持不变**:
- SID: `products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt` (baseline K=256, 与 #87/#107 同)
- Stage 1 T5 embedding: `logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt`
- 模型架构, lr, batch size, num_hierarchies=4, sequence_length=120

**启动命令** (5 个 seed):
```bash
# seed=7
CUDA_VISIBLE_DEVICES=0 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt \
    sequence_length=120 num_hierarchies=4 \
    seed=7 task_name=task31_seed7_s3

# seed=99
CUDA_VISIBLE_DEVICES=1 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt \
    sequence_length=120 num_hierarchies=4 \
    seed=99 task_name=task31_seed99_s3

# seed=2024
CUDA_VISIBLE_DEVICES=2 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt \
    sequence_length=120 num_hierarchies=4 \
    seed=2024 task_name=task31_seed2024_s3

# seed=2025
CUDA_VISIBLE_DEVICES=3 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt \
    sequence_length=120 num_hierarchies=4 \
    seed=2025 task_name=task31_seed2025_s3

# seed=2026
CUDA_VISIBLE_DEVICES=0 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt \
    sequence_length=120 num_hierarchies=4 \
    seed=2026 task_name=task31_seed2026_s3  # 等 seed=7 完再启动
```

**注**: seed=2026 必须等 seed=7 跑完才能启动 (GPU 0 共用).

---

## 3. 决策触发 (vs Task #87 baseline seed=42)

| 指标条件 | 结果指标 | 决策 |
|----------|----------|------|
| 5 个新 seed R@5 mean ∈ [0.0155, 0.0233] (±20% of baseline 0.01937) | 6 个 seed R@5 std/mean CV < 30% | ✅ seed variance 可控, n=8 数据扩展有效 |
| 5 个新 seed R@5 mean ∉ [0.0155, 0.0233] | CV > 30% | ⚠️ seed variance 太大, n=8 仍不够, 需更大 n |
| 任一 seed R@5 < 0.0097 (50% baseline) | 同 | ❌ 跑出失败 run, 写入失败记录 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 5× Stage 3 TIGER 训练 (单 seed ~6h, 串行) | ~30 h |
| 5× Stage 4 推断 + Recall/NDCG 评估 | ~30 min |
| Task #27 enhanced_stats 重跑 (n=8) | ~10 min |
| **总计** | **~30.5 h 墙钟** |

**串行原因**: GPU 0 需等 seed=7 完成才能跑 seed=2026, GPU 1/2/3 各跑一个 seed. 实际并行 4 个 + 串行 1 个 = 4×6h + 6h = **~30h 串行 or 24h 4-way 并行**.

---

## 5. 风险与缓解

**风险 1**: seed=2024/2025/2026 大数值 seed 可能导致 TIGER Adafactor 收敛慢 → 缓解: 加 patience=8 evals×5 epochs=40 epoch 上限, 必要时早停
**风险 2**: 5 个 seed 都跑出 R@5 < baseline (类似 seed=123) → n=8 数据全部在 baseline 附近, 仍能给出 R@5 分布估计
**风险 3**: 4 个 GPU 并行跑 TIGER 可能互抢显存 → 同卡 Stage 3 不可与 Stage 3 并行 (loop.md §3), 不同卡 OK, 监控 nvidia-smi

---

## 6. 完成度跟踪

- [ ] 5 个新 seed 启动 + 跑完 Stage 3 + Stage 4
- [ ] n=8 数据上 Task #27 enhanced_stats 重跑
- [ ] n=8 power 重新计算 (预期 ≥ 0.30)
- [ ] 写 `verdicts/task31_baseline_seed_extension_result.md`
- [ ] 更新 Task #27 verdict 引用新 n=8 结果

---

## 7. 关联

| 任务 | 角色 |
|------|------|
| Task #87 | 当前 baseline anchor (seed=42, R@5=0.01937) |
| Task #107 | seed=123 variance probe (R@5=0.01489) |
| Task #27 | 邻域假设验证 (n=4 power=0.127, 需扩 n) |
| Task #31 (本任务) | 加 5 seed 让 n=8 |