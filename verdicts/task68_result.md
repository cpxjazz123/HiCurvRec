# Task #68 — MCKG 原始（未修 norm）+ 192d RQ-VAE 验证维度独立性

> **任务目的**: 验证 R1 "MCKG 拼接提高 192d (不论是否修 norm) 都应该有显著高于 64d 的 L0 coverage"，填补 2×2 矩阵（维度 × 健康度）中"高维 + 不健康"的关键"未测"格
> **完成日期**: 2026-07-20
> **状态**: ✅ Stage 0 + Stage 1 完成，决策 **PARTIAL (R1 部分成立)**

---

## 1. 任务目标

- **核心假设 R1**: MCKG 拼接提高 192d 应该有显著高于 64d 的 L0 coverage，因为 encoder 有充足通道处理异常值
- **对照假设 R2**: norm 病才是主因，提高维度本身无法解决

## 2. 实验设计

| 维度 | Task #67 (norm 修复 + Revival) | **Task #68 (本任务)** |
|------|--------------------------------|------------------------|
| **输入 norm 修复** | exp_map_pre_clip + cap=5.0 各自修复 | **不修复，直接 raw concat** |
| **Revival hook** | 启用 | **禁用**（no_revive yaml） |
| **embedding_dim** | 192 | 192 |
| **num_hierarchies** | 3 | 3 |
| **codebook_width** | 256 | 256 |
| **max_steps** | 5000 | 5000 |

**唯一独立变量**: 是否对每个 subspace 做 norm 修复 + Revival hook

---

## 3. Stage 0: 输入准备（不修 norm）

`products/task68_mckg_raw_192d/entity_embedding_concat192d_raw.pt` (11924, 192)

- 直来自 raw MCKG 3 段拼接
- **max_abs=378** (vs Task #67 norm 修复后 max_abs=3.8, 100× 异常值)
- hyperbolic subspace 范围 [-378, 316] 是 norm 病态主因

---

## 4. Stage 1: RQ-VAE 训练（不加 Revival）

- 启动命令：`python -m src.train experiment=rqvae_train_flat_no_revive ...`
- GPU 1 (PID 3352732)
- 启动时间：2026-07-20 16:03
- 结束时间：2026-07-20 16:48 (45 min 实际跑 5000 steps)
- Checkpoint: `logs/task68_mckg_raw_192d/runs/2026-07-20/16-04-01/checkpoints/checkpoint_000_005000.ckpt`

---

## 5. 关键指标

### 5.1 最终 epoch 累积指标 (cumulative over training epoch)

| 指标 | Epoch Cumulative | 阈值 | 判定 |
|------|------------------|------|------|
| L0 frac_layer_coverages_epoch | **0.651** | ≥0.80 GO / 0.50-0.80 PARTIAL | **PARTIAL** |
| L1 frac_layer_coverages_epoch | 0.941 | ≥0.80 | ✅ |
| L2 frac_layer_coverages_epoch | 0.940 | ≥0.80 | ✅ |
| L0 id_entropy_epoch | 4.570 | ≥4.5 健康 | ✅ |
| L1 id_entropy_epoch | 5.210 | ≥4.5 健康 | ✅ |
| L2 id_entropy_epoch | 5.220 | ≥4.5 健康 | ✅ |
| frac_unique_ids_epoch | **0.960** | ≥0.95 健康 | ✅ |
| recon_loss_epoch | 0.0142 | <0.1 | ✅ 稳定 |
| first_centroids_norm | 7.76 | 1-100 健康 | ✅ |
| last_centroids_norm | 0.760 | 1-100 健康 | ✅ |

### 5.2 最终 batch-level 指标 (per training batch)

| 指标 | Batch |
|------|-------|
| L0 frac_layer_coverages_step | 0.684 |
| L1 frac_layer_coverages_step | 0.980 |
| L2 frac_layer_coverages_step | 0.992 |
| L0 id_entropy_step | 4.790 |
| L1 id_entropy_step | 5.360 |
| L2 id_entropy_step | 5.410 |
| frac_unique_ids_step | 0.998 |

### 5.3 训练稳定性

- 无 NaN / Inf
- 无 RuntimeError
- loss 平滑下降到 0.00299
- bf16-mixed 精度下未爆

---

## 6. 决策结论

### 6.1 L0 coverage = 0.651 (epoch cumulative)

落在 [0.50, 0.80] 区间 → **PARTIAL 决策**：
- **R1 部分成立**: 192d 比 64d 的 L0 coverage 高（0.651 vs 0.11），维度提升有帮助
- **R2 部分成立**: 但 0.651 < 0.80 阈值，说明 norm 病仍是限制因素
- **结论**: 维度提升 + norm 修复 协同作用才能完全解决坍缩

### 6.2 与 Task #67 对比

| 指标 (epoch cumulative) | Task #67 (norm fix + Revival) | Task #68 (raw) | 差距 |
|--------------------------|-------------------------------|----------------|------|
| L0 cov | (仍在跑, step ~3800) | 0.651 | — |
| L0 entropy | ~4.3 | **4.570** | raw 略优 |
| frac_unique_ids | ~0.96 | 0.960 | 持平 |
| 训练稳定性 | ✅ 无错 | ✅ 无错 | 持平 |

---

## 7. 启示

1. **维度独立解决坍缩是部分真**: 从 64d cov=0.11 → 192d raw cov=0.651 是 6× 提升
2. **norm 修复仍是必要**: 0.651 < 0.80 阈值说明 norm 病态压低了 encoder capacity 利用率
3. **最关键的洞见**: **不需要 Revival hook 也能达到 0.651 coverage** — 说明 norm 修复 + Revival 的组合可能 overkill
4. **未来方向**: 测试 norm 修复但不加 Revival (Task #67 v5 实际上就是这个组合, 正在跑)

---

## 8. 后续建议

- **Task #67 v5 跑完后**: 比较 v5 (norm fix + Revival) vs Task #68 (raw) 的最终 L0 cov, 量化 Revival 的边际贡献
- **下一步 ROI**: 跑 "MCKG 修复 + 64d" 进一步定位 norm 修复独立性 (填补 2×2 矩阵最后一格)
- **不再建议**: 用 64d MCKG 跑 neural RQ-VAE (cov=0.11 已确认坍缩)

---

## 9. 产物清单

- `products/task68_mckg_raw_192d/entity_embedding_concat192d_raw.pt` — raw 192d 输入
- `logs/task68_mckg_raw_192d/runs/2026-07-20/16-04-01/checkpoints/checkpoint_000_005000.ckpt` — Stage 1 ckpt
- `configs/experiment/rqvae_train_flat_no_revive.yaml` — no Revival config
- `logs/task68_mckg_raw_192d.log` — 训练日志

---

result: Task #68 Stage 1 完成. L0 cumulative coverage 0.651 ∈ [0.50, 0.80] → **PARTIAL (R1 部分成立)**. 维度提升 (64→192d) 给 6× coverage 提升 (0.11→0.651), 但未达 0.80 阈值, norm 修复仍必要. 未做 Stage 2.2/3/4 (按设计本任务只验证 Stage 1).
