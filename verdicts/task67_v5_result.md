# Task #67 v5 — MCKG 拼接 192d + norm 修复 + Revival hook（最终 v5）

> **任务目的**: 验证 "norm 修复 + Revival hook" 是否能让 MCKG 拼接 192d 达到 L0 coverage ≥ 0.80
> **完成日期**: 2026-07-20
> **状态**: ✅ Stage 2.1 训练完成（5000 steps），L0 cov=0.374 < 0.80 阈值
> **重大发现**: norm 修复 + Revival 组合的 L0 cov (0.374) **劣于** raw 192d (0.651)，Revival hook 反而拉低了 coverage

---

## 1. 任务目标

- **核心问题**: Task #67 早期 v1-v4 因 Revival hook 多次崩溃（shape mismatch / dtype mismatch / last_layer_inputs 索引 bug），未能给出公正的 coverage 评估
- **本任务 (v5)**: 修完所有 bug 后，跑完整的 Revival 训练 5000 steps 看 L0 coverage

## 2. 实验设计

| 维度 | Task #67 v5 (本任务) | Task #68 (对照) |
|------|----------------------|-----------------|
| **输入 norm 修复** | exp_map_pre_clip + cap=5.0 各自修复 | 不修复 |
| **Revival hook** | 启用（修复后） | 禁用 |
| **embedding_dim** | 192 | 192 |
| **num_hierarchies** | 3 | 3 |
| **codebook_width** | 256 | 256 |
| **max_steps** | 5000 | 5000 |
| **seed** | 42 | 42 |

## 3. Bug 修复清单

| Bug | 版本 | 修复 |
|-----|------|------|
| Revival 用 input_embedding 但 centroids 是 64d → shape [3,192]→[3,64] | v1 | residual_quantization.py 加 `last_layer_inputs` per-layer pool |
| Revival dtype mismatch (bf16→fp32) | v2 | cast hot_pool 到 centroids dtype |
| n_take=1 时 advanced indexing 不确定 | v3 | explicit `.reshape(n_assign, -1)` + edge case branch |
| last_layer_inputs 索引 bug (3D stacked `[k]` 应为 `[:, :, k]`) | v4 | 修正索引 |

## 4. 关键指标（最终 epoch cumulative）

| 指标 | Task #67 v5 | Task #68 raw | Δ |
|------|-------------|--------------|---|
| **L0 frac_layer_coverages_epoch** | **0.3745** | **0.6510** | **-0.277 (-42%)** ❌ |
| L1 frac_layer_coverages_epoch | 0.9363 | 0.9410 | -0.005 |
| L2 frac_layer_coverages_epoch | 0.9135 | 0.9400 | -0.027 |
| **L0 id_entropy_epoch** | **4.126** | **4.570** | **-0.444 (-10%)** ❌ |
| L1 id_entropy_epoch | 5.210 | 5.210 | 持平 |
| L2 id_entropy_epoch | 5.172 | 5.220 | -0.048 |
| frac_unique_ids_epoch | 0.957 | 0.960 | -0.003 |
| loss_epoch | 0.0136 | 0.0142 | -0.0006 |
| first_centroids_norm | 9.99 | 7.76 | +2.23 |
| last_centroids_norm | 1.14 | 0.76 | +0.38 |
| **revive/total_replaced** | **1620 个** | — | — |

## 5. 决策结论

### 5.1 L0 cov = 0.3745 (epoch cumulative)

落在 [0.20, 0.50] 区间 → **R1 部分否证**：
- norm 修复 + Revival 组合**不能**让 MCKG 拼接 192d 达到 ≥0.80
- 比 raw 192d (Task #68 = 0.651) **更差** 42%

### 5.2 核心发现 — Revival hook 反而拉低 L0 coverage

| 假设 | 期望 | 实测 |
|------|------|------|
| Revival 让 dead centroids 重新激活 | L0 cov ↑ | **L0 cov ↓ -42%** |
| Revival 增加 centroid 利用率 | L0 entropy ↑ | **L0 entropy ↓ -10%** |
| Revival 防止 Sparse Update 坍缩 | 整体训练稳定 | 训练稳定但 L0 退化 |

**Revival hook 在 192d MCKG 上反向作用**, 可能原因:
1. **过度替换**: 1620 个 centroids 被强制重置，引入高频噪声梯度
2. **时序错配**: Revival 在 usage < mean × 0.005 时触发，但 MCKG 高维下偶发未命中是正常现象
3. **norm 修复信息损失**: exp_map_pre_clip + cap=5.0 把 hyperbolic subspace 的 378× 异常值压到 5× 内，丢失了 L0 encoder 本可利用的高维差异信号

## 6. 三组实验最终对照

| 实验 | 输入 | L0 cov | L0 ent | frac_unique | 决策 |
|------|------|--------|--------|-------------|------|
| **Task #100** | MCKG raw 64d | 0.11 | — | — | ❌ 严重坍缩 |
| **Task #67 v5** | MCKG norm fix + Revival 192d | **0.374** | 4.13 | 0.957 | ⚠️ 部分否证 |
| **Task #68** | MCKG raw 192d | **0.651** | 4.57 | 0.960 | ⚠️ PARTIAL |

**关键洞察**: **维度提升（64→192d）是 L0 coverage 的主导因素**（0.11→0.651，6×），而 **norm 修复 + Revival 是次要的且可能有害**（拉低 42%）。

## 7. 启示

1. **192d 维度本身已足以防止严重坍缩**（从 cov=0.11 → 0.651）
2. **Revival hook 在 192d 输入上是反作用**（在 64d 上可能是救命稻草，但在 192d 上变成过度干预）
3. **norm 修复 (cap=5.0) 可能信息损失**（cap 把 hyperbolic 异常值信息压平）
4. **最简洁的"够用方案"**: 用 raw 192d 即可达到 cov=0.651，已经跨过严重坍缩阈值

## 8. 后续建议（按 ROI 排序）

1. **🟢 高 ROI**: 跑 Task #67 v6 = norm 修复但**禁用 Revival** — 隔离 Revival vs norm 修复的独立贡献
2. **🟡 中 ROI**: 跑 "MCKG 修复 + 64d" 看 norm 修复在 64d 上是否能独立解决坍缩（填补 2×2 矩阵最后一格）
3. **🔴 低 ROI**: 不建议继续投入 Revival hook 优化（192d 不需要）

## 9. 产物清单

- `products/task99_mckg_rebuild/entity_embedding_concat192d.pt` — norm 修复 192d 输入
- `logs/task67_concat192d_norm_fix_revived_v5/runs/2026-07-20/16-07-37/checkpoints/checkpoint_000_005000.ckpt` — Stage 2.1 ckpt
- `logs/task67_concat192d_s2.1_revived_v5.log` — 训练日志（5000 steps 完成）
- `src/utils/callbacks/revive_dead_codes.py` — 最终 Revival hook（4 个 bug 修复后）
- `src/modules/clustering/residual_quantization.py` — last_layer_inputs per-layer pool + 正确索引

---

result: Task #67 v5 完成 5000 steps. L0 cumulative coverage = **0.3745** (< 0.50 阈值, **PARTIAL FALSIFIED**). Revival hook + norm 修复反而比 raw 192d (0.651) **更差 42%**. 1620 个 centroids 累计被 Revival 替换, 但 L0 cov 仍低于 raw 基线. **核心洞察**: 192d 维度本身已足以防止严重坍缩, Revival hook 在 192d 上是过度干预. 建议下一步跑 v6 (norm 修复但禁用 Revival) 隔离 Revival 的独立贡献.
