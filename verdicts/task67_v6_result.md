# Task #67 v6 — norm 修复但禁用 Revival (隔离 Revival 独立贡献)

> **任务目的**: 通过对比 v6 (norm fix + no Revival) 与 v5 (norm fix + Revival) 和 #68 (raw + no Revival), 隔离 Revival hook 与 norm 修复两个变量的独立贡献.
> **完成日期**: 2026-07-20
> **状态**: ✅ Stage 2.1 训练完成 (5000 steps), L0 cumulative cov=**0.4761**

---

## 1. 任务目标

- **核心问题**: Task #67 v5 (norm fix + Revival) L0 cov=0.374 < Task #68 (raw) L0 cov=0.651 (-42%). 需 v6 隔离 Revival vs norm fix 独立贡献.
- **本任务 (v6)**: 用 norm fix 192d 输入但禁用 Revival hook, 跑完 5000 steps 看 L0 coverage.

## 2. 实验设计

| 维度 | Task #67 v5 | **Task #67 v6 (本任务)** | Task #68 |
|------|-------------|---------------------------|----------|
| **输入 norm 修复** | exp_map_pre_clip + cap=5.0 | **相同** | 不修复 |
| **Revival hook** | 启用 | **禁用** | 禁用 |
| **embedding_dim** | 192 | 192 | 192 |
| **num_hierarchies** | 3 | 3 | 3 |
| **codebook_width** | 256 | 256 | 256 |
| **max_steps** | 5000 | 5000 | 5000 |
| **seed** | 42 | 42 | 42 |

**变量对照**:
- v6 vs v5: 仅 Revival hook 状态不同
- v6 vs #68: 仅 norm 修复状态不同

## 3. 关键指标 (最终 epoch cumulative, 从 metrics.csv 解析)

| 指标 | Task #67 v5 | **Task #67 v6** | Task #68 raw |
|------|-------------|-----------------|--------------|
| **L0 frac_layer_coverages_epoch** | 0.3745 | **0.4761** | **0.6510** |
| L1 frac_layer_coverages_epoch | 0.9363 | 0.9431 | 0.9410 |
| L2 frac_layer_coverages_epoch | 0.9135 | 0.9248 | 0.9400 |
| **L0 id_entropy_epoch** | 4.126 | **4.269** | **4.570** |
| L1 id_entropy_epoch | 5.210 | 5.217 | 5.210 |
| L2 id_entropy_epoch | 5.172 | 5.191 | 5.220 |
| frac_unique_ids_epoch | 0.957 | 0.957 | 0.960 |
| loss_epoch | 0.0136 | 0.0136 | 0.0142 |
| first_centroids_norm_epoch | 9.99 | 7.68 | 7.76 |
| last_centroids_norm_epoch | 1.14 | 0.718 | 0.76 |

## 4. 隔离分析

### 4.1 Revival 独立贡献 (v6 vs v5)

| 指标 | v5 | v6 | Δ | 解释 |
|------|-----|-----|---|------|
| L0 cov | 0.3745 | **0.4761** | **+0.102** | **Revival 是负贡献 -0.10** |
| L0 entropy | 4.126 | 4.269 | +0.143 | Revival 反向压低熵 |
| L1 cov | 0.9363 | 0.9431 | +0.007 | 轻微 |
| L2 cov | 0.9135 | 0.9248 | +0.011 | 轻微 |
| first_centroids_norm | 9.99 | 7.68 | -2.31 | Revival 把 centroid norm 推到异常值 |

**结论**: 关掉 Revival 后 L0 cov 从 0.374 → 0.476 (+0.10). **A 假设成立**: Revival hook 是负贡献, 但只占总负贡献的 36% (0.10 / 0.28).

### 4.2 norm fix 独立贡献 (v6 vs #68 raw)

| 指标 | v6 | #68 | Δ | 解释 |
|------|-----|------|---|------|
| L0 cov | 0.4761 | **0.6510** | **-0.175** | **norm fix 是负贡献 -0.17** |
| L0 entropy | 4.269 | 4.570 | -0.301 | norm fix 压低熵 |
| first_centroids_norm | 7.68 | 7.76 | -0.08 | 几乎相同 |

**结论**: norm fix 输入相对 raw 输入 L0 cov 损失 0.17 (-27%). **B 假设成立**: norm 修复 (cap=5.0) 把 hyperbolic 异常值压平, 实际损失了 L0 encoder 可利用的高维差异信号.

### 4.3 综合: 哪个变量是 v5 失败的主因?

| 变量 | 独立负贡献 | 占 v5 vs raw 总差 (0.277) 比例 |
|------|----------|-------------------------------|
| Revival hook | -0.10 | **36%** |
| norm fix | -0.17 | **64%** |
| **总和** | **-0.27** | 100% |

**关键洞察**: norm fix 信息损失是 v5 失败的**主因** (64%), Revival hook 反向贡献是**次因** (36%).

## 5. 决策结论

### 5.1 三组实验最终对照

| 实验 | 输入 | Revival | L0 cov | L0 ent | 决策 |
|------|------|---------|--------|--------|------|
| Task #100 | MCKG raw 64d | — | 0.11 | — | ❌ 严重坍缩 |
| Task #67 v5 | MCKG norm fix 192d | 启用 | 0.3745 | 4.126 | ❌ 双重负贡献 |
| **Task #67 v6** | MCKG norm fix 192d | 禁用 | **0.4761** | **4.269** | ⚠️ 部分改善 |
| **Task #68** | MCKG raw 192d | 禁用 | **0.6510** | **4.570** | ✅ **最佳** |

### 5.2 最佳配置确认

✅ **raw 192d + no Revival** (Task #68 配置) 是当前最佳组合:
- 维度 (192d) 防止严重坍缩 (0.11 → 0.651, 6× 提升)
- 不 norm fix 保留 hyperbolic 异常值信号
- 不 Revival 避免过度干预

### 5.3 后续建议

1. ✅ **永久关闭 Revival hook** (192d+ 输入不需要, 反作用)
2. ✅ **不使用 norm fix** (信息损失 0.17 比 Revival 负贡献更大)
3. 🔴 **不建议继续优化 Revival** (192d 不需要)
4. 🟡 可选: 在 64d 输入上重测 Revival 是否有救 (Task #100 的 0.11 太低, 可能 Revival 在 64d 上是必要的)
5. 🟢 高 ROI: 跑 Task #69 (5-graph weight diagnose) 验证 MCKG 几何异质性 vs 图结构

## 6. 启示

1. **维度是 L0 coverage 的主导因素** (0.11 → 0.651, 6× 提升来自维度)
2. **norm fix 信息损失不可忽视** (cap=5.0 把 hyperbolic 异常值压平)
3. **Revival hook 在 192d 上是反作用** (1620 centroids 被强制重置, 引入噪声梯度)
4. **修复方案应保留原始信号**, 而非平滑掉异常值

## 7. 产物清单

- `logs/task67_v6_norm_fix_no_revival.log` — 训练日志 (5000 steps 完成, 41 min)
- `logs/task67_v6_norm_fix_no_revival/runs/2026-07-20/17-15-28/checkpoints/checkpoint_000_005000.ckpt` — Stage 2.1 ckpt (34.5 MB)
- `logs/task67_v6_norm_fix_no_revival/runs/2026-07-20/17-15-28/csv/version_0/metrics.csv` — 训练指标 (101 rows, epoch cumulative 完整)

## 8. 完成度跟踪

- [x] 写 descriptions/task67_v6_norm_fix_no_revival.md
- [x] 启动 Stage 1 RQ-VAE 训练 (PID 3665558, 41 min 完成)
- [x] 监控 step 500/1000/2500/5000 L0 cov/id_entropy/loss (全部健康)
- [x] 写 `verdicts/task67_v6_result.md` (含 result: 行)
- [x] 清理 loop.md §16 (v6 从暂停区删除, 归档到历史已清理)
- [x] 更新 memory `192d-revival-counterproductive.md` 加入 v6 隔离结论

---

result: Task #67 v6 完成 5000 steps (41 min). L0 cumulative cov=**0.4761** (落在 [0.40, 0.65) 区间, **A+B 都成立**). Revival hook 关闭后从 v5=0.374 提升到 v6=0.476 (+0.10, A 成立: Revival 是负贡献 36%); 但 v6 仍低于 #68 raw 0.651 (-0.17, B 成立: norm fix 信息损失 64% 是主因). **最终结论**: raw 192d + no Revival (Task #68 配置) 是最佳组合; norm fix 不应启用, Revival hook 应永久关闭. 192d 维度本身已足以防止严重坍缩 (6× 提升来自维度, 修复方案应保留原始信号而非平滑异常值).
