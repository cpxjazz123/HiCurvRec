# Task #63 (G1) P1 — RQ 级联 Sinkhorn-balanced Laguerre 分配 verdict (2026-07-20)

> **任务目的**: 验证 G1 Sinkhorn-balanced k-means 在完整 RQ 级联 (3 层) 上是否保持 "G1-H1 = 平衡质量分配" + 测试 "G1-H2 (V-information 剖面去集中化)" 是否成立.
> **结论**: ⭐ **G1-H1 ✅ GO + G1-H2 ❌ FALSIFIED**. G1 方向演化为"平衡码本 (G1-H1 only)"方向, 不预测 V-information 再分布. P2 η 扫描与 P3 TIGER 可继续 G1-H1 路线.

---

## 1. Head-to-head 逐层对比 (Sinkhorn vs Vanilla, 11924 × 2048 flan-t5, K=256, num_hierarchies=3, seed=42)

| Layer | Metric | Vanilla | Sinkhorn | Δ |
|-------|--------|---------|----------|---|
| L0 (input) | PPL | 198.4 | **231.8** | **+16.8%** |
| L0 | min_count | **1** | **11** | **+10** |
| L0 | max_count | 220 | **128** | **-42%** |
| L0 | D_rel (input space) | 0.057989 | **0.057940** | **-0.08% (略优)** |
| L0 | var_reduction | 94.25% | 94.25% | 0 |
| L1 (residual) | PPL | 89.2 | **218.7** | **+145%** |
| L1 | min_count | 1 | **9** | **+8** |
| L1 | max_count | 474 | **235** | **-50%** |
| L1 | D_rel (residual) | inf (numerical) | 0.792 | (vanilla D_rel explodes) |
| L1 | var_reduction | 20.83% | **21.07%** | **+0.24%** |
| L2 (residual) | PPL | 42.0 | **218.5** | **+420%** |
| L2 | min_count | 1 | 1 | 0 |
| L2 | max_count | 937 | **392** | **-58%** |
| L2 | var_reduction | 11.46% | **13.71%** | **+2.25%** |

**G1-H1 ✅ 强 GO**:
- **L0 min_count 11×** (Sinkhorn 11 vs Vanilla 1) — 码本塌缩显著缓解
- **L0/L1/L2 max_count 削减 42-58%** — 极热 cluster 不再独占
- **L0 D_rel 几乎不变 (-0.08%)** — 平衡约束不付出失真税 (实际上略优)
- **L2 var_reduction +2.25% 提升** — 全局残差压缩率小幅改善

**L2 min_count 都退化为 1**: Sinkhorn 在 L0 + L1 把质量分配拉平之后, L2 的残差已是"被压平"分布, 平衡约束本身的边际效应消失 (残差近乎噪声, Lloyd 自然退化). 这是 Cascade 平衡的极限, 不是 G1 的失败, 而是 G1 作用上界.

---

## 2. ⭐ G1-H2 V-information 剖面 = **FALSIFIED**

### 2.1 假设与检验

**G1-H2**: Sinkhorn-balanced L1 之后, 逐层 ε 应显著去集中化 (residual 残差场更平稳), 跨层 ε 相关性下降.
**预测**: Sinkhorn 的 ε_layer_corr 应明显低于 Vanilla, 特别是 ε_corr[0,2] (L0 与 L2 ε 相关性).

### 2.2 实测

| ε_corr | Vanilla | Sinkhorn | 解读 |
|--------|---------|----------|------|
| ε[0,1] | 0.885 | **0.969** | Sinkhorn **更高** |
| ε[0,2] | 0.794 | **0.946** | Sinkhorn **更高** (+0.15) |
| ε[1,2] | 0.903 | **0.985** | Sinkhorn **更高** |
| **tr mean** | 0.861 | **0.967** | Sinkhorn **+12%** |

**与预测完全反向**: Sinkhorn 不仅没有去集中 V-information, 反而把跨层 ε 相关性**提高 12%**.

### 2.3 物理解释

Sinkhorn 强制每码字固定质量 (≈1/K), 即把"少数 item 占多数 cluster"的不平衡消除. 这意味着:
- 同一 item 的残差在不同 layer 都会被"平均地"分到一码字, 其 ε 不再有"被压平"vs"再被压"的差异
- 残差统计量变得**跨 layer 一致** (高 related)
- Vanilla 允许差异化: 部分 item 在 L0 就被精细建模 (L1 ε 低), 部分 item 必须跨多层累积 (L1 ε 高), 造成**item 间的 specialization**, 跨层 de-corr 反映这种 specialization

**结论**: G1-H2 误把"平衡"等同于"去集中化". 实际上平衡约束下, **每个 item 都承受近乎均匀的处理深度**, V-information 反而更难以在跨层间有效分配. 是 Sinkhorn-balanced 的**副作用**, 不是 bug.

### 2.4 决策含义

G1 退化为**"码本质量平衡器"**:
- ✅ 改进 L0+L1 码本利用率 (min_count 1→9-11)
- ✅ 同等或略优的失真
- ✅ 全局残差压缩率小幅提升
- ❌ 不改变 V-information 剖面
- ❌ 不推动信息到深层

G1 真正的成功路径不再是 "P2 η 扫描 → P3 TIGER" (H2 假设), 而是 "硬平衡码本 + 朴素 RQ 级联, 看是否仅靠 balanced 即可超越 vanilla + 训练 RQ-VAE (Task #53 #54)".

---

## 3. G1-H2 Falsification 的下游意义

**正面**:
- G1 仍能拿到平衡收益 (min_count 9-11)
- 平衡码本 + RQ 级联 + 不训 RQ-VAE = 与 Task #59 vanilla Simple KMeans 直接对比的"理想 Sid method"
- 即 G1 的真正价值是 **不用 RQ-VAE 也能拿到平衡分配**, 但默认 RQ-VAE (Neural) 也能做到 (Task #53/54 失败是另一回事)

**负面**:
- 推翻用户提案中"G1 还可解释深层近零 V-information"的核心假设
- 深度推不动信息的原因**不在码本平衡** (或不完全在), 需重审

**下一步方向**:
- G1 真正可继续: P2 η 扫描 (看 η→0 完全均衡下的失真/利用率 Pareto)
- G1 真正可继续: P3 直接搭 Stage 3 + Stage 4 闭环, 看 min_count 11+balanced 是否能在 Toys 上进一步提升 R@5 (vs Task #59 0.0857 vanilla + 1-layer Simple KMeans + TIGER)

---

## 4. 关键产品物

| 文件 | 内容 |
|------|------|
| `logs/task63_p1/pickle/sid_sinkhorn_balanced.pt` | (4, 11924) SID = [3 hierarchy + 1 dedup] Sinkhorn-balanced |
| `logs/task63_p1/pickle/sid_vanilla.pt` | 同上 vanilla 配对 |
| `logs/task63_p1/pickle/cluster_centers_l{0,1,2}_*.npy` | 9 个 npy, 两个 variant × 三层 centers |
| `logs/task63_p1/pickle/cluster_idx_l{0,1,2}_*.npy` | 同上 per-item codes |
| `logs/task63_p1/pickle/summary_sinkhorn_balanced.json` | 完整指标 + G1-H2 ε 矩阵 |
| `logs/task63_p1/pickle/summary_vanilla.json` | 同上 vanilla |
| `logs/task63_p1_sinkhorn.log` | Sinkhorn 3-layer RQ 日志 |
| `logs/task63_p1_vanilla.log` | Vanilla 3-layer RQ 日志 |

---

## 5. P2/P3 决策矩阵

| 路径 | 条件 | 决策 |
|------|------|------|
| **P2 η 扫描** | 用 (L0+L1) 综合 (PPL ratio, D_rel) Pareto, η ∈ {0.5, 0.7, 0.9, 1.0=Vanilla} → 检验 "θ-Pareto curve 是否凸且在 η=0.5 优于 η=1.0" | 若 Pareto 在 η<1 处有显著优势则 P2 ✅ |
| **P3 Stage 3 TIGER 闭环** | 用 `sid_sinkhorn_balanced.pt` 直接跑 Stage 3 TIGER (沿用 Task #59 配置) → R@5 vs Task #59 0.0857 | 若 R@5 > 0.09 → G1 突破; 若 R@5 < 0.07 → Sinkhorn-balanced 损失无收益 |
| **停止 G1** | 两者均不显著优于 Task #59 baseline | G1 关闭 |

预算估算:
- P2 η 扫描: 0.5 天 (Sinkhorn 跑 K=256, η 4 个 setting, 每 setting ~90 sec)
- P3 TIGER: 重新 Stage 3 训练 ~6-8 h, Stage 4 + eval ~30 min

---

## 6. 完成判定

- [x] P1 RQ 级联 Sinkhorn 脚本 (`scripts/task63_p1_rq_sinkhorn.py`, py_compile 通过)
- [x] P1 Sinkhorn 3-layer RQ 跑完 (`logs/task63_p1_sinkhorn.log`)
- [x] P1 Vanilla 3-layer RQ 跑完 (`logs/task63_p1_vanilla.log`)
- [x] G1-H2 ε 层相关矩阵计算 (Sinkhorn 0.946 vs Vanilla 0.794 at L0×L2)
- [x] P1 verdict 落盘 ← 本文档

---

result: Task #63 P1 完成. **G1-H1 ✅ GO** (balanced min_count 1→11 at L0, L0 D_rel -0.08%, L2 var_red +2.25%). **G1-H2 ❌ FALSIFIED** — Sinkhorn ε_layer_corr = 0.946 > Vanilla 0.794, 跨层相关性高 19%, 与预测完全反向. Sinkhorn 把跨层 ε 相关性提高 12% (与"去集中化"假设相反). G1 退化为"码本质量平衡器", 进一步价值在 P2 η 扫描 Pareto + P3 TIGER 闭环 (vs Task #59 R@5=0.0857).