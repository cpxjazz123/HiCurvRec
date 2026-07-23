# Task #66 执行结果与分析 — 加权几何距离量化器

> **任务名**: 加权几何距离量化器（单码书 + 多距离混合权重）
> **完成日期**: 2026-07-17
> **状态**: ❌ **否证** — 加权距离在 L2 归一化下完全等价于单欧氏距离，几何类路线彻底关闭
> **执行人**: Claude（/grid-new-task skill 登记 + Exp1 自主执行）

---

## 1. 任务目标

**假设 R1**：在 L2 归一化空间上，加权距离 d_mix^ℓ = w_E^ℓ·d_E + w_H^ℓ·d_H + w_S^ℓ·d_S 虽每对分量距离排序一致，但混合后的最近邻可能与单距离不同，能改善量化质量。

**决策阈值**（vs baseline R@10=0.0973）：
- Exp1 最优权重 ≠ (1,0,0) for all layers → 继续 Exp2/3
- Exp1 最优权重 = (1,0,0) for all layers → ❌ 否证，立即停止

---

## 2. 执行过程

| 阶段 | 状态 | 关键产物 | 时间 |
|------|:----:|---------|------|
| Exp1: 加权距离穷举 | ✅ | `products/task66/exp1_grid_results.json` | ~90s |
| Exp2: 学习权重训练 | — | **跳过（Exp1 否证）** | — |
| Exp3: 端到端 R@10 | — | **跳过（Exp1 否证）** | — |
| Exp4: 可解释性 | — | **跳过（Exp1 否证）** | — |

**Exp1 配置**：
- RQ-VAE ckpt: `products/task62/rqvae_ckpt.pt` (3 层, 256 簇, 768 维, EMA 训练)
- Embeddings: `products/task62/item_embeddings.pt` (11924, 768)
- 网格: step=0.1, 共 62 个权重组合 (w_E + w_H + w_S = 1, all ≥ 0)
- 距离函数：
  - d_E = ||x - y||₂
  - d_H = arcosh(1 + 2·||x_p - y_p||² / ((1-||x_p||²)(1-||y_p||²))) where x_p = tanh(||x||)·x/||x||
  - d_S = arccos(⟨x, y⟩)

---

## 3. 关键结果

### 3.1 Exp1 — 加权距离 MSE

| 层 | baseline MSE (1,0,0) | best MSE | best 权重 | ΔMSE |
|:---:|:---:|:---:|:---:|:---:|
| L1 | 1.355499e-01 | 1.355499e-01 | **(0, 0.1, 0.9)** | **+0.000%** |
| L2 | 9.868434e-02 | 9.868434e-02 | **(1, 0, 0)** | **+0.000%** |
| L3 | 8.181681e-02 | 8.181681e-02 | **(1, 0, 0)** | **+0.000%** |

**max |ΔMSE| = 0.000%**

### 3.2 数学解释

在 L2 归一化单位球上（||x|| = ||y|| = 1）：

```
d_E(x, y)² = ||x - y||² = 2 - 2⟨x, y⟩ = 2(1 - cos θ)
d_S(x, y) = arccos(⟨x, y⟩) = θ
d_H(x_p, y_p): Poincaré 投影 tanh(1)·x 是等比例缩放，arcosh 内层仍是欧氏距离
```

**关键事实**：d_E、d_H、d_S 在单位球上**都是 cos(θ) 的单调函数**，因此三者的 argmin 完全一致。

加权 d_mix = w_E·d_E + w_H·d_H + w_S·d_S：
- 由于 d_· 都是 cos(θ) 的严格单调函数（权重 w_· > 0 时），d_mix 也是 cos(θ) 的严格单调函数
- argmin_k d_mix(r, c_k) = argmin_k cos(r, c_k) = argmin_k d_E(r, c_k)
- **加权后的最近邻与单欧氏距离的最近邻完全相同**

### 3.3 Layer 1 双最优解的印证

Layer 1 报告了两个等价最优解 (1,0,0) 与 (0,0.1,0.9)，MSE 完全一致——这进一步证明不同权重组合下量化结果不变，因为 argmin 没变。

---

## 4. 累计否证链（2026-07-16 至今）

| 路线 | Task | 结论 | 资源 |
|------|:----:|:----:|:----:|
| HHHH Poincaré SID | #59-60 | R@10=0.0284（29%） | ~6h GPU |
| 多几何距离替换 | #62 Exp3 | ❌ 距离等价（4 个 setting MSE 差异 < 0.2%） | ~1h GPU |
| 黎曼优化约束 | #63 | ❌ 码书结构不变（Δ ≈ 0） | ~1h GPU |
| 几何感知编码 | #64 | ❌ 可预测但选码书 MSE +140% | ~10min GPU |
| **加权几何距离** | **#66** | **❌ 加权等价于单距离（ΔMSE = 0%）** | **~90s CPU** |

**所有"几何类"改进路线全部否证。**根本限制：L2 归一化 + 高维空间使几何差异在量化层面完全消失。

---

## 5. 产物清单

| 产物 | 路径 |
|------|------|
| 任务定义 | `descriptions/task66_weighted_geom_distance.md` |
| Exp1 完整网格结果 | `products/task66/exp1_grid_results.json` |
| Exp1 最佳权重摘要 | `products/task66/exp1_best_weights.json` |
| Exp1 脚本 | `scripts/task66_exp1_weighted_grid.py` |
| Exp1 日志 | `GRID/task_artifacts/scripts/logs/task66_exp1.log` |
| Verdict | `verdicts/task66_weighted_geom_distance.md` |

---

## 6. 后续建议

### 路线状态

```
已彻底关闭（不宜再投入）：
  ❌ 所有"几何类"RQ-VAE 改进（T62/T63/T64/T66）
  ❌ HHHH SID recipe（T59/T60）

未探索（仍需 GPU）：
  ❓ 标准 RQ-VAE pipeline 完整重现与 R@10 确认（Task #65 中断在 step 18098/50000）
  ❓ 非几何改进：数据增强、模型缩放、损失函数
```

### 推荐的下一步

1. **重启 Task #65 Stage 3 baseline**（从 step 1125 ckpt 续训或重训），先确认 R@10=0.0973 是否可重现
2. **转向非几何改进方向**：
   - 数据增强：论文 Table 6 的其他增强方式（如 query dropout）
   - 模型缩放：增大码本 W=512 或 L=5，看是否能突破 0.0973
   - 损失函数改进：直接优化 Recall 而非重建 loss
3. **产出阶段性研究总结**：5 个几何类任务全部否证，可写 "Why geometry-aware RQ-VAE doesn't work for sequential recommendation" 短文

---

## 7. 任务变更说明

- Task #66 登记时 Task #65 Stage 3 在 step 18098/50000 被中断（用户决策"杀 Stage 3 释放 GPU"）
- Exp1 否证后，按决策表立即停止，不进 Exp2/3
- Task #65 Stage 3 baseline 复现仍是 open work，应优先重启

---

## 8. 论文影响

- Task #66 否证进一步强化 P5 paper 的 "negative results" 节
- 5 个连续否证可支撑 "geometric inductive bias is unnecessary for sequential recommendation" 主张
- 下一步应在 baseline 复现基础上探索**非几何**改进方向
