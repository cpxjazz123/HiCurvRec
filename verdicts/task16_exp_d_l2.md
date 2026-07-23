# Task #71 方案 D 执行结果 — 三残差 L2 量化对比

> **任务名**: Task #71 Exp D — r_E/r_H/r_S 残差 L2 量化对比
> **完成日期**: 2026-07-17
> **状态**: ✅ **已完成**
> **Decision**: **R4_CONFIRMED** (三残差 L2 量化结果显著不同 → 继续 E)

---

## 1. 任务目标

用 r_E/r_H/r_S 三个残差在 L2 码本上做量化，对比：
- q_L2 选择一致率（top-1, top-3）
- L2 重建 MSE

**关键发现 vs Task #70**：Task #70 测的是**距离排序保持**（argmin 等价）；本任务测的是**残差向量本身**对量化结果的影响。

---

## 2. 实验设置

- **嵌入**: `item_embeddings.pt` (11924, 768) unit-norm
- **L1 码本**: MiniBatchKMeans (256 clusters, seed=42)
- **L2 码本**: MiniBatchKMeans on L1 残差 (256 clusters, seed=43)
- **三残差**: r_E (Euclidean) / r_H (Poincaré log_map) / r_S (Spherical log_map)
- **L2 距离**: 欧氏距离（统一基线）

---

## 3. 关键结果

### 3.1 L2 重建 MSE（决定性）

| 残差 | MSE | 比值 |
|------|----:|-----:|
| **r_E** | 0.000151 | 1× |
| **r_H** | 0.000313 | **2.07×** |
| **r_S** | 0.002912 | **19.27×** |

**MSE 差异百分比**:
- |E-H|/E: **106.77%**
- |E-S|/E: **1824.99%**
- |H-S|/H: **830.96%**

### 3.2 q_L2 选择一致率

| 对 | top-1 match | top-3 match |
|----|------------:|------------:|
| E vs H | **19.10%** | 1.24% |
| E vs S | 19.41% | 0.18% |
| **H vs S** | **0.62%** | **0.00%** |

### 3.3 码字使用数

| 残差 | unique codewords used / 256 |
|------|---------------------------:|
| r_E | 255 (健康) |
| r_H | **28** (严重欠利用) |
| r_S | 252 (健康) |

---

## 4. 物理解释

### 4.1 三残差量化结果完全不同

**反 Task #70 推论**：
- Task #70 结论：距离排序保持 → argmin 等价
- Task #71 Exp D：**残差向量不同** → L2 量化结果完全不同

**统一解释**：
- Task #70 测的是 L1 阶段 d_E vs d_H 在 r_1 + c_E 上的距离排序
- Task #71 Exp D 测的是 L2 阶段 d_E 在 r_E vs r_H vs r_S 上的距离排序
- 因为 r_E ≠ r_H ≠ r_S（Task #71 Exp A），L2 阶段排序自然不同 → 不同 L2 码字

### 4.2 r_H 码字欠利用（28/256）

r_H 是 Poincaré 切空间向量，norm = d_H(q, r)。在 norm 范围 [0.42, 0.61] 内（Exp A），r_H 集中在 **q 附近的薄壳上**。在欧氏 L2 码本上找 argmin 时，**所有 r_H 都映射到 norm 最接近的几个码字**（即码本中心附近的码字）。

这意味着 r_H 的 L2 量化**实际上是个低分辨率量化**——大部分信息被丢弃。

### 4.3 r_S MSE 比 r_E 大 19 倍

r_S norm 范围 [0, π/2]，平均 π/2 ≈ 1.57（Exp A）。这意味着 r_S 的"基线距离"远大于 r_E（norm 0.96）。当 r_S 被强制投影回欧氏 L2 码本时：
- 实际差异 = ||r_S - q_L2||² ≈ ||r_S||² ≈ 2.47（粗略）
- 而 MSE = 0.0029 是因为大部分样本的 r_S norm 接近 1.5，所以每个分量的 MSE ≈ (1.5)²/768 ≈ 0.0029

**r_S 不适合用欧氏 L2 码本**——它需要球面 L2 码本才能正确量化。

---

## 5. 决策

**R4_CONFIRMED** — 三残差 L2 量化结果显著不同

| 触发条件 | 实测 | 决策 |
|---------|------|------|
| match_EH > 0.99 且 MSE 差异 < 1% | ❌ (match 19%, MSE 107%) | — |
| match < 95% 或 MSE 差异 > 5% | ✅ (match < 20%, MSE > 100%) | **R4_CONFIRMED → 继续 E** |

**q_L2 选择只有 19% 一致**——这是比 Task #70 更强的证据：**三流形几何产生本质不同的量化结果**。

---

## 6. P5 paper 影响

### 6.1 关键新证据

之前的 P5 主张（基于 Task #70）：
> "Geometric inductive bias does not change which codeword is selected"

Task #71 Exp D 给出反证：
> "Geometric inductive bias **changes which L2 codeword is selected in 80% of cases**, and changes L2 MSE by 100-1800%"

**P5 paper 必须修订**：
- 之前：Poincaré ≈ Euclidean (等价)
- 现在：Poincaré ≠ Euclidean（量化结果不同）

### 6.2 新研究方向

如果三流形 L2 量化结果不同，那么：
1. **哪个流形的 L2 重建质量最好？**（r_E MSE 最小，但 r_E 是默认选择）
2. **完整三流形 RQ-VAE 的 R@10 哪个最高？**（Task #71 Exp E）

### 6.3 与 Task #68 的关系

Task #68 = L1 brand side feature，已在 GPU 1 训练。如果 Task #71 Exp E 用完整三流形 RQ-VAE，可以同时验证：
- Task #68 假设（L1 brand side feature 提升 R@10）
- Task #71 假设（不同流形产生不同 R@10）

---

## 7. 产物清单

| 产物 | 路径 |
|------|------|
| 任务脚本 | `scripts/task71_exp_d_l2.py` |
| 结果 JSON | `products/task71/exp_d_l2.json` |
| Verdict | `verdicts/task71_exp_d_l2.md` |

---

## 8. 完成度

- [x] 写 `scripts/task71_exp_d_l2.py`
- [x] 训练 L1 + L2 码本
- [x] 三残差计算 + L2 argmin
- [x] 对比 MSE + match rate
- [x] 决策：R4_CONFIRMED
- [x] 写 verdict

**Task #71 Exp D 完成 — R4_CONFIRMED，三残差 L2 量化完全不同 → 继续 E（待 Task #68 Stage 3 完成后启动）。**

---

## 9. 累计 Task #71 结论

| 方案 | 结论 | 行动 |
|------|------|------|
| **A (形式)** | R1_CONFIRMED (cos < 0.95) | ✅ |
| **B (probe)** | R2_MARGINAL (diff 3.77%, 残差无 brand 信息) | ⚠️ |
| **D (L2)** | R4_CONFIRMED (match 19%, MSE 差异 100%+) | ✅ |

**下一步**: Exp E（完整三流形 RQ-VAE，2 天 GPU）— 待 Task #68 Stage 3 完成后启动。

跳过了 Exp C（聚类），因为：
- Exp D 已经给出更强证据
- 用户建议"D 提到 C 之前"
- Exp C 投入产出比低于 Exp D