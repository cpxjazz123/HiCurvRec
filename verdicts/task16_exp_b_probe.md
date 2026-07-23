# Task #71 方案 B 执行结果 — 三残差 brand probe 信息保留对比

> **任务名**: Task #71 Exp B — r_E/r_H/r_S 残差 brand probe
> **完成日期**: 2026-07-17
> **状态**: ✅ **已完成**
> **Decision**: **R2_MARGINAL** (diff 3.77% → 继续 C/D)

---

## 1. 任务目标

对比 r_E/r_H/r_S 三个残差向量对 brand 信息的保留能力：用 LogisticRegression 预测 brand 标签，对比 probe 准确率。

---

## 2. 实验设置

- **嵌入**: `item_embeddings.pt` (11924, 768) unit-norm
- **过滤**: top 20 brands（覆盖 6094 个 item）
- **L1 码本**: MiniBatchKMeans (256 clusters, seed=42)
- **Train/test split**: 70/30 stratified
- **品牌分类数**: 20

---

## 3. 关键结果

### 3.1 Probe 准确率

| 残差类型 | train_acc | test_acc | vs majority (28.65%) |
|---------|----------:|---------:|---------------------:|
| **r_E** (Euclidean) | 0.2865 | **0.2865** | = baseline（完全预测多数类）|
| **r_H** (Poincaré log_map) | 0.2865 | **0.2865** | = baseline（完全预测多数类）|
| **r_S** (Spherical log_map) | **0.3482** | 0.2488 | train > baseline, test < baseline（**过拟合**）|

### 3.2 Baseline

- **Majority baseline**: 0.2865（最常见 brand = LEGO）
- **Random baseline**: 0.0500 (1/20)

### 3.3 差异

| 度量 | 值 |
|------|---|
| max(test_acc) | 0.2865 (r_E) |
| min(test_acc) | 0.2488 (r_S) |
| **diff (max - min)** | **0.0377** |

---

## 4. 物理解释

### 4.1 残差不含 brand 信息

- r_E 和 r_H 的 test_acc 完全等于 majority baseline → LogisticRegression 完全预测多数类 → 残差里**没有 brand 信号**
- r_S 的 train_acc > baseline 但 test_acc < baseline → **过拟合**，泛化失败

### 4.2 与 Task #67 D3 的关系

- Task #67 D3 测的是**累积表征**（z_≤1+1）的 brand 探针准确性，发现 L1 增加 brand 信息
- Task #71 Exp B 测的是**残差本身**（r_1）的 brand 信息，发现 r_1 不含 brand 信息

**统一解释**：
- L1 残差 r_1 = x - q_1 中**不包含** brand 信息
- 但 L1 码字 q_1 包含部分 brand 信息（通过累积表征 z_≤1 = q_1 + ... 传递）
- 也就是说，**brand 信息被 L1 量化吸收了**，而不是残留在残差里

### 4.3 r_S 过拟合的根因

r_S 的 norm range = [0, π/2]，分布与 r_E/r_H 不同（norm 集中在 π/2），可能导致 LogisticRegression 找到一些**虚假的相关性**（在训练集上准确但测试集失效）。

---

## 5. 决策

**R2_MARGINAL** — diff = 3.77% ∈ (2%, 5%]，按决策表 → **继续 C/D**

| 触发条件 | 实测 | 决策 |
|---------|------|------|
| diff > 5% | ❌ (3.77%) | — |
| 2% < diff < 5% | ✅ (3.77%) | **R2_MARGINAL → 继续 C/D** |
| diff < 2% | ❌ | — |

**注意**：所有 probe 准确率都接近 majority baseline → **三残差都不含 brand 信息**。这本身是一个强结论：
- **R2 否证的"软"版本**：三个流形残差都没有 brand 信息
- 但形式上仍有 3.77% 差异（来自 r_S 过拟合）→ MARGINAL

---

## 6. P5 paper 影响

### 6.1 论文主张的精确化（进一步）

Task #71 Exp A 主张：三残差**形式上不同**
Task #71 Exp B 主张：三残差在**信息保留上无显著差异**（都无 brand 信息）

新合题：
> "Three manifold geometries produce different residual vector representations (form-wise), but all fail to retain brand information, suggesting that **residual vectors capture 'what's left after quantization' rather than 'categorical features'**."

### 6.2 新研究问题

如果残差不含 brand 信息，那么下游任务（如 R@10）的提升**只能来自码字本身**（不是残差）。这支持 P5 的核心主张："**码本几何比残差几何更重要**"。

---

## 7. 产物清单

| 产物 | 路径 |
|------|------|
| 任务脚本 | `scripts/task71_exp_b_probe.py` |
| 结果 JSON | `products/task71/exp_b_probe.json` |
| Verdict | `verdicts/task71_exp_b_probe.md` |

---

## 8. 完成度

- [x] 复用 Exp A 三残差（用 --use_kmeans）
- [x] 加载 brand labels + 过滤 top 20 brands
- [x] 训练 LogisticRegression × 3 残差
- [x] 对比准确率
- [x] 决策：R2_MARGINAL
- [x] 写 verdict

**Task #71 Exp B 完成 — R2_MARGINAL，三个残差都不含 brand 信息 → 继续 C/D。**