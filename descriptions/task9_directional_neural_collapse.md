# Task 26 (Idea 2): Directional Neural Collapse — 方向性方差分解

> **状态**: 🟡 backlog

> **目的**：构造"任务相关方向" (品类标签 / co-purchase 聚类)，看第一层残差是不是"独裁抓错了对象"——是抓了任务相关方差，还是抓了 nuisance 方差？
> **依赖**：Stage 2 SID tensor 的 `r_lst` + 任务相关标签 (需查 GRID 数据是否有 category metadata)
> **环境**：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`
> **目标仓库**：`/fs04/ar57/wenyu/GeneRec/GRID`

---

## 记号约定

- `r_l`: 第 l 层量化前残差 (来自 Stage 2 RQ/AQ 的输出)
- `q_l`: 第 l 层量化输出
- `k ∈ {1..K}`: 类别标签 (品类 / co-purchase 聚类)
- `μ_k = E[r_l | k]`: 类均值
- `μ_G`: 全局均值
- `S_B = Σ_k (n_k / N) (μ_k - μ_G)(μ_k - μ_G)ᵀ`: 类间散度矩阵

**任务相关子空间** `U ∈ ℝ^{d×m}`: 取 `S_B` 的 top-m 特征向量 (m 通常取 10-50)
**正交补** `U_⊥`: 任务无关子空间

---

## 现象 1 (核心分解): 逐层残差方差的方向分解

### 测什么

逐层算：

$$\rho_l^{task} = \frac{\mathbb{E}\big[\|U^\top r_l\|^2\big]}{\mathbb{E}\big[\|r_l\|^2\big]}, \quad \rho_l^{nuis} = 1 - \rho_l^{task}$$

并对 `q_1` (L1 量化输出) 算 `ρ^{task}(q_1)`：

$$\rho^{task}(q_1) = \frac{\mathbb{E}\big[\|U^\top q_1\|^2\big]}{\mathbb{E}\big[\|q_1\|^2\big]}$$

### 要看到的现象 (三种结局)

- **(a)** `ρ^{task}(q_1)` 高 (e.g., > 0.7) → L1 抓的主要是分离方向 → 坐实降级叙事，且升级为"L1 抓的方差恰好任务相关"
- **(b)** `ρ^{task}(q_1)` 低 (e.g., < 0.3) → **大发现**：第一层独裁抓错了对象，开出"方差预算重定向"新方法线
- **(c)** 中间 → `ρ_l^{task}` vs l 的曲线本身成为新诊断量

### Kill 线

- 若 `ρ_l^{task}` 各层无差异 (e.g., std < 0.05) → 方向不携带层间差异信息，"范数解释一切"彻底坐实
- 若 `ρ^{task}(q_1)` 与 `ρ_1^{task}` 差距很大 → L1 抓的方向 ≠ 残差本身的方向（需要更深入分析）

### 复用

- `result/task18/A_baseline_rqidx.pt` 里的 `r_lst` 和 `q_lst`
- `result/task21_aq_run_info.json` 里的 AQ 对应数据
- 类别标签：需先查 `data/amazon_data/toys/` 是否有 category mapping

### 产物

- `result/idea2_dnc/idea2_phen1_per_layer_task_ratio.json`: per-layer ρ_l^{task}, ρ_l^{nuis}, ρ^{task}(q_l)
- `result/idea2_dnc/idea2_phen1_verdict.md`: (a)/(b)/(c) 判定
- `result/idea2_dnc/per_layer_task_ratio.png`: 跨层 ρ_l^{task} 曲线

### 单成本

- CPU only (含 SVD on 11924 × 2048 矩阵): < 1 h
- 依赖品类标签可用性；若不可用，**fallback** 用 co-purchase 矩阵 K-means (K=20) 构造伪标签

---

## 现象 2 (directional CDNV, 防止 Spearman 陷阱): 归一化后的方向分解

### 测什么

**归一化残差** `r̂_l = r_l / ‖r_l‖` (剥范数) → 重新跑现象 1 的 ρ_l^{task} 计算

**方向性类内-类间方差比**:

$$\text{CDNV}_l^{dir} = \frac{\mathbb{E}_k\big[\text{Var}(U^\top r_l \mid k)\big]}{\mathbb{E}_{k\neq k'}\big[\|U^\top(\mu_k^{(l)} - \mu_{k'}^{(l)})\|^2\big]}$$

(分子: 任务相关方向上的类内方差; 分母: 任务相关方向上的类间距离)

### 要看到的现象

- 若归一化 (`r̂_l`) 后 `ρ_l^{task}` 各层无差异且 CDNV 各层持平 → **方向不携带层间差异信息，"范数解释一切"彻底坐实** (这就是上一轮要求的"缩放到同大小再测"的正式版)
- 若归一化后 L1 的 task 占比仍显著更高 → **范数解释不了全部，方向有独立贡献**

### Kill 线

- 归一化后各层 ρ_l^{task} 差距 < 0.05 且 CDNV 差距 < 0.05 → 方向层间差异被范数吃掉了，判死"DNC 是 first-layer 独裁的独立机制"
- 反之 → 方向有独立贡献，必须保留为机制解释

### 复用

- 复用现象 1 的 `r_lst`，只需额外除以范数
- 复用类别标签

### 产物

- `result/idea2_dnc/idea2_phen2_normalized_task_ratio.json`: 归一化后的 per-layer ρ̂_l^{task}, CDNV_l^{dir}
- `result/idea2_dnc/idea2_phen2_verdict.md`: kill 判定
- `result/idea2_dnc/normalized_cdnv_comparison.png`: 归一化前后对比

### 单成本

- CPU only, < 30 min (复用现象 1)

---

## 完成判定

| 子任务 | pass 条件 | 主线动作 |
|--------|-----------|----------|
| 现象 1 | (a) 或 (c): ρ_l^{task} 跨层有差异且 ρ^{task}(q_1) 揭示明确语义 | ✅ 方向分解机制成立 |
| 现象 1 (b) | ρ^{task}(q_1) < 0.3 → 大发现 | 🎯 开辟"方差预算重定向"方法线 |
| 现象 1 kill | ρ_l^{task} 跨层无差异 | ❌ 范数解释一切 |
| 现象 2 | 归一化后 ρ_l^{task} 仍有差异 | ✅ 方向有独立贡献 |
| 现象 2 fail | 归一化后 ρ_l^{task} 各层持平 | ❌ "范数主导"彻底坐实 |

---

## 执行顺序

1. 加载类别标签 (若无, K-means fallback) → 构造 S_B → 取 U
2. 现象 1: 对每层 r_l / q_l 算 ρ_l^{task}, ρ_l^{nuis} (含 ρ^{task}(q_1))
3. 现象 2: r̂_l = r_l / ‖r_l‖ → 重算 ρ̂_l^{task} + CDNV_l^{dir}