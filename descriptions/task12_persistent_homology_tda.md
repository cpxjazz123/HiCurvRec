# Task 29 (Idea 5): 持续同调 (TDA) — 数据层级深度的拓扑判据

> **状态**: 🟡 backlog

> **目的**：用拓扑数据分析 (TDA) 看 item embedding 的**真实层级深度**——合并尺度有几个显著峰, 决定"3 层里只有 1 层有用"是不是数据本身的特性
> **依赖**：Stage 1 item embedding (`merged_predictions_tensor.pt`)
> **环境**：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys` (需装 `ripser` 或 `gudhi`)
> **目标仓库**：`/fs04/ar57/wenyu/GeneRec/GRID`

---

## 记号约定

- `x_i ∈ ℝ^d`: item embedding (d=2048)
- **Vietoris-Rips filtration**: 对样本点云, 在每个尺度 ε 构造 simplicial complex
- **H0 持续图** `{(b_i, d_i)}`: H0 的诞生-死亡对; H0 中所有 `b_i = 0`, `d_i` = 该连通分量被合并的尺度
- **持续度**: `ℓ_i = d_i - b_i`
- **持续熵**: `E_pers = -Σ_i (ℓ_i / L_tot) log(ℓ_i / L_tot)`, `L_tot = Σ_i ℓ_i`

---

## 现象 1 (H0 多尺度合并结构, 总闸): KDE 峰计数 n_peaks

### 测什么

1. **子采样** (TDA 对大数据敏感): 随机取 ~5000 个 item embedding
2. **白化预处理** (避免范数差异导致距离谱膨胀)
3. **Vietoris-Rips filtration**: 算 H0 持续图 `{(b_i=0, d_i)}`
4. **死亡时刻 KDE**: 对 `{d_i}` 做核密度估计 `f(t) = KDE({d_i})`
5. **峰计数**: `n_peaks` = KDE 显著峰的个数 (峰 = 一次大规模的层级合并事件, 对应一个真实的聚类尺度)
6. **辅助量**: `E_pers` (持续熵, 衡量尺度分布的分散度)

### 要看到的现象与分流

- **`n_peaks = 1`** (合并集中在单一尺度) → **数据真实层级深度 ≈ 1**, 这直接为"为什么 3 层里只有 1 层有用"提供**数据侧**解释: 不是量化算法失败, 是数据本来就没有多尺度结构可编. **同时判死一切"设计更深层级"的方向**
- **`n_peaks ≥ 2`** 且尺度清晰分离 → **层级是真的**, 问题在量化没对准尺度 → 开出新方法线: 把第 l 层码本大小 `M_l` 对准第 l 个拓扑尺度上的聚类数

### Kill 线

- 取决于 n_peaks: 若 n_peaks=1 → 整个 idea 5 降级为"数据本身不支持多层级"
- 若 n_peaks ≥ 2 但尺度间距 < 0.1 → 几乎不分离, 等价于 n_peaks=1

### 复用

- `logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt` (Stage 1 flan-t5-xl embedding, 11924 × 2048)
- 子采样 5000: 随机种子 42, 复用 task20 / task21 子采样脚本

### 产物

- `result/idea5_tda/idea5_phen1_h0_persistence.json`: H0 持续图 + KDE + n_peaks + E_pers
- `result/idea5_tda/idea5_phen1_verdict.md`: n_peaks 数值 + 判定 (1 vs ≥2)
- `result/idea5_tda/h0_persistence_kde.png`: 死亡时刻 KDE 曲线

### 单成本

- CPU only (Ripser 5000 点: < 5 min, Gudhi < 2 min)
- 需先 `pip install ripser` 或 `gudhi`

---

## 现象 2 (尺度-层对齐检验, 仅当 n_peaks ≥ 2): misalign_l

### 测什么

设拓扑测出的合并尺度为 `t_1 < t_2 < ... < t_{n_peaks}`, 对应聚类数 `k_l = (尺度 t_l 时的连通分量数)`.

检验现有 RQ 各层的有效码字数 (`exp(H_l^{util})`) 是否与 `k_l` 对齐:

$$\text{misalign}_l = \frac{|\exp(H_l^{util}) - k_l|}{k_l}$$

### 要看到的现象

- 各层 misalign 都很小 (e.g., < 0.3) → RQ 已经自发对准了拓扑尺度, "对准尺度"没有改进空间
- L1 对准但深层严重错位 → 又一个"深层失败"的独立机制解释 + 可操作的修法

### Kill 线

- 各层 misalign 都 < 0.1 → "对准尺度"没有改进空间, 关闭该方法线
- L1 misalign < 0.2 且 L2+ misalign > 0.5 → 判死深层结构 + 升级新方法线

### 复用

- 复用现象 1 算出的 `k_l`
- 复用 task18 的 `H_l^{util}` (从 idx_lst 统计码字被选频率)

### 产物

- `result/idea5_tda/idea5_phen2_layer_alignment.json`: per-layer exp(H_util), k_l, misalign_l
- `result/idea5_tda/idea5_phen2_verdict.md`: per-layer misalign + 判定
- `result/idea5_tda/layer_topology_alignment.png`: 码字数 vs 拓扑聚类数

### 单成本

- CPU only, < 10 min (复用现象 1 结果)

---

## 完成判定

| n_peaks | 现象 1 判定 | 主线动作 |
|---------|------------|----------|
| n_peaks = 1 | 数据层级 ≈ 1, 量化算法无责 | ✅ 数据侧解释 + 关闭深层方向 |
| n_peaks ≥ 2 | 层级是真, 量化未对准 | 🎯 开"对准拓扑尺度"方法线 |

| 子任务 | pass 条件 | 主线动作 |
|--------|-----------|----------|
| 现象 2 (L1) | misalign_1 < 0.2 | ✅ L1 自发对准 |
| 现象 2 (深层) | misalign_l > 0.5 | ✅ 深层结构独立失败 |
| 现象 2 kill | 全部 misalign < 0.1 | ❌ 对准尺度无改进空间 |

---

## 执行顺序

1. 加载 Stage 1 embedding → 子采样 5000 → 白化 → Ripser → H0 持续图
2. KDE 死亡时刻 → 数 n_peaks
3. **if** n_peaks ≥ 2 → 进现象 2 (对齐检验); **else** → 关闭

---

## 备选 (若 Ripser/Gudhi 不可用)

**Fallback**: 手工合并尺度检测
1. 对子采样 5000 点, 算 pairwise Euclidean 距离 (5000 × 5000 矩阵 = 200MB)
2. 在多个尺度 ε 上聚类 (DBSCAN 或 single-linkage) 数连通分量
3. 画连通分量数 vs ε, 数显著拐点 → n_peaks

这等价于手算 H0, 单连通分量计数