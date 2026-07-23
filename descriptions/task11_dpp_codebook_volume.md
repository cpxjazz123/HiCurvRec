# Task 28 (Idea 4): DPP 码本体积 (碰撞的几何成因)

> **状态**: 🟡 backlog

> **目的**：把码本视为点集，测其**归一化 Gram log-volume** 与碰撞率的关系 —— 体积小是否就是碰撞高的成因？
> **依赖**：Stage 2 SID tensor 及其 `codebooks` 列表 (≥6 算法: A/B/C/AQ/HRQ/WF)
> **环境**：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`
> **目标仓库**：`/fs04/ar57/wenyu/GeneRec/GRID`

---

## 记号约定

- `C_l = {c_j^(l)}_{j=1}^{M_l}`: 第 l 层码本
- `ĉ_j = c_j / ‖c_j‖`: 单位化码字 (逐码字, 剥离尺度——避免范数混杂)
- `C̃_l = [ĉ_1, ..., ĉ_{M_l}]`: 单位化码字矩阵 (M_l × d)
- **归一化 Gram**:

$$G_l = \tilde{C}_l^\top \tilde{C}_l + \epsilon I, \quad \epsilon \text{小常数防 log(0)}$$

- **log-volume**:

$$V_l = \log\det(G_l)$$

- **总体积**: `V = Σ_l V_l`
- **利用率熵**: `H_l^{util} = -Σ_j p_j \log p_j`, `p_j` = 码字 j 被选频率

---

## 现象 1 (体积-碰撞相关, 总闸): V vs 碰撞率

### 测什么

对每套 checkpoint 每层码本, 算 `V_l`, 总体积 `V = Σ_l V_l`。

对全部已有算法 (A/B/C/AQ/HRQ/WF, ≥ 6 个数据点) 画 `V` vs 碰撞率散点图, 算 Spearman 相关。

### 要看到的现象

- **强负相关** (Spearman ρ < -0.6, 体积小 → 碰撞高)
- GSRQ (碰撞 0.65) 应该是体积最小的之一

### Kill 线

- 无相关 (|ρ| < 0.3) 或正相关 → 码字扎堆不是碰撞成因, 判死 DPP 正则这条方法线

### 复用

- `result/task18/{A,B,C}_baseline_rqidx.pt` 里的 `codebooks`
- `result/task21_*.pt` (AQ)
- `result/task20_*.pt` (HRQ)
- `result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt` (WF)

碰撞率数据: task17 已测 (A=0.16, B=?, C=0.65)

### 产物

- `result/idea4_dpp/idea4_phen1_volume_vs_collision.json`: per-algorithm V_l, V_total, collision_rate
- `result/idea4_dpp/idea4_phen1_verdict.md`: Spearman ρ + kill 判定
- `result/idea4_dpp/volume_vs_collision_scatter.png`: 6+ 算法散点

### 单成本

- CPU only (涉及 Gram 行列式 on 256×256 矩阵): < 5 min
- **核心改进**: 这次有 ≥ 6 个数据点, Spearman 才真正有意义 (不再是 3 点陷阱)

---

## 现象 2 (区分"体积"与"利用率"): 控制利用率的偏相关

### 测什么

同时算利用率熵 `H_l^{util} = -Σ_j p_j \log p_j`, 然后算**偏相关**:

$$\rho_{partial}(V, \text{collision} \mid H^{util})$$

即在控制 `H_l^{util}` 后, V 与 collision 的偏相关系数

### 要看到的现象

- 控制利用率后, 体积与碰撞的偏相关仍显著 (|ρ_partial| > 0.4) → 体积是**独立于利用率**的信息, DPP 抓到了现有指标抓不到的东西
- 反之 → V 只是利用率的换皮, 无增量价值

### Kill 线

- 偏相关消失 (|ρ_partial| < 0.2) → V 与 H_util 完全共线, DPP 正则不提供新信号

### 复用

- 复用现象 1 的 V_l
- `p_j` 从 SID tensor 的 idx_lst 统计: `p_j = count(idx == j) / N`
- 复用碰撞率

### 产物

- `result/idea4_dpp/idea4_phen2_partial_correlation.json`: per-algorithm H_util, ρ_partial
- `result/idea4_dpp/idea4_phen2_verdict.md`: ρ_partial + kill 判定
- `result/idea4_dpp/util_vs_volume_vs_collision.png`: 3D scatter (或两两 scatter)

### 单成本

- CPU only, < 10 min

---

## 完成判定

| 子任务 | pass 条件 | 主线动作 |
|--------|-----------|----------|
| 现象 1 | Spearman ρ < -0.6 | ✅ DPP 正则有理论依据 |
| 现象 1 fail | |ρ| < 0.3 | ❌ 码字扎堆不是碰撞成因 |
| 现象 2 | |ρ_partial| > 0.4 | ✅ 体积是独立信号 |
| 现象 2 fail | |ρ_partial| < 0.2 | ❌ V 只是利用率的换皮 |

---

## 执行顺序

1. 现象 1 (~5 min): 加载 6 算法 codebooks → 算 V_l → 画 V vs collision 散点 → Spearman
2. 现象 2 (~10 min): 复用 V_l + 计算 H_util + 偏相关