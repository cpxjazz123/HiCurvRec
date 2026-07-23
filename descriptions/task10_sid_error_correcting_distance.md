# Task 27 (Idea 3): 纠错码 / 最小距离 (SID 作为信道码)

> **状态**: 🟡 backlog

> **目的**：把 SID 序列视为信道码字，测码字之间的最小距离分布是否构成 e2e 错误的"瓶颈"——距离谱 + 错误距离因果检验
> **依赖**：Stage 2 SID tensor (`merged_predictions_tensor.pt`) + Stage 4 推断错误日志
> **环境**：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`
> **目标仓库**：`/fs04/ar57/wenyu/GeneRec/GRID`

---

## 记号约定

- `SID_i = (z_i^(1), ..., z_i^(L))`: semantic ID
- **汉明距离** `d_H(SID_i, SID_j) = Σ_l 𝟙[z_i^(l) ≠ z_j^(l)]`
- **距离谱** `P(h)`: 所有 (i,j) 对中距离为 h 的占比
- **P_err(h)**: 在 e2e 推断中"生成错误 SID 与真实 SID 距离为 h"的占比

---

## 现象 1 (距离谱): P(h) 分布

### 测什么

$$P(h) = \frac{|\{(i,j): d_H(SID_i, SID_j) = h\}|}{N(N-1)/2}, \quad h \in \{0, 1, ..., L\}$$

**重点看**：
- `P(0)`: SID 重复率 (碰撞率) — 已知 task17 Group C = 0.65 异常高
- `P(1)`: **危险对占比** —— 一字之差就撞上另一合法商品
- `P(2) ~ P(L)`: 健康距离谱

对 A (RQ-VAE baseline) / AQ / HRQ / WF 各自算一遍。

### 要看到的现象

- 健康算法 (A/AQ) `P(1) > 1%` (足够多"近邻对") —— 这是纠错的瓶颈
- GSRQ (C) `P(0)` 极高但 `P(1)` 反而低 → 大量重复但重复之间挤在一起

### Kill 线

- `P(1) < 1%` → 一字之差几乎不可能 (码字离得远, 但也意味着 SID 稀疏, 利用率低) → "距离谱"不是错误来源

### 复用

- `result/task18/A_baseline_rqidx.pt` (A) 的 `idx_lst`
- `result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt` (WF) 的 `idx_lst`
- `result/task21_*.pt` (AQ)
- `result/task20_*.pt` (HRQ)

### 产物

- `result/idea3_ecc/idea3_phen1_distance_spectrum.json`: per-algorithm P(h)
- `result/idea3_ecc/idea3_phen1_verdict.md`: 各算法 P(1) 对比 + kill 判定
- `result/idea3_ecc/distance_spectrum_4algos.png`: 4 算法距离谱叠加图

### 单成本

- CPU only, < 5 min (11924 × 11924 / 2 = 71M 对, 向量化距离计算)

---

## 现象 2 (错误-距离相关, 核心因果检验): lift(1)

### 测什么

从 e2e 推断日志提取生成错误案例。对每个错误, 记录：
- "模型生成的错误 SID" `SID_err`
- "真实目标 SID" `SID_true`
- "它们之间的汉明距离" `d_H(SID_err, SID_true)`

得到错误距离分布 `P_err(h)`，然后算：

$$\text{lift}(1) = \frac{P_{err}(1)}{P(1)}$$

即"生成错误中距离 1 的占比" vs "全 SID 中距离 1 的占比" 之比

### 要看到的现象

- `lift(1) ≫ 1` (例如 > 5) → 错误**不成比例地**落在距离 1 的邻居上 → 证明"码字挨太近"是真实的错误来源
- 不同距离的 lift 值应呈现明显峰值在 h=1

### Kill 线

- `lift(1) ≈ 1` (错误均匀分布, 与距离无关) → 最小距离不是瓶颈, 判死
- 两者都高 (`P(1)` 高 + `lift(1)` 高) → 打开"距离约束 tokenizer 训练"新方法线 (在量化 loss 里加 `max(0, δ_min − d_H)` 型惩罚的松弛版)

### 复用

- 现象 1 算出的 `P(1)`
- e2e 推断错误日志: `logs/inference/runs/task21_aq_s4*/pickle/eval_errors.json` (若已记录错误案例)
- 备用: 用 TIGER 模型在 diag_val 上跑一次 inference, 手动收集错误

### 产物

- `result/idea3_ecc/idea3_phen2_error_distance_lift.json`: per-distance lift(h)
- `result/idea3_ecc/idea3_phen2_verdict.md`: lift(1) 数值 + kill 判定
- `result/idea3_ecc/error_distance_lift.png`: lift(h) 曲线

### 单成本

- 现象 2 需跑 1 次小规模 inference (或在已有 eval log 解析): 1-2 h GPU / CPU-only log 解析 < 30 min
- 若无现成错误日志, **fallback**: 直接生成所有可能的距离 1 邻居 (对每个 SID, 把 4 个位置各替换为其他 3 个值), 检查这些邻居是否确实出现在错误列表中

---

## 完成判定

| 子任务 | pass 条件 | 主线动作 |
|--------|-----------|----------|
| 现象 1 | `P(1) > 1%` 且跨算法对比有差异 | ✅ 进入现象 2 |
| 现象 1 fail | `P(1) < 1%` 所有算法 | ❌ 距离谱不是问题 |
| 现象 2 | `lift(1) > 5` | ✅ "码字挨太近"是错误主因 |
| 现象 2 fail | `lift(1) ≈ 1` | ❌ 最小距离不是瓶颈 |

---

## 执行顺序

1. 现象 1 (~5 min): 加载 4 算法的 SID → 算 P(h) 谱
2. 现象 2 (~1-2 h 或 <30 min if logs available): 跑/解析错误日志 → 算 lift(h)