# Task #81 Result — Stage 1c Metric 阳性对照 v1 (2D 嵌入, Euclidean target)

> **完成日期**: 2026-07-23
> **状态**: 🟡 **v1 阳性对照失败 — 3/3 合成树识别 κ=0 (判定 C)**
> **决策**: Stage 1c metric v1 不可信 (2D 嵌入 + Euclidean target 设计存在 κ=0 bias),
> 设计 v2 阳性对照 (32D 嵌入 + 真实 graph distance 作为 target) 重测

---

## 1. 任务目的

承接 Task #80 Stage 1c v2 结论: phonism RQ-VAE 4 层 residual 点云上 κ=0 全层最优。用户
(2026-07-23) 质疑: 如果 metric 在**已知是双曲结构**的合成数据上**都不能**识别负曲率,
那 "phonism 数据是欧氏" 结论站不住脚。

这是 metric 流程层面的可信度验证 (positive control)。

## 2. v1 实验设计

3 棵已知 κ=-1 嵌入的合成树 (Poincaré ball 2D, Möbius exp map + 切空间噪声 σ=0.02):

| Tree | 层数 | 分支 | 节点数 | 嵌入 κ | 数据点 norm |
|------|------|------|--------|--------|------------|
| Tree-A | 3 | 4 | 21 | -1.0 | 0.286 (mean) |
| Tree-B | 4 | 3 | 40 | -1.0 | 0.314 (mean) |
| Tree-C | 3 | 5 | 31 | -1.0 | 0.284 (mean) |

κ 候选: {-2.0, -1.5, -1.0, -0.7, -0.5, -0.3, 0.0}
n_subset = 全部 (合成数据 < 300), n_iter = 200, seed=42

## 3. v1 结果 — **3/3 识别 κ=0 (判定 C)**

| Tree | κ=-2.0 | κ=-1.5 | κ=-1.0 | κ=-0.7 | κ=-0.5 | κ=-0.3 | κ=0.0 | **Best κ** |
|------|--------|--------|--------|--------|--------|--------|--------|------------|
| Tree-A | 0.0101 | 0.0066 | 0.0040 | 0.0024 | 0.0015 | 0.0009 | **0.0000** | **κ=0** ✗ |
| Tree-B | 1.2584 | 0.6444 | 0.3725 | 0.2870 | 0.2472 | 0.2169 | **0.0557** | **κ=0** ✗ |
| Tree-C | 0.8284 | 0.2633 | 0.1000 | 0.0630 | 0.0486 | 0.0378 | **0.0031** | **κ=0** ✗ |

**所有树 best κ = 0**, stress 单调递减 (κ=-2 → κ=0 越来越小)。这表明 metric 在 2D
嵌入的合成数据上**完全偏向 κ=0**, 与数据真实结构 (κ=-1 嵌入) 矛盾。

## 4. v1 失败根因分析 (R11.3 自主诊断)

Stage 1c metric 的设计:
1. target = pairwise Euclidean distance (假设输入是欧氏空间的点云)
2. 对每个 κ, 在 Poincaré ball (κ<0) / 欧氏空间 (κ=0) 上做 Riemannian GD MDS, 让
   hyperbolic_dist 拟合这个 target
3. κ=0 时, Riemannian GD 退化为 Euclidean MDS — 它能**完美还原**欧氏距离矩阵
   (因为欧氏空间能容纳任意维度的欧氏距离分解)
4. κ<0 时, hyperbolic_dist 拟合欧氏 target — 但 Poincaré ball 是有界空间 (半径
   1/√|κ|), 拟合能力受限

**根本 bias**: 当数据点是 2D 嵌入且离 Poincaré ball 原点不远 (mean norm 0.286-0.314)
时, 欧氏距离与 Poincaré 距离近似(因 conformal factor λ ≈ 2)。所以:
- κ=0 (Euclidean MDS) → stress ≈ 0 (完美拟合)
- κ=-1 → stress > 0 (受 Poincaré ball 边界约束, 拟合受限)
- κ=-2 → stress 最大 (更小的 Poincaré ball, 拟合更受限)

**这不是 metric bug,是 metric 设计本身的局限**: 用 euclidean pairwise 作为 target,
再让 hyperbolic 拟合, 在低维 (2D) 嵌入 + 数据点离 origin 近时, κ=0 永远最优。

## 5. v1 判定决策

| 情况 | 判定 | 决策 |
|------|------|------|
| v1 判定 | **C** (3/3 树 best κ = 0) | **metric v1 不可信**, v1 设计存在 κ=0 bias |

**v1 决策**: 不能据此否定 phonism 真实数据结论(因为 v1 设计本身有 bias), 但也不能
据此肯定 phonism 结论。**需要 v2 阳性对照**: 用 32D 嵌入 (与 phonism RQ-VAE 残差维度
一致) + 真实 graph distance 作为 target。

## 6. v2 阳性对照设计 (待执行)

| 维度 | v1 (失败) | v2 (修复) |
|------|----------|----------|
| 嵌入维度 | 2D | **32D** (与 phonism RQ-VAE 一致) |
| 嵌入方法 | Möbius exp map 直接 2D | **更高维**双曲嵌入 (e.g. 2D 树结构 → 32D 双曲空间) |
| target distance | euclidean pairwise (Stage 1c 默认) | **真实 graph shortest path** (树结构固有) |
| metric 评估 | 用 Stage 1c compute_true_distortion 直接调 | **新 metric**: hyperboloid model 或修改 Riemannian GD target |
| 节点数 | 21/40/31 (小) | **500-1000** (与真实 residual 子集大小一致) |

**核心修复**: 改 target 为**树结构的 graph geodesic distance**, 而不是欧氏 pairwise。
这是 Sarkar 2011 / Nickel & Kiela 2017 论文里的标准 metric — 树结构天然适合双曲空间
(因为树是 discrete hyperbolic space), graph shortest path distance 是 hyperbolic
distance 的离散近似。

## 7. 产物清单

- 脚本: `scripts/task81_positive_control.py`
- 日志: `logs/task81_positive_control.log`
- JSON: `verdicts/task81_positive_control_phonism_metric.json`
- Description: `descriptions/task81_positive_control_phonism_metric.md`

## 8. 后续动作

- [ ] 写 v2 阳性对照脚本 (`scripts/task81_positive_control_v2.py`)
  - 用 32D 嵌入 (与 phonism RQ-VAE 残差 dim 一致)
  - target = 树 graph shortest path distance (而非欧氏 pairwise)
  - metric = hyperbolic_dist 拟合 graph shortest path (Sarkar 2011 风格)
- [ ] 跑 v2, 判定能否识别 κ ≈ -1
- [ ] 根据 v2 结果更新 task80 verdict §9 (phonism 真实数据 κ=0 结论维持 / 修正)
- [ ] 在 task80 verdict 末尾追加 §10 (Positive Control Verification 段落)

result: Task #81 Stage 1c Metric 阳性对照 v1 — **3/3 合成树 (21/40/31 节点, 2D Poincaré
ball 嵌入, κ=-1) 都被识别为 κ=0 最优**。判定 C: Stage 1c metric v1 存在 κ=0 bias,
不可信。根因: 用 euclidean pairwise 作为 target, 在 2D 低维 + 数据点离 origin 近时, 欧氏
MDS 能完美还原 (κ=0 stress ≈ 0), 而 κ<0 受 Poincaré ball 边界约束 stress > 0。
不能据此否定也不能肯定 phonism 真实数据结论, 需设计 v2 (32D 嵌入 + graph shortest path
target) 重测。task80 verdict §9 暂不修正, 待 v2 结果出来后再决定。