# Task #81 — Stage 1c Metric 阳性对照 (Positive Control for Phonism Metric Trust)

> **任务目的**: 验证 Task #80 Stage 1c 的 Riemannian GD MDS + Kruskal stress-1 metric 在已知
> 是双曲结构 (κ=-1 嵌入) 的合成数据上能否正确识别 κ ≈ -1, 决定 phonism 真实数据 "κ=0
> 全层最优" 结论是否可被采信。

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #80 (Stage 1c v2) 结论: 在 phonism RQ-VAE 4 层 residual 点云上, **所有层都 κ=0
stress 最低** (L0=13.449 / L1=85.123 / L2=174.343 / L3=332.019), 且 κ≠0 候选 stress 显著
更差 5.58-5.87×。

用户 (2026-07-23) 提出**核心质疑**: 这套 Stage 1c metric 本身是否可信? 如果 metric 在已知是
双曲结构的合成数据上**都不能**识别出负曲率, 那 "phonism 数据是欧氏" 结论就站不住脚 — 是
metric 流程有 bug, 不是数据本身的几何。

这是 Stage 1c metric 的**阳性对照** (positive control) 验证 — 流程层面的可信度问题。

---

## 2. 实验设计

### 合成数据生成 (3 棵已知是双曲结构的树)

| Tree | 层级 | 分支 | 节点数 | 嵌入曲率 κ | 切空间噪声 σ |
|------|------|------|--------|-----------|------------|
| Tree-A | 3 层 | 4 | 1+4+16 = **21** | -1.0 | 0.02 |
| Tree-B | 4 层 | 3 | 1+3+9+27 = **40** | -1.0 | 0.02 |
| Tree-C | 3 层 | 5 | 1+5+25 = **31** | -1.0 | 0.02 |

**生成方法** (Nickel & Kiela 2017 风格, Ganea 2018 Eq. 7):
1. 根节点在 Poincaré ball 原点 (0, 0)
2. 每层节点从父节点的切空间随机方向走固定 step, 通过 Möbius exp map 嵌入到双曲空间
3. step_per_level = [0, 0.40, 0.25, 0.15] (层越深 step 越小 — 树状收缩)
4. 每个节点加切空间高斯噪声 (σ=0.02) 后再 exp map 一次

**正确答案**: 3 棵树的真实曲率都是 **κ = -1**, 嵌入时设定。如果 Stage 1c metric 可信, 应
该识别 κ ≈ -1 (允许小幅偏差, 例如 -0.7 ~ -1.5 都算"识别出负曲率")。

### 复用 Stage 1c metric 流程

```python
from task80_stage1c_true_distortion import compute_true_distortion

for tree in [tree_a, tree_b, tree_c]:
    for kappa in [-2.0, -1.5, -1.0, -0.7, -0.5, -0.3, 0.0]:
        stress = compute_true_distortion(
            tree, kappa=kappa, n_subset=len(tree),
            seed=42, n_iter=200, lr=0.005, d=2, device='cpu'
        )
```

### 决策表

| 情况 | 判定条件 | 决策 |
|------|----------|------|
| A (流程可信) | ≥ 2/3 树 best κ ≤ -0.5 | 维持 phonism κ=0 结论, 写 §10 positive control verification |
| B (部分可信) | 1/3 树 best κ ≤ -0.5 | 流程部分可信, 加跑更多合成配置 (5/10 节点, branching 2/6, 不同 σ) |
| C (流程有 bug) | 0/3 树 best κ ≤ -0.5 (即 3 棵都 κ=0 最优) | metric 有 bug, **task80 verdict 修正**, 需先修流程 |
| D (识别更负曲率) | ≥ 2/3 树 best κ ≤ -1.5 | metric 偏向负, 同样流程 bug, task80 verdict 修正 |

---

## 3. 决策触发 (vs Stage 1c metric 可信度)

| 情况 | 触发条件 | 决策 |
|------|----------|------|
| 流程可信 | ≥ 2/3 树 best κ ∈ [-2, -0.5] | 维持 task80 verdict §9 (phonism 真实数据 κ=0 最优) |
| 流程有 bug | 3/3 树 best κ ∈ {-0.3, 0, +0.5} | task80 verdict §9 失效, 需先修 metric 流程 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 合成数据生成 | ~1 min |
| 3 棵树 × 7 κ × 200 iter | ~5 min (CPU) |
| 输出 JSON + 写 verdict | ~2 min |
| **总计** | **~10 min** |

(单次 evaluate 不到 1s/iter, 200 iter × 7 κ × 3 trees ≈ 4200 次评估 ≈ 5 min)

---

## 5. 风险与缓解

**风险 1**: 合成数据太"完美", metric 可能 trivial 识别。
- 缓解: 加切空间噪声 σ=0.02 (5% step 大小, 不太小)
- 缓解: 选 3 种不同树结构 (覆盖层级 / 分支数变化)

**风险 2**: Stage 1c metric 在小数据上不稳定 (n<300).
- 缓解: n_subset 设为全部点 (21/40/31 都 < 300, 不用 sample)

**风险 3**: Poincaré ball 半径约束 (数据可能接近边界).
- 缓解: step_per_level 选小值, 加上 max_norm projection (在 compute_true_distortion
  内部 project_to_poincare_ball 已处理)

---

## 6. 完成度跟踪

- [ ] descriptions/task81_positive_control_phonism_metric.md 写入
- [ ] scripts/task81_positive_control.py 写入 + py_compile 通过
- [ ] 3 棵树合成数据生成 (Tree-A/B/C)
- [ ] 跑 compute_true_distortion (3 树 × 7 κ)
- [ ] 输出 JSON: verdicts/task81_positive_control_*.json
- [ ] 写 verdict verdicts/task81_positive_control_phonism_metric_result.md
- [ ] 根据情况 A/B/C/D 更新 task80 verdict §10 (positive control 验证段落)
- [ ] 更新 loop.md §16 (登记 + 完成清理)

---

## 7. 引用

- Task #80 Stage 1c v2 metric: `scripts/task80_stage1c_true_distortion.py`
  (Riemannian GD MDS + Kruskal stress-1, Sarkar 2011 / Nickel & Kiela 2017 / Gu 2019)
- Phonism RQ-VAE 残差结构判定: task80 verdict §9
- 双曲树嵌入: Ganea 2018 "Poincaré GloVe" Eq. 7 (Möbius exp map)