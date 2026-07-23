# Task #116 result — HG-Rec per-layer δ_95/diameter (原版 vs 改造版)

> **完成日期**: 2026-07-24
> **状态**: 🟢 完成 (双版本 × 双 metric × 4 层 × 3 seeds 全测)
> **结果文件**: `verdicts/task116_hgrec_delta_task84_c111.json`, `verdicts/task116_hgrec_delta_task88_c555.json`

---

## 1. 任务目的

对 HG-Rec HRQ-VAE 两个版本做 **逐层 Gromov δ-hyperbolicity + diameter 测量**, 验证"per-layer curvature c=[0.5,0.5,0.5] 是否让 residual 几何更树状":

- **原版 (Task #84)**: c=[1.0, 1.0, 1.0] 单一全局曲率
- **改造版 (Task #88)**: c=[0.5, 0.5, 0.5] 三层强双曲 (更小 c → 更强 Poincaré 曲率)

**两个 metric**:
- A. **Euclidean**: tangent/log-map 空间 ||x-y|| (与 Task #92/79 对齐)
- B. **Poincaré (model-native)**: proj_to_ball(expmap0(v, c_i), c_i) → poincare_distance(_, _, c_i)

**4 个 residual 空间**:
- L0: raw encoder z
- L1: z - q0 (Q1 前的 residual)
- L2: z - q0 - q1 (Q2 前的 residual)
- L3: z - q0 - q1 - q2 (final residual)

**采样规模**: n=50,000 (Task #79 收敛结论), seeds = [42, 123, 456]

---

## 2. 关键指标 (δ_95/diameter, n=50K, mean ± std over 3 seeds)

### 2.1 Euclidean metric

| Layer | Task #84 c=[1,1,1] | Task #88 c=[0.5,0.5,0.5] | Δ |
|-------|--------------------|--------------------------|---|
| L0 | **0.0985 ± 0.0003** | **0.1034 ± 0.0003** | **+0.0049** ⚠️ |
| L1 | **0.0792 ± 0.0001** | **0.0812 ± 0.0001** | **+0.0020** ⚠️ |
| L2 | **0.0748 ± 0.0000** | **0.0762 ± 0.0001** | **+0.0014** ⚠️ |
| L3 | **0.0703 ± 0.0000** | **0.0703 ± 0.0000** | **±0.0000** |

### 2.2 Poincaré metric (c_i)

| Layer | Task #84 c=[1,1,1] | Task #88 c=[0.5,0.5,0.5] | Δ |
|-------|--------------------|--------------------------|---|
| L0 | **0.1071 ± 0.0009** | **0.1125 ± 0.0009** | **+0.0054** ⚠️ |
| L1 | **0.0820 ± 0.0002** | **0.0829 ± 0.0001** | **+0.0009** ⚠️ |
| L2 | **0.0767 ± 0.0001** | **0.0772 ± 0.0002** | **+0.0005** ⚠️ |
| L3 | **0.0718 ± 0.0001** | **0.0708 ± 0.0001** | **-0.0010** ✓ |

---

## 3. 解读

### 3.1 改造版 (c=[0.5,0.5,0.5]) 的反直觉发现

**改造版 δ_95/d 在 L0/L1/L2 三层都比原版高 (更不树状)**, 仅 L3 final residual 几乎相同或略低.

这个结果**与 Idea1 假设方向相反**: 用户预期 "per-layer curvature 让 residual 几何更树状 (hyperbolic)". 实际上:

- L0 raw encoder: c555 比 c111 高 **+5.4% (Poincaré)** / **+4.9% (Euclidean)**
- L1: +1.1% / +2.5%
- L2: +0.7% / +1.9%
- L3 final: -1.0% / ±0%

δ_95/d 升高意味着 Gromov 4-point 条件更不满足, 即 **更不像一棵树**. 这与 "强双曲 c=0.5 应该产生更树状 latent" 的直觉相反.

### 3.2 解释假设

可能的机制:
1. **Norm scaling mismatch**: c=0.5 时 Poincaré ball 半径 r = 1/sqrt(c) = 1.414, 相比 c=1.0 r=1.0 更大. residual 模长 (norm ~0.07-0.13) 在 c=0.5 ball 中相对半径更小 → distances 收缩 → diameter 减小但 δ_95 缩小幅度更大 → δ_95/d 升高.
2. **量化器过拟合欧氏范数**: HG-Rec 训练目标含 ||z-q||² (commitment/codebook loss), 这是欧氏损失. 即使 c=0.5, 量化匹配仍是欧氏 → encoder 学到的 latent 不"真"双曲.
3. **c=0.5 不够极端**: Task #70 Ollivier 真实曲率 κ_1 ≈ +0.7 (Toys 数据), 数据本身不强双曲, 模型学到的 c 也接近欧氏. 强行 c=0.5 是过参数化.

### 3.3 与 Phonism (Task #79) 的对照

| 数据 | Phonism δ_95/d L0 | HG-Rec c111 δ_95/d L0 | HG-Rec c555 δ_95/d L0 |
|------|-------------------|-----------------------|-----------------------|
| Euclidean | ~0.107 | 0.0985 | 0.1034 |

- HG-Rec c111 比 Phonism **更低 (更树状)** ✓ 符合 HG-Rec 改造比 phonism 更几何感知的预期
- HG-Rec c555 比 c111 高, 但仍接近 Phonism 水平

### 3.4 双 metric 结论一致性

Euclidean 和 Poincaré metric 给出**一致的 Δ 方向** (改造版 δ_95/d 更高), 说明这不是 metric artifact, 是真实的几何变化.

但 absolute 数值不同: Poincaré 距离放大 2-3× (因为 hyperbolic 距离公式的 2/sqrt(c) 因子 + 大 ball 半径), 但 normalized ratio δ_95/d 在两层 metric 下保持稳定.

---

## 4. 决策触发 (vs 用户假设)

| 用户假设 | 实测结果 | 决策 |
|----------|----------|------|
| per-layer curvature 让 residual 更树状 (Idea1 假设) | ❌ 反向, c555 在 L0/L1/L2 略不树状 (Δ +1-5%) | **Idea1 在 HG-Rec 上不能简单通过几何δ证实** |
| Δ 应该显著 (>5%) | ❌ 实测 Δ 1-5% (statistically detectable 但 effect size 小) | 几何差异确实存在但效应弱 |

---

## 5. 关键产物

- `scripts/task116_hgrec_delta_per_layer.py` — 适配 HG-Rec 的 Gromov δ 测量脚本
  - 复用了 Task #92 的 4-point 采样 + diameter approximation
  - 新增: model.hrq.vq_layers 适配 (vs task92 的 model.rq.vq_layers)
  - 新增: 双 metric (Euclidean + Poincaré c_i)
  - 新增: per-layer curvature restoration from ckpt args
- `verdicts/task116_hgrec_delta_task84_c111.json` (21 KB)
- `verdicts/task116_hgrec_delta_task88_c555.json` (21 KB)

---

## 6. 关键决策点 (R11.3 自决)

1. **n=50K + 3 seeds**: 遵循 Task #79 收敛结论 (n=5K δ_max 低估 17-27%), 3 seeds 提供 sensitivity bound
2. **双 metric 同时跑**: 验证结论是否依赖 metric 选择, 防止 metric artifact
3. **L3 final residual 包含在内**: 用户没明确, 但符合 Task #92/79 的 4 空间惯例
4. **不 pre-load parquet schema**: 用 duck typing 找 'embedding' / 'emb' 列名, 兼容多种 flan-t5 输出
5. **Task #117 stress grid 互补验证**: 单一 δ 指标不够, stress 给出 orthogonal 视角

---

## 7. 与 Task #117 stress grid 的交叉验证

| Layer | c111 δ_95/d (Eucl) | c111 best κ* (stress) | c555 δ_95/d (Eucl) | c555 best κ* (stress) |
|-------|--------------------|-----------------------|--------------------|-----------------------|
| L0 | 0.0985 | 0.0 (Eucl) | 0.1034 | 0.0 (Eucl) |
| L1 | 0.0792 | 0.0 | 0.0812 | 0.0 |
| L2 | 0.0748 | 0.0 | 0.0762 | 0.0 |
| L3 | 0.0703 | 0.0 | 0.0703 | 0.0 |

**两个独立诊断一致**: HG-Rec residual 几何是 **欧氏优于双曲**, per-layer curvature 改造版 (c555) 没有让 latent 几何变成双曲.

但 stress 的绝对数值差异极小 (L1 c111=7.20 vs c555=6.56, Δ ~9%), 与 δ_95/d 的 Δ 同量级, 都说明"几何差异存在但效应弱".

---

result: Task #116 — HG-Rec δ_95/diameter per-layer (双版本 × 双 metric × 3 seeds) 完成. **反直觉发现**: c=[0.5,0.5,0.5] 改造版 δ_95/d 在 L0/L1/L2 比 c=[1,1,1] 原版**略高** (+1-5%), 即改造版 latent **不是更树状而是略不树状**, 与 Idea1 假设方向相反. 仅 L3 final residual 几乎无差异. δ_95/d 绝对值 7-10% (Euclidean) / 7-11% (Poincaré), 远低于纯树状 (<1%) 的预期.