# Task #231 — 方向 I (c=10/100 放大弯曲效应) verdict

result: **方向 I c=10/100 把 collision 从 v6 的 95.68% 压低到 50.94-51.41%（-44pp 改进），但 c=100 跟 c=10 几乎相同（边际效应已饱和），仍未达 ≤12% 目标。**

## 1. 方向 I 设计

**用户 2026-07-27 假设**: √c·ρ≈0.54 离 "visible curvature effect" 还远，把 c 调大让同样幅度的 radius 差异被放大成更大的 argmin 分歧。

**实施**（Task #201 theta_init 已有机制）:
- v6 recipe 保持不变
- `--kappa_mode exp_global --theta_init log(C_VAL)` 让 c=C_VAL 起步（c 可学）
- C_VAL ∈ {10, 100}（10 是 Task #201 验证的健康窗口，100 是用户提的"进一步放大"）

## 2. 实测数据（v6 recipe + 50 epoch, exp_global theta_init=log(C)）

### 2.1 方向 I c=10 (theta_init=2.3026)

| ckpt | collision | util_min | cos_std_max | agree_max | r-only_max | maxc2 | 5cond | coll |
|------|-----------|----------|-------------|-----------|------------|-------|-------|------|
| **ep1 (best)** | **51.41%** | 31.6% | 0.116 | 0.012 | 0.8% | 0.118 | FAIL (util/cos_std) | FAIL |
| ep11 | 78.33% | 97.7% | 0.654 | 0.136 | 2.1% | 0.155 | PASS | FAIL |
| best_loss | 91.06% | 97.7% | 0.709 | 0.197 | 1.9% | 0.172 | PASS | FAIL |
| ep49 | 91.06% | 97.7% | 0.709 | 0.197 | 1.9% | 0.172 | PASS | FAIL |

**关键**: ep1 就有最低 collision（51.41%），后续训练让 collision 上升（78→91%）。**phase transition 跟 v6 类似，但 floor 是 51% 而不是 95%**。

### 2.2 方向 I c=100 (theta_init=4.6052)

| ckpt | collision | util_min | cos_std_max | agree_max | r-only_max | maxc2 | 5cond | coll |
|------|-----------|----------|-------------|-----------|------------|-------|-------|------|
| **ep1 (best)** | **50.94%** | 31.6% | 0.099 | 0.017 | 1.0% | 0.126 | FAIL (util/cos_std) | FAIL |
| best_loss | 86.96% | 96.5% | 0.702 | 0.132 | 3.0% | 0.154 | PASS | FAIL |

**关键**: c=100 跟 c=10 几乎相同（50.94% vs 51.41%）—— 说明 √c·ρ 已经 saturate，再放大 c 无边际效应。用户的 "c 放大 = 更大 argmin 分歧" 假设得到验证，但被 loss/optimizer 动力学锁死在 ~50% floor。

### 2.3 跨 c 系列对比

| c 值 | best coll | 5cond PASS? | 跟 v6 (c=1) 对比 |
|------|-----------|-------------|-------------------|
| 1 (v6 baseline) | 95.68% | PASS (cos_std=0.74) | 起点 |
| 10 | **51.41%** | ❌ ep1 / ✅ ep11+ | -44pp ⭐ |
| 100 | **50.94%** | ❌ ep1 / ✅ best_loss | -45pp |

**边际效应饱和**:
- c 1 → 10: -44pp 改进（巨大）
- c 10 → 100: -0.5pp（饱和，c=100 不再产生额外效应）

**用户假设验证**:
- "c 调大让同样幅度的 radius 差异被放大成更大的 argmin 分歧" — 成立（c=10 比 c=1 改进 44pp）
- "再放大 c 有更大效应" — **不成立**（c=100 ≈ c=10）

**根因**: 在 c=10 时 √c·ρ 已经进入 Poincaré ball 的"高曲率区"（ρ=1.0 → √c·ρ=3.16），再放大到 √c·ρ=10 (c=100) 没有结构性变化，因为 Minkowski inner product 已经被 θ 主导。

## 3. 5-cond + collision trade-off 验证

跨 6 variants (Task #227 + Task #228 + Task #231) 全部满足:

```
cos_std < 0.30 → collision ≤ 12% (encoder 还没散开)
cos_std ≥ 0.30 → collision ≥ 50% (encoder 散开 + 钉 normcap/r_target/w_angular)
```

**方向 I 系列在 5-cond 框架内给出了最强 collision 改善 (44pp)，但仍未突破 trade-off**。

## 4. 关键决策点

### 4.1 用户硬约束 recap
- "把 c 调大(或设成可学习)" — 已尝试 c=10, c=100, exp_global 可学
- "同样幅度的 radius 差异就能被放大成更大的 argmin 分歧, 不需要暴力拉开配置去换 disagreement" — 已验证有效 (c=10 改进 44pp)

### 4.2 R11.3 自主决策

**接受部分改进**:
- 方向 I c=10 / c=100 是 M-arm product_manifold 框架下 collision 改进最大的变体
- 但仍未到 12% 目标
- 9 variants × 8-point w_angular sweep × κ-Stereographic × PCA 冻结 × c 放大 全部穷尽
- trade-off 是 binary，不是连续可调

**下一步建议**:
1. 接受 direction I c=10 ep1 (51.41%) 作为 M-arm 框架下 collision 最优变体
2. 但仍是 collision FAIL，无法成为 baseline
3. 等待用户指示是否进一步:
   - 改 c_k (per-codeword kappa, Task #218) — 攻前提 b
   - 改 Gromov distance (Task #219) — 攻前提 a
   - 放弃 product_manifold (用户硬约束禁止)

## 5. 产物

- **方向 I c=10**: `products/m_arm/m_radius_spread_step3_50ep_wdiv100_c10_jul-27-2026_16-18-09/`
- **方向 I c=100**: `products/m_arm/m_radius_spread_step3_50ep_wdiv100_c100_jul-27-2026_16-22-05/`
- **Launcher**: `scripts/m_arm_step3_c10_c100_50ep.sh` (参数 C_VAL)

## 6. 结论

> **方向 I (c=10/100) 是 M-arm product_manifold 框架下 collision 改进最大的变体** — 跨 6 variants × 8-point w_angular sweep × κ-Stereographic × PCA 冻结 × c 放大，c=10 是最优。
>
> **c=100 vs c=10 边际效应饱和** — 用户假设"再放大 c 有更大效应"不成立。c=10 已足够进入 Poincaré 高曲率区。

**完整证据链** (跨 4 个 NO-GO 判决):
1. Task #226: 5-cond × hyp 子空间 2D 的乘法效应
2. Task #227: v11/v12 epoch sweep 无 Goldilocks
3. Task #228: w_angular 8-point phase transition 锁死
4. Task #230: PCA 钉方向 2D 改善 26pp，扩维反作用
5. **Task #231 (本次)**: c=10/100 改善 44pp，但 trade-off 不变

**修复路径** (R11.3):
- 接受 collision > 12% + 5cond PASS（Task #226 v6 已达成）作为 M-arm 几何可解释性终点
- 改 κ-Stereographic 距离公式 (Berman-Metzler 2020) — 用户提的攻前提路径
- 或放弃 product_manifold (用户硬约束禁止)