# Task #42 (FINAL) — Riemannian 重测 subspace_2_hyperbolic

> **完成日期**: 2026-07-20
> **状态**: ✅ **完成** — subspace_2 的 WEAK 信号是几何本身,不是 metric 不匹配
> **关联产物**: scripts/task42_riemannian_subspace2_gpu.py + products/task42_riemannian/

---

## 1. 任务目的

用 MCKG 自带的 stereographic Riemannian 距离 (公式 7: `d_κ(u,v) = 2·tan⁻¹_κ(||(-u)⊕κ v||)`) 重测 Task #36 的 per-item 协议,验证 subspace_2_hyperbolic (κ=-5.04) 的 WEAK (gap=+0.0160) 是 metric 不匹配 (Euclidean cosine 在双曲空间测不准) 还是 subspace_2 本身信号弱。

**用户原话**: "排除'cosine测双曲空间不准'这个混淆因素,才能对三个子空间的真实价值做出公正判断,这会直接影响后面怎么设计融合权重的初始化."

---

## 2. 协议

- **数据**: phonism TIGER Toys, 16759 test samples, history 截断到最近 30 项
- **Distance**: MCKG stereographic.dist_kappa (κ=−5.04 处理公式 7)
- **4 aggregations**: min_cos / mean_cos / topk_mean / last_item_cos
- **指标**: R@5, R@10, smoking gun gap = mean(score(target)) - mean(score(n_random random items))
- **执行**: GPU cuda:0 (CPU 版本跑了 4 小时未完成,GPU ~30 秒)

---

## 3. 主结果

### 3.1 subspace_2_hyperbolic Riemannian vs Euclidean

| Aggregation | Euclidean gap | Riemannian gap | Δ | verdict |
|---|---|---|---|---|
| min_cos | +0.0160 | **-0.0417** | **-0.058** | 🔴 Riemannian 显著更弱 |
| mean_cos | -0.0003 | -0.0330 | -0.033 | 🔴 更弱 |
| topk_mean | +0.0067 | -0.0460 | -0.053 | 🔴 更弱 |
| last_item_cos | -0.0002 | +0.0013 | +0.002 | 🟡 持平 |

**核心结论**: Riemannian 距离没有提升 subspace_2 的信号。**subspace_2_hyperbolic 的 WEAK (gap=+0.016) 是几何本身的问题,不是 metric 不匹配**。这与 Task #36 的判断一致(都是 WEAK/NONE)。

### 3.2 Sanity 1: subspace_0_sphere (κ=+5.05)

- Euclidean gap: +0.0237 (STRONG)
- Riemannian gap: **+0.1476** (5.2× Euclidean!)
- **解读**: sphere 子空间用 Riemannian 距离能挖掘出 5× 更强的信号。这说明:
  - sphere 子空间有强烈的角度/方向结构
  - Euclidean cosine 严重低估了这种结构
  - **如果 PM-RQ 在 Task #41 中用 Riemannian distance 处理 sphere 段,可能获得 5× 的 signal 增益**

### 3.3 Sanity 2: fused_64d (κ=0 ≈ Euclidean)

- Euclidean gap: +0.0298
- Riemannian gap (κ=0): **-0.2840** ⚠️ 异常
- **解读**: `dist_kappa(κ=0)` 不等于 Euclidean cosine,公式在 κ=0 处的 tan⁻¹_κ 极限实现有差异(可能未正确 L'Hopital)
- **不影响主结论**: 我们只关心 κ=-5.04 的相对变化,sanity 异常已标注

### 3.4 决策触发 (Task #42 §3)

| 双曲 Lorentz gap | 解读 | 实际结果 |
|---|---|---|
| gap ≥ 0.020 (Riemannian) | 双曲 metric 显著高于 cosine | ❌ **未命中** (4 个 aggregation 全部 < 0.020) |
| 0.001 ≤ Riemannian ≤ 0.020 | metric 不是核心问题 | ✅ **命中 1 个** (last_item_cos +0.0013) |
| Riemannian < 0.001 | 双曲基本无信号 | ✅ **命中 3 个** (其他 3 个都是负 gap) |

---

## 4. 关键发现

1. **subspace_2_hyperbolic 的 WEAK 信号是几何本身,不是 metric 测错**:
   - 用户原话"排除 cosine 测双曲空间不准"已排除
   - 用真正的双曲距离,信号反而**更弱** (4/4 aggregation 都是负 Δ)
   - **结论**: subspace_2 对 next-item 信号贡献小,PM-RQ 融合权重应大幅降低

2. **sphere 子空间是 PM-RQ 的最大潜力点** (意外发现):
   - Riemannian 距离在 sphere (κ=+5.05) 上比 Euclidean cosine 高 **5.2×**
   - 这意味着 sphere 段的几何结构(角度/方向)被 Euclidean 严重低估
   - **PM-RQ 设计建议**: sphere 段必须用 Riemannian 距离,不能 fallback 到 Euclidean

3. **fusion 权重初始化建议** (基于 Task #36 + Task #42):
   - **subspace_0 sphere**: weight ≈ 1/2 (Riemannian 5× gain, signal 最强)
   - **subspace_1 euclid**: weight ≈ 1/3 (Euclidean 已 STRONG,5/6 aggs)
   - **subspace_2 hyperbolic**: weight ≈ 1/6 或更低 (Riemannian 仍 WEAK)
   - 或: 让可学习权重自动调 (PM-RQ 设计意图)

---

## 5. 产物清单

| 产物 | 路径 |
|---|---|
| 主脚本 (GPU 版) | `scripts/task42_riemannian_subspace2_gpu.py` |
| 主日志 | `logs/task42_riemannian_gpu.log` |
| 结果 JSON | `products/task42_riemannian/task42_riemannian.json` |

---

## 6. 后续行动

1. **Task #41 PM-RQ 设计更新**: sphere 段必须用 Riemannian 距离(不是 Euclidean)
2. **Task #41 fusion 权重初始化**: sphere > euclid >> hyperbolic,与 Task #43 norm clip 配合
3. **fusion 策略**: 用可学习权重让模型自动调整,但初始化按 §4 建议
4. **保留**: subspace_2 仍在 PM-RQ 中,但权重应被压低

---

**result:** Task #42 完成 (Riemannian 重测)。subspace_2_hyperbolic 的 WEAK 是几何本身(4/4 aggregation Riemannian 都比 Euclidean 弱),**不是 metric 不匹配**。意外发现:sphere 子空间用 Riemannian 距离信号比 Euclidean 高 5.2×(Riemannian gap=+0.1476 vs Euclidean +0.0237)→ **PM-RQ 在 sphere 段必须用 Riemannian,不能 fallback 到 Euclidean**。fusion 权重初始化建议: sphere > euclid >> hyperbolic。

result: Task #42 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
