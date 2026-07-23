# Task #42 — 排除 "cosine 在双曲空间不准" 混淆因素 (Riemannian 重测 subspace_2)

> **任务目的**: 在 subspace_2_hyperbolic (κ=-5.04) 上用真正的双曲距离 (Poincaré 或 Lorentz) 重跑 task36 的 per-item dense retrieval 协议,验证 task36 报的 WEAK (gap=+0.0160) 是 metric 不匹配还是 subspace_2 本身信号弱

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (CPU-only, ~10 min)

---

## 1. 背景

承接 Task #36: subspace_2_hyperbolic 在 per-item protocol 下 min_cos/max_cos gap=+0.0160 (WEAK),mean_cos/recency_min gap=-0.0003/-0.0005 (NONE)。

**潜在混淆**: cosine similarity 是欧氏空间度量,在负曲率流形 (hyperbolic) 上不直接适用:
- 双曲空间的两点"距离"应该用 Poincaré disk 距离或 Lorentz inner product
- 用 Euclidean cosine 在 hyperbolic 上测 = 把曲面拉直 → 可能高估或低估真实距离
- 这会**直接影响 PM-RQ 融合权重的初始化**:若 subspace_2 实际信号强 (只是 cosine 测不准),则融合权重不应该和 sphere/euclid 等权初始化

**用户原话**: "排除'cosine测双曲空间不准'这个混淆因素,才能对三个子空间的真实价值做出公正判断,这会直接影响后面怎么设计融合权重的初始化."

## 2. 实验设计

**变量**: 仅 distance metric (Euclidean cosine → Riemannian)
**保持不变**:
- 16759 test samples (Task #36 同一集)
- history 截断到最近 30 项
- min_cos / max_cos / topk_mean / mean_cos 4 个 aggregation (省略 last_item_cos, padding 干扰严重)
- subspace_2_hyperbolic (κ=-5.04) item embedding

**实现方案**:
- 加载 `subspace_item[2]` (N=11924, D=64)
- 假设这是 Lorentz model (last coord is time-like, Minkowski 内积)
- 公式: `dist(x, y) = arccosh(-<x, y>_L) / sqrt(|κ|)`,其中 `<x, y>_L = -x_0 y_0 + Σ x_i y_i`
- 转换: `score(query, item) = -dist(query, item)` (距离越小 → 分数越高)
- per-item protocol 保持

**启动命令**:
```bash
python3 scripts/task42_riemannian_subspace2.py
```

## 3. 决策触发 (vs Task #36 Euclidean 结果)

| 双曲 Lorentz gap | 解读 | 后续行动 |
|---|---|---|
| gap ≥ 0.020 (Riemannian) 显著高于 Task #36 (0.016) | 双曲空间确实携带信号,只是 cosine 测错 | ✅ 双曲权重在 PM-RQ 融合中应保留 ≥ 1/3 权重 |
| 0.001 ≤ Riemannian gap ≤ 0.020, 与 Task #36 ±10% | metric 不是核心问题,subspace_2 本身就是 WEAK | 双曲权重可降至 1/6 (或依赖可学习权重自动调) |
| Riemannian gap < 0.001 | 双曲子空间基本无信号,可大幅降低 | 双曲权重可接近 0,优先 sphere+euclid 融合 |

## 4. 预算

| 阶段 | 估算 |
|---|---|
| 数据加载 + test 构造 | ~1 min |
| 4 aggregations × 1 space (Riemannian metric) | ~5 min |
| R@5/R@10 + smoking gun gap | ~2 min |
| verdict | ~2 min |
| **总计** | **~10 min CPU** |

## 5. 风险与缓解

**风险 1**: MCKG embedding 可能不是严格 Lorentz model (time-like coord 不一定在 last dim) → 缓解: 在 verdict 中明确假设,如果不符合 Lorentz 范式,使用 Poincaré disk fallback
**风险 2**: 双曲距离计算比 cosine 慢 ~10× (arccosh) → 缓解: 历史截断到 30 后 batch size 64 可控,实测
**风险 3**: fused_item 不是双曲 (它是 tangent 拼接),所以只对 subspace_2 重测 → 不影响其他三个 space 的结论

## 6. 完成度跟踪

- [ ] 加载 MCKG embedding, 验证 Lorentz 范式
- [ ] 实现 Riemannian per-item score (4 aggregations)
- [ ] R@5/R@10 + smoking gun gap
- [ ] 对比 Task #36 cosine 结果, 输出 verdict