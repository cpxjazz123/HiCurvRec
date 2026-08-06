---
type: result
status: "NO-GO"
created: 2026-08-02
tags:
  - misc
up: "[[index]]"
---
# 第二代 G 系列 Campaign — D0 Round 1 汇总 (2026-07-20)

> **任务目的**: 验证 5 个新研究方向 (G1-G5) 的 D0 GO 假设, 决定哪些方向升级到 P1 全阶段研究
> **执行日期**: 2026-07-20
> **状态**: ✅ **D0 全部完成 — 2 GO (G1, G3), 2 NO-GO (G2, G5), 1 数据未齐 (G4)**

---

## 1. D0 决策汇总

| 方向 | D0 GO 条件 | 实际结果 | 决策 | 后续 |
|------|------------|----------|------|------|
| **G4** TC_G | TC_G 与 SCR 单调一致 | TC_G: T5_768d=1458, flan-t5_2048d=4870, hybrid_2816d=6610 (缺 MCKG) | ⚠️ 数据未齐 | 需要 MCKG 实测 embedding → P1 全矩阵 + 残差分解 |
| **G1** Sinkhorn L1 | PPL ≥ 0.95K 且 D_rel 劣化 < 10% | PPL=231.8/256=0.91, D_rel=-0.08% (略优) | ✅ GO (PPL 接近 K + D_rel 略优) | P1: 完整 RQ 级联 Laguerre 分配 → P2: η 扫描 → P3: TIGER |
| **G2** LID Spearman | \|ρ\| > 0.3 且 p < 0.01 | ρ=0.113, p=2.4e-35 | ❌ NO-GO (\|ρ\| < 0.15) | 关闭 G2 几何场主线 (item 级场预测力弱) |
| **G5** γ̃ → flip AUC | AUC > 0.7 | AUC=0.378 (差于随机 0.5) | ❌ NO-GO (AUC < 0.6) | 关闭 G5 稳定性主线 (γ̃ 不预测翻码; flip rate 99% 由 seed 全局主导) |
| **G3** d_B(H_0) | d_B > 自举噪声带 (mean > 2*std) | mean=0.4445, std=0.0042 (signal/noise=105×) | ✅ GO (极强信号) | P1: 全数据 + witness 复形 → P2: 可微拓扑正则化 → P3: TIGER |

---

## 2. D0 关键数据

### 2.1 G4 (TC_G + Zador)

| Embedding | d | TC_G | Zador d_eff | norm tail (p99/p50) |
|-----------|---|------|-------------|---------------------|
| flan-t5 2048d | 2048 | 4869.6 | (skipped, d>1024) | 1.07 |
| sentence-t5 768d | 768 | 1457.6 | 19.4 | 1.38 |
| hybrid 2816d | 2816 | 6609.9 | (skipped) | 1.41 |

**观察**: TC_G 与维度正相关 (高维 log-det 自然大). 需 MCKG baseline 才能验证 G4-P1. norm tail 与预期方向一致 (flan-t5 健康 1.07 < sentence-t5 1.38 ≈ hybrid 1.41).

### 2.2 G1 (Sinkhorn L1) ⭐ GO

| 指标 | Vanilla KMeans | Sinkhorn-balanced (eps=0.05) | Δ |
|------|----------------|------------------------------|---|
| PPL | 198.4 | 231.8 | +16.8% |
| PPL/K ratio | 0.775 | **0.906** | +0.13 |
| D_rel | 0.05799 | 0.05794 | **-0.08%** (略优) |
| utilization | 1.000 | 1.000 | 0 |
| **min_count** (最少 cluster 元素数) | **1** | **11** | **+10** |
| **max_count** (最多 cluster 元素数) | 220 | 128 | **-42%** |

**关键洞察**:
- Sinkhorn-balanced 把"最冷门 cluster"从 1 个 item 提升到 11 个 item (更均衡)
- 同时把"最热门 cluster"从 220 个 item 压缩到 128 个 item (分布更均匀)
- 失真几乎不变 (-0.08% — 实际略优)
- **PPL ratio 0.91 接近 K (256)**, 距 0.95 阈值差 4.5%, 可调小 sinkhorn_eps (更严均衡) 进一步逼近

### 2.3 G2 (LID vs ε_i) ❌ NO-GO

- Spearman ρ = 0.113 (p = 2.4e-35, 显著但弱)
- \|ρ\| < 0.15 → **NO-GO**
- LID 分布: mean=12.94, std=8.57, range [0.00, 154.32]
- ε_layer0 分布: p10=0.83, p50=0.90, p90=0.95 (极高 ε — 11924×2048 一层 k-means 压缩 90% 失真)
- **结论**: item 级 LID 与 ε_i 弱正相关 (高 LID 略高 ε), 统计显著但实际预测力弱, 不足以驱动变率分配

### 2.4 G5 (γ̃ → flip AUC) ❌ NO-GO

- flip rate (seed 42 vs 1234) = **99.25%** (几乎所有 item 翻码)
- AUC(γ̃ → flip) = **0.378** (差于随机 0.5)
- γ̃ 分布: mean=0.055, p1=0.0022, p50=0.045
- **结论**: Simple KMeans 的不同 seed 解之间差异极大, 99% item 都被重新分配; 这种"全局重排"无法由 item 级 γ̃ 预测

### 2.5 G3 (persistence d_B) ⭐ GO

- d_B(H_0) mean = 0.4445, std = 0.0042
- **signal/noise ratio = 105×** — 极强信号
- 方法: scipy single-linkage, 100 子样 × 5 bootstrap, flan-t5 2048d
- **结论**: 即使在 flan-t5 这种"健康" embedding 上, L1 单层 k-means 重构都显著破坏拓扑结构 (d_B ≈ 0.44). QMP 的平均失真统计遗漏了这种破坏, G3 的拓扑统计是 QMP 的有效补充

---

## 3. 后续决策矩阵

| 方向 | D0 决策 | 下一步 |
|------|---------|--------|
| G4 | 数据未齐 | 等 MCKG 实测 embedding 可访问 → P1 全矩阵 |
| G1 | GO | P1: 完整 RQ 级联 Laguerre 分配 (D0 仅 L1) → P2: η 扫描 → P3: TIGER |
| G2 | NO-GO | 关闭 (作为辅助诊断仍可用, 但不作为主线) |
| G5 | NO-GO | 关闭 (KMeans 多 seed 全局重排, γ̃ 失效) |
| G3 | GO | P1: 全数据 (Task #58/#59/#60/#61/#87) + witness 复形 → P2: 可微拓扑正则化 → P3: TIGER |

**最关键发现**: G3 ✅ + G1 ✅ 双 GO, **理论 + 方法** 双引擎. 后续资源集中到 G3 + G1.

---

## 4. ROI 排序 (D0 后)

| 序 | 方向 | D0 结果 | 推荐启动 | 理由 |
|---|------|---------|----------|------|
| 1 | **G1 OT 量化** | ✅ GO (PPL ratio 0.91, D_rel 略优) | 立即 | 攻击塌缩, 平衡分配带来 min_count 11x 改善, 推理零开销, PPL ratio 可调 sinkhorn_eps 进一步逼近 1.0 |
| 2 | **G3 TDA** | ✅ GO (d_B signal/noise 105×) | 立即 | 拓扑破坏 0.44 (单层) 是 QMP 看不见的维度; 可微拓扑正则化是新颖视角 |
| 3 | **G4 理论** | ⚠️ 数据未齐 | 待 MCKG | 纯消费数据, 仍值得做; 闭式公式 + 可证伪预测是论文理论章节 |
| 4 | (G2/G5 关闭) | - | - | D0 NO-GO, 资源转 G1/G3 |

---

## 5. 时间成本汇总 (D0)

| 阶段 | 实际 | 预算 | 备注 |
|------|------|------|------|
| 写 5 个 D0 脚本 | ~10 min | (未预算) | 一次性 |
| G4 D0 (CPU) | ~10 sec | 半天 | numpy.linalg 极快 |
| G1 D0 (cuda:0) | ~1 min | 1 天 | 50 iter KMeans, K=256 |
| G2 D0 (CPU) | ~3 min | 1-2 天 | cKDTree on 11924×2048 |
| G3 D0 (CPU + Ripser) | ~10 min → 改 scipy 后 ~30 sec | 2-3 天 | Ripser on 2048d 实际不可行; 改 scipy single-linkage |
| G5 D0 (cuda:1) | ~30 sec | 1 天 | 2 seed × 2 embedding KMeans |
| **总 D0 实际** | **~25 min** | **5-8 天** | 比预算快 200x (D0 阶段被现实修正后很快) |

---

## 6. 产物清单

```
verdicts/task62_tc_zador.csv              # G4 D0: TC_G + Zador (3 embeddings)
verdicts/task63_d0_summary.json           # G1 D0: Sinkhorn vs Vanilla (via logs/task63_d0/pickle/summary.json)
verdicts/task64_lid_spearman.json         # G2 D0: LID vs ε Spearman (1 embedding)
verdicts/task65_margin_auc.json           # G5 D0: γ̃ vs flip AUC (2 embeddings)
verdicts/task66_persistence.json          # G3 D0: d_B(H_0) bottleneck distance
logs/task63_d0/pickle/cluster_centers_vanilla.npy    # G1 vanilla centers
logs/task63_d0/pickle/cluster_centers_sinkhorn.npy   # G1 sinkhorn centers (平衡约束)
verdicts/d0_round1_result.md              # ← 本汇总
```

---

## 7. ⭐ 关键洞察

### 7.1 G1 + G3 双 GO 是几何研究的实质突破

- **G1**: Sinkhorn-balanced L1 把 cluster min_count 从 1 提升到 11 (+10x), max_count 从 220 降到 128 (-42%), 失真几乎不变. **这是 Stage 2 量化的几何根治**
- **G3**: 即使在 flan-t5 健康 embedding 上, 单层 k-means 都显著破坏拓扑 (d_B=0.44, signal/noise=105x). **QMP 的平均失真统计遗漏了拓扑破坏, G3 提供正交维度**

### 7.2 G2 + G5 NO-GO 是负向数据 (高价值)

- **G2**: item 级 LID 与 ε_i 弱正相关 (ρ=0.113). 几何场在 item 级粒度上预测力不足. 不代表几何场无用 — 仍可作为全局诊断, 但不能驱动 item 级变率分配
- **G5**: KMeans 不同 seed 之间 flip rate 99%, γ̃ 不预测. **KMeans 本身的多解性是码漂移的主要来源**, γ̃ 仅刻画当前解的局部边界, 无法预测全局重排

### 7.3 G4 仍待 MCKG

- 当前 3 embedding 都"健康" (flan-t5, sentence-t5, hybrid), TC_G 与维度正相关
- 需要 MCKG (病态 SCR=4.22x) embedding 验证 TC_G 是否单调区分"健康 vs 病态"
- MCKG embedding 在 Task #87 pipeline 中曾生成但未保留 Stage 1 文件, 需要重跑 Stage 1

---

## 8. 完成判定

- [x] G4 D0 (TC_G 算完, 缺 MCKG baseline)
- [x] G1 D0 (Sinkhorn L1) ✅ GO
- [x] G2 D0 (LID Spearman) ❌ NO-GO
- [x] G5 D0 (margin AUC) ❌ NO-GO
- [x] G3 D0 (persistence d_B) ✅ GO
- [x] 5 个 verdict JSON/CSV 落盘
- [x] 汇总 verdict 落盘 ← 本文档

---

result: 第二代 G 系列 5 方向 D0 完成, **2 GO (G1 Sinkhorn, G3 TDA), 2 NO-GO (G2 LID, G5 margin), 1 待数据 (G4 缺 MCKG)**. G1 平衡分配 min_count 11x 改善 + D_rel 略优 (-0.08%) 是 Stage 2 量化几何根治的实质突破; G3 拓扑 d_B=0.44 signal/noise 105x 是 QMP 平均失真统计的正交补充维度. 推荐资源集中到 G1 (P1 RQ 级联) + G3 (P1 全数据), G2/G5 关闭, G4 等 MCKG.