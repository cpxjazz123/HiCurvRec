# Issue #213 Equal-K Codebook Geometry Audit: PASS

日期: 2026-08-08
Issue: #213 (work_items/81)
状态: **6 phases 全部完成, G(E_0) ≠ G(E_1) ≠ G(E_2) 强验证**

---

## 1. 背景

之前 Issue #210/#211 已验证:
- κ heterogeneity 1.49 (v15) vs 0.017 (equal) = 87x 差异
- K 是 κ heterogeneity 主因 (限于 K 顺序 + 残差累积 + REC_LAYER_W 协同)

**未解问题**: 这是否意味着"三层 codebook geometry 完全不同"是个 misnomer? 即如果 K 一致, 三层 codebook 几何是否仍显著不同?

**Issue #213 目标**: 在 K=(128,128,128) controlled 条件下, 检查 G(E_0) ?= G(E_1) ?= G(E_2)。

---

## 2. Phase A 准备

**ckpt**: `/home/wlia0047/ar57/.../taskA_stage2_equal128/hrqvae_kappa_sync.ckpt`
- num_emb_list = [128, 128, 128]
- final_kappas = [0.4509, 0.4670, 0.4681] (三层 κ 几乎一致, diff=0.017)
- 1000 epochs, single-card bs=1024, Stage1 hyp_v2

---

## 3. Phase B: Codebook Health Check

| 层 | K | util | dead_ratio | entropy | entropy_max | max_share |
|---|---|---|---|---|---|---|
| L0 | 128 | **1.000** | **0.000** | 4.779 | 4.852 | 0.0167 |
| L1 | 128 | **1.000** | **0.000** | 4.807 | 4.852 | 0.0172 |
| L2 | 128 | **1.000** | **0.000** | 4.833 | 4.852 | 0.0124 |

**判定**: 三层完美训练, **无 collapse**, 可信的几何分析。

---

## 4. Phase C: Global Geometry (pairwise distance)

| 层 | mean | std | q50 | q05 | q95 |
|---|---|---|---|---|---|
| L0 | **0.1946** | 0.0385 | 0.1943 | 0.1394 | 0.2507 |
| L1 | 0.1060 | 0.0146 | 0.1056 | 0.0846 | 0.1283 |
| L2 | 0.0839 | 0.0096 | 0.0834 | 0.0697 | 0.0986 |

**Pairwise distance 显著差异 (即使 K 一致)**:

| 比较 | KS stat | p-value | Wasserstein |
|---|---|---|---|
| L0 vs L1 | 0.9133 | 0.00e+00 | 0.0886 |
| L1 vs L2 | 0.6361 | 0.00e+00 | 0.0221 |
| L0 vs L2 | 0.9824 | 0.00e+00 | 0.1107 |

**强结论**: L0 距离 L2 多达 2.3 倍 (0.19 vs 0.08), Wasserstein 0.11. **不是 K cardinality 解释** (K 都 128).

---

## 5. Phase D: Local Geometry (NN / 2NN / margin / density)

| 层 | nn_mean | nn_median | margin_mean | density_5 | density_10 |
|---|---|---|---|---|---|
| L0 | **0.1044** | 0.1034 | 1.1150 | 8.47 | 6.85 |
| L1 | 0.0767 | 0.0758 | 1.0409 | 12.46 | 10.41 |
| L2 | 0.0641 | 0.0638 | 1.0336 | 14.96 | 12.71 |

**观察**:
- L0: 局部结构最松散 (margin 1.12, density 8.5)
- L1: 中等 (margin 1.04, density 12.5)
- L2: 局部最密 (margin 1.03, density 15.0)

---

## 6. Phase E: Radial Geometry (||e||_2)

| 层 | mean | std | q50 | q05 | q95 |
|---|---|---|---|---|---|
| L0 | **0.1389** | 0.0265 | 0.1369 | 0.0990 | 0.1842 |
| L1 | 0.0747 | 0.0105 | 0.0743 | 0.0590 | 0.0917 |
| L2 | 0.0592 | 0.0056 | 0.0589 | 0.0500 | 0.0690 |

**判定**:

| 比较 | KS stat | p-value | Wasserstein |
|---|---|---|---|
| L0 vs L1 | 0.9219 | 9.67e-59 | 0.0641 |
| L1 vs L2 | 0.7031 | 1.20e-30 | 0.0155 |
| L0 vs L2 | 0.9844 | 1.13e-71 | 0.0796 |

**含义**: L0 码字在 encoder 输出空间, L2 码字在深层精细残差空间. 函数 funnel 模式.

---

## 7. Phase F: Spectral / Anisotropy

| 层 | top1 | top5 | top10 | eff_rank | spectral_entropy | cond |
|---|---|---|---|---|---|---|
| L0 | **0.2054** | 0.5876 | 0.7852 | **15.65** | 0.7936 | 5844 |
| L1 | 0.0836 | 0.3879 | 0.5898 | 22.17 | 0.8941 | 4735 |
| L2 | 0.0633 | 0.3285 | 0.5263 | **23.64** | 0.9126 | 4602 |

**含义**:
- L0: top1 占 20.5% (强方向性), effective rank 仅 15.65 (低有效维度)
- L2: top1 仅 6.3% (各向同性), effective rank 23.64 (高有效维度)
- 三层各向异性 / 有效维度 / 谱形状 **完全不同**

---

## 8. Main Table

| Metric | L0 | L1 | L2 |
|---|---:|---:|---:|
| Pairwise dist mean | 0.1946 | 0.1060 | 0.0839 |
| Pairwise dist std | 0.0385 | 0.0146 | 0.0096 |
| NN dist mean | 0.1044 | 0.0767 | 0.0641 |
| NN margin | 1.1150 | 1.0409 | 1.0336 |
| Local density (k=5) | 8.47 | 12.46 | 14.96 |
| Codeword norm mean | 0.1389 | 0.0747 | 0.0592 |
| Effective rank | 15.65 | 22.17 | 23.64 |
| Spectral entropy | 0.7936 | 0.8941 | 0.9126 |
| Utilization | 1.000 | 1.000 | 1.000 |
| Assignment entropy | 4.779 | 4.807 | 4.833 |

---

## 9. Gate 评估

| Gate | 判定 | 结果 |
|---|---|---|
| **Gate 1** | 三层 codebook 无 collapse | ✅ PASS (util=1.0, dead=0) |
| **Gate 2** | pairwise distance 显著异质 (KS p<1e-3 + Wasserstein) | ✅ PASS (KS 0.64-0.98, W 0.022-0.111) |
| **Gate 3** | NN / margin / density 显著异质 | ✅ PASS (margin 1.12/1.04/1.03, density 8.5/12.5/15.0) |
| **Gate 4** | radial / effective rank / spectral entropy 显著异质 | ✅ PASS (norm 0.14/0.07/0.06, eff_rank 15.6/22.2/23.6) |

**4/4 GATE PASS**.

---

## 10. 关键结论

### 10.1 强结论: RQ depth 驱动 codebook geometry 异质

```
G(E_0) ≠ G(E_1) ≠ G(E_2)
```

即使 K cardinality 完全消除 (K=128 全部), 三层 codebook 在 4 个几何维度上仍然显著不同:

1. **Pairwise distance**: L0/L2 差异 2.3 倍
2. **Local density**: L0/L2 差异 1.76 倍 (密度)
3. **Radial spread**: L0/L2 差异 2.35 倍 (norm)
4. **Effective dimensionality**: L0/L2 差异 1.51 倍 (eff_rank)

### 10.2 物理直觉 — Funnel 结构

- L0 收到 encoder 直接输出 → 残差最大 → codeword 空间大 (norm=0.14, dist=0.19)
- L1 收到 L0 残差 → 收窄 (norm=0.07, dist=0.11)
- L2 收到 L0+L1 残差 → 最小 (norm=0.06, dist=0.08)

**等于一个 funnel 结构**: 浅层捕获 broad semantic, 深层捕获 fine-grained detail.

### 10.3 对曲率框架的指导

Issue #210 验证 κ heterogeneity 主要由 K cardinality 驱动 (在 K 不一致时).
Issue #213 验证 codebook geometry heterogeneity **由 RQ depth 驱动** (即使 K 一致).

**推论**:
- 曲率异质设计有 **两层独立支撑**:
  1. K cardinality 驱动 κ (Issue #210)
  2. RQ depth 驱动 codebook geometry (Issue #213, **本 Issue**)
- 即使 K 全均匀, 仍可基于 geometry 异质性设计 **per-layer type-specific curvature**
- 这进一步支持 "三层码本不应共享同一曲率" 的论文动机

### 10.4 对 v15 capmatch 的解释

v15 K=(64,128,256) 递增 + capmatch:
- K 异质 → κ 异质 (Issue #210)
- RQ depth → geometry 异质 (Issue #213)
- 两者协同 → 最佳 SID 链路 (test R@10=0.1080)

但 equal128 K=(128,128,128) 也具有 codebook geometry 异质 (本 Issue), 但 κ 几乎一致 (0.45/0.47/0.47). 后续 Stage3 训练 (Phase D) 将验证 κ 异质 vs geometry 异质何者对端到端性能更重要.

---

## 11. NO-GO 检查

| NO-GO 条件 | 实际 |
|---|---|
| 1. 三层 pairwise distribution 重合 | ❌ (L0/L2 KS=0.98) |
| 2. NN distance 一致 | ❌ (L0/L2 0.104 vs 0.064) |
| 3. radial distribution 一致 | ❌ (L0/L2 norm 0.14 vs 0.06) |
| 4. spectrum/eff_rank 一致 | ❌ (L0/L2 eff_rank 15 vs 24) |
| 5. 仅显著 p 但 effect size 极小 | ❌ (Wasserstein 0.02-0.11) |
| 6. 差异来自 collapse | ❌ (util=1.0, dead=0) |

**全部 NO-GO 条件未触发** → 强 GO.

---

## 12. 文件清单

- ckpt: `taskA/_history/issue210_equal_codebook/taskA_stage2_equal128/hrqvae_kappa_sync.ckpt`
- summary JSON: `taskA/_history/issue213_equal_k_geometry_audit/codebook_geometry_summary.json`
- 5 CSV: `taskA/_history/issue213_equal_k_geometry_audit/{pairwise_distance,nearest_neighbor,radial,spectral,statistical_tests}_stats.csv`
- 4 figures: `taskA/_history/issue213_equal_k_geometry_audit/fig{1,2,3,4}_*.png`
- verdict: 本文件

---

## 13. DECOR BAN

Issue #213 严格在曲率框架内推进, **不引入任何 DECOR 机制**.

---

## 14. 后续

- 写论文段落: "即使 K 一致, RQ depth 仍驱动 codebook geometry 异质 → 支持 per-layer curvature"
- 探索: equal128 SID + Stage3 (Phase D) → 验证 κ 异质缺失 (但 geometry 异质保留) 时代价
- 探索: K=(128,128,128) + per-layer type-specific curvature (no K-driven κ) → 验证纯 geometry-driven 曲率
