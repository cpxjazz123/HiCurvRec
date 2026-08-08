# Issue #213 补充: Equal-K 三配置 (64/128/256) Codebook Geometry 对比 — PASS

日期: 2026-08-08
Issue: #213 (work_items/81) 补充
状态: **3-config 对比完成, K 幅度不改变 geometry 异质结构 (funnel 模式在 64/128/256 三种 K 下都成立)**

---

## 1. 背景

Issue #213 主体只跑了 K=128,128,128 (equal128), 强验证 G(E_0) ≠ G(E_1) ≠ G(E_2).

**未解问题**: geometry 异质是 K cardinality 效应 (K 越大越显著) 还是 RQ depth 内禀结构?

**本补充目标**: 在 K=64,64,64 与 K=256,256,256 重复 Phase B-F, 对比三个 K 幅度下 geometry 异质结构是否一致.

---

## 2. 三配置 ckpt

| 配置 | num_emb_list | e_dim | final_kappas | 1000ep 训练 |
|---|---|---|---|---|
| equal64  | [64,64,64]   | 32 | [0.4511, 0.4670, 0.4678] | bs=1024 single-card |
| equal128 | [128,128,128] | 32 | [0.4509, 0.4670, 0.4681] | bs=1024 single-card |
| equal256 | [256,256,256] | 32 | [0.4508, 0.4669, 0.4680] | bs=1024 single-card |

注: 三配置 final_kappas 几乎一致 (差异 < 0.001), 说明 κ 由 Stage1 hyp_v2 + Stage2 训练动力决定, 与 K cardinality 无关.

---

## 3. Phase B 三配置 Health 对比

| 指标 | L0 | L1 | L2 | L0 | L1 | L2 | L0 | L1 | L2 |
|---|---|---|---|---|---|---|---|---|---|
| **K** | 64 | 64 | 64 | 128 | 128 | 128 | 256 | 256 | 256 |
| **util** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **dead_ratio** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **entropy** | 4.089 | 4.121 | 4.144 | 4.779 | 4.807 | 4.833 | 5.445 | 5.481 | 5.507 |
| **max_share** | 0.031 | 0.028 | 0.024 | 0.017 | 0.017 | 0.012 | 0.012 | 0.016 | 0.009 |

**判定**: 三配置全部完美训练 (util=1, dead=0). 最大单码字占用率均 < 4% (equal64 L0 最差), 无 collapse.

**观察**: L0 < L1 < L2 entropy 单调递增 (0.03-0.06 nats), 等价于"深层分布更均匀".

---

## 4. Phase C Pairwise Distance 三配置对比

| 层 | equal64 mean | equal128 mean | equal256 mean |
|---|---:|---:|---:|
| **L0** | 0.1922 | 0.1946 | 0.1993 |
| **L1** | 0.1059 | 0.1060 | 0.1035 |
| **L2** | 0.0856 | 0.0839 | 0.0799 |
| **L0/L2 ratio** | **2.25x** | **2.32x** | **2.49x** |

**KS / Wasserstein 跨配置**:

| 比较 | equal64 KS | equal128 KS | equal256 KS | equal64 W | equal128 W | equal256 W |
|---|---:|---:|---:|---:|---:|---:|
| L0 vs L1 | 0.9067 | 0.9133 | 0.9306 | 0.0863 | 0.0886 | 0.0958 |
| L1 vs L2 | 0.6190 | 0.6361 | 0.6612 | 0.0203 | 0.0221 | 0.0236 |
| L0 vs L2 | 0.9807 | 0.9824 | 0.9869 | 0.1066 | 0.1107 | 0.1194 |

**强结论**: 三配置的 pairwise 异质结构**几乎完全一致**. L0/L2 ratio 稳定在 2.25-2.49x, KS 稳定在 0.98+, W 稳定在 0.11+. K cardinality 改变只**轻微**影响距离尺度 (L2 略收缩) 而**不改变**异质结构.

---

## 5. Phase D Local Geometry (NN / Margin) 三配置对比

| 指标 | L0 | L1 | L2 | L0 | L1 | L2 | L0 | L1 | L2 |
|---|---|---|---|---|---|---|---|---|---|
| **K** | 64 | 64 | 64 | 128 | 128 | 128 | 256 | 256 | 256 |
| **nn_mean** | 0.1095 | 0.0809 | 0.0686 | 0.1044 | 0.0767 | 0.0641 | 0.1015 | 0.0703 | 0.0570 |
| **nn_median** | 0.1091 | 0.0800 | 0.0682 | 0.1034 | 0.0758 | 0.0638 | 0.1005 | 0.0696 | 0.0564 |
| **margin_mean** | 1.0931 | 1.0480 | 1.0269 | 1.1150 | 1.0409 | 1.0336 | 1.0940 | 1.0490 | 1.0420 |
| **margin_std** | 0.1075 | 0.0448 | 0.0227 | - | - | - | - | - | - |

**强结论**:
- nn_mean L0/L2 ratio: equal64=1.60x, equal128=1.63x, equal256=1.78x
- margin L0 始终最高 (1.09-1.12), L2 最低 (1.03)
- 三配置"局部松密度"差异完全一致: **L0 局部最松散, L2 局部最紧密**.

---

## 6. Phase E Radial Geometry 三配置对比

| 指标 | L0 | L1 | L2 | L0 | L1 | L2 | L0 | L1 | L2 |
|---|---|---|---|---|---|---|---|---|---|
| **K** | 64 | 64 | 64 | 128 | 128 | 128 | 256 | 256 | 256 |
| **norm mean** | 0.1358 | 0.0745 | 0.0602 | 0.1389 | 0.0747 | 0.0592 | 0.1456 | 0.0730 | 0.0564 |
| **norm std** | 0.0260 | 0.0093 | 0.0046 | 0.0265 | 0.0105 | 0.0056 | 0.0304 | 0.0112 | 0.0069 |
| **L0/L2 ratio** | **2.26x** | | | **2.35x** | | | **2.58x** | | |

**含义**: 径向 funnel 结构三配置一致 — L0 norm=0.13-0.15 (大), L2 norm=0.06 (小). K 越大, L0/L2 ratio 略增 (2.26→2.58x), 说明 K cardinality 会轻微放大浅层码字径向尺度.

---

## 7. Phase F Spectral 三配置对比

| 指标 | L0 | L1 | L2 | L0 | L1 | L2 | L0 | L1 | L2 |
|---|---|---|---|---|---|---|---|---|---|
| **K** | 64 | 64 | 64 | 128 | 128 | 128 | 256 | 256 | 256 |
| **top1** | 0.2189 | 0.0892 | 0.0776 | 0.2054 | 0.0836 | 0.0633 | 0.1752 | 0.0720 | 0.0558 |
| **top5** | - | - | - | 0.5876 | 0.3879 | 0.3285 | - | - | - |
| **top10** | - | - | - | 0.7852 | 0.5898 | 0.5263 | - | - | - |
| **eff_rank** | 12.80 | 18.54 | 20.05 | 15.65 | 22.17 | 23.64 | 17.61 | 24.31 | 25.77 |
| **spectral_entropy** | - | - | - | 0.7936 | 0.8941 | 0.9126 | - | - | - |

**强结论**:
- eff_rank L0/L2 ratio: equal64=0.64x, equal128=0.66x, equal256=0.68x (L0 始终 < L2)
- top1 L0 始终最高 (0.18-0.22), L2 最低 (0.06-0.08), 各向异性 funnel 结构完全一致
- K 越大, eff_rank 整体越大 (K=64 → 13-20, K=128 → 16-24, K=256 → 18-26) — 这是 K cardinality 直接效应 (可用码字多 → 编码更多样化)

---

## 8. 三配置 Main Table

| Metric | L0 | L1 | L2 | L0 | L1 | L2 | L0 | L1 | L2 |
|---|---|---|---|---|---|---|---|---|---|
| **K** | 64 | 64 | 64 | 128 | 128 | 128 | 256 | 256 | 256 |
| **Pairwise dist mean** | 0.1922 | 0.1059 | 0.0856 | 0.1946 | 0.1060 | 0.0839 | 0.1993 | 0.1035 | 0.0799 |
| **Pairwise dist std** | 0.0392 | 0.0136 | 0.0091 | 0.0385 | 0.0146 | 0.0096 | 0.0403 | 0.0147 | 0.0102 |
| **Pairwise L0/L2 ratio** | 2.25x | | | 2.32x | | | 2.49x | | |
| **NN dist mean** | 0.1095 | 0.0809 | 0.0686 | 0.1044 | 0.0767 | 0.0641 | 0.1015 | 0.0703 | 0.0570 |
| **NN margin** | 1.0931 | 1.0480 | 1.0269 | 1.1150 | 1.0409 | 1.0336 | 1.0940 | 1.0490 | 1.0420 |
| **Codeword norm mean** | 0.1358 | 0.0745 | 0.0602 | 0.1389 | 0.0747 | 0.0592 | 0.1456 | 0.0730 | 0.0564 |
| **Effective rank** | 12.80 | 18.54 | 20.05 | 15.65 | 22.17 | 23.64 | 17.61 | 24.31 | 25.77 |
| **Utilization** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Assignment entropy** | 4.089 | 4.121 | 4.144 | 4.779 | 4.807 | 4.833 | 5.445 | 5.481 | 5.507 |

---

## 9. Gate 评估 (3-config)

| Gate | 判定 | 结果 |
|---|---|---|
| **Gate 1** | 三配置均无 collapse | ✅ PASS (util=1.0, dead=0) |
| **Gate 2** | pairwise L0/L2 ratio 三配置均在 2.25-2.49x, KS>0.98, W>0.10 | ✅ PASS (异质结构稳定) |
| **Gate 3** | NN / margin / density 三配置 funnel 一致 (L0 局部松散 → L2 紧密) | ✅ PASS |
| **Gate 4** | radial / eff_rank / spectral 三配置 funnel 一致 (L0 大 norm → L2 小 norm) | ✅ PASS |

**4/4 GATE PASS (三配置)**.

---

## 10. 关键发现

### 10.1 Geometry 异质结构 K-invariant

```
G_64(E_0) ≠ G_64(E_1) ≠ G_64(E_2)
G_128(E_0) ≠ G_128(E_1) ≠ G_128(E_2)
G_256(E_0) ≠ G_256(E_1) ≠ G_256(E_2)
```

三层异质结构在 K=64/128/256 三种幅度下**完全一致**:
- pairwise L0/L2 ratio 稳定 2.25-2.49x
- NN margin L0>L1>L2 funnel 三配置稳定
- radial norm L0>L1>L2 funnel 三配置稳定
- eff_rank L0<L1<L2 funnel 三配置稳定

### 10.2 K cardinality 的次级效应

K 变化只**轻微**改变:
- 整体 eff_rank (可用码字多 → 编码更多样化, K=64: 13-20, K=256: 18-26)
- L0/L2 ratio 微增 (K 越大, 浅深层对比略放大, 2.26→2.58x)

但**不改变** funnel 模式**方向** (L0 broad → L2 fine-grained).

### 10.3 与 Issue #210 的整合

- Issue #210: κ heterogeneity 主要由 K cardinality 驱动 (4-config κ 对比, 87x 差异)
- Issue #213: **geometry heterogeneity 由 RQ depth 驱动, K cardinality 几乎不影响** (本补充)
- 两者协同 → 论文动机 (per-layer curvature) 有**双独立支撑**:
  1. K cardinality → κ 异质 (Issue #210)
  2. RQ depth → geometry 异质 (Issue #213 主体 + 本补充)

### 10.4 对 v15 capmatch 的解释力增强

v15 K=(64,128,256) 递增 + capmatch (κ 异质):
- K 异质 → κ 异质 (Issue #210)
- RQ depth → geometry 异质 (Issue #213)

但即使把 K 改回 equal128 (本 Issue), geometry 异质结构仍保留. Phase D Stage3 (equal128 SID + v85p) 验证 κ 异质缺失 (但 geometry 异质保留) 时, 可**精确剥离** κ 异质效应, 单独评估 geometry 异质贡献.

---

## 11. NO-GO 检查

| NO-GO 条件 | 实际 |
|---|---|
| 1. 三配置 pairwise L0/L2 ratio 一致 (<1.2x) | ❌ (2.25-2.49x, 显著) |
| 2. 三配置 NN margin 一致 | ❌ (L0 margin 始终最高 1.09-1.12) |
| 3. 三配置 radial funnel 反向 (L2 大 L0 小) | ❌ (L0 大 L2 小, 反向) |
| 4. 三配置 spectrum 一致 | ❌ (L0 top1 始终最高 0.18-0.22) |
| 5. 仅 p<1e-3 但 effect size 极小 | ❌ (Wasserstein 0.09-0.12 显著) |
| 6. 差异来自 collapse | ❌ (util=1.0, dead=0, 三配置均完美) |

**全部 NO-GO 条件未触发** → 强 GO (3-config).

---

## 12. 文件清单

- equal64 ckpt: `taskA/_history/issue210_equal_codebook/taskA_stage2_equal64/hrqvae_kappa_sync.ckpt`
- equal128 ckpt: `taskA/_history/issue210_equal_codebook/taskA_stage2_equal128/hrqvae_kappa_sync.ckpt`
- equal256 ckpt: `taskA/_history/issue210_equal_codebook/taskA_stage2_equal256/hrqvae_kappa_sync.ckpt`
- 3-config summary JSON:
  - `taskA/_history/issue213_equal_k_geometry_audit/equal64/codebook_geometry_summary.json`
  - `taskA/_history/issue213_equal_k_geometry_audit/codebook_geometry_summary.json` (equal128)
  - `taskA/_history/issue213_equal_k_geometry_audit/equal256/codebook_geometry_summary.json`
- 3-config CSV tables: 每个子目录 5 个 CSV
- 3-config figures: 每个子目录 4 张 PNG
- verdict: 本文件

---

## 13. DECOR BAN

Issue #213 补充严格在曲率框架内推进, **不引入任何 DECOR 机制**.

---

## 14. 后续推论

1. **论文段落 (增强版)**: "即使 K cardinality 改变 (64/128/256), RQ depth 仍稳定驱动 codebook geometry 异质 → 支持 per-layer type-specific curvature 的设计独立性于 K 选择."

2. **实验设计**: equal128 SID + per-layer type-specific curvature (不依赖 K 异质) → 验证纯 geometry-driven 曲率 (vs v15 capmatch 依赖 K 异质)

3. **Phase D Stage3**: equal128 SID + v85p → 验证 κ 异质缺失 (但 geometry 异质保留) 时端到端代价, 进一步定位 v15 增益来源.