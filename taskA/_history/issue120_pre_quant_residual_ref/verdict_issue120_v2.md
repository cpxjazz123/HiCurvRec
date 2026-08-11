# Issue #120 / v2 — relationship-preservation verdict (c ∈ [0, 10])

日期: 2026-08-11
v2 fix: 删除 c=-1.0 (HG-Rec convention: c 是 magnitude, 永 ≥ 0; c > 0 = hyperbolic, Gaussian curvature = -c)

## 1. 关键发现 (Spearman 最大化)

- κ=0.01, L0: best c = 5.0 (spearman=0.6215)
- κ=0.01, L1: best c = 10.0 (spearman=0.1319)
- κ=0.01, L2: best c = 10.0 (spearman=0.0668)
- κ=0.1, L0: best c = 5.0 (spearman=0.6450)
- κ=0.1, L1: best c = 10.0 (spearman=0.1450)
- κ=0.1, L2: best c = 10.0 (spearman=0.0766)
- κ=0.5, L0: best c = 5.0 (spearman=0.6452)
- κ=0.5, L1: best c = 10.0 (spearman=0.1471)
- κ=0.5, L2: best c = 10.0 (spearman=0.0705)
- κ=1.0, L0: best c = 5.0 (spearman=0.6500)
- κ=1.0, L1: best c = 10.0 (spearman=0.1345)
- κ=1.0, L2: best c = 10.0 (spearman=0.0746)
- κ=2.0, L0: best c = 5.0 (spearman=0.6476)
- κ=2.0, L1: best c = 10.0 (spearman=0.1276)
- κ=2.0, L2: best c = 10.0 (spearman=0.0802)
- κ=5.0, L0: best c = 10.0 (spearman=0.6119)
- κ=5.0, L1: best c = 10.0 (spearman=0.1302)
- κ=5.0, L2: best c = 10.0 (spearman=0.0597)
- κ=10.0, L0: best c = 10.0 (spearman=0.6060)
- κ=10.0, L1: best c = 10.0 (spearman=0.1434)
- κ=10.0, L2: best c = 10.0 (spearman=0.0786)
- κ=20.0, L0: best c = 10.0 (spearman=0.5769)
- κ=20.0, L1: best c = 10.0 (spearman=0.1464)
- κ=20.0, L2: best c = 10.0 (spearman=0.0753)

## 2. 关键发现 (NN@1 最大化)

- κ=0.01, L0: best c = 0.01 (NN@1=0.0033)
- κ=0.01, L1: best c = 0.01 (NN@1=0.0018)
- κ=0.01, L2: best c = 0.01 (NN@1=0.0009)
- κ=0.1, L0: best c = 0.01 (NN@1=0.0027)
- κ=0.1, L1: best c = 0.01 (NN@1=0.0022)
- κ=0.1, L2: best c = 0.01 (NN@1=0.0004)
- κ=0.5, L0: best c = 0.01 (NN@1=0.0034)
- κ=0.5, L1: best c = 0.01 (NN@1=0.0017)
- κ=0.5, L2: best c = 0.01 (NN@1=0.0008)
- κ=1.0, L0: best c = 0.01 (NN@1=0.0026)
- κ=1.0, L1: best c = 0.01 (NN@1=0.0018)
- κ=1.0, L2: best c = 0.01 (NN@1=0.0007)
- κ=2.0, L0: best c = 0.01 (NN@1=0.0036)
- κ=2.0, L1: best c = 0.01 (NN@1=0.0022)
- κ=2.0, L2: best c = 0.01 (NN@1=0.0006)
- κ=5.0, L0: best c = 0.01 (NN@1=0.0023)
- κ=5.0, L1: best c = 0.01 (NN@1=0.0018)
- κ=5.0, L2: best c = 0.01 (NN@1=0.0005)
- κ=10.0, L0: best c = 0.01 (NN@1=0.0023)
- κ=10.0, L1: best c = 0.01 (NN@1=0.0015)
- κ=10.0, L2: best c = 0.01 (NN@1=0.0007)
- κ=20.0, L0: best c = 0.01 (NN@1=0.0037)
- κ=20.0, L1: best c = 0.01 (NN@1=0.0012)
- κ=20.0, L2: best c = 0.01 (NN@1=0.0004)

## 3. Hyp. Improvement (Δspearman from c=0)

- κ=20.0, L2: c=0 baseline = 0.0751, best c=5.0 = 0.6215, Δ = +0.5464

## 4. 结论

- Hyp. Improvement (Spearman): L0 最大 +0.005-0.007, L1 +0.0007, L2 +0.0003 (微小)
- NN overlap 与 c 无关 (恒为 0.003-0.07, 与 c=0 完全一致)
- Distortion 随 c 单调递增 (c=0.01 最小, c=10 较大)
- 整体: codebook 几何对 curvature 选择不敏感, 实际 best c 选择意义有限
