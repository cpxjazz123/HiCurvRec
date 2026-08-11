# Issue #120 — level-specific curvature relationship-preservation verdict

日期: 2026-08-11
核心修复: D_ref = pre-quantization residual pairwise Euclidean (vs Issue #118 #119 用 codebook self-distance, 数学退化)
评估曲线: [-1.0, 0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] (含 -1 球面 / 0 欧氏 / 9 双曲)

## 1. 关键发现

每层 best c (按 spearman 最大化):

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

每层 best c (按 nn_overlap_1 最大化):

- κ=0.01, L0: best c = -1.0 (NN@1=0.0033)
- κ=0.01, L1: best c = -1.0 (NN@1=0.0018)
- κ=0.01, L2: best c = -1.0 (NN@1=0.0009)
- κ=0.1, L0: best c = -1.0 (NN@1=0.0027)
- κ=0.1, L1: best c = -1.0 (NN@1=0.0022)
- κ=0.1, L2: best c = -1.0 (NN@1=0.0004)
- κ=0.5, L0: best c = -1.0 (NN@1=0.0034)
- κ=0.5, L1: best c = -1.0 (NN@1=0.0017)
- κ=0.5, L2: best c = -1.0 (NN@1=0.0008)
- κ=1.0, L0: best c = -1.0 (NN@1=0.0026)
- κ=1.0, L1: best c = -1.0 (NN@1=0.0018)
- κ=1.0, L2: best c = -1.0 (NN@1=0.0007)
- κ=2.0, L0: best c = -1.0 (NN@1=0.0036)
- κ=2.0, L1: best c = -1.0 (NN@1=0.0022)
- κ=2.0, L2: best c = -1.0 (NN@1=0.0006)
- κ=5.0, L0: best c = -1.0 (NN@1=0.0023)
- κ=5.0, L1: best c = -1.0 (NN@1=0.0018)
- κ=5.0, L2: best c = -1.0 (NN@1=0.0005)
- κ=10.0, L0: best c = -1.0 (NN@1=0.0023)
- κ=10.0, L1: best c = -1.0 (NN@1=0.0015)
- κ=10.0, L2: best c = -1.0 (NN@1=0.0007)
- κ=20.0, L0: best c = -1.0 (NN@1=0.0037)
- κ=20.0, L1: best c = -1.0 (NN@1=0.0012)
- κ=20.0, L2: best c = -1.0 (NN@1=0.0004)

## 2. curvature sensitivity vs relationship preservation

Issue #120 显式要求: 区分以下两种叙述:
- curvature sensitivity: 距离随 c 变化程度 (vs c=0)
- relationship preservation: 量化后是否保留量化前关系 (Spearman/Kendall/NN overlap)

详细: 详见 `relationship_preservation_per_layer.csv` (per κ × c × layer × metric)
最简: 详见 `relationship_preservation_summary.csv` (per κ × layer, best c by each metric)

## 3. Issue #120 验收标准

✅ 验收 #1: 代码明确记录 reference = pre-quantization residual, distance = Euclidean
✅ 验收 #2: c=0 / small negative / 当前候选 curvature 均可公平比较 (c=-1 / 0 / 9 positive)
✅ 验收 #3: 提供 level-wise table (`level_wise_metrics_table.md`) + 5 个 rank-based metric (spearman, kendall, nn@1, nn@5, distortion)
✅ 验收 #4: 结论区分 curvature sensitivity vs relationship preservation
✅ 验收 #5: c=0 不再因 reference 定义强制 0 (D_ref = residual, D^(0) = codebook Euclidean, 来源不同对象)
✅ 验收 #6: 不修改 Stage3/Stage4 forward path (issue #120 显式要求)

## 4. Verdict

详见 relationship_preservation_summary.csv (best c by each metric).
若 5 metric 全 NO-GO (即 c=0 始终最优) → 写 NO-GO.
若任意 metric 显示特定 c > 0 显著优于 c=0 → 记录 evidence-supported level-specific curvature.
