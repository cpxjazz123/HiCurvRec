# PGD / q-kNN 审计记录（2026-11-16）

关联旧报告：`analysis_sid_recall/QKNN_PGD_MOTIFS.md`。旧报告的结论**不再使用**；本文件给出修正后的统计口径和结果。

## 1. 发现的问题与处理

| 问题 | 证据 | 处理 |
|---|---|---|
| PGD `4-tailed-triangle` 全为 0 | 源码中 `local_4_tailed_tris` 从未赋值；合成图 `K3 + pendant` 的 micro 输出 4 个边全为 0，但 PGD 全局 `total_4_tailed_tris = 1` | **绕过**：不再使用该列；不基于它做结论 |
| PGD `3-star` 因此被多算 | 同一合成图的 pendant 边输出 `3-star=1`，正确 induced K1,3 应为 0 | **绕过**：不再使用 `3-star`；只使用独立列 triangle / 4-clique / 4-chordal-cycle / 4-cycle / 4-path |
| CPU kNN 自排除依赖“自身在第一位” | 原 `[:,1:]` 只是位置切片；若查询结果不是自身首位就会把自身算进邻居 | **修复**：按行删除 `neighbours == row_index`，不足 k 直接报错；新增 `test_qknn_self_exclusion.py` |
| PGD 重新编号可能配错边 | 原脚本没有逐边唯一性检查 | 显式反解映射，要求 315,391 条边一一对应；重复/缺失即报错 |
| 匹配覆盖未记录 | `nanmean` 可能静默丢单元 | 记录每类边的 matched coverage；四类主 motif 为 1.0000，4-path 为 0.9919 |
| 原 CV 是随机 5 折 | 同一模型系列的不同臂可能同时进训练和测试 | 改为按模型系列的 `GroupShuffleSplit`，30 次，Ridge(alpha=1) |
| 原 partial 控制的 overall 包含目标类边 | 目标类自身进入对照，产生机械耦合 | 改为 leave-target-out overall recovery |
| 4 个高碰撞臂主导 | collision ≥ 0.90 的 4 个臂同时有低 recall、低 overall recovery、极高 q-edge 数 | 主表并列 `collision < 0.90` 敏感性分析；结论只以该子集为准 |

## 2. 主结果（48 臂，PGD 可靠列）

部分相关控制 leave-target-out overall、q-edge 数和 L3 collision。CV 为按系列分组的 30 次 GroupShuffleSplit；p 为单侧 Wilcoxon 配对检验。

| motif | partial rho (p) | CV ΔR² rate (p) | CV ΔR² excess (p) |
|---|---:|---:|---:|
| triangle | -0.1539 (0.296) | +0.1346 (1.3e-8) | +0.1726 (5.0e-7) |
| 4-clique | -0.1598 (0.278) | +0.1979 (4.7e-9) | +0.2304 (1.3e-7) |
| 4-chordal-cycle | -0.1783 (0.225) | +0.1697 (2.3e-8) | +0.2004 (1.3e-7) |
| 4-cycle | -0.0207 (0.889) | +0.1360 (3.1e-5) | +0.2199 (5.0e-7) |
| 4-path | -0.1712 (0.245) | +0.1846 (1.7e-5) | +0.0417 (0.214) |

**不能用“CV 显著为正”解释为“结构保留越高，recall 越高”**：partial 相关是负的（虽然多数不显著），CV 只说明这些特征对 arm 间差异有预测力。方向与预测力必须分开报告。

## 3. 去掉高碰撞臂后，正 CV 增量消失

`collision < 0.90` 的 44 臂：

| motif | partial rho | CV ΔR² rate (p) | CV ΔR² excess (p) |
|---|---:|---:|---:|
| triangle | -0.1215 | -0.0185 (1.000) | -0.0209 (1.000) |
| 4-clique | -0.1520 | -0.0241 (1.000) | -0.0330 (1.000) |
| 4-chordal-cycle | -0.1893 | -0.0182 (1.000) | -0.0201 (1.000) |
| 4-cycle | +0.0618 | -0.0324 (1.000) | -0.0501 (1.000) |
| 4-path | -0.1914 | -0.0347 (1.000) | +0.0031 (0.017) |

被排除的 4 个高碰撞臂：

| arm | tree | collision | recall@10 | q-edges | overall recovery |
|---|---|---:|---:|---:|---:|
| 37 | `TIGER_CURVALLOC-LORENTZ-RQ` | 0.9977 | 0.04189 | 242,842 | 0.00123 |
| 34 | `TIGER_BRad-RQ` | 0.9931 | 0.03061 | 236,520 | 0.00409 |
| 36 | `TIGER_HGS-TREE` | 0.9511 | 0.05125 | 224,828 | 0.00413 |
| 28 | `TIGER_cogeo_region_argmax` | 0.9367 | 0.03983 | 225,859 | 0.00116 |

这 4 个臂同时表现为“码本坍缩、q 边数高、整体恢复低、recall 低”。因此原 PGD 的显著 CV 增量主要是**高碰撞/低恢复状态**的可识别性，不是结构保真度的有效信号。去掉它们后，五类 motif 的 CV 增量全部为负（4-path excess 的 +0.003 是唯一例外，但同时 rate 为负，不能作为证据）。

## 4. 合成图复现 PGD 缺陷

输入：

```
0,1
1,2
2,0
2,3
```

即三角形 `0-1-2` 加 pendant `2-3`。PGD micro 输出：

```
src,dst,triangle,2-star,4-clique,4-chordal-cycle,4-tailed-triangle,4-cycle,3-star,4-path
2,3,1,1,0,0,0,0,0,0
1,3,1,1,0,0,0,0,0,0
4,3,0,2,0,0,0,0,1,0
2,1,1,0,0,0,0,0,0
```

- 4 条边都参与同一个 4-node tailed triangle，`4-tailed-triangle` 应各为 1，实际全为 0；
- pendant 边 `2-3`（PGD 内部编号 4-3）的 `3-star` 为 1，但 induced K1,3 要求三个叶子互不相邻；这里两个三角形顶点相邻，应为 0。

结论：**PGD 的 `4-tailed-triangle` 和 `3-star` 不能用于该分析**。若以后需要，必须在 PGD 的 triangle 枚举里为 `local_4_tailed_tris` 赋值，而不是靠 `3-star` 的全局差补。

## 5. 对 Issue 215 / 后续设计的含义

1. 现有 PGD 结果不能支持“稠密结构保留是 SID 结构的独立有效信号”。去掉退化臂后没有可重复的正向增量。
   精确 K1,3 的修复与复算见 `analysis_sid_recall/QKNN_K13_AUDIT.md`；精确 K1,3 同样不支持
   “K1,3 富集边恢复更好”，其 top 类匹配 excess 为负。
2. 不能把 PGD 的 4-node motif 直接当成 Edge-ORCA 或 5-node 结构；它只是边级 4-node 计数。
3. 后续若继续做“Graphlet Role × 商品关系 × SID 前缀”，结构统计应先经过：合成图单测、自排除 kNN、逐边一一映射、匹配覆盖率、高碰撞敏感性、分组 CV 六项检查；否则很容易把码本退化当成结构规律。
4. 当前可保留的结论只有：高碰撞臂会制造“结构恢复 vs recall”的伪相关；正常碰撞臂内没有证据表明 PGD 的 4-node 结构保留预测 recall。

## 6. 产物

| 文件 | 内容 |
|---|---|
| `qknn_pgd_audited.py` | 审计版分析，含一一映射、覆盖率、leave-target-out、分组 CV |
| `qknn_pgd_significance.py` | 配对 Wilcoxon/符号检验、`collision < 0.90` 敏感性 |
| `results/qknn_pgd_audited.json/.md` | 审计统计 |
| `results/qknn_pgd_significance.json/.md` | 显著性与碰撞敏感性 |
| `test_qknn_self_exclusion.py` | kNN 自排除回归 |
