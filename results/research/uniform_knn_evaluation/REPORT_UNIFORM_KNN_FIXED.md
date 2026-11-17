# Uniform-KNN 修复后正式训练结果

日期：2026-11-17  
范围：Stage2 单次正式训练（3000 epoch）+ 离线硬 q 评估；**未运行 Stage3**。

## 1. 训练

| 项目 | 值 |
|---|---:|
| 臂 | `TIGER_Uniform_KNN_RQ` |
| 代码 | q memory 同步、argmin 统一、tau detach、`KNN_WEIGHT=1e-4` |
| 训练预算 | 3000 epoch，单卡全局 batch 4096 |
| 墙钟 | **1214.6 秒** |
| final total loss | 0.00034592 |
| recon | 7.3444e-5 |
| KNN KL | 0.261786 |
| memory age | mean 2.51, max 6 |
| 最终 distinct collision | **0.37776** |
| smoke | `SMOKE_PASS`，无 dead/non-finite term |

训练没有早停、没有调参扫描、没有多种子。

## 2. 导出完整性

本次导出使用 **CPU canonical checkpoint-bound export**；离线 q 统一由导出的三层 SID 查码本求和，不再使用带 STE 的 `model.rq(z)` 输出：

| 校验 | 结果 |
|---|---|
| checkpoint SHA-256 | `7aaf8961...` |
| export token SHA-256 | `31af1356...` |
| re-encoded token SHA-256 | `31af1356...` |
| 两次 fresh load `tokens_match` | **true** |
| 外部 CPU 重编码 mismatch | **0 条** |
| `build_vectors` `tokens_match_export` | **true** |
| q source | `exported_sid_codebooks` |
| q unique / SID unique | **15,338 / 15,338** |
| `export_manifest.json` | 已写 |

旧的 GPU 导出 SID 已保存到 `results/research/uniform_knn_evaluation/pre_cpu_export/`；当前 SID 是 CPU canonical 版本。基线也用导出 SID 查码本生成 q（`q_unique=20,425`，与其 SID 数一致）。

## 3. 离线硬 q 结果

| 指标 | 实验 0 基线 | 修复前 Uniform | 本次 Uniform |
|---|---:|---:|---:|
| q 行为边恢复率 | **1.1279%** | 0.4582% | **0.9377%** |
| z 行为边恢复率 | 1.7113% | 0.5276% | **1.6151%** |
| x_hat 行为边恢复率 | 1.1169% | 0.6992% | **0.9479%** |
| x → z 条件保留 | 65.674% | 9.606% | **60.787%** |
| x → z → q 条件保留 | 28.938% | 7.902% | **21.605%** |
| z → q（给定 x→z） | 44.063% | 82.268% | **35.542%** |
| q → x_hat（给定 x→z→q） | 87.784% | 64.976% | **85.263%** |
| q 唯一向量 | 20,425 | — | **15,338** |
| q 重复行 | 4,162（16.93%） | 3,415（13.89%） | **9,249（37.62%）** |
| q K=10 边界并列率 | 22.99% | 14.02% | **48.29%** |
| q 同向量行为边 | 525 | — | **744** |
| q 双端重复组行为边 | 25,821（8.19%） | — | **97,027（30.76%）** |

结论：

- 修复后比**修复前**明显好；
- 但仍**未超过**纯欧氏基线：q 低 0.19 个百分点，x→z→q 低 7.33 个百分点；
- 量化仍是主要损失：x→z→q 仅 21.6%，z→q 仅 35.5%。

另外，q 的 9,249 条重复行造成 744 条同向量行为边（命中率 95.1%）；这些边贡献约 707 个 q 命中，占总 q 命中的约 23.9%。因此 0.9377% 仍包含明显的“撞码字”成分，不能当作纯结构改善。

## 4. 码本容量

| 指标 | 基线 | 修复前 | 本次 |
|---|---:|---:|---:|
| L1 已用码字 | 249 / 256 | 216 / 256 | **114 / 256** |
| L2 已用码字 | 256 / 256 | 31 / 256 | **256 / 256** |
| L2 有效码字 | 93.9 | 18.6 | **108.3** |
| L3 已用码字 | 256 / 256 | 256 / 256 | **69 / 256** |
| L2 不同前缀 | 7,374 | 3,307 | **4,988** |
| L3 不同三元组 | 20,425 | 21,172 | **15,338** |
| L3 distinct collision | 16.93% | 13.89% | **37.62%** |
| L3 item-weighted collision | 26.53% | 24.53% | **53.98%** |

**新的失败方向变了：L2 恢复正常，但 L1 和 L3 严重坍缩。**  
`TIGER_Uniform_KNN_RQ` 的 Stage2 质量门记录为 `technically_valid_sid=false`；本次不进入 Stage3。

## 5. 最终判断

1. **导出一致性问题已经解决**：同一 checkpoint 的 CPU 重编码与导出 SID 完全一致，manifest、token hash、SID hash 均已记录。
2. **Uniform-KNN 仍不是成功的机制**：q 恢复 0.9377% 低于基线 1.1279%；L1/L3 码本容量严重下降；L3 碰撞率逼近/超过质量边界。
3. **不启动 Stage3、Graphlet-KNN 或曲率实验**。当前应先解决 L1/L3 坍缩和硬 q 仍不改善的问题。

## 6. 产物

- `results/stage2_RQ-VAE/TIGER_Uniform_KNN_RQ/export_manifest.json`
- `results/stage2_RQ-VAE/TIGER_Uniform_KNN_RQ/stage2_metrics/064e34db...json`
- `results/research/uniform_knn_evaluation/uniform_knn_metrics.json`
- `results/research/uniform_knn_evaluation/uniform_knn_sid_usage.json`
- `uniform_knn_metrics_pre_fix.*`：修复前对照
