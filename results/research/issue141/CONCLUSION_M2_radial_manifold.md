# Issue #141 结论（异质径向流形）

**机制**：E15 x H16 x R1，d^2 = E + H/4 + (dr)^2，R_M = R_H + R_phi(r)

**范围**：仅 Stage 2；不启动、不设计 Stage 3，不把结构代理的改善表述为 Recall 提升。

## 运行事实

- epoch：3000 / 3000
- 是否结束：True
- 累计墙钟：3425.592 s
- 最后 epoch 耗时：0.85 s
- 机制事件条数：3000

| 量 | 首条 | 末条 |
|---|---|---|
| epoch | 1 | 3000 |
| mean_scale | 2.3086034147926467e-10 | 1.488022939620024e-11 |
| step_scale | 2.3086034147926467e-10 | 1.488022939620024e-11 |

## 强制分层 SID 记录（CLAUDE.md 63）

- 记录文件：`/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_M2_radial_manifold/stage2_metrics/3ede5bd240da7eae1f2ceddb9343e38750dd8a3c0c95c63932f0216324d28055.json`
- SID SHA-256：`3ede5bd240da7eae1f2ceddb9343e38750dd8a3c0c95c63932f0216324d28055`

| 指标 | 值 |
|---|---|
| admission_policy | technical validity only; no quality floor |
| baseline_metrics | ... |
| diagnostics | ... |
| diagnostics_only | ... |
| methodology | ... |
| metrics | ... |
| provenance | ... |
| quality_source | ... |
| report_path | /fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_M2_radial_manifold/stage2_metrics/3ede5bd240da7eae1f2ceddb9343e38750dd8a3c0c95c63932f0216324d28055.json |
| schema | stage2_sid_layer_metrics_v1 |
| status | complete |
| technically_valid_sid | False |

## 预注册反证与实测回答

- 若径向只在连续 r 上拟合、最终码字没有携带 ⇒ 记 NO_QUANTIZED_CURVATURE_LEVERAGE（半径由三个码字重构）
- 若径向退化为常数 ⇒ 报告半径分布（radial 事件记录 radius_mean）

## 诚实边界

- 单次运行即定论，种子噪声计入不确定性，不做多种子。
- 关闭开关的臂未训练（CLAUDE.md 19g），开关仅做代码内接线检查。
- 结构指标只作诊断，不作晋级闸门；本 Issue 不包含 Stage 3。

