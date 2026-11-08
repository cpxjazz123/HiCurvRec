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

注：`technically_valid_sid` 是闸门（SID 可解析 + owner 的学习碰撞阈值 ≤35%）的布尔结论，**不是**可解析性；为 False 时原因即上表 `metrics.l3_collision` 超阈，按诊断记录，不作本 Issue 的准入门槛。

## 结构诊断（冻结支持、共享校准、Huber）

- 报告：`/fs04/ar57/wenyu/GeneRec/results/research/issue143/structure_TIGER_M2_radial_manifold.json`
- SID SHA-256：`3ede5bd240da7eae1f2ceddb9343e38750dd8a3c0c95c63932f0216324d28055`
- 支持：940498 对（真边 315391）

| 指标 | 值 |
|---|---|
| distortion | 0.28354286206610724 |
| edge_half | 0.5627994053233527 |
| unobserved_half | 0.004286318808861772 |
| adjacency_brier | 0.24720888040028594 |
| usage | [15, 64, 254] |
| level2_prefixes | 593 |
| level3_triples | 9007 |
| collision_full | 0.8012364257534469 |
| collision_first | 0.994997356326514 |
| max_triple_bucket | 77 |
| max_level_code_usage | [5043, 1970, 259] |
| neighbour_retention@16 | 0.037353515625 (k=16, sample=2048) |
| reconstruction_mse | 9.590831177774817e-05 |
| sid_changed_share | 1.0 (levels=3) |

度/三角形误差**未计算**：a full 24587x24587 SID graph would be rebuilt from hard distances; the issue's D_AF3/retention/Brier are measured on the frozen stratified support, and no unbiased degree/triangle estimator exists on that support, so none is reported as if it were the full-graph number

对照（冻结 TIGER，同一支持/同一校准/同一代码；诊断，不是无条件控制组）：

| 指标 | 冻结 TIGER | 本臂 |
|---|---|---|
| distortion | 0.2886732654457991 | 0.28354286206610724 |
| adjacency_brier | 0.24827616140270206 | 0.24720888040028594 |
| collision_full | 0.2745353235449628 | 0.8012364257534469 |
| level2_prefixes | 7414 | 593 |
| usage | [247, 256, 256] | [15, 64, 254] |
| neighbour_retention@16 | 0.049346923828125 | 0.037353515625 |
| reconstruction_mse | 6.712303002132103e-05 | 9.590831177774817e-05 |

注：`collision_full` 与分层记录的 `metrics.l3_collision` 是同一口径（重复三级 SID 的商品占比），两处数值应一致；不一致即为计算缺陷，不得并列解释。训练日志里的 `collision=` 是 `1 − 不同三元组/商品数`，会低估（每个重复桶里有一个商品被当成不碰撞），不得用来替代上述冻结口径。

## 预注册反证与实测回答

- 预注册：若径向只在连续 r 上拟合、最终码字没有携带 ⇒ 记 NO_QUANTIZED_CURVATURE_LEVERAGE（半径由三个码字重构）
- 预注册：若径向退化为常数 ⇒ 报告半径分布（radial 事件记录 radius_mean）

**实测回答（来自分层记录与结构报告的实际数值）**

- 机制事件只记录 `curvature_mean=-1.9997`（首末相同），**没有 radius_mean**；「半径是否由三个码字承载」无法从现有日志回答，记 NOT_RECORDED。
- `curvature_mean` 首末同为 -1.9997，未观测到非恒定曲率；结合 hard SID 的usage [15,64,254] 与 collision 0.8012，判 **NO_QUANTIZED_CURVATURE_LEVERAGE**。
- hard SID：D_AF3 0.2835（基线 0.2887）、retention@16 0.0374（基线 0.0493）、collision 0.8012（基线 0.2745）。

## 诚实边界

- 单次运行即定论，种子噪声计入不确定性，不做多种子。
- 关闭开关的臂未训练（CLAUDE.md 19g），开关仅做代码内接线检查。
- 结构指标只作诊断，不作晋级闸门；本 Issue 不包含 Stage 3。

