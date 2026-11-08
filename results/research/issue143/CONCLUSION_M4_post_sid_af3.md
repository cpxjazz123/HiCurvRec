# Issue #143 结论（量化后 AF3）

**机制**：独立 Bernoulli 边的 AF3 期望经软读出反传，含邻域保持与用量项

**范围**：仅 Stage 2；不启动、不设计 Stage 3，不把结构代理的改善表述为 Recall 提升。

## 运行事实

- epoch：3000 / 3000
- 是否结束：True
- 累计墙钟：2866.491 s
- 最后 epoch 耗时：0.971 s
- 机制事件条数：3000

| 量 | 首条 | 末条 |
|---|---|---|
| epoch | 1 | 3000 |
| mean_scale | 4.374861838537184e-07 | 1.36395599776021e-06 |
| step_af3 | 0.700154607644064 | 0.31719414818140196 |
| step_mean_probability | 0.010580824936138877 | 0.09465041361972415 |
| step_neighbour | 2.5325958308256666 | 0.5441243409097316 |
| step_scale | 4.374861838537184e-07 | 1.3639559977602103e-06 |
| step_usage | 0.001538097858428955 | 0.008285224437713623 |

## 重复运行与证据修复

- 共享 metrics 流中的运行（按每次运行自己的起点分组）：
  - `1794130544.71`：920 行，epoch 743–1202，完整 1..3000 = False
  - `1794139058.79`：6003 行，epoch 1–3000，完整 1..3000 = True
- 交付运行：`1794139058.7920477`（6003 行，events={'ckpt_saved': 1, 'post_sid': 3000, 'train': 3000, 'train_end': 1, 'train_start': 1}）
- 首次完成运行的产物保存于 `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_M4_post_sid_af3/run_a_complete/item_sids.json`，SID SHA-256 `c6a0bfe3b25ae12e5db11384f3d5b67eb3ffbdc838428663f8b103d8e379453e`；与交付运行相同：False
- 22:57 因误判进程被回收而重复启动了同一臂；重复运行不参与选优，交付运行按“证据完整”这一与结果无关的规则选取，两次运行的差异如实并列。

## 强制分层 SID 记录（CLAUDE.md 63）

- 记录文件：`/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_M4_post_sid_af3/stage2_metrics/6f71e6b20b42c48866809c454551a2be9ee600d3746ba2bdb110330bfafd15a3.json`
- SID SHA-256：`6f71e6b20b42c48866809c454551a2be9ee600d3746ba2bdb110330bfafd15a3`

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
| report_path | /fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_M4_post_sid_af3/stage2_metrics/6f71e6b20b42c48866809c454551a2be9ee600d3746ba2bdb110330bfafd15a3.json |
| schema | stage2_sid_layer_metrics_v1 |
| status | complete |
| technically_valid_sid | True |

注：`technically_valid_sid` 是闸门（SID 可解析 + owner 的学习碰撞阈值 ≤35%）的布尔结论，**不是**可解析性；为 False 时原因即上表 `metrics.l3_collision` 超阈，按诊断记录，不作本 Issue 的准入门槛。

## 结构诊断（冻结支持、共享校准、Huber）

- 报告：`/fs04/ar57/wenyu/GeneRec/results/research/issue143/structure_TIGER_M4_post_sid_af3.json`
- SID SHA-256：`6f71e6b20b42c48866809c454551a2be9ee600d3746ba2bdb110330bfafd15a3`
- 支持：940498 对（真边 315391）

| 指标 | 值 |
|---|---|
| distortion | 0.27814316245671156 |
| edge_half | 0.5516009839990778 |
| unobserved_half | 0.004685340914345399 |
| adjacency_brier | 0.2485925872555836 |
| usage | [156, 127, 256] |
| level2_prefixes | 5222 |
| level3_triples | 19471 |
| collision_full | 0.3236263065847806 |
| collision_first | 0.9087322568837191 |
| max_triple_bucket | 29 |
| max_level_code_usage | [829, 1103, 133] |
| neighbour_retention@16 | 0.04791259765625 (k=16, sample=2048) |
| reconstruction_mse | 7.30770334484987e-05 |
| sid_changed_share | 1.0 (levels=3) |

度/三角形误差**未计算**：a full 24587x24587 SID graph would be rebuilt from hard distances; the issue's D_AF3/retention/Brier are measured on the frozen stratified support, and no unbiased degree/triangle estimator exists on that support, so none is reported as if it were the full-graph number

对照（冻结 TIGER，同一支持/同一校准/同一代码；诊断，不是无条件控制组）：

| 指标 | 冻结 TIGER | 本臂 |
|---|---|---|
| distortion | 0.28865508197737616 | 0.27814316245671156 |
| adjacency_brier | 0.2482683951655989 | 0.2485925872555836 |
| collision_full | 0.2745353235449628 | 0.3236263065847806 |
| level2_prefixes | 7414 | 5222 |
| usage | [247, 256, 256] | [156, 127, 256] |
| neighbour_retention@16 | 0.049224853515625 | 0.04791259765625 |
| reconstruction_mse | 6.665317778242752e-05 | 7.30770334484987e-05 |

注：`collision_full` 与分层记录的 `metrics.l3_collision` 是同一口径（重复三级 SID 的商品占比），两处数值应一致；不一致即为计算缺陷，不得并列解释。训练日志里的 `collision=` 是 `1 − 不同三元组/商品数`，会低估（每个重复桶里有一个商品被当成不碰撞），不得用来替代上述冻结口径。

## 预注册反证与实测回答

- 预注册：若 soft AF3 降而 hard D_AF3 不降 ⇒ 判未证实
- 预注册：若 D_AF3 降而邻域保持/重建/用量恶化 ⇒ 判未证实

**实测回答（来自分层记录与结构报告的实际数值）**

- soft AF3 项 0.700 → 0.317（下降）；hard D_AF3 0.2781 < 基线 0.2887，hard 也随之下降，因此「soft 降而 hard 不降」这一条**不成立**。
- 但 retention@16 0.0479 < 基线 0.0493、collision 0.3236 > 0.2745、一级码 156/256 < 247/256、重建 MSE 7.31e-5 > 6.71e-5：按第二条反证判 **未证实的结构收益**。
- test_gradient 7/7 通过（encoder/E-H 码本/逐层软分配均收到梯度）；test_entrypoint 1/1 通过（真实入口一 epoch 并导出）。
- 首次完成运行的产物原被放在 `_aborted_620epochs/`（名称误导：内容是 3000 epoch 的完整导出），已原样复制到 `run_a_complete/` 并保留原目录；两次运行的 SID 分别为 `c6a0bfe3…`（首次）与 `6f71e6b2…`（交付），不做择优。

## 诚实边界

- 单次运行即定论，种子噪声计入不确定性，不做多种子。
- 关闭开关的臂未训练（CLAUDE.md 19g），开关仅做代码内接线检查。
- 结构指标只作诊断，不作晋级闸门；本 Issue 不包含 Stage 3。

