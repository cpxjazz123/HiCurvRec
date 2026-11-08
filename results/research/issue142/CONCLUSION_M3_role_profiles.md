# Issue #142 结论（局部曲率 profile）

**机制**：TRAIN-only 七维结构角色经量化读出回传到码本

**范围**：仅 Stage 2；不启动、不设计 Stage 3，不把结构代理的改善表述为 Recall 提升。

## 运行事实

- epoch：3000 / 3000
- 是否结束：True
- 累计墙钟：3577.692 s
- 最后 epoch 耗时：1.015 s
- 机制事件条数：3000

| 量 | 首条 | 末条 |
|---|---|---|
| epoch | 1 | 3000 |
| mean_scale | 0.00046216631088167866 | 2.8262368777192584e-10 |
| step_scale | 0.0003643001265538134 | 6.343946043275867e-10 |

## 重复运行与证据修复

- 共享 metrics 流中的运行（按每次运行自己的起点分组）：
  - `1794139055.16`：6003 行，epoch 1–3000，完整 1..3000 = True
- 交付运行：`1794139055.15563`（6003 行，events={'ckpt_saved': 1, 'role': 3000, 'train': 3000, 'train_end': 1, 'train_start': 1}）
- 首次完成运行的产物保存于 `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_M3_role_profiles/run_a_complete/item_sids.json`，SID SHA-256 `56277eaf6e54ef9875ea0214d9f682de4ebc8586a5cab4edc18d22ea17062d90`；与交付运行相同：False
- 22:57 因误判进程被回收而重复启动了同一臂；重复运行不参与选优，交付运行按“证据完整”这一与结果无关的规则选取，两次运行的差异如实并列。

## 强制分层 SID 记录（CLAUDE.md 63）

- 记录文件：`/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_M3_role_profiles/stage2_metrics/4dcbe0d969b7587a280c7b635256753e0fa9f684e5a799e8a31b94801cf246df.json`
- SID SHA-256：`4dcbe0d969b7587a280c7b635256753e0fa9f684e5a799e8a31b94801cf246df`

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
| report_path | /fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_M3_role_profiles/stage2_metrics/4dcbe0d969b7587a280c7b635256753e0fa9f684e5a799e8a31b94801cf246df.json |
| schema | stage2_sid_layer_metrics_v1 |
| status | complete |
| technically_valid_sid | False |

注：`technically_valid_sid` 是闸门（SID 可解析 + owner 的学习碰撞阈值 ≤35%）的布尔结论，**不是**可解析性；为 False 时原因即上表 `metrics.l3_collision` 超阈，按诊断记录，不作本 Issue 的准入门槛。

## 结构诊断（冻结支持、共享校准、Huber）

- 报告：`/fs04/ar57/wenyu/GeneRec/results/research/issue143/structure_TIGER_M3_role_profiles.json`
- SID SHA-256：`4dcbe0d969b7587a280c7b635256753e0fa9f684e5a799e8a31b94801cf246df`
- 支持：940498 对（真边 315391）

| 指标 | 值 |
|---|---|
| distortion | 0.28408535804782736 |
| edge_half | 0.5645746984350147 |
| unobserved_half | 0.0035960176606399676 |
| adjacency_brier | 0.23369620970706803 |
| usage | [13, 169, 256] |
| level2_prefixes | 982 |
| level3_triples | 11783 |
| collision_full | 0.6900394517427909 |
| collision_first | 0.9895066498556148 |
| max_triple_bucket | 46 |
| max_level_code_usage | [9118, 1048, 157] |
| neighbour_retention@16 | 0.036468505859375 (k=16, sample=2048) |
| reconstruction_mse | 0.00015287735732272267 |
| sid_changed_share | 1.0 (levels=3) |

度/三角形误差**未计算**：a full 24587x24587 SID graph would be rebuilt from hard distances; the issue's D_AF3/retention/Brier are measured on the frozen stratified support, and no unbiased degree/triangle estimator exists on that support, so none is reported as if it were the full-graph number

对照（冻结 TIGER，同一支持/同一校准/同一代码；诊断，不是无条件控制组）：

| 指标 | 冻结 TIGER | 本臂 |
|---|---|---|
| distortion | 0.2886732654457991 | 0.28408535804782736 |
| adjacency_brier | 0.24827616140270206 | 0.23369620970706803 |
| collision_full | 0.2745353235449628 | 0.6900394517427909 |
| level2_prefixes | 7414 | 982 |
| usage | [247, 256, 256] | [13, 169, 256] |
| neighbour_retention@16 | 0.049346923828125 | 0.036468505859375 |
| reconstruction_mse | 6.712303002132103e-05 | 0.00015287735732272267 |

注：`collision_full` 与分层记录的 `metrics.l3_collision` 是同一口径（重复三级 SID 的商品占比），两处数值应一致；不一致即为计算缺陷，不得并列解释。训练日志里的 `collision=` 是 `1 − 不同三元组/商品数`，会低估（每个重复桶里有一个商品被当成不碰撞），不得用来替代上述冻结口径。

## 预注册反证与实测回答

- 预注册：若只有辅助头变好、量化器无梯度 ⇒ 判机制未实现（test_gradient 要求 encoder 与码本都收到梯度）
- 预注册：若 profile 回归改善而 hard SID 结构指标不改善 ⇒ 必须报告为未证实的量化收益

**实测回答（来自分层记录与结构报告的实际数值）**

- test_gradient 5/5 通过（encoder 与码本都收到梯度），耦合通路存在。
- 机制事件只记录 `scale`（0.000462 → 2.8e-10），**没有 profile 预测误差**，记 NOT_RECORDED；hard SID：D_AF3 0.2841（基线 0.2887）略好，但 retention@16 0.0365（基线 0.0493）更差、collision 0.6900（基线 0.2745）、一级码 13/256。判 **未证实的量化收益**，且码本塌缩。

## 诚实边界

- 单次运行即定论，种子噪声计入不确定性，不做多种子。
- 关闭开关的臂未训练（CLAUDE.md 19g），开关仅做代码内接线检查。
- 结构指标只作诊断，不作晋级闸门；本 Issue 不包含 Stage 3。

