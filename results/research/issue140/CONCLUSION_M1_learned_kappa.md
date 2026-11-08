# Issue #140 结论（学习曲率）

**机制**：kappa_l = KAPPA_MAX tanh(theta_l) 进入层代价、原生残差与读出

**范围**：仅 Stage 2；不启动、不设计 Stage 3，不把结构代理的改善表述为 Recall 提升。

## 运行事实

- epoch：3000 / 3000
- 是否结束：True
- 累计墙钟：4097.945 s
- 最后 epoch 耗时：1.186 s
- 机制事件条数：3000

| 量 | 首条 | 末条 |
|---|---|---|
| epoch | 1 | 3000 |
| kappa | [-0.9999901056289673, -0.9999978542327881, -0.9999904036521912] | [2.7137824076817196e-07, -3.715461389219854e-06, 4.4076468839193694e-06] |
| mean_scale | 4.072893866866965e-05 | 1.3785624600831374e-05 |
| step_af3 | 0.6934558926620042 | 0.3137827671531862 |
| step_kappa_gradient | 1.773612900990429e-08 | 0.00012439228885341436 |
| step_mean_probability | 0.4265811877146608 | 0.4265738402225073 |
| step_neighbour | 0.6362551233798617 | 0.6270109455955392 |
| step_scale | 3.8987883887838e-05 | 1.7583173866147636e-05 |
| step_usage | 2.656380456755869e-05 | 0.00022687515593133867 |

## 重复运行与证据修复

- 共享 metrics 流中的运行（按每次运行自己的起点分组）：
  - `1794131858.65`：270 行，epoch 2867–3000，完整 1..3000 = False
  - `1794139056.76`：6003 行，epoch 1–3000，完整 1..3000 = True
- 交付运行：`1794139056.7617216`（6003 行，events={'ckpt_saved': 1, 'curvature': 3000, 'train': 3000, 'train_end': 1, 'train_start': 1}）
- 首次完成运行的产物保存于 `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_M1_learned_kappa/run_a_complete/item_sids.json`，SID SHA-256 `8f6ff43f46bd4b666459dfd5718b6d2b79f7d2ed6f1d9849945a7bf0f3f2bf2c`；与交付运行相同：False
- 22:57 因误判进程被回收而重复启动了同一臂；重复运行不参与选优，交付运行按“证据完整”这一与结果无关的规则选取，两次运行的差异如实并列。

## 强制分层 SID 记录（CLAUDE.md 63）

- 记录文件：`/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_M1_learned_kappa/stage2_metrics/92ca528bd9b962bea44c42806db7fded6afa12cd84886d7ef0e2be9b7a81c012.json`
- SID SHA-256：`92ca528bd9b962bea44c42806db7fded6afa12cd84886d7ef0e2be9b7a81c012`

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
| report_path | /fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_M1_learned_kappa/stage2_metrics/92ca528bd9b962bea44c42806db7fded6afa12cd84886d7ef0e2be9b7a81c012.json |
| schema | stage2_sid_layer_metrics_v1 |
| status | complete |
| technically_valid_sid | False |

注：`technically_valid_sid` 是闸门（SID 可解析 + owner 的学习碰撞阈值 ≤35%）的布尔结论，**不是**可解析性；为 False 时原因即上表 `metrics.l3_collision` 超阈，按诊断记录，不作本 Issue 的准入门槛。

## 结构诊断（冻结支持、共享校准、Huber）

- 报告：`/fs04/ar57/wenyu/GeneRec/results/research/issue143/structure_TIGER_M1_learned_kappa.json`
- SID SHA-256：`92ca528bd9b962bea44c42806db7fded6afa12cd84886d7ef0e2be9b7a81c012`
- 支持：940498 对（真边 315391）

| 指标 | 值 |
|---|---|
| distortion | 0.287975828777752 |
| edge_half | 0.5721077954622962 |
| unobserved_half | 0.003843862093207871 |
| adjacency_brier | 0.24530364716912575 |
| usage | [42, 221, 256] |
| level2_prefixes | 2933 |
| level3_triples | 16834 |
| collision_full | 0.46178875015251963 |
| collision_first | 0.9537560499450929 |
| max_triple_bucket | 39 |
| max_level_code_usage | [3456, 719, 144] |
| neighbour_retention@16 | 0.04559326171875 (k=16, sample=2048) |
| reconstruction_mse | 7.429486868204549e-05 |
| sid_changed_share | 1.0 (levels=3) |

度/三角形误差**未计算**：a full 24587x24587 SID graph would be rebuilt from hard distances; the issue's D_AF3/retention/Brier are measured on the frozen stratified support, and no unbiased degree/triangle estimator exists on that support, so none is reported as if it were the full-graph number

对照（冻结 TIGER，同一支持/同一校准/同一代码；诊断，不是无条件控制组）：

| 指标 | 冻结 TIGER | 本臂 |
|---|---|---|
| distortion | 0.2886732654457991 | 0.287975828777752 |
| adjacency_brier | 0.24827616140270206 | 0.24530364716912575 |
| collision_full | 0.2745353235449628 | 0.46178875015251963 |
| level2_prefixes | 7414 | 2933 |
| usage | [247, 256, 256] | [42, 221, 256] |
| neighbour_retention@16 | 0.049346923828125 | 0.04559326171875 |
| reconstruction_mse | 6.712303002132103e-05 | 7.429486868204549e-05 |

注：`collision_full` 与分层记录的 `metrics.l3_collision` 是同一口径（重复三级 SID 的商品占比），两处数值应一致；不一致即为计算缺陷，不得并列解释。训练日志里的 `collision=` 是 `1 − 不同三元组/商品数`，会低估（每个重复桶里有一个商品被当成不碰撞），不得用来替代上述冻结口径。

## 预注册反证与实测回答

- 预注册：若 κ 不改变任何一次选码 ⇒ 记 NO_GEOMETRIC_LEVERAGE（test_gradient 度量至少一个 token 改变）
- 预注册：若 κ 越界或投影冲突 ⇒ 记数值失败（KAPPA_MAX 由投影半径推导，test_math 覆盖边界）

**实测回答（来自分层记录与结构报告的实际数值）**

- κ 轨迹 [-1,-1,-1] → [2.71e-7, -3.72e-6, 4.41e-6]，末步 κ 梯度 1.24e-4（非零）；test_gradient 4/4 通过（要求扰动改变至少一个 token）。结论不是「无作用」，而是**训练把曲率推到 0，几何退化为欧氏**。
- test_math 11/11 通过（κ→0± 极限、投影与定义域覆盖）；未观察到越界。
- hard SID：D_AF3 0.2880（基线 0.2887）、retention@16 0.0456（基线 0.0493）、collision 0.4618（基线 0.2745）、一级码 42/256（基线 247/256）。判 **NO_GAIN**：几何自退化，且码本明显变差。

## 诚实边界

- 单次运行即定论，种子噪声计入不确定性，不做多种子。
- 关闭开关的臂未训练（CLAUDE.md 19g），开关仅做代码内接线检查。
- 结构指标只作诊断，不作晋级闸门；本 Issue 不包含 Stage 3。

