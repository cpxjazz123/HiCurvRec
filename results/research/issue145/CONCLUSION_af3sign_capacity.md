# Issue #145 结论（AF3 符号容量分配）

**机制**：L1=E32、L2/L3=H32（由 TRAIN 边 AF3 符号比例的 largest remainder 冻结），训练不含行为信号

**范围**：仅 Stage 2；不启动、不设计 Stage 3，不把结构代理的改善表述为 Recall 提升。

## 运行事实

- epoch：3000 / 3000
- 是否结束：True
- 累计墙钟：936.986 s
- 最后 epoch 耗时：0.353 s
- 机制事件条数：0

## 重复运行与证据修复

- 共享 metrics 流中的运行（按每次运行自己的起点分组）：
  - `1794147897.47`：3003 行，epoch 1–3000，完整 1..3000 = True
- 交付运行：`1794147897.4681897`（3003 行，events={'ckpt_saved': 1, 'train': 3000, 'train_end': 1, 'train_start': 1}）

## 强制分层 SID 记录（CLAUDE.md 63）

- 记录文件：`/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_af3sign_capacity/stage2_metrics/b9083f4dd9cb32f7c51ad586da15af2574b4b0e3d677cc89524c8a9af6ad7855.json`
- SID SHA-256：`b9083f4dd9cb32f7c51ad586da15af2574b4b0e3d677cc89524c8a9af6ad7855`

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
| report_path | /fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_af3sign_capacity/stage2_metrics/b9083f4dd9cb32f7c51ad586da15af2574b4b0e3d677cc89524c8a9af6ad7855.json |
| schema | stage2_sid_layer_metrics_v1 |
| status | complete |
| technically_valid_sid | True |

注：`technically_valid_sid` 是闸门（SID 可解析 + owner 的学习碰撞阈值 ≤35%）的布尔结论，**不是**可解析性；为 False 时原因即上表 `metrics.l3_collision` 超阈，按诊断记录，不作本 Issue 的准入门槛。

## 结构诊断（冻结支持、共享校准、Huber）

- 报告：`/fs04/ar57/wenyu/GeneRec/results/research/issue143/structure_TIGER_af3sign_capacity.json`
- SID SHA-256：`b9083f4dd9cb32f7c51ad586da15af2574b4b0e3d677cc89524c8a9af6ad7855`
- 支持：940498 对（真边 315391）

| 指标 | 值 |
|---|---|
| distortion | 0.28763382065219895 |
| edge_half | 0.571461840221648 |
| unobserved_half | 0.0038058010827498865 |
| adjacency_brier | 0.2446691319034031 |
| usage | [177, 253, 256] |
| level2_prefixes | 6701 |
| level3_triples | 20420 |
| collision_full | 0.27038678976695 |
| collision_first | 0.869809248790011 |
| max_triple_bucket | 22 |
| max_level_code_usage | [640, 959, 144] |
| neighbour_retention@16 | 0.05084228515625 (k=16, sample=2048) |
| reconstruction_mse | 6.689198198728263e-05 |
| sid_changed_share | 1.0 (levels=3) |

度/三角形误差**未计算**：a full 24587x24587 SID graph would be rebuilt from hard distances; the issue's D_AF3/retention/Brier are measured on the frozen stratified support, and no unbiased degree/triangle estimator exists on that support, so none is reported as if it were the full-graph number

对照（冻结 TIGER，同一支持/同一校准/同一代码；诊断，不是无条件控制组）：

| 指标 | 冻结 TIGER | 本臂 |
|---|---|---|
| distortion | 0.28865508197737616 | 0.28763382065219895 |
| adjacency_brier | 0.2482683951655989 | 0.2446691319034031 |
| collision_full | 0.2745353235449628 | 0.27038678976695 |
| level2_prefixes | 7414 | 6701 |
| usage | [247, 256, 256] | [177, 253, 256] |
| neighbour_retention@16 | 0.049224853515625 | 0.05084228515625 |
| reconstruction_mse | 6.665317778242752e-05 | 6.689198198728263e-05 |

注：`collision_full` 与分层记录的 `metrics.l3_collision` 是同一口径（重复三级 SID 的商品占比），两处数值应一致；不一致即为计算缺陷，不得并列解释。训练日志里的 `collision=` 是 `1 − 不同三元组/商品数`，会低估（每个重复桶里有一个商品被当成不碰撞），不得用来替代上述冻结口径。

## 预注册反证与实测回答

- 预注册：若 AF3 符号比例极端导致某分量维数为 0 ⇒ 按确定性规则保留并报告
- 预注册：若结构指标未改善 ⇒ 说明静态签名映射不受支持；D_AF3 改善但邻域/用量/碰撞恶化也不能称整体更好

**实测回答（来自分层记录与结构报告的实际数值）**

- AF3 符号分布极极端：315391 条 TRAIN 边中 315390 负、1 零、0 正；largest remainder 给出 L2/L3 = H32 × E0 × S0，按确定性规则保留，球面分量为 0，因此没有球面代码路径。
- 结构确实小幅改善：D_AF3 0.28763（基线 0.28866）、ret@16 0.05084（0.04922，首次超过基线）、Brier 0.24467（0.24827）、L3 碰撞 0.2704（0.2745）、最大三级桶 22（42）、Hard ER 6.296（6.468）。
- 代价同时记录：一级码 177/256（基线 247）、L2 前缀 6701（7414）、重建 MSE 6.689e-5（6.665e-5）。判 **小幅结构改善但粗层容量与重建有代价**；效应量小，不宣称整体更好，也不宣称 Recall。
- test_math 6/6、test_gradient 3/3、test_ddp 1/1、test_entrypoint 1/1；碰撞闸门 True（0.2704 < 0.35）。

## 诚实边界

- 单次运行即定论，种子噪声计入不确定性，不做多种子。
- 关闭开关的臂未训练（CLAUDE.md 19g），开关仅做代码内接线检查。
- 结构指标只作诊断，不作晋级闸门；本 Issue 不包含 Stage 3。

