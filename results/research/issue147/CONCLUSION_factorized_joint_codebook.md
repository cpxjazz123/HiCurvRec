# Issue #147 结论（因子化 E×H 联合码本）

**机制**：每层 16×16 分支码本对 → token=16*k_E+k_H，联合代价 + 基线 Sinkhorn

**范围**：仅 Stage 2；不启动、不设计 Stage 3，不把结构代理的改善表述为 Recall 提升。

## 运行事实

- epoch：3000 / 3000
- 是否结束：True
- 累计墙钟：921.936 s
- 最后 epoch 耗时：0.331 s
- 机制事件条数：0

## 重复运行与证据修复

- 共享 metrics 流中的运行（按每次运行自己的起点分组）：
  - `1794148034.43`：3003 行，epoch 1–3000，完整 1..3000 = True
- 交付运行：`1794148034.4348488`（3003 行，events={'ckpt_saved': 1, 'train': 3000, 'train_end': 1, 'train_start': 1}）

## 强制分层 SID 记录（CLAUDE.md 63）

- 记录文件：`/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_factorized_joint_codebook/stage2_metrics/994d87ca7b8d39f0a0fcaf734180297534a5aeb44bd486a2d7050983d61305ba.json`
- SID SHA-256：`994d87ca7b8d39f0a0fcaf734180297534a5aeb44bd486a2d7050983d61305ba`

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
| report_path | /fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_factorized_joint_codebook/stage2_metrics/994d87ca7b8d39f0a0fcaf734180297534a5aeb44bd486a2d7050983d61305ba.json |
| schema | stage2_sid_layer_metrics_v1 |
| status | complete |
| technically_valid_sid | False |

注：`technically_valid_sid` 是闸门（SID 可解析 + owner 的学习碰撞阈值 ≤35%）的布尔结论，**不是**可解析性；为 False 时原因即上表 `metrics.l3_collision` 超阈，按诊断记录，不作本 Issue 的准入门槛。

## 结构诊断（冻结支持、共享校准、Huber）

- 报告：`/fs04/ar57/wenyu/GeneRec/results/research/issue143/structure_TIGER_factorized_joint_codebook.json`
- SID SHA-256：`994d87ca7b8d39f0a0fcaf734180297534a5aeb44bd486a2d7050983d61305ba`
- 支持：940498 对（真边 315391）

| 指标 | 值 |
|---|---|
| distortion | 0.28593212331590767 |
| edge_half | 0.5679953865914091 |
| unobserved_half | 0.0038688600404062283 |
| adjacency_brier | 0.2432409196773433 |
| usage | [220, 246, 256] |
| level2_prefixes | 5376 |
| level3_triples | 18939 |
| collision_full | 0.35201529263431897 |
| collision_first | 0.886118680603571 |
| max_triple_bucket | 40 |
| max_level_code_usage | [1263, 920, 325] |
| neighbour_retention@16 | 0.04205322265625 (k=16, sample=2048) |
| reconstruction_mse | 7.482642104150727e-05 |
| sid_changed_share | 1.0 (levels=3) |

度/三角形误差**未计算**：a full 24587x24587 SID graph would be rebuilt from hard distances; the issue's D_AF3/retention/Brier are measured on the frozen stratified support, and no unbiased degree/triangle estimator exists on that support, so none is reported as if it were the full-graph number

对照（冻结 TIGER，同一支持/同一校准/同一代码；诊断，不是无条件控制组）：

| 指标 | 冻结 TIGER | 本臂 |
|---|---|---|
| distortion | 0.28865508197737616 | 0.28593212331590767 |
| adjacency_brier | 0.2482683951655989 | 0.2432409196773433 |
| collision_full | 0.2745353235449628 | 0.35201529263431897 |
| level2_prefixes | 7414 | 5376 |
| usage | [247, 256, 256] | [220, 246, 256] |
| neighbour_retention@16 | 0.049224853515625 | 0.04205322265625 |
| reconstruction_mse | 6.665317778242752e-05 | 7.482642104150727e-05 |

注：`collision_full` 与分层记录的 `metrics.l3_collision` 是同一口径（重复三级 SID 的商品占比），两处数值应一致；不一致即为计算缺陷，不得并列解释。训练日志里的 `collision=` 是 `1 − 不同三元组/商品数`，会低估（每个重复桶里有一个商品被当成不碰撞），不得用来替代上述冻结口径。

## 预注册反证与实测回答

- 预注册：若联合对明显稀疏或量化后结构未改善 ⇒ 如实记录为负结果
- 预注册：若联合分配退化为分别 argmin 或绕过 Sinkhorn ⇒ 机制未实现

**实测回答（来自分层记录与结构报告的实际数值）**

- 联合分配没有退化为分别 argmin：代价是 16×16 展开后的 256 个联合候选，Sinkhorn 仍按基线在最后一层执行，token 仍是 3 个、每层 256 个联合码（token=16*k_E+k_H）。
- 联合对不稀疏：usage [220, 246, 256]，三级用满 256 个联合对；Hard ER 4.096（基线 6.468）、D_AF3 0.28593（0.28866）、Brier 0.24324（0.24827）、L2 前缀 5376（7414）改善。
- 但 ret@16 0.04205（基线 0.04922）明显变差、L3 碰撞 0.3520（0.2745，闸门 False）、重建 MSE 7.48e-5（6.665e-5）变差：按预注册反证判 **未证实的整体结构改善**。
- test_math 5/5、test_gradient 3/3、test_ddp 1/1、test_entrypoint 1/1。

## 诚实边界

- 单次运行即定论，种子噪声计入不确定性，不做多种子。
- 关闭开关的臂未训练（CLAUDE.md 19g），开关仅做代码内接线检查。
- 结构指标只作诊断，不作晋级闸门；本 Issue 不包含 Stage 3。

