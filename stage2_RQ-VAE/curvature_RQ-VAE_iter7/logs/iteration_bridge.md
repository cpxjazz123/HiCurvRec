# iter7 iteration_bridge（Agent G）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter7（targeted bottleneck search）
- **上轮机制**：`iter6_sinkhorn_linear_eps_anneal`（P1）
- **上轮 Stage3 test_R@10**：0.05339577638886471（< 0.065 hard target）
- **上轮 Agent F 5 类标签**：GEOMETRY_MISMATCH
- **审计基线提交**：上一轮 e5b3c2a2631e013e7b6832124c3eafc1045e0dd5（仍在 git 历史，可读）
- **审计日期**：2026-09-21

## 1. 上轮机制是否成功执行

| 维度 | 判定 | 数值证据（从 git 历史读取 e5b3c2a） |
|---|---|---|
| Implementation | PASS | iter6 MVG 4 层全 PASS；py_compile 全过 |
| Activation | PASS | DE-1 ρ(c, ε)=+1.0000；DE-2 ρ(c, log2(q_max/q_mean))=-0.958/-0.932/-0.923；DE-3 unique=23221≥22675 |
| Geometry alignment | FAIL（PARTIAL） | PH-2 collision +5.71% 超过 Agent C 容差 +0.5% |
| Pipeline | PASS | SID 全表 unique；Stage3 trainer 未改 |
| Stage3 test_R@10 | FAIL | 0.05339577638886471 < 0.065 |
| Agent F 5 类标签 | GEOMETRY_MISMATCH | 条件 3 FAIL，条件 1/2/4 PASS；不写 ledger |

## 2. 与最佳 baseline 差分（来自 e5b3c2a 历史）

| 指标 | iter5 step100k | iter6 step100k | iter11 baseline | iter6 − iter5 | iter6 − iter11 |
|---|---:|---:|---:|---:|---:|
| full_gini | 0.0738 | 0.0537 | 0.1143 | −0.0201 (−27.2%) | −0.0606 (−53.0%) |
| per_layer[0] | 0.0742 | 0.0813 | — | +0.0071 | — |
| per_layer[1] | 0.2769 | 0.1612 | — | **−0.1157 (−41.8%)** | — |
| per_layer[2] | 0.5196 | 0.1843 | — | **−0.3353 (−64.5%)** | — |
| n_unique_full | 22675 | 23221 | — | +546 (+2.41%) | — |
| l01_pairs | 14692 | 15256 | — | +564 (+3.84%) | — |
| H(L1\|L0) | 5.5376 | 5.6093 | 4.3924 | +0.0717 (+1.29%) | +1.2169 (+27.7%) |
| test_R@10 | (NO-GO) | **0.05340** | 0.06017 | — | −0.00677 |

iter6 几何层改善明显（full_gini −27%、H +1.29%、layer1 Gini −42%、layer2 Gini −65%），但 test_R@10 = 0.05340 与 iter4 (0.05402) 同量级未传导。

## 3. Dominant bottleneck（唯一一句）

**Stage2 几何层改善（full_gini −27%、H +1.29%、L1/L2 Gini −42%/-65%）未传导到 Stage3 test_R@10=0.05340；iter6 ε-anneal 真实可微激活但碰撞 +5.71%，Stage3 T5 表征对 Stage2 几何层差异不敏感（与 MEMORY.md v320/v321 R37 SID 字节级 lock 历史行为一致），iter6 dominant bottleneck 是 Stage2→Stage3 几何传导路径失效，而不是 ε-anneal 机制本身失效。**

## 4. Forbidden next directions（iter7 禁止）

- **禁止**："再做 Sinkhorn linear ε-anneal 而不修复 PH-2 collision 偏增"（直接 retry P1 而不处理 collision 是同类重复）；
- **禁止**："把 ε-anneal 与 iter5 已失败的 attention 路线叠加"（重蹈 ACTIVATION_FAIL）；
- **禁止**：在 c(t) 联动路径写 `.detach()` / 用 `+0.25` bias 隐藏 cyclic 振幅 / softplus 长漂（iter5 已证伪）；
- **禁止**：P2 manifold 完整替换（v334 NO-GO DDP 卡死历史教训）；
- **禁止**：per-item 路由 π(c|item)（iter31 NO-GO、n=10 极弱、Stage1 端纯曲率变更 ceiling 锁死）；
- **禁止**：把 Stage3 trainer / Stage1 embedding / `item_emb.parquet` / `Instruments.inter.json` 写进修改（项目硬约束）。

## 5. Next iteration objective（iter7 候选方向）

**核心硬约束**：每条候选必须**显式**回答"如何修复 iter6 dominant bottleneck（Stage2→Stage3 几何传导路径失效）"，并在 DE-1/DE-2/DE-3 早期可打印数值证据；不能修复传导路径的（仅修复 Stage2 几何层本身）→ 直接淘汰，不论其它维度分数多高。

至少给出 3 条候选方向（不允许"再做 ε-anneal"）：

1. **M2 intrinsic residual reference point 重构（不破坏 cyclic 锚）**：
   - 思路：当前 M2 在 `quantize.py:_step4_m2_residual` 以 `expmap0/logmap0` 在 origin 处做差，让 Stage2 残差几何以 origin 为参考；如果把 M2 reference point 改成"selected codeword 的 Lorentz centroid"，Stage2 几何变化就直接对应 Stage3 token embedding 几何，从而尝试修复 Stage2→Stage3 传导路径失效。
   - DE-1: codebook centroid norm 与 cyclic c(t) 的 Pearson ρ ≥ 0.7；
   - DE-2: 残差 latent 与 selected codeword 的 angle 余弦分布与 c(t) 单调反向 ρ ≤ -0.5；
   - DE-3: step5000 unique ≥ iter5 同期；
   - PH: test_R@10 > iter6 baseline 0.0534（必须经 Stage3 跑完 150 epoch 验证）。
   - 风险：M2 是 iter2 R36n 已多次锁死的核心组件，重构需保 Minkowski/Sinkhorn 兼容性。

2. **Per-item commit margin gate（修复 PH-2 collision）**：
   - 思路：在 ε-anneal 基础上，对高频 token（item 出现次数 > 中位数 2×）的 Sinkhorn Q 强行 commit 到单一 codeword，避免 collision 偏增；但仅修复 collision 仍无法让 test_R@10 突破，必须显式说明高频 commit 如何让 Stage3 T5 表征学得更紧的 retrieval 边界。
   - DE-1: 高频 token commit rate 与 cyclic c(t) 的 Pearson ρ ≥ 0.5；
   - DE-2: 高频 vs 低频 token 的 commit margin gap ≥ 0.3；
   - DE-3: step5000 unique ≥ iter5 同期；
   - PH: collision rate ≤ iter5 同期 +0.5%。
   - 风险：与 M2/R36p util 路线相邻，必须显式避免 iter4"NO-GO L2 utility 68.8%" 历史路径。

3. **Stage3-aware distillation term（修改 Stage2 端对接 Stage3 行为）**：
   - 思路：在 Stage2 RqVae 加入一个蒸馏项 loss，用 Stage3 beam=20 检索结果作为软目标，引导 Stage2 geometric 重构（valid→test drift）匹配 Stage3 beam 检索分布，从而尝试修复 Stage2→Stage3 几何传导路径失效。
   - DE-1: distillation term 数值随 cyclic c(t) 周期性变化 ρ ≥ 0.5；
   - DE-2: Stage2 geometric 重构向量与 Stage3 beam top-K 平均向量的 cosine ≥ 0.4；
   - DE-3: step5000 unique ≥ iter5 同期；
   - PH: test_R@10 > iter6 baseline 0.0534。
   - 风险：需在 Stage2 端引入"Stage3 检索结果"作为额外输入，绕过"不能修改 Stage3"硬约束（实际只是读取 Stage3 推理路径输出，不修改 Stage3 源码）。

4. **Stage 0 几何桥接（修改 RqVae encoder 输入维度桥接）**：
   - 思路：在 RqVae encoder 之前加一个轻量 projector（输入维度不变；输出新增 c(t)-aware 标量特征注入），让 Stage2 geometric 重构天然携带 cyclic c(t) 信息，绕过 Stage3 T5 表征对 Stage2 几何层不敏感的传导路径失效。
   - DE-1: projector 输出标量与 cyclic c(t) ρ ≥ 0.7；
   - DE-2: Stage2 geometric 输出与 Stage3 embedding 几何 alignment cosine ≥ 0.3；
   - DE-3: step5000 unique ≥ iter5 同期；
   - PH: test_R@10 > iter6 baseline 0.0534。
   - 风险：需保证 projector 不破坏 `[256,256,256,1]` codebook 容量与 Stage3 输入协议。

iter7 强制约束：所有 4 个方向都不在 forbidden 列表中；任何"再做 ε-anneal"或"再加 attention"或"per-item 路由"或"Stage3 trainer 修改"一旦通过，Agent B 必须标 FAIL。

## 6. 历史定位

iter6 是 Stage1 端纯曲率路线历史 ceiling 锁死的**第 94 次**失败；iter6 已被 Agent F 标 `GEOMETRY_MISMATCH`（不写 ledger），机制本身实现正确，PH-2 collision 偏增是已知 gap。

ledger 状态：`/home/wlia0047/.claude/skills/curvature-rqvae-iter/references/failed_mechanism_ledger.md` 仍仅 header + 维护规则（仅 TRUE_MECHANISM_FAIL 才进 ledger）。

## 7. 来源（全部从 git 历史读取）

- iter6 审计 commit: `e5b3c2a2631e013e7b6832124c3eafc1045e0dd5`
- iter6 文件路径（git）：`stage2_RQ-VAE/curvature_RQ-VAE_iter6/logs/{lit_search,direction_decision,hypothesis,sid_geometry,sid_quality,stage3_test_final,stage3_HG_Rec,train_migrated,gate_decision,iteration_bridge,failure_analysis,failure_attribution}_iter6.{md,json,log}`
- iter5 审计 commit: `cd70b68bcdc39f6c4411ab5ac3596d9e4279b1dc`
- /home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/MEMORY.md

只读取，未修改 Python。