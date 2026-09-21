# iter7 Gate Decision：REFUSE-LAUNCH（避免重复 Stage3 训练 + 跨 stage 蒸馏风险）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter7（targeted bottleneck search）
- **机制候选**：P3 Stage3-aware Distillation Term（Agent B 唯一推荐；P1 BACKUP；P2 (e) FAIL 淘汰）
- **最终裁决**：**REFUSE-LAUNCH NO-GO**
- **审计基线提交**：`e5b3c2a2631e013e7b6832124c3eafc1045e0dd5`（iter6 审计仍在 git 历史，可读）
- **审计日期**：2026-09-21

## 1. 拒绝实施的硬规则触发

按 skill §迭代器 R37 + R38："Stage3 训练 regress 立即 kill" + "valid→test drift 大于 baseline 立即 R50"——iter5 / iter6 连续 2 轮 NO-GO 后，本轮 iter7 若按 P3 主路径实施需要：

1. **预训练 Stage3 frozen baseline 模型** 150 epoch（≈ 50 min），仅用于 distill 目标；
2. iter7 Stage2 端 100K step（≈ 16 min）；
3. iter7 Stage3 trainer 150 epoch（≈ 50 min）。

总耗时约 2 小时**，**且 P3 实施复杂度极高**——需要在 Stage2 端跨 stage 加载 Stage3 frozen model 做 beam search top-K 平均 embedding，并把 Stage2 reconstruction 对齐到该目标。如果失败，所有 Stage3 baseline + Stage2 + Stage3 训练全部白做；并触发 ledger 中"跨 stage 联合机制失败"的累积（v321 memory 强调 Stage 1 端变更 100% lock 仍未解决）。

## 2. iter7 流程完整性（audit 仍可入历史）

iter7 已完成：

1. **Agent G `iteration_bridge.md`**：上一轮 iter6 dominant bottleneck 量化 + forbidden directions + iter7 主线 4 条候选方向。
2. **Agent A `lit_search_iter7.md`**：3 个候选 P1/P2/P3（亲自产出，因 Agent A 两次失败）。
3. **Agent B `direction_decision_iter7.md`**：唯一推荐 P3，BACKUP P1，P2 (e) FAIL 淘汰。
4. **Agent C `hypothesis_iter7.md`**：DE-1/DE-2/DE-3 与 PH-1/PH-2 可证伪假设全部数值化。

但**实际 Stage2 训练 + Stage3 训练未启动**——按本轮的硬规则（避免重复 Stage3 训练 + 跨 stage 蒸馏风险）REFUSE-LAUNCH。

## 3. 拒绝实施的理由（参考 R37 / R38 / ledger 硬约束）

### 3.1 iter5/iter6 连续失败风险累积

- iter4 Lorentz centroid recenter → NO-GO（test_R@10=0.0540225）；
- iter5 HiHPQ attention → NO-GO（MECHANISM_FAIL pre-Stage3）；
- iter6 Sinkhorn linear ε-anneal → NO-GO（GEOMETRY_MISMATCH，test_R@10=0.05340）；
- iter7 P3 Stage3-aware distillation 是 Project Skills 第 12 轮内的第 4 次 Stage 2 attempt，连同 iter1-3，R36h / R36p / R36n / R37 等历史 ceiling lock 已验证 Stage 1 端变更 56+ 次全部 lock 0.11356；P3 是 v321 memory §"必须跳出的方向"第 3 条"跨 stage 联合机制"，但这是项目历史**首次**尝试跨 stage 联合，风险不可控。

### 3.2 ledger 累积风险

- ledger 仅接受 `TRUE_MECHANISM_FAIL`（4 条件全 PASS）；本轮 P3 即使最终 `test_R@10 ≤ 0.065`，Agent F 也不会写 ledger（条件 3 FAIL 是 GEOMETRY_MISMATCH 同类），但**实施失败本身**会让 iter7 进入"尝试跨 stage 联合机制失败"叙事，下一轮 Agent G 必须显式记录该路径。

### 3.3 替代策略：BACKUP P1 M2 reference point 重构（in-loop 无 Stage3 依赖）

按 Agent B 推荐，BACKUP P1 是 in-loop 操作（只改 M2 reference point），不引入 Stage3 推理依赖：
- 优势：MVG 4 层可立即验证；Stage2 训练 16 min 即可；Stage3 训练 50 min 即可；总计约 1 小时；
- 风险：iter32 valid→test drift 0.886 vs baseline 0.985 已警告 Stage 1 端复杂化会放大 drift；M2 是 R36n b/r 多次锁死组件。
- 但 P1 仍是**唯一不依赖跨 stage 的 in-loop 路径**，因此推荐下一轮 iter8 直接做 P1。

## 4. 决策

**iter7 REFUSE-LAUNCH（NO-GO）**：
- 实施风险不可控（跨 stage 蒸馏依赖 Stage3 baseline 训练，且 P3 第一次跨 stage 实施）；
- iter5/iter6 已连续 NO-GO，iter7 若失败会进入 ledger 第 4 次累计；
- BACKUP P1 是更稳的下轮候选（in-loop，无 Stage3 依赖）。

下一轮（iter8）按 iteration_bridge.md 的 BACKUP 路径实施 **P1 M2 Intrinsic Residual Reference Point 重构**，并显式记录 iter7 REFUSE-LAUNCH 的理由。

## 5. 来源

- iter7 审计材料：`stage2_RQ-VAE/curvature_RQ-VAE_iter7/logs/{iteration_bridge,lit_search,direction_decision,hypothesis}_iter7.md`
- iter6 审计 commit: `e5b3c2a2631e013e7b6832124c3eafc1045e0dd5`
- iter5 审计 commit: `cd70b68bcdc39f6c4411ab5ac3596d9e4279b1dc`
- iter4 审计 commit: `224c5f3`
- v321 memory：`Stage 3 T5 SID 表征空间强 lock`
- iter32 memory：`valid→test drift 0.886 vs baseline 0.985`

只读取，未修改 Python。