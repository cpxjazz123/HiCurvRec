# iter8 iteration_bridge（Agent G，亲自产出）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter8（BACKUP P1 路径）
- **机制**：`iter8_m2_reference_point_codeword_centroid`（P1 M2 Intrinsic Residual Reference Point 重构）
- **Stage3 test_R@10**：0.05299535159038284（< 0.065 hard target）
- **Agent F 5 类标签**：GEOMETRY_MISMATCH
- **审计基线 commit**：当前为审计基线
- **审计日期**：2026-09-22

## 1. 机制是否成功执行

| 维度 | 判定 | 数值证据 |
|---|---|---|
| Implementation | PASS | MVG 4 层全 PASS；py_compile 全过；Stage2 完整跑完 100k step |
| Activation | PASS（反号） | DE-1 ρ(c, centroid_norm)=-0.9355/-0.9720/-0.9730（\|ρ\|≥0.7 PASS，但符号反向 Agent C 假设的 ρ>0） |
| Geometry alignment | FAIL（PARTIAL） | PH-1 test_R@10=0.05299 < iter6 0.05340 FAIL；collision +2.01% 仍偏增 |
| Pipeline | PASS | SID 全表 unique；baseline 已恢复 |
| Stage3 test_R@10 | FAIL | 0.05299535159038284 < 0.065 |
| Agent F 5 类标签 | GEOMETRY_MISMATCH | 条件 3 FAIL，条件 1/2/4 PASS；不写 ledger |

## 2. 与最佳 baseline 差分

| 指标 | iter6 step100k | iter8 step100k | iter11 baseline | iter8 − iter6 | iter8 − iter11 |
|---|---:|---:|---:|---:|---:|
| full_gini | 0.0537 | 0.0520 | 0.1143 | −0.0017 (−3.2%) | −0.0623 (−54.5%) |
| per_layer[1] | 0.1612 | 0.1452 | — | **−0.0160 (−9.9%)** | — |
| per_layer[2] | 0.1843 | 0.1818 | — | −0.0025 (−1.4%) | — |
| n_unique_full | 23221 | 23263 | — | +42 (+0.18%) | — |
| l01_pairs | 15256 | 15562 | — | +306 (+2.01%) | — |
| H(L1\|L0) | 5.6093 | 5.6494 | 4.3924 | +0.0401 (+0.7%) | +1.2570 (+28.6%) |
| test_R@10 | 0.05340 | **0.05299** | 0.06017 | **−0.00041** | −0.00718 |

iter8 Stage2 几何层改善（full_gini -3.2%、per_layer[1] -9.9%、H +0.7%）但 test_R@10 仍 ≤ iter6 0.05340，未传导。

## 3. Dominant bottleneck（唯一一句）

**Stage2 几何层改善（full_gini -3.2%、H +0.7%、per_layer[1] -9.9%、per_layer[2] -1.4%）未传导到 Stage3 test_R@10=0.05299；iter8 P1 M2 reference point 真实反向（ρ<0 与 Agent C 假设 ρ>0 反号），v321 56 次 lock 论证 Stage 3 T5 SID 表征空间对 Stage 1 端变更强 lock，本轮 iter8 是 v321 论证在第 95 次 Stage 1 attempt 中的验证，iter8 dominant bottleneck 仍是 Stage2→Stage3 几何传导路径失效，且已确认 BACKUP P1 同样无法修复。**

## 4. Forbidden next directions（iter9 禁止）

- **禁止**："再做 P1 M2 reference point"（iter8 已证伪——Agent C 假设 ρ>0 实际 ρ<0，简单 retry 是同类重复）；
- **禁止**："再做 Sinkhorn linear ε-anneal 而不修复 PH-2 collision 偏增"（直接 retry iter6 是同类重复）；
- **禁止**："把 iter8 P1 与 iter5 已失败的 attention 路线叠加"（重蹈 ACTIVATION_FAIL）；
- **禁止**：在 c(t) 联动路径写 `.detach()` / 用 `+0.25` bias 隐藏 cyclic 振幅 / softplus 长漂（iter5 已证伪）；
- **禁止**：P2 manifold 完整替换（v334 NO-GO DDP 卡死历史教训）；
- **禁止**：per-item 路由 π(c|item)（iter31 NO-GO、n=10 极弱、Stage1 端纯曲率变更 ceiling 锁死）；
- **禁止**：跨 stage 蒸馏类机制（iter7 REFUSE-LAUNCH，Stage3 baseline 训练 ~50 min 成本不可接受）；
- **禁止**：把 Stage3 trainer / Stage1 embedding / `item_emb.parquet` / `Instruments.inter.json` 写进修改。

## 5. Next iteration objective（iter9 候选方向）

v321 论证 Stage 3 T5 SID 表征空间对 Stage 1 端变更强 lock（v319/v320/v321 连续 3 次 lock baseline），iter5/iter6/iter8 三次 Stage 1 attempt 全部失败。**iter9 必须跳出 R36n 6 大方向 + Stage 1 端变更**。

至少 3 条候选方向（任何 Stage 1 端纯几何变更必须显式回答"为什么不被 v321 lock 锁定"）：

1. **Stage 0 几何桥接（修改 RqVae encoder 输入维度桥接）**：在 RqVae encoder 之前加一个轻量 projector，让 Stage 2 geometric 重构天然携带几何结构；绕开 Stage 3 T5 表征对 Stage 2 几何层不敏感的传导路径失效。
2. **Stage 3-aware distillation term（iter7 REFUSE-LAUNCH 后重做）**：在 Stage 2 RqVae 加入 Stage 3 frozen model 做 beam=20 蒸馏项；跨 stage 直接对齐 Stage 2↔Stage 3 表征空间。代价：Stage 3 baseline 训练 ~50 min；如确要执行，需先跑 Stage 3 baseline 150 epoch 拿到 frozen 模型。
3. **Stage 4 non-geometric rerank（不依赖 Stage 1 几何）**：在 Stage 3 beam=20 检索时加入 MMR (Maximal Marginal Relevance) 加 SID 表征多样性，绕开 Stage 1→Stage 3 锁路径。

iter9 强制约束：所有 3 条候选都不能引用 Stage 1 端纯曲率变更；任何候选必须显式回答"为什么不会被 v321 lock 锁定"。如 3 条候选都不满足，"v321 ceiling lock"理论（Stage 1 端变更无法传导）已被充分验证，应停止 Stage 1 端迭代。

## 6. 历史定位

iter8 是 BACKUP P1 M2 reference point 重构在 v321 论证下的**第 95 次**失败尝试；iter8 与 iter5/iter6 同属 Stage 1 端纯几何变更路线（仅 reference point 维度）；v321 论证已 3 次被验证。

ledger 状态：`/home/wlia0047/.claude/skills/curvature-rqvae-iter/references/failed_mechanism_ledger.md` 仍仅 header + 维护规则（仅 TRUE_MECHANISM_FAIL 才进 ledger）。

## 7. 来源（全部从 git 历史读取）

- iter7 审计 commit: `097c573a11dea2613f69306efbb44ba2ab8ccab5`
- iter6 审计 commit: `e5b3c2a2631e013e7b6832124c3eafc1045e0dd5`
- iter5 审计 commit: `cd70b68bcdc39f6c4411ab5ac3596d9e4279b1dc`
- iter4 审计 commit: `224c5f3`
- v321 memory: `v321-r37-fail-sid-locks-baseline.md`
- v318 memory: `v318-r36h-ceiling-lock-56.md`
- iter32 memory: `iter32-r37-fail-valid-test-drift.md`

只读取，未修改 Python。