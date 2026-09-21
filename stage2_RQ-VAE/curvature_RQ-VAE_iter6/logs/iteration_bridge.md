# iter6 iteration_bridge（Agent G）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter6
- **机制**：`iter6_sinkhorn_linear_eps_anneal`（P1）
- **Stage3 test_R@10**：0.05339577638886471
- **Agent F 5 类标签**：GEOMETRY_MISMATCH
- **审计日期**：2026-09-21

## 1. 机制是否成功执行

| 维度 | 判定 | 数值证据 |
|---|---|---|
| Implementation | PASS | MVG L1/L2/L3/L4 全过；py_compile 全过；counterfactual rollback OFF 锁定 `get_eps=sk_eps_min` 可恢复 baseline |
| Activation | PASS | DE-1 ρ(c, ε)=+1.0000（n=300）；DE-2 ρ(c, log2(q_max/q_mean))=-0.958/-0.932/-0.923；DE-3 unique=23221≥22675 |
| Geometry alignment | FAIL（PARTIAL） | PH-2 collision 偏增 5.71% 超过 Agent C 容差 +0.5% |
| Pipeline | PASS | SID (24587,4) 全表 unique；Stage3 trainer 未改；baseline 已恢复 |

## 2. 与最佳 baseline 差分

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

iter6 几何改善明显（full_gini −27%、H +1.29%、layer1 Gini −42%、layer2 Gini −65%），但 test_R@10 = 0.05340，与 iter4 (0.05402)、iter5 (NO-GO) 处于同 0.054±0.001 区间，未传导。

## 3. Dominant bottleneck（唯一一句）

**Sinkhorn ε-anneal 在 cyclic c(t) 路径上完全可微且 ρ(c, ε)=1.0 真正激活，但把 Q 分布拉平 → collision rate 偏增 5.71%，导致 SID 几何方向虽然改善（full_gini −27%、H +1.29%）却未形成 iter11 的"强 collapse 路线 + L1 diverse 双特征"，test_R@10 与 iter4 同量级未被传导，证实 Stage3 T5 表征对 Stage2 几何层差异不敏感。**

## 4. Forbidden next directions

- **禁止**："再做 Sinkhorn linear ε-anneal 而不修复 PH-2 collision 偏增"（直接 retry P1 而不处理 collision 是同类重复）；
- **禁止**："把 ε-anneal 与 iter5 已失败的 attention 路线叠加"（会重蹈 ACTIVATION_FAIL）；
- **禁止**：在 c(t) 联动路径写 `.detach()` / `+0.25` bias / softplus 长漂（iter5 已证伪）；
- **禁止**：扩展到 P2 manifold 替换（v334 NO-GO DDP 卡死历史教训）；
- **禁止**：per-item 路由 π(c|item)（iter31 NO-GO、n=10 极弱、Stage1 端纯曲率变更 ceiling 锁死）。

## 5. Next iteration objective（iter7 候选方向）

至少 3 条候选方向必须能**显式回答**"如何修复 iter6 dominant bottleneck（Stage2 geometric layer 改善未传导到 Stage3 test_R@10）"：

1. **Stage 0 几何桥接**：绕开 Stage2 内部 c(t) 联动，转而修改 Stage1 embedding（item_emb）使其天然携带几何结构；直接对应 test_R@10 传导路径。
2. **per-item commit gate（修复 PH-2）**：在 ε-anneal 基础上增加"高频 token 强制 commit margin"，让 ε 拉宽只对低频 token 生效，避免 collision 偏增 5.71%；Agent A 必须显式回答"为什么这个 gate 可以让 test_R@10 突破"。
3. **T5 表征端联调（不修 Stage2）**：直接调整 Stage3 beam search 或 HF cache；绕开 Stage2 ceiling；Agent A 必须给出实证。

iter7 强制约束：候选不能解释"如何修复 Stage2 geometric layer 改善未传导到 Stage3 test_R@10" → 直接淘汰；不允许"再做 ε-anneal"。

## 6. 历史定位

iter6 是 Stage1 端纯曲率路线历史 ceiling 锁死的**第 94 次**失败（沿用 MEMORY v318-v337 计数）；但与 iter5 不同，iter6 已被 Agent F 标 `GEOMETRY_MISMATCH`（不写 ledger），机制本身实现正确，可在修复 PH-2 collision 漂移后重做。

ledger 状态：`/home/wlia0047/.claude/skills/curvature-rqvae-iter/references/failed_mechanism_ledger.md` 仍仅 header + 维护规则（仅 TRUE_MECHANISM_FAIL 才进 ledger）。

## 7. 来源

- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter6/logs/{hypothesis_iter6,sid_geometry_iter6,failure_analysis_iter6,failure_attribution_iter6,sid_quality_iter6,stage3_test_final_iter6}.{md,json}
- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/iteration_bridge.md
- /home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/MEMORY.md

只读取，未修改 Python。