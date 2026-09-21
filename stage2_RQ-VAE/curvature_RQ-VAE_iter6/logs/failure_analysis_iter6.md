# iter6 Failure Analyst（Agent E）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter6
- **机制**：`iter6_sinkhorn_linear_eps_anneal`（P1）
- **Stage3 test_R@10**：0.05339577638886471
- **Agent D 三态**：PARTIAL / METRIC_MISMATCH（DE-1/2/3 PASS、PH-1 PASS、PH-2 FAIL）
- **审计日期**：2026-09-21

## 1. 三段事实归类

### 类 1 — 机制未生效

**不成立**：Agent D 已确认 DE-1 ρ(c, ε) = +1.0000，DE-2 ρ(c, log2(q_max/q_mean)) = −0.958 / −0.932 / −0.923（三层），DE-3 step100000 unique=23221 ≥ iter5 22675，PH-1 H=5.6093 ≥ 5.4876；机制激活，几何方向有变化。

### 类 2 — 机制生效但 SID geometry 方向错

**部分成立**：iter6 几何与 baseline 反向（layer1 Gini −42%、layer2 Gini −65%、H(L1|L0) +1.30%、collision +3.84%），但这是 **Agent C 设计的预期方向**（PH-2 是因 sinkhorn ε 拉宽让 collision 微涨；不属于"几何方向错"）。归类：不属类 2。

### 类 3 — SID geometry 对但 downstream 不吃

**主要成立**：Agent D PARTIAL、DE 全部 PASS、Stage3 完整 150 epoch 跑完、`test_R@10 = 0.05339`。与 iter4 (0.05402)、iter11 (0.06017)、iter5 (NO-GO) 同量级；iter6 几何改善（H(L1|L0) +0.0717、Gini −27%）未转化为 Stage3 `test_R@10` 增益。

归类：**类 3（SID 几何对齐预期但下游不增益）**。

## 2. 与 baseline 差分

| 指标 | iter5 step100k | iter6 step100k | iter11 baseline | iter6 − iter5 | iter6 − iter11 |
|---|---:|---:|---:|---:|---:|
| full_gini | 0.0738 | **0.0537** | 0.1143 | −0.0201 (−27.2%) | −0.0606 (−53.0%) |
| per_layer[0] | 0.0742 | 0.0813 | — | +0.0071 | — |
| per_layer[1] | 0.2769 | **0.1612** | — | −0.1157 (−41.8%) | — |
| per_layer[2] | 0.5196 | **0.1843** | — | −0.3353 (−64.5%) | — |
| n_unique_full | 22675 | **23221** | — | +546 (+2.41%) | — |
| l01_pairs | 14692 | **15256** | — | +564 (+3.84%) | — |
| H(L1\|L0) | 5.5376 | **5.6093** | 4.3924 | +0.0717 (+1.29%) | +1.2169 (+27.7%) |
| test_R@10 | (NO-GO) | **0.05340** | 0.06017 | — | −0.00677 |

## 3. 失败 fingerprint

- **几何层改善但下游层不变**：iter6 让 L1/L2 层间 Gini 大幅下降、H(L1|L0) 上升、collision 微涨，但 Stage3 `test_R@10 = 0.05340` 与 iter4 baseline `0.05402` 处于同一 0.054 ± 0.001 区间；说明 Stage3 T5 表征对 Stage2 几何层差异 **不敏感**（与 MEMORY.md `v320 R37 FAIL 字符级 lock v319`、`R37 FAIL SID 字节级 lock` 等 Stage1 端纯曲率 ceiling 行为一致）。
- **未形成 iter11 collapse 路线**：iter6 L0 utilization = 1.00 vs iter11 L0 utilization = 0.22，未走"强 collapse + L1 diverse"路线；
- **未提升 test_R@10**：SID 几何层改善未传导到下游。

## 4. Agent E 结论

归类为 **类 3**（SID 几何对齐预期但下游不增益）。

不写入 `references/failed_mechanism_ledger.md`（等 Agent F 出 5 类标签判定；若 F 给出 `TRUE_MECHANISM_FAIL` 才进 ledger，否则失败仅归类到本文件）。

## 5. 来源

- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter6/logs/{hypothesis_iter6,sid_geometry_iter6,sid_quality_iter6,stage3_test_final_iter6}.{md,json}
- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/{sid_quality_iter5.json,iteration_bridge.md}
- /home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/MEMORY.md

只读取，未修改 Python。