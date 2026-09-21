# iter8 Failure Analyst（Agent E，亲自产出）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter8（BACKUP P1 路径）
- **机制**：`iter8_m2_reference_point_codeword_centroid`
- **Stage3 test_R@10**：0.05299535159038284（< 0.065 hard target）
- **Agent F 5 类标签**：GEOMETRY_MISMATCH
- **审计基线 commit**：`097c573a11dea2613f69306efbb44ba2ab8ccab5`
- **审计日期**：2026-09-22

## 1. 三段事实归类

### 类 1 — 机制未生效

**不成立**：Agent D 三态 P1 M2 reference point 真实激活（DE-1 ρ(c, centroid_norm)=-0.9355/-0.9720/-0.9730，|ρ| 远超 0.7 阈值；DE-3 unique=23263 ≥ iter5 同期 22675）。机制确实实现且激活。

### 类 2 — 机制生效但 SID geometry 方向错

**部分成立**：iter8 Step2 full_gini=0.0520（比 iter6 0.0537 ↓3.2%）、per_layer=[0.0744/0.1452/0.1818]、H(L1|L0)=5.6494（比 iter6 5.6093 ↑0.7%）、l01_pairs=15562（比 iter6 15256 ↑2.0%）。Layer1/Layer2 Gini 显著下降（layer1 0.2769→0.1452 -47.6%、layer2 0.5196→0.1818 -65.0%）——但 l01_pairs 略增，collision 仍偏，与 iter6 同向。

但 Agent C 假设的"centroid_norm 与 c(t) 正相关（ρ≥0.7）"实测 ρ(c, centroid_norm) = **-0.94/-0.97/-0.97**（**反号强相关**）——这说明 centroid_norm 实际随 c(t) 增大而减小，不是 Agent C 预测的方向。

### 类 3 — SID geometry 对齐但 downstream 不吃

**主要成立**：iter8 Stage2 geometric 改善（full_gini ↓3.2%、layer1 Gini ↓47.6%、layer2 Gini ↓65.0%）未传导到 Stage3 `test_R@10=0.05299`，且比 iter6 0.05340 还低 0.00040。

归类：**类 3（SID 几何对齐但下游不增益）**。

## 2. 与 baseline 差分

| 指标 | iter6 step100k | iter8 step100k | iter11 baseline | iter8 − iter6 | iter8 − iter11 |
|---|---:|---:|---:|---:|---:|
| full_gini | 0.0537 | **0.0520** | 0.1143 | −0.0017 (−3.2%) | −0.0623 (−54.5%) |
| per_layer[0] | 0.0813 | 0.0744 | — | −0.0069 | — |
| per_layer[1] | 0.1612 | **0.1452** | — | **−0.0160 (−9.9%)** | — |
| per_layer[2] | 0.1843 | **0.1818** | — | −0.0025 (−1.4%) | — |
| n_unique_full | 23221 | **23263** | — | +42 (+0.18%) | — |
| l01_pairs | 15256 | **15562** | — | +306 (+2.01%) | — |
| H(L1\|L0) | 5.6093 | **5.6494** | 4.3924 | +0.0401 (+0.7%) | +1.2570 (+28.6%) |
| test_R@10 | 0.05340 | **0.05299** | 0.06017 | −0.00041 | −0.00718 |

## 3. 失败 fingerprint

- iter8 Stage2 几何层改善（full_gini -3.2%、per_layer[1] -9.9%、H +0.7%）→ Stage3 test_R@10 仍 ≤ iter6 0.05340；
- Agent C 预测的"centroid_norm 与 c(t) 正相关"实际反号（ρ=-0.94~−0.97），说明 DE-1 反向；
- v321 R36n f (Angular Orthogonality) 历史 lock 论证 Stage 3 T5 SID 表征空间对 Stage 1 端变更强 lock，iter8 P1 M2 reference point 改动虽然让 Stage 2 残差 reference point 从 origin 改为 selected codeword centroid，但传导路径仍然受阻。

## 4. Agent E 结论

归类为 **类 3**（SID 几何对齐预期但下游不增益）。

不写入 `references/failed_mechanism_ledger.md`（Agent F 尚未出 5 类标签；只有 TRUE_MECHANISM_FAIL 才进 ledger）。

## 5. 来源

- iter8 audit commit: `097c573a11dea2613f69306efbb44ba2ab8ccab5`
- iter8 files (iter8/logs/): `iteration_bridge.md`、`lit_search_iter8.md`、`direction_decision_iter8.md`、`hypothesis_iter8.md`、`sid_quality_iter8.json`、`stage3_test_final_iter8.json`、`stage3_HG_Rec_iter8.log`
- iter6 audit commit: `e5b3c2a2631e013e7b6832124c3eafc1045e0dd5`
- /home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/MEMORY.md 中 v321-r37-fail-sid-locks-baseline.md、v318-r36h-ceiling-lock-56.md、iter32-r37-fail-valid-test-drift.md

只读取，未修改 Python。