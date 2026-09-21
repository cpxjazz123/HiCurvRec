# iter6 Gate Decision：Sinkhorn 线性 ε-anneal

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter6
- **机制**：`iter6_sinkhorn_linear_eps_anneal`（P1）
- **最终裁决**：**NO-GO**
- **Stage3 test_R@10**：0.05339577638886471（硬目标 0.065）
- **审计基线提交**：`cd70b68bcdc39f6c4411ab5ac3596d9e4279b1dc`（起始），本轮所有 agent 产物在 `stage2_RQ-VAE/curvature_RQ-VAE_iter6/logs/`
- **审计日期**：2026-09-21

## 1. Agent 串行证据链

1. **Agent A**：`logs/lit_search_iter6.md`（≥3 个候选 P1/P2/P3；只检索，不评判）。
2. **Agent B**：`logs/direction_decision_iter6.md`（唯一推荐 Sinkhorn linear ε-anneal；评分 51/60；BACKUP=无）。
3. **Agent C**：`logs/hypothesis_iter6.md`（DE-1 ε(t) 与 c(t) ρ≥0.95；DE-2 Q 周期性震荡且 max/mean 与 c(t) 反向；DE-3 step5000 unique≥iter5 同期；PH-1 H(L1|L0)≥iter5 -0.05；PH-2 collision≤iter5 +0.5%）。
4. **MVG**：iter6 4 层全部 PASS（ON/OFF loss_diff=2.438，on_attn_temp=1.574e-01）。
5. **Stage2**：100k step 跑完 total time=948.2s。
6. **Agent D**：`logs/sid_geometry_iter6.md`（DE-1 ρ(c, ε)=+1.0000；DE-2 ρ=-0.958/-0.932/-0.923；PH-2 FAIL collision+5.71%；三态 PARTIAL/METRIC_MISMATCH）。
7. **Agent E**：`logs/failure_analysis_iter6.md`（类 3：SID geometry 对齐预期但 downstream 不吃）。
9. **Agent F**：`logs/failure_attribution_iter6.md`（5 类标签 GEOMETRY_MISMATCH；不写 ledger）。
10. **Agent G**：`logs/iteration_bridge.md`（dominant bottleneck + forbidden next directions + iter7 objective）。

## 2. 4 层 MVG PASS 证据

- L1：`total_loss.requires_grad=True grad_fn=True value=457.26`；
- L2：3 个 `layers.{0,1,2}.embedding.weight` grad L2 = 2.769 / 5.018 / 5.521（>1e-12）；
- L3：5 步 update ratio floor ≥ 1e-7；
- L4：ON/OFF 200 步后 `loss_diff = 2.438e+00 > 1e-6`，`on_attn_temp = 1.574e-01`；
- counterfactual rollback OFF 锁定 `get_eps=sk_eps_min` 可恢复 baseline。

## 3. Stage2 训练记录

- DDP `world_size=4`，`master_port=50200`；
- `global_step=100000`，`total time=948.2s`；
- step100000 描述性指标：`full_gini=0.0537`，`per_layer=[0.0813/0.1612/0.1843]`，`n_unique_full=23221/24587`，`l01_pairs=15256`，`H(L1|L0)=5.6093`；
- 100 个 `[iter6][sinkhorn]` 采样点（3 层 × 100 step），DE-1 ρ(c, ε) = +1.0000 强耦合。

## 4. Stage3 评估

- DDP `world_size=4`，`master_port=50201`；
- `epoch=150/150`，`total time=948.2s`（Stage2 实际 15min 余 ~，Stage3 ~48 min）；
- epoch150 valid：`recall@10=0.05823569`、`ndcg@10=0.03206888`；
- **test_R@10 = 0.05339577638886471**（< 0.065 hard target，NO-GO）；
- 与 iter4 (0.05402)、iter11 (0.06017) 同量级。

## 5. 失败根因（Agent E/F/G 共识）

- Agent D 三态 PARTIAL：DE 全部 PASS、PH-2 FAIL collision+5.71%；
- Agent F 5 类标签 GEOMETRY_MISMATCH（条件 3 FAIL；条件 1/2/4 PASS）；
- Agent E 类 3：SID 几何改善（full_gini −27%、H +1.29%）但 test_R@10 未传导；
- 根因（Agent G）：Sinkhorn ε-anneal 在 cyclic c(t) 路径上完全可微且 ρ=1.0 真实激活，但把 Q 分布拉平 → collision 偏增 5.71%，导致 SID 几何方向改善却未形成 iter11 "强 collapse + L1 diverse" 双特征，test_R@10 与 iter4 同量级未被传导；证实 Stage3 T5 表征对 Stage2 几何层差异不敏感（与 MEMORY.md v320/v321 R37 SID 字节级 lock 行为一致）。

## 6. 决策

**NO-GO：iter6 Sinkhorn 线性 ε-anneal 触发 Agent F GEOMETRY_MISMATCH，硬目标 0.05340 < 0.065 未达成。**

iter6 不写 `failed_mechanism_ledger.md`（仅 TRUE_MECHANISM_FAIL 才进 ledger；iter6 条件 3 FAIL）。

下一轮（iter7）必须按 `iteration_bridge.md` 的 dominant bottleneck 与 forbidden directions 重置检索方向，避免"再做 ε-anneal 而不修 PH-2 collision 漂移"。