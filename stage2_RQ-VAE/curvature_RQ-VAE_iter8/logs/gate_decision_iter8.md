# iter8 Gate Decision：P1 M2 Intrinsic Residual Reference Point 重构

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter8（BACKUP P1 路径）
- **机制**：`iter8_m2_reference_point_codeword_centroid`（M2 reference point 从 Poincaré origin 改为 selected codebook codeword 的 Lorentz centroid）
- **最终裁决**：**NO-GO**
- **Stage3 test_R@10**：0.05299535159038284（硬目标 0.065）
- **审计基线 commit**：当前为审计基线
- **审计日期**：2026-09-22

## 1. Agent 串行证据链

1. **Agent G `iteration_bridge.md`**：iter6 dominant bottleneck 量化 + iter8 BACKUP P1 主线 + forbidden directions；
2. **Agent A `lit_search_iter8.md`**：3 个候选 P1/P2/P3（亲自产出，因 Agent A 多次失败）；
3. **Agent B `direction_decision_iter8.md`**：唯一推荐 P1，BACKUP P2，P3 因 v334 DDP 卡死历史不推荐；
4. **Agent C `hypothesis_iter8.md`**：DE-1/DE-2/DE-3 与 PH-1/PH-2 可证伪假设全部数值化；
5. **MVG**：iter8 4 层全部 PASS（L1=401.93；L2 grad L2 2.77/4.27/4.75；L3 update ratio floor ≥ 1e-7；L4 ON/OFF loss_diff=31.01）；
6. **Stage2**：100k step 跑完 total time=981.6s；
7. **Stage3**：150 epoch 跑完，`test_R@10=0.05299535159038284` < 0.065 hard target；
8. **Agent E `failure_analysis_iter8.md`**（类 3：SID 几何对齐预期但下游不增益）；
9. **Agent F `failure_attribution_iter8.md`**（5 类标签 = GEOMETRY_MISMATCH，条件 3 FAIL；不写 ledger）；
10. **Agent G `iteration_bridge.md`**（dominant bottleneck + forbidden directions + iter9 objective）。

## 2. 4 层 MVG PASS 证据

- L1：`total_loss.requires_grad=True grad_fn=True value=401.93`；
- L2：3 个 `layers.{0,1,2}.embedding.weight` grad L2 = 2.769/4.266/4.745（>1e-12）；
- L3：5 步 update ratio floor ≥ 1e-7；
- L4：ON/OFF 200 步后 `loss_diff = 3.101e+01 > 1e-6`，`on_attn_temp = 8.011e-01`（用 c(t) 作为机制直接输出证据，因 iter8 无 attention / get_eps）；
- counterfactual rollback OFF 把 `m2_reference_point` 重置为 `'origin'` 路径能让 layer 走 baseline `_step4_m2_residual` 分支，与 ON 路径分歧显著（loss_diff=31.01）。

## 3. Stage2 训练记录

- DDP `world_size=4`，`master_port=50200`；
- `global_step=100000`，`total time=981.6s`；
- step100000 描述性指标：`full_gini=0.0520`，`per_layer=[0.0744/0.1452/0.1818]`，`n_unique_full=23263/24587`，`l01_pairs=15562`，`H(L1|L0)=5.6494`；
- 100 个 `[iter8][m2]` 采样点（3 层 × 100 step），DE-1 ρ(c, centroid_norm)=-0.9355/-0.9720/-0.9730（三层 n=100）。

## 4. Stage3 评估

- DDP `world_size=4`，`master_port=50201`；
- `epoch=150/150` 跑完；
- epoch150 valid：`recall@10=0.057278155956754125`，`ndcg@10=0.031402785186594624`；
- **test_R@10 = 0.05299535159038284**（< 0.065 hard target，NO-GO）；
- 与 iter4 (0.05402)、iter6 (0.05340) 同量级未传导；比 iter6 还低 0.00041。

## 5. 失败根因（Agent E/F/G 共识）

- **Agent E 类 3**：SID 几何对齐预期（full_gini -3.2%、per_layer[1] -9.9%、H +0.7%）但下游 test_R@10 未传导；
- **Agent F 5 类标签 GEOMETRY_MISMATCH**（条件 3 FAIL：PH-1 test_R@10=0.05299 < iter6 0.05340；DE-1 符号反向 ρ<0 与 Agent C 假设 ρ>0 反号）；
- **Agent G dominant bottleneck**：v321 56 次 lock 论证 Stage 3 T5 SID 表征空间对 Stage 1 端变更强 lock，iter8 是 BACKUP P1 在 v321 论证下的第 95 次失败尝试，Stage 1 端任何变更（包括 M2 reference point）均无法修复 Stage2→Stage3 几何传导路径失效。

## 6. 决策

**NO-GO：iter8 P1 M2 Intrinsic Residual Reference Point 重构触发 Agent F GEOMETRY_MISMATCH，硬目标 0.05299 < 0.065 未达成。**

iter8 不写 `failed_mechanism_ledger.md`（仅 TRUE_MECHANISM_FAIL 才进 ledger）。

下一轮（iter9）按 `iteration_bridge.md` 的 dominant bottleneck 重置检索方向：必须跳出 Stage 1 端纯几何变更 + R36n 6 大方向；候选方向（Stage 0 几何桥接 / Stage 3-aware distillation 重做 / Stage 4 non-geometric rerank）必须显式回答"为什么不被 v321 lock 锁定"。

## 7. 来源

- iter8 logs: `stage2_RQ-VAE/curvature_RQ-VAE_iter8/logs/{iteration_bridge,lit_search,direction_decision,hypothesis,failure_analysis,failure_attribution,gate_decision}_iter8.md`、`sid_quality_iter8.json`、`stage3_test_final_iter8.json`、`stage3_HG_Rec_iter8.log`
- iter7 audit commit: `097c573a11dea2613f69306efbb44ba2ab8ccab5`
- iter6 audit commit: `e5b3c2a2631e013e7b6832124c3eafc1045e0dd5`
- iter5 audit commit: `cd70b68bcdc39f6c4411ab5ac3596d9e4279b1dc`
- iter4 audit commit: `224c5f3`

只读取，未修改 Python。