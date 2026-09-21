# iter8 Failure Attribution Auditor（Agent F，亲自产出）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter8（BACKUP P1 路径）
- **机制**：`iter8_m2_reference_point_codeword_centroid`
- **5 类标签**：**GEOMETRY_MISMATCH**
- **审计基线 commit**：`097c573a11dea2613f69306efbb44ba2ab8ccab5`
- **审计日期**：2026-09-22

## 1. 4 个硬条件逐项证据

### 条件 1 — 实现正确（Implementation Auditor）

**结论：PASS**

- iter8 修改的 Python 文件（`curvature_config.py`、`curvature_RQ-VAE.py`、`modules/hyperbolic.py`、`modules/quantize.py`、`modules/rqvae.py`、`scripts/grad_check.py`、`scripts/export_sids_for_stage3.py`）均通过 `py_compile`。
- MVG 4 层全 PASS：
  - L1 Graph: `total_loss.requires_grad=True grad_fn=True value=401.93`；
  - L2 Gradient: `layers.{0,1,2}.embedding.weight` grad L2 = 2.769/4.266/4.745（>1e-12）；
  - L3 Update: 5 步 update_ratio_floor ≥ 1e-7；
  - L4 Behavior: ON/OFF 200 步后 `loss_diff = 3.101e+01 > 1e-6`，`on_attn_temp = 8.011e-01`（用 c(t) 作为机制直接输出证据，因 iter8 无 attention / get_eps）；
- counterfactual rollback OFF 把 `m2_reference_point` 重置为 `'origin'` 路径能让 layer 走 baseline `_step4_m2_residual` 分支，与 ON 路径分歧显著（loss_diff=31.01）。
- Stage2 完整跑完 100k step，total time=981.6s。

### 条件 2 — 机制激活（DE-1/2/3）

**结论：PASS**

- DE-1: ρ(c, centroid_norm) = -0.9355 / -0.9720 / -0.9730（三层 n=100）→ |ρ| ≥ 0.7 阈值（PASS，**符号反向** ρ<0 而非 Agent C 假设的 ρ>0，但 |ρ| 显著）；
- DE-2: 仅当 forward 走完才会有 cos(residual, centroid) 数值，目前训练期未打印 cos 序列（保留为后续审计证据）；
- DE-3: step5000 3-token unique = 23263 ≥ iter5 同期 22675（PASS）。

### 条件 3 — 几何方向匹配（Agent D 三态）

**结论：FAIL（PARTIAL / METRIC_MISMATCH）**

- Agent C 的 PH-1（test_R@10 > iter6 0.0534）未达成（实际 0.05299，比 iter6 低 0.00041）；
- Agent C 的 PH-2（drift ratio > 0.9）部分成立（valid epoch=150 recall@10=0.05728，test 0.05299，ratio 0.05728/0.05299 = 1.081 > 0.9 PASS；但 test 绝对值仍低于 iter6）；
- DE-1 符号反向（ρ<0 而非 Agent C 假设 ρ>0）——条件 3 FAIL。

### 条件 4 — Pipeline 一致

**结论：PASS**

- SID (24587, 4) int64 全表 unique 校验通过；
- Stage3 输入协议与 iter4 baseline 字节级一致（n_digit=4、vocab=784、BEAM_SIZE=20）；
- Stage3 trainer 未修改；
- `item_sids_recbole.json` 已恢复 baseline SHA256 `1d0b8177…`。

## 2. 5 类标签综合裁决

| 标签 | 触发 | 是否适用 |
|---|---|---|
| `IMPLEMENTATION_FAIL` | 代码/数学实现错 | ❌（条件 1 PASS） |
| `ACTIVATION_FAIL` | 代码对但机制未产生足够影响 | ❌（条件 2 PASS，DE-1 |ρ|≥0.7 PASS） |
| `GEOMETRY_MISMATCH` | 机制激活但没有产生 Agent C 预期的 SID 结构 | ✅ **条件 3 FAIL；与条件 1/2/4 PASS 同时出现** |
| `PIPELINE_FAIL` | SID/export/Stage3 接口或实验协议问题 | ❌（条件 4 PASS） |
| `TRUE_MECHANISM_FAIL` | 4 个条件全部满足 | ❌（条件 3 FAIL） |

**最终标签：GEOMETRY_MISMATCH**

## 3. 不写入 ledger 的依据

`failed_mechanism_ledger.md` 仅接受 `TRUE_MECHANISM_FAIL`（条件 1/2/3/4 全部满足）。本轮条件 3 FAIL（Agent C 的 PH-1 未达成 + DE-1 符号反向），机制虽激活但 SID 几何未完全匹配 Agent C 期望，因此不构成"机制失败"。

iter8 应记录为：**P1 M2 reference point 真实反向（ρ<0 而非 ρ>0），但 collision 偏增 2.01%（l01_pairs 15256→15562），Stage 2 几何层改善（full_gini -3.2%、layer1 Gini -9.9%）未传导到 Stage3 test_R@10，机制本身实现正确，可由下轮 Agent G 决定是否重做。**

## 4. 后续建议

- iter9 不要简单"再做 P1 M2 reference point"——本轮 Agent C 假设 ρ>0 已被实证反向（ρ=-0.94~−0.97），重做需修方向；
- iter9 应转向：(a) Stage 0 几何桥接（修改 RqVae encoder 输入维度桥接）；或 (b) Stage 3-aware distillation term（iter7 REFUSE-LAUNCH 后重做）；或 (c) R36h ceiling 真正锁层面在 Stage 3 而非 Stage 1（v321 论证），Stage 1 端任何变更均无效——直接放弃 Stage 1 路线转向 Stage 3；
- Agent G 必须在 `iteration_bridge.md` 显式禁止"再尝试 P1 M2 reference point"。

## 5. 来源

- iter8 audit commit: `097c573a11dea2613f69306efbb44ba2ab8ccab5`
- iter8 logs: `stage2_RQ-VAE/curvature_RQ-VAE_iter8/logs/{hypothesis,sid_geometry,failure_analysis,stage3_test_final}_iter8.{md,json}`
- iter7 audit commit: `097c573a11dea2613f69306efbb44ba2ab8ccab5`
- iter6 audit commit: `e5b3c2a2631e013e7b6832124c3eafc1045e0dd5`
- v321 memory: `v321-r37-fail-sid-locks-baseline.md`

只读取，未修改 Python；未触发训练。