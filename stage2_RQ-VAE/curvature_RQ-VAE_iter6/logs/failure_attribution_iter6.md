# iter6 Failure Attribution Auditor（Agent F）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter6
- **机制**：`iter6_sinkhorn_linear_eps_anneal`（P1）
- **5 类标签**：**GEOMETRY_MISMATCH**（条件 1/2/4 PASS、条件 3 FAIL）
- **审计日期**：2026-09-21

## 1. 4 个硬条件逐项证据

### 条件 1 — 实现正确（Implementation Auditor）

**结论：PASS**

- iter6 修改的 Python 文件（`curvature_config.py`、`curvature_RQ-VAE.py`、`modules/quantize.py`、`modules/rqvae.py`、`scripts/grad_check.py`、`scripts/mvg_check.py`、`scripts/export_sids_for_stage3.py`）均通过 `py_compile`。
- MVG 4 层全 PASS：
  - L1 Graph: `total_loss.requires_grad=True grad_fn=True value=457.26`；
  - L2 Gradient: 3 个 `layers.{0,1,2}.embedding.weight` grad L2 = 2.769 / 5.018 / 5.521（>1e-12）；
  - L3 Update: 5 步 update_ratio_floor ≥ 1e-7；
  - L4 Behavior: ON/OFF 200 步后 `loss_diff = 2.438e+00 > 1e-6`，`on_attn_temp = 1.574e-01`（P1 路径无 attention，用 ε 替代）；
- counterfactual rollback（OFF 路径把 `get_eps` 锁在 `sk_eps_min`）能恢复 v318 baseline 行为。

### 条件 2 — 机制激活（DE-1~3）

**结论：PASS**

- DE-1：ρ(c, ε) = +1.0000（n=300）≥ 0.95；
- DE-2：ρ(c, log2(q_max/q_mean)) = −0.958 / −0.932 / −0.923（三层）≤ −0.7；
- DE-3：step100000 unique=23221 ≥ iter5 22675。

### 条件 3 — 几何方向匹配（Agent D 三态）

**结论：FAIL**

- Agent D 三态 = **PARTIAL / METRIC_MISMATCH**；
- DE 全部 PASS 但 PH-2（collision rate ≤ iter5 +0.5%）FAIL（实际 +5.71%）；
- Agent C 的 proxy 假设 PH-2 不达成 → 几何方向不完全匹配。

### 条件 4 — Pipeline 一致

**结论：PASS**

- SID (24587, 4) int64 全表 unique 校验通过；
- Stage3 输入协议与 iter4 baseline 字节级一致（n_digit=4、vocab=784、BEAM_SIZE=20）；
- Stage3 trainer 配置未修改；
- `item_sids_recbole.json` 已恢复 baseline SHA256 `1d0b8177…`。

## 2. 5 类标签综合裁决

| 标签 | 触发 | 是否适用 |
|---|---|---|
| `IMPLEMENTATION_FAIL` | 代码/数学实现错 | ❌（条件 1 PASS） |
| `ACTIVATION_FAIL` | 代码对但机制未产生足够影响 | ❌（条件 2 PASS） |
| `GEOMETRY_MISMATCH` | 机制激活但没有产生 Agent C 预期的 SID 结构 | ✅ **条件 3 FAIL；与条件 1/2/4 PASS 同时出现** |
| `PIPELINE_FAIL` | SID/export/Stage3 接口问题 | ❌（条件 4 PASS） |
| `TRUE_MECHANISM_FAIL` | 4 个条件全部满足 | ❌（条件 3 FAIL） |

**最终标签：GEOMETRY_MISMATCH**

## 3. 不写入 ledger 的依据

`failed_mechanism_ledger.md` 仅接受 `TRUE_MECHANISM_FAIL`（条件 1/2/3/4 全部满足）。本轮条件 3 FAIL（Agent C 的 proxy 假设 PH-2 不达成），机制虽然激活（条件 2 PASS）但 SID 几何未完全匹配 Agent C 期望，因此不构成"机制失败"。

iter6 应记录为：**Sinkhorn 线性 ε-anneal 真实反向（de-Δ-full_gini=-27%），但 collision proxy +5.71% 超过 Agent C 容差，机制路径激活但 SID 几何与 Agent C 预期不完全一致**。机制本身（ε-anneal + cyclic c(t) 乘法耦合）实现正确，可由下轮 Agent G 决定是否重做。

## 4. 后续建议

- iter7 不要简单"再做 ε-anneal"，因为该路线 PH-2 已知 collision 偏增 5.71%；
- iter7 应转向"绕开 ε-anneal 但保留 cyclic c(t) 联动"的方向，例如：(a) per-item 路由但用更强 base；(b) c(t)-dependent VQ-VAE 的 commitment loss；(c) Stage 0 几何桥接（不修改 Stage2 内部）；
- Agent G 必须在 `iteration_bridge.md` 显式禁止"再尝试线性 ε-anneal 而不修复 PH-2 collision 漂移"。

## 5. 来源

- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter6/logs/{hypothesis_iter6,sid_geometry_iter6,failure_analysis_iter6,sid_quality_iter6,stage3_test_final_iter6}.{md,json}
- /home/wlia0047/.claude/skills/curvature-rqvae-iter/references/failed_mechanism_ledger.md
- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/iteration_bridge.md

只读取，未修改 Python。