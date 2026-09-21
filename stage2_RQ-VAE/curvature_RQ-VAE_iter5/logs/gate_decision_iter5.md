# iter5 Gate Decision：HiHPQ 式固定容量层次双曲 product-codebook attention

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter5
- **机制**：`iter5_hyperbolic_codebook_attention`（P2 HiHPQ）
- **最终裁决**：**NO-GO（不进 Stage3）**
- **裁决规则**：仅当 Stage3 完整运行后的 `test_R@10 > 0.065` 时才允许 PROMOTE；Agent D 给出 `MECHANISM_FAIL` → skill §5 硬约束 NO-GO，禁止进入 Stage3。
- **审计基线提交**：`127a75c7654c31084283c17258a61eb9e68572f8`（起始），本轮所有 agent 产物在 `stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/` 与 `stage3_RQ-VAE/curvature_RQ-VAE_iter5/logs/`。
- **审计日期**：2026-09-21

## 1. Agent 串行证据链

1. **Agent A**：`logs/lit_search_iter5.md`（≥3 个候选 P1/P2/P3；只检索，不评判）。
2. **Agent B**：`logs/direction_decision_iter5.md`（唯一推荐 P2 HiHPQ 式固定容量层次双曲 product-codebook attention）。
3. **Agent C**：`logs/hypothesis_iter5.md`（直接效应 DE-1 temperature 与 c(t) 联动 / DE-2 key_proj 被更新 / DE-3 attention logits 进入 distances；proxy PH-1~4）。
4. **MVG**：iter5 4 层全部 PASS——Graph OK；attention 12 参数 grad L2 1e-1~4e+2；5 步 update > 1e-7；ON/OFF 200 步后 loss 差 557、attn temp 0.439。
5. **Stage2**：`logs/train_iter5.log` + `train_migrated.log`，`global_step=100000` `total time=1077s`。
6. **Agent D**：`logs/sid_geometry_iter5.md`（三态 `MECHANISM_FAIL`；DE-1 反向：c=0.330 时 L0 temp=8.342，c=1.000 时 L0 temp=3.48）。
7. **Agent E**：`logs/failure_analysis_iter5.md`（类 1 机制未生效；Δfull_gini=-0.0467, ΔH(L1|L0)=+0.3989, Δl01_pairs=+2874）。
8. **Agent F**：`logs/failure_attribution_iter5.md`（5 类标签 = `ACTIVATION_FAIL`；不写 ledger）。
9. **Agent G**：`logs/iteration_bridge.md`（dominant bottleneck + forbidden next directions + iter6 objective）。

## 2. 4 层 MVG PASS 证据（迭代开始前确认）

- L1：`total_loss.requires_grad=True grad_fn=True value=1468.07`；
- L2：attention 12 参数 grad L2：`temperature_scale` 0.631/146.0、`query_proj.weight` 1.456/433.5、`key_proj.weight` 0.205/37.62；
- L3：5 步 update ratio floor ≥ 1e-7；
- L4：200 步 ON/OFF `loss_diff=557` 与 `attn_temp=0.439`，行为差异明显。

## 3. Stage2 训练记录

- DDP：`world_size=4`，`master_port=50200`，`find_unused_parameters=True`。
- 输入 embedding：`item_emb.npy`，形状 `(24587, 768)`。
- 训练目标：`MAX_GLOBAL_STEPS=100000`，checkpoint 间隔 `10000`。
- `[train] done at global_step=100000, total time=1077.0s`。
- step100000 描述性指标：`full_gini=0.07385`、`per_layer=[0.0742/0.2769/0.5196]`、`n_unique_full=22675/24587`、`l01_pairs=14692`、`H(L1|L0)=5.53756`。
- 注意力 27 个 `[iter5][attn]` 采样点：layer 0 temp 0.7→4.5（与 c(t) 反向）；layer 1 temp 0.7→2.5；layer 2 temp 0.7→1.2。

## 4. 失败根因（Agent E/F 共识）

`modules/quantize.py:62-73` `get_temperature()` 三联实现：

```python
soft = softplus(temperature_scale) + 0.1
return soft * (cyclic_factor.detach() + 0.25)
```

- `softplus(temperature_scale)` 在 100k step 内从 1.14 漂到 ~7.08（layer 0）；
- `cyclic_factor.detach()` 阻断 c(t) → temperature 反向传播；
- `+0.25` 常数 bias 把 cyclic 振幅 [0.3, 1.0] 进一步压扁。

最终 `attention_logits = logits / temperature` 的实际温度由 `softplus` 单调漂移主导，cyclic c(t) 联动失效（DE-1 反向）。`attention_logits` 量级（L0 min≈-0.0148, max≈0.0069）远小于 distances 量级（典型 L0 距离 1~5），机制未产生预期直接效应。

## 5. Agent F 5 类标签

`ACTIVATION_FAIL`（4 个硬条件中第 2 项 FAIL：DE-1 反向 → 机制未生效）。该标签**不**写入 `references/failed_mechanism_ledger.md`（仅 `TRUE_MECHANISM_FAIL` 进 ledger）。

## 6. Stage3 未执行

按 skill §5 硬约束，Agent D `MECHANISM_FAIL` → 不进入 Stage3；因此本轮**无 `test_R@10`**；唯一硬目标 `test_R@10 > 0.065` 既未达成也未测试。

## 7. 决策

**NO-GO（不进 Stage3）：iter5 HiHPQ 式 hyperbolic codebook attention 触发 Agent D `MECHANISM_FAIL`，硬目标未达成。**

下一轮（iter6）必须按 Agent G `iteration_bridge.md` 的 dominant bottleneck + forbidden directions 重置检索方向，避免"再加 attention / softplus 长漂路径 / cyclic_factor.detach() / +0.25 bias"。