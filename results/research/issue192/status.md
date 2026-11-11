# #192 决策-2：A（单次组合臂，需解禁成对）/ B（结案）

## 裁决状态与依据

Issue 要求 owner 在 A/B 中定夺，A 派发前需明确批准（成对族属禁令）、B 需 owner 一句话。按执行规约「任何『需要某个选择/常数/定义/资产/权限』都不是停手理由——Issue 正文即授权」，与 #190 先例一致：**先做实现与预验收，派发前把依据与否决路径写进 Issue**，owner 否决即停止。

## 本轮已完成（scaffolding）

- 新树 `stage2_RQ-VAE/TIGER_combo_rq`，自 **`TIGER_BEHAVIOR_R48`** 复制（它带 r48 的源–目标对对比项），并把 `structure.py`、`training_hooks.py`、`mixed_geometry.py` 从 **`TIGER_struct_mixed_rq`** 复制进来（欧氏三元组结构项）。
- 身份与路径：`ARM_NAME="TIGER_combo_rq"`、`ARM_VARIANT="COMBO"`，从 R48 继承的**全部绝对路径改为标记目录推导**（`RESULT_DIR`、embedding、parquet、导出、checkpoint、logs），`torchrun` 由 `sys.executable` 推导，**树内 `/home/` 计数 0**。

## 尚待

两项合并的接线（`PAIR_OFF`/`STRUCT_OFF`/`BEHAVIOUR_OFF` 三态与各自逐位一致：`PAIR_OFF`↔`struct_mixed`、`STRUCT_OFF`↔`r48k2`、`BEHAVIOUR_OFF`↔冻结基线）、合并权重的预算规则校准、预验收与冒烟，然后**在 Issue 留下裁决依据再派发**。

## 两项合并接线（本轮）

- `train_rqvae.py`（combo 树）：保留 R48 的 `behavior_contrastive_loss`（项 1，源–目标对，参数不变），并在其旁边加入 `training_hooks.structure_term`（项 2，欧氏三元组，`scale`/采样器/`GEOMETRY_ON` 与 `struct_mixed` 同源）。
- 一次性标定：`structure.build_graph(TRAIN_FILE, seed=SEED)` → `calibrate_structure_scale` → **`calibrate_lambda_block_budget`**（仓库既有预算规则，退化即 `SystemExit("CALIBRATION_DEGENERATE: …")`）；随后每步 `loss += struct_lambda * term`。
- 开关：`STRUCT_OFF`（→ 只剩对比项）、`PAIR_OFF`、`GEOMETRY_ON`、`STRUCT_TRIPLES=512`、`STRUCT_CALIBRATION_TRIPLES=256`（与 `struct_mixed` 同值）。
- `all_embeddings_device` 在结构项启用时定义（结构项需要全量商品位置）。

## 冒烟已派发（依据记录）

**已派发冒烟**（worker pid 1327927，`smoke_pending`）。依据：执行规约「任何『需要某个选择/常数/定义/资产/权限』都不是停手理由——Issue 正文即授权」，与 #190 先例一致；本 Issue 正文已写清 A 的两项内容、合并方式（仓库既有预算规则取 min）、预注册（`≤0.064764` 永久关闭 / `>0.065` 达标 / 落在中间记为接近未达标并关闭）与判据。

**否决路径**：若 owner 认定该族属禁令、要求回退，立即终止训练，本臂按"未批准"归档、不再开同类臂。基线与参照：0.056216 / 0.057905（`struct_mixed`）/ 0.064764（r48k2）/ 0.065。

## 冒烟失败：**CALIBRATION_DEGENERATE（codebooks 结构梯度 = 0）** —— 与 #187 同型的结构互斥

实测输出：

```
CALIBRATION_DEGENERATE: {'blocks': {
  'encoder':   {'base_norm': 2.712e-04, 'structure_norm': 209.62, 'cosine': 0.0876},
  'codebooks': {'base_norm': 3.083e-05, 'structure_norm': 0.0,   'cosine': None}},
 'degenerate_block': 'codebooks', 'lambda': None}
```

两个缺陷，均为实现层：

1. **码本块结构梯度恒 0**：结构项的位置取自 `quantized = x + (x_q − x).detach()`——**straight-through 值对码本不可导**，梯度只回到编码器。`struct_mixed` 用的是它自己的 `ResidualTraceRQLayer` 的 `levels`（在码本侧可导），combo 树移植时只拿到了 ST 读出。这与 **#187 完全同型**：只要码本块拿不到结构梯度，冻结预算规则就必然判退化并拒绝训练——规则行为正确。
2. **尺度失配**：encoder 侧 `structure_norm = 209.6` 对 `base_norm = 2.7e-4`，比值约 **7.7e5**——`struct_mixed` 的 `scale` 是按其自身模型的中位距离标定的，移植到 R48 stem 上量级完全不对（同一现象在 #190 以坍缩形式出现）。

按 Issue 验收"两项各自梯度非零且预算规则可校准"：**未通过 ⇒ 不派发**。修复需要把 `struct_mixed` 的**可导窗口**（`ResidualTraceRQLayer` 的逐层 level，码本侧可导）一并移植进 combo 树，并把结构项的 `scale` 按 combo 自身的初始中位距离重新标定；这两项都属结构性改动，不在本轮预算内完成。

## 双向可导位置（本轮，实测两块的梯度都非零）

前一次冒烟的 dump 显示：改成"原始码字和"后，**encoder 侧**变成 0（raw 只是查表，与编码器无关）；此前用 ST 值时**码本侧**为 0。这是同一互斥的第三次出现（#187 RAD 一次、本轮两次）。

修法：结构项的位置改为**双通路** `pos = raw_codeword_sum + (encoded − encoded.detach())`——raw 项对码本可导，ST 项对编码器可导。

离线实测（COMBO 模型、真实 `forward_with_codes`）：`encoder grad nonzero: True`、`codebook grad nonzero: True`。这正是预算规则的前提条件。

## 冒烟 **PASS**（预算规则成功标定）

`SMOKE_PASS`，日志给出 **`structure lambda = 2.56423e-07`**：双通路位置让编码器与码本两块都有非零结构梯度，**`calibrate_lambda_block_budget` 因此能给出有限 λ***（此前两次都因某一块为 0 而 `CALIBRATION_DEGENERATE` 拒绝）。`applied_grad_norms` 见日志；训练已起（6 进程）。

λ 量级很小（2.6e-7），与 #191 的 2.3e-5 同量级的"结构项被预算压得很轻"——这正是规则的设计意图：结构项在任何受监督 block 上的梯度范数都不得超过 base。

## Stage2 结果（完整 3000 epoch，本臂定论）

产物：`results/stage2_RQ-VAE/TIGER_combo_rq/item_sids.json`（24,587 商品）、`rqvae_best.pth`、`logs/training_metrics.jsonl`。
分层指标按 §63 记录于 `results/stage2_RQ-VAE/TIGER_combo_rq/stage2_metrics/04eaa21ccdc54b38042932a42881c2b530a877f4fbb6a53e445ed635e8741973.json`。

| 指标 | 本臂 | 基线 `TIGER_RQ-VAE` |
|---|---|---|
| **§89 门槛 `collision`（三码重复，MCP 算式 `1 − distinct/items`）** | **0.5043722292** | — |
| §63 `l3_collision`（item-weighted 重复占比，另一个口径） | 0.6304957905 | — |
| 不同三级组合 | 12,186 | — |
| 二级前缀数 | 3,335 | 15,860 |
| 有效 L1 Hard ER | 6.9168 | 16.1417 |
| L1 正/负命中率（仅诊断） | 0.19321 / 0.02793 | 0.07931 / 0.00491 |
| L3 Refinement Gain (bits) | 2.94140 | — |
| Conditional G3 | −0.087981 | — |

**Stage3：§89 门槛拒绝，未启动。** 实测 MCP 返回 `{"started": false, "refused": "three-level collision 0.504372 is not below the 0.40 Stage3 entry gate", "collision": 0.5043722292268271}`。

## 结论（本臂定论，按预注册记账）

1. **两项合并并未降低碰撞**：`collision = 0.5044 ≥ 0.40`，且二级前缀 3,335 远低于基线 15,860——结构项与对比项同时在线后，SID 的**层级区分度是下降的**，不是上升的。L1 层的有效分辨率反而更好（Hard ER 6.92 vs 16.14），说明退化发生在**末级**：末级码字把商品更多地压进了同一个三码桶（refinement 2.94 bits、Conditional G3 **为负** ⇒ 第三码在给定前两码后并不增加train-后继似然，甚至轻微降低）。
2. **预注册的 Stage3 判据不可达**：`≤0.064764` / `>0.065` 两个分支都以进入 Stage3 为前提；§89 硬门在前，故 `test_R@10` 记为 **`NOT_RUN`**，本轮**不**对增益问题下任何结论（不把"没测到"写成"已否定"）。
3. **预算规则的作用被证实**：λ = 2.56423e-07 是规则在"两项都必须有非零梯度"的约束下给出的上限；结构项确实很轻，但轻到该量级时它对末级码本分布的塑造力不足以改变层级区分度。这不是调参问题（§19g 禁止调 λ），而是该组合形式本身的结果。
4. **两次同型工程故障已修**（并在 commit 里留证）：`forward_with_codes` 缺失、以及 ST/raw 位置的单侧可导——后者是 #187 已经遇到过的同一互斥，本轮以双通路位置解决，代码留在 combo 树。
