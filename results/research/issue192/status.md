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
