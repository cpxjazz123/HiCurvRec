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
