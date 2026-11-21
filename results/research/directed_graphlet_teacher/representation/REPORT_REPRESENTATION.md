# 表示层实验：冻结 GNN 表、KNN 角色预测、checkpoint 与 SID 的配对

三件互不相同的事，各自有数字。

## 1. 冻结 GNN 表（`build_gnn_embedding.py`、`manifest.json`）

| 项 | 值 |
|---|---|
| 来源 | `C_residual_edge_main_seed226.pth`，其 `param_sha` 在加载时自检通过 |
| 图 | 训练时的消息图，源图 sha `1ffe2f5a…`（与 `data_contract.json` 的 `graph_sha` 一致），192528 行有向边 |
| 语义输入 | `sentence_t5.npy` sha `6490c717…`，(24587, 768) |
| 输出 | (24587, 128) float32，sha `13dc755f…`，**与既有 relation 目标逐行余弦 = 1.000** |
| 覆盖 | 133 个孤立商品（度 0），有边商品 24454，度中位 8 |
| 训练后是否还需要图 | 否：表是冻结文件，RQ-VAE 只读它 |

因为与 relation 目标逐行相同，这份表就是"GNN 输出"本身；用它做输入等价于把 GNN 表示喂给 RQ-VAE，
不需要改任何 Stage2 源码。

## 2. KNN 输入 → 预测真实 Orbit 角色（`role_probe.py`、`role_probe.json`）

**协议**：输入是状态自身构建的**互惠** kNN 图（k=10；随机 32 维输入平均度 8.3，语义/GNN 只有 3.9–4.0，
说明真实表示都集中在低维流形上）+ 节点向量；答案 `own_orbit`（376 角色）**只来自真实行为图**；
12266 个有角色商品，80/20 划分；300 epoch；对照为打乱标签。

| 状态 | 交叉熵 | 常数先验 | 打乱标签对照 | 相对先验 |
|---|---|---|---|---|
| 随机 32 维 | 3.6220 | 3.4794 | 3.6055 | −0.143 |
| 语义 768 | 3.4691 | 3.4794 | 3.5327 | **+0.010** |
| GNN 冻结 128 | 3.4776 | 3.4794 | 3.5150 | +0.002 |

**结论：目前无法用这条测试判定表示好坏，因为标签本身不合格。**

| 标签诊断（`headroom`） | 值 |
|---|---|
| 中位有效角色数 | 1.0（46% 商品有 ≥2 个角色） |
| 常数先验交叉熵 | 3.487 nats |
| 用该表自己的先验算的 oracle 交叉熵 | **7.257 nats** |
| 逐行行和 | 被更新的商品 >1（表混入了被删除实例的计数） |

只要一行里出现被删除的实例，行和就大于 1，先验就被压扁，oracle 比常数先验差一倍。**修复顺序**：
先把 `own_orbit` 的行和归一化（或改用可预测的聚合角色，例如 family 级 5 维），再重跑同一协议。
在此之前，"语义 +0.010 nats"这种量级不能解读为预测能力（同量级的正负号已经在两个状态间翻转）。

## 3. checkpoint 能否复现它旁边的 SID 表（`checkpoint_sid_pairs.py`、`checkpoint_sid_pairs.json`）

判定：用 `rqvae_best.pth` 在冻结语义输入上前向，`infer_use_sk=False`（与 trainer 导出 SID 时同一推理路径），
逐项比较三码；一致率 > 0.999 记 MATCH。

**31 个臂：7 个 MATCH，18 个 0.99–0.999（边界），6 个 checkpoint 无法载入。**

- MATCH：`TIGER_drq_D1`、`g4_closure_A/B/C/D`、`sinkhorn_all_levels_rq`、`Uniform_KNN_control`
- 0.99 附近：这些差异量级与"浮点/集合通信导致的近边界跳码"一致，**不能据此判为错配**
- 明确错配：**`TIGER_RQ-VAE`（一致率 0.003）**——仓库里那份 bootstrap 产物 SID 不是这个 checkpoint 导出的
- 无法载入：`graphlet_codebook*`、`mixed_rq_F0`、`role_geom*`，这些臂的 checkpoint 结构与冻结基线不同，
  本脚本无法验证，记为 `CHECKPOINT_UNUSABLE`（不当作通过）

**这条结论取代**先前"只有 0.30% 一致"的说法：那次跑分用了 `infer_use_sk=True`，与 trainer 的
`infer_use_sk=False` 不同，错在比对方。现在 25 个臂能被验证，其中 7 个完全一致。

## 未做

- `rqvae_latent_32` / `rqvae_q_32` 两行角色测试没有跑：`rqvae_latent.npz` 从 `TIGER_RQ-VAE` 导出，
  而该臂的 SID 配对已被判为错配，等定下用哪个臂的 checkpoint 再跑。
- 未启动正式 Stage2，未走 `verification_check`。
