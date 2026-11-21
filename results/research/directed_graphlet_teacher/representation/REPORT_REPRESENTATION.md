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

**协议**：输入是状态自身构建的**互惠** kNN 图（k=10）+ 节点向量；答案 `own_orbit`（376 角色）只来自
真实行为图；12266 个有角色商品，80/20 划分；300 epoch；打乱标签对照组**复用同一初始权重、同一梯度裁剪**。

| 状态 | 交叉熵 | 常数先验 | 打乱标签对照 | 相对先验 | 训练/评估子图平均度 |
|---|---|---|---|---|---|
| 随机 32 维 | 3.6220 | 3.4794 | 3.6080 | −0.1426 | 8.3 / 8.2 |
| 语义 768 | 3.4691 | 3.4794 | 3.5417 | +0.0103 | 3.8 / 3.9 |
| GNN 冻结 128 | 3.4776 | 3.4794 | 3.5138 | +0.0018 | 4.3 / 4.3 |

**上一版的"标签损坏"结论已撤回。** 直接检查（`target_diagnostics`）：12266 行**全部**行和为 1
（最大偏差 5e-8）、无负值、无非有限值；归一化后熵中位 0 bit、46.1% 行有效角色 ≥2。
之前报告的 7.26 nats 来自把常数先验当 oracle、又混用 log 底数，与标签正确性无关。

**现在能说**：三种状态相对常数先验的增益都在 0.01 nats 量级，`gnn` 与打乱对照只差 0.036 nats。
这条测试不够灵敏，**既不能证明 GNN/KNN 帮助保留角色，也不能证明无帮助**。要定论需改目标
（只保留 ≥2 有效角色的行，或改用可预测的聚合角色），并把训练/评估子图规模对齐。

## 3. checkpoint 能否复现它旁边的 SID 表（`checkpoint_sid_pairs.py`、`checkpoint_sid_pairs.json`）

判定：用 `rqvae_best.pth` 在冻结语义输入上前向，`infer_use_sk=False`（trainer 导出 SID 的同一推理路径），
分别报告 **token 一致率**与**完整三层 SID 一致率**；只有完整 SID 100% 才记 MATCH。

| 判定 | 臂数 | 说明 |
|---|---|---|
| MATCH（完整 SID 100%） | 6 | `drq_D1`、`g4_closure_A/B/C/D`、`sinkhorn_all_levels_rq` |
| 可复现但非完全一致 | 19 | token 0.9886–0.9995，完整 SID 0.9786–0.9995 |
| 明确错配 | 1 | `af3_curvature_factor_mass_rq`（token 0.8389 / SID 0.6845） |
| 疑似错配，配置未恢复 | 1 | `TIGER_RQ-VAE`（0.0030 / 0.0000） |
| checkpoint 结构与基线不同，无法验证 | 6 | `graphlet_codebook*`、`mixed_rq_F0`、`role_geom*` |

**边界**：本脚本对每个臂都用**基线 config 与欧氏距离**重建模型（`arm_space` 字段记录了每臂的
导出 manifest 与 checkpoint `selection`）。因此"可复现但非完全一致"只能说明基线结构能复现该臂
98% 以上的 SID，**不能**据此判为错配；`TIGER_RQ-VAE` 的 0.3% 同样需要按它自己的导出配置复核后
才能定性。上一版"浮点误差导致小差异"的说法也收回——每臂差异量级已单独记录，未做归因。

## 未做

- `rqvae_latent_32` / `rqvae_q_32` 两行角色测试没有跑：`rqvae_latent.npz` 从 `TIGER_RQ-VAE` 导出，
  而该臂的 SID 配对已被判为错配，等定下用哪个臂的 checkpoint 再跑。
- 未启动正式 Stage2，未走 `verification_check`。
