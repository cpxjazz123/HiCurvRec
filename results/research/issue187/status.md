# #187 RAD：半径型结构损失（一级码分组、非成对）

## 本轮已完成

- 新树 `stage2_RQ-VAE/TIGER_rad_rq`，自**冻结基线 `TIGER_mixed_rq`** 复制（`ARM_NAME="TIGER_rad_rq"`、`ARM_VARIANT="RAD"`）。
- `model/rad_loss.py`：
  - `hyperbolic_radius(pos_h) = 2·atanh(‖pos_h‖)`（钳到 `1−1e-6`，`p=0 ⇒ r=0`）；
  - `radii(pos_e, pos_h)`：`GEOMETRY_OFF` 时改用联合欧氏范数；
  - `radius_loss(r_i, group_ids)`：`R_k = mean_{i∈G_k} r_i`（**detach**）、`scale = mean_i r_i`（detach、下限 1e-12），`L_rad = mean_i (r_i − R_{k(i)})²/scale²`；**空码剔除**、**单样本组贡献 0**；返回 `{groups, empty, singletons, scale}` 记录；`BEHAVIOUR_OFF` 时**精确返回 0**；
  - `is_pairwise_free(source)`：对源码做机器断言，禁止出现 `triple / margin / InfoNCE / BCE / edge_index / source_target`。
- 测试 `tests/test_rad.py` 8 项：半径恒等式（rtol 1e-12）、零位置零半径、原点与边界处梯度有限、手算组均值与损失、组内等半径 ⇒ 0、**无量纲性**（整体 ×10 损失不变）、开关关闭精确 0、无成对机制。

## 尚待

把 `radius_loss` 接进训练循环（一级 token 作为 `group_ids`、λ 用仓库既有 `calibrate_lambda_block_budget` 初始化一次后冻结、`CALIBRATION_DEGENERATE` 时拒绝训练）、`BEHAVIOUR_OFF` 与冻结基线逐位一致、冒烟后派发 Stage2。
