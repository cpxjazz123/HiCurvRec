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

## 测试修正后 **8 passed**

两处失败都是测试/检查器自身的问题，不是机制：
1. `singletons` 断言写成 1，实际组 `[0,0,1,2]` 有两个单样本组（组 1、组 2），**2 才是对的**；
2. `is_pairwise_free` 把自己模块的 docstring 与它自己的 token 列表也扫进去了——改为**只扫可执行代码**（去掉模块 docstring 与本函数自身的字面量）后通过。

## λ 预算（本轮，**10 passed**）

- 仓库既有规则 `calibrate_lambda_block_budget` 在 `TIGER_struct_mixed_rq/training_hooks.py:60`，但它内部调用的是 **`structure_term`（抽样三元组成对项）**——本臂禁止使用。
- 因此按"同一规则、换结构项"实现 `rad_loss.calibrate_budget(blocks, base_loss, struct_loss)`：`λ* = min_over_blocks ‖g_block^base‖/‖g_block^struct‖`，**无 ε、无种子切换、无回退**；任一 block 的结构梯度为 0 时抛 `CalibrationDegenerate`（`status="CALIBRATION_DEGENERATE"`），**不得静默继续训练**。
- 新测试两项：预算等于最紧 block 的比值；结构梯度为 0 时必须拒绝（而不是返回猜测值）。

## 训练循环接线（本轮）

- `rad_loss.step_terms(module, embeddings)`：走模块自身的 `encoder → rq → decoder`，返回 `(base_term, radius_term, record)`；`positions` 按 16/16 切分量化读出（欧氏块/球面块）。
- 训练器：`LAMBDA_RAD` 在首个 batch 上用 `rad_loss.calibrate_budget` 标定一次（encoder 与 codebook 两个 block），退化为 `SystemExit("CALIBRATION_DEGENERATE: …")`、**不训练**；随后每个 batch `loss += LAMBDA_RAD · radius_term`。`BEHAVIOUR_OFF` 时 `LAMBDA_RAD=0` 且分支不进入（与冻结基线逐位一致）。
- 训练器在项目解释器下可导入（`raw_module.rq.vq_layers` 为正确属性名）。
