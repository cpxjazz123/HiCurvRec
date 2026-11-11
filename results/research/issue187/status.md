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

## 冒烟：**CALIBRATION_DEGENERATE 生效并拒绝了训练** —— 并暴露一处实现缺陷

第二次冒烟输出 `CALIBRATION_DEGENERATE: block encoder has no structure gradient (norm=0.0)`，**没有开始训练**——预算规则按 Issue 要求 fail-closed，行为正确。

原因是我把位置取自**量化读出**：残差栈返回 `x + (x_q − x).detach()`，半径定义在它上面时对编码器的梯度精确为 0。Issue 要求 `pos_i` 与 `struct_mixed_rq` 的 `_positions` 同口径（编码器侧位置）。已改为从**编码器输出**取位置：`pos_e = z[:, :16]`、`pos_h = expmap0(z[:, 16:], curvature)`（同一指数映射、留在球内）。测试仍 **10 passed**。

## 预派发负结果：冻结预算规则与半径项在结构上不相容

改到编码器侧位置后，冒烟给出 **`CALIBRATION_DEGENERATE: block codebooks has no structure gradient (norm=0.0)`**，同样**拒绝训练**。

机制原因（不是实现缺陷）：
- 半径 `r_i = 2·atanh(‖pos_i^H‖)` 只依赖**位置**，而位置来自编码器输出；因此梯度**只到编码器**，对码本参数**恒为 0**；
- 反过来把位置取自量化读出（`x + (x_q − x).detach()`）时，编码器侧恒为 0（上一次冒烟已实测）；
- 仓库既有冻结规则 `calibrate_lambda_block_budget` 取 `min over {encoder, codebooks}`，任一块结构梯度为 0 即 `CALIBRATION_DEGENERATE`；Issue 明确该状态**不得训练**且**不得静默换规则/回退**。

结论：**按 Issue 规定的半径形式（只用位置、非成对）+ 冻结的 block 预算规则，二者在结构上不相容**——不存在既满足"半径只依赖位置"又让 codebook block 收到非零结构梯度的配置。按 Issue 第 5 条（λ 退化即拒绝训练、不得换规则），本臂记为**预派发负结果**并停止，不修改半径定义、不改预算规则、不偷换 block 集合。
