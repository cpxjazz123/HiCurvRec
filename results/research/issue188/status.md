# #188 RCS：径向一致性损失（码与商品同壳、非成对）

## 本轮已完成

- 新树 `stage2_RQ-VAE/TIGER_rcs_rq`，自**冻结基线 `TIGER_mixed_rq`** 复制（`ARM_NAME="TIGER_rcs_rq"`、`ARM_VARIANT="RCS"`）。
- `model/rcs_loss.py`：
  - `radial(x, κ) = (2/√κ)·atanh(√κ·√(‖x‖²+eps))`，`atanh` 参数钳到 `1−1e-6`（`GEOMETRY_OFF` 时改用欧氏范数）；
  - `loss(item_positions, code_vectors, κ)`：`code_radius` 由 **含梯度的码向量** 计算（`c.detach() + (c − c.detach())` 保留原梯度通路），`scale = mean ρ_i`（detach、下限 1e-12），`L = mean[(ρ_i − ρ_{c(k)})²]/scale²`；`TERM_OFF` 精确返 0；
  - `calibrate_budget`：**同一冻结规则**（`λ* = min_over_blocks ‖g_base‖/‖g_struct‖`，无 ε/无回退），任一块结构梯度为 0 即抛 `CalibrationDegenerate`；
  - `is_pairwise_free`：对可执行代码断言无 `triple/margin/InfoNCE/BCE/edge_index/source_target`。
- 测试 `tests/test_rcs.py` **9 passed**，其中最关键一项是 **`test_both_blocks_receive_gradient_the_rad_failure_point`**：encoder 与 codebook 两个 block 的 `‖g_struct‖` 都 > 0，且 `λ* > 0` —— 这正是 #187 卡死的地方，本臂通过了。

## 一处诚实的更正

`atanh` 非线性 ⇒ 该损失是**无量纲但非尺度不变**的。我最初的测试断言"整体缩放后损失不变"，实测 0.01491 vs 0.01184 不同；已把测试改为断言**有限、为正、且两者确实不同**，并在注释里写明原因——不把数学上不成立的性质写成通过。

## 尚待

接进训练循环（位置取编码器输出、码向量取一级码含梯度、λ 首个 batch 标定后冻结、退化即 `SystemExit`）、`TERM_OFF` 与冻结基线逐位一致、冒烟后派发 Stage2。
