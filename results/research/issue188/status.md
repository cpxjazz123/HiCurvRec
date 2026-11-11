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

## 训练循环接线（本轮）

- `rcs_loss.step_terms(module, embeddings, κ)`：位置取**编码器输出**（欧氏半原样、双曲半过 `expmap0`），码向量取 `module.rq.levels[0].get_code_embs()[token_1]`——**带梯度**，这正是让 codebook block 进入预算视野的关键。
- 训练器：数据集改为 `(embedding, id)`；λ 在首个 batch 上标定一次（encoder + codebooks 两 block），退化即 `SystemExit("CALIBRATION_DEGENERATE: …")`；随后每步 `loss += λ·L_rcs`；`TERM_OFF` 时 λ=0、分支不进入（与冻结基线逐位一致）。
- 训练器在项目解释器下可导入。

## 冒烟 **PASS**：两块梯度都非零（#187 的失败点已清除）

`SMOKE_PASS`，日志给出 **`rcs lambda=2.55933e-05`** —— 说明预算规则**成功标定**，即 encoder 与 codebook 两个 block 的结构梯度都 **> 0**（若任一为 0，规则会 `CALIBRATION_DEGENERATE` 并拒绝训练；#187 正是死在这里）。

`applied_grad_norms` 八项全为有限正值（0.00075–0.00269）。训练已起（6 进程），epoch 265/3000、loss 1.42e-4。λ 与各 block 比值已写入 metrics（`rcs_lambda` 事件）。

### 两点必须记清

1. **比值**：metrics 的 `rcs_lambda` 显示 `ratios = {encoder: 2.2974e-05, codebooks: 4.5491e-04}`，`λ*` 取最小者（encoder 侧）。codebook 侧比值大约 20 倍，说明**码本块的结构梯度比编码器块小约 20 倍但确实非零**——这正是本臂相对 #187 的差别所在，也说明 λ 实际被编码器块支配。
2. **rank 间差异**：日志打印 `2.55933e-05`、metrics 记录 `2.2974e-05`——每个 rank 用**自己**的首个 batch 标定，因此各 rank 的 λ 略有不同（同一规则、不同样本）。本轮如实并列记录，不取平均、不事后统一。
