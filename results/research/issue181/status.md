# #181 现状：AF3 结构教师已实现（toy 图验收中）

## 已完成

- 新树 `stage2_RQ-VAE/TIGER_af3_structdistill_rq`（`ARM_NAME="TIGER_af3_structdistill_rq"`、`ARM_VARIANT="AF3SD"`），自 #180 树复制（其一级几何选码与数值验收已完成，符合 Issue "待 #180 可用后以其代码为只读起点"）。
- `model/af3_teacher.py`：
  - `af3_values`：`AF3(i,j) = 4 − deg(i) − deg(j) + 3m_ij`；
  - `quantiles`：TRAIN 图 AF3 的 q05/q95（冻结进 manifest）；
  - `weights`：`W_ij = 1 + clip((AF3−q05)/(q95−q05), 0, 1)`，**观测边权重恒 ≥1、桥梁边绝不为 0**（不把负 AF3 当负样本、不删边），对称化，未观测边为 0；稀疏 COO；
  - `normalized`：`P = D^{-1/2} W D^{-1/2}`，**零度节点保留全零行**（Issue 要求的显式退化规则）；
  - `teacher`：`T = (P + P²)/2`；
  - `build`：返回教师与要绑进 checkpoint 的 manifest（物品数、边数、q05/q95、规则原文）。
- 测试 `tests/test_af3_teacher.py`（6 项）：4-环上 AF3 手算恒等、q05/q95、桥梁边权重严格 ≥1 且未观测边为 0 且对称、教师对称且与定义式手算一致、孤立节点零行、确定性。

## 尚待

学生侧 STE 读出 `z_hat = z + stopgrad(q − z)`、候选集合采样（16 一跳 + 16 两跳 + 32 全局）、逐行 zscore+softmax 的 KL、λ* 探针标定、SD_OFF/GEOMETRY_OFF 两个开关的逐位一致、梯度与 DDP 检查、一 epoch 冒烟，然后 Stage2/Stage3 与结构保真报告。

## 学生侧已实现（12 passed / 1 failed）

新增 `model/struct_distill.py`：`ste_readout(z,q)=z+stopgrad(q−z)`（前向等于量化读出、梯度回编码器）、`candidate_sets`（16 一跳 + 16 两跳 + 32 全局、去重、排除自身、固定 seed、有界）、`zscore`（逐行、方差下限）、`relational_loss`（`KL(stopgrad p_G ‖ p_Q)`、教师侧 `detach`）、`student_cosines`（接受共享池或逐 anchor 集合）。

测试 `tests/test_struct_distill.py`（7 项）全部通过：STE 前向等于量化值（按浮点舍入 `allclose`）且梯度为 1、候选集合确定且有界且排除自身、学生=教师时 KL≈0、学生被置换时 KL 显著变大、**教师侧无梯度**、`zscore` 对常量行不产生 NaN、学生余弦读的是 STE 读出而不是量化前值。

## 仍未通过的一项：教师与定义式**不一致**（已记录，未修）

`test_teacher_is_symmetric_and_reproduces_the_definition` 失败，实测数字：

- `build()` 给出的 `T[0,0] = 0.25` ⇒ `p[0,1] = 0.7071` ⇒ `W[0,1] = 1.2247`（即 `u ≈ 0.2247`）；
- 用同一模块的公开助手（`quantiles` → `weights` → `normalized`）在测试里重算得到 `p[0,1] = 0.6325` ⇒ `W[0,1] ≈ 1.0954`（`u ≈ 0.0954`）。

两者仅差在 **u**，即 q05/q95 的取值不同，而两条路径本应传入同一组 `coo` 与 `degrees`——这是一个尚未定位的实现差异（可能出在 `weights` 内部重算度数与 `build` 传入度数的不一致，或 `coalesce` 对同向重复边的合并）。**未修，未放宽断言**：教师是整条机制的信号源，在它与定义式逐元素一致之前不能派发。

对称性、桥梁权重 ≥1、未观测为 0、孤立节点零行、确定性这些性质在另一项测试里已通过。

## 结论（本轮）

实现进度：教师（待修）+ 学生侧损失（已通过）。按 Issue 的"实现前验收"要求，第 1 项未完全通过即**不派发**；下一步是定位 `u` 的差异（打印两条路径的 q05/q95 与度数），修好后重跑 6 项教师验收。

## 定位结果（本轮）：**q05/q95 在 `build()` 内部与定义式不同**

用与失败测试完全相同的图 `[(0,1),(1,2),(2,3),(1,3)]` 复现，数字如下：

- AF3 值（`4−deg−deg+3`，度数 0:1、1:3、2:2、3:3）为 `{3, 2, 2, 1}`；
- 由该 AF3 表直接取分位数：**q05 = 1.15、q95 = 2.85**；
- 而 `build()` 返回的 **manifest 里 q05 = 2.0、q95 = 3.0**；
- 因此 `build()` 的 `T[0,0] = 0.25` 与定义式给的 `0.2` 不一致（`W[0,1]` 1.2247 vs 1.0954）。

即：**教师的分位数路径与 AF3 表本身对不上**，问题定位在 `build()` 内部（`degrees` 重算或传入顺序），不在 `weights`/`normalized`/`teacher` 这条已经与定义式逐元素相符的路径上。**未修、未放宽断言**，教师仍未通过验收 1，本臂**不派发**。

## 更正（本轮）：**教师是对的，是我的测试度数写错了，上一轮的"定位"作废**

上一轮把差异归到 `build()` 的分位数路径，**错了**。实测打印：

- `[(0,1),(1,2),(2,3),(1,3)]` 的真实度数是 **0:1、1:3、2:2、3:2**（节点 3 只出现在 (2,3) 与 (1,3)）；
- 因此 AF3 = `{3, 2, 3, 2}`，`np.quantile` 的 0.05/0.95 = **2.0 / 3.0**，与 manifest 完全一致；
- 我在测试里手写的度数表把节点 3 记成 3（`{0:1,1:3,2:2,3:3}`），据此算出的 AF3 = `{3,2,2,1}`、分位数 1.15/2.85 —— 这才是那组"对不上"的数字的来源。

修法：把两处手写度数改为 **3:2**，并在测试注释里写明这次误判的原因（不得靠放宽断言掩盖）。修正后 **13 passed / 0 failed**（教师 6 项 + 学生 7 项）。

教师当前状态：`T = (P+P²)/2` 与定义式逐元素一致、对称、桥梁边权重恒 ≥1、未观测边为 0、孤立节点零行、确定性 —— **验收 1 通过**。

## λ 调度与 λ* 标定（本轮）：**37 passed / 0 failed**

`model/struct_distill.py` 追加：
- `lambda_schedule(epoch)`：`epoch ≤ 1500 → 0`、`1500 < epoch < 3000` 线性升、`≥ 3000 → 1`（乘在 λ* 上），沿用 #179 的时间窗；
- `calibrate(grad_rq, grad_sd) = 0.1 · ‖∇L_RQ‖ / (‖∇L_SD‖ + 1e-12)`，即 Issue 的公式，分母有护栏、**不设上限**（测试按此语义断言有限而非被截断）。

测试：`lambda_schedule` 在 1/1500/2250/3000/3001 五点分别给出 0/0/0.5/1/1；`calibrate(2,4)=0.05`、分母为 0 时有限不抛。全树 **37 passed**（教师 6 + 学生 9 + #179/#180 遗留 22）。

## 尚待

把 SD 损失接进训练循环（替换 R48 对比损失、`SD_OFF` 时完全不执行）、λ* 探针实测并写 manifest、`SD_OFF`/`GEOMETRY_OFF` 逐位一致测试、4-rank DDP 与梯度方向检查、一 epoch 冒烟，然后 Stage2/Stage3 与结构保真报告。

## 稀疏教师 + 行式查询（本轮）

- `build_sparse(coo, n_items)`：同一教师，但 **`T=(P+P²)/2` 全程保持稀疏**（`P²` 用 `torch.sparse.mm`），避免 24k 商品时 576M 元素的稠密矩阵；返回 manifest 与稠密版同源。
- `SparseRows`：把合并后的 `indices` 按 `(row,col)` 排序，用 `searchsorted` 定位每行区间，再对候选列做二分查找；**取一行是 O(log nnz + C log C)，不做任何 N×N 分配**。`teacher_scores` 即其封装（调用方可缓存该 lookup）。
- 测试：**稀疏教师与稠密版逐元素一致**（`allclose 1e-12`，q05/q95 相同）、`teacher_scores` 对指定 (anchor, candidate) 与稠密矩阵取值逐项一致。教师套件 **8 passed**。

> 说明：我第一版 `teacher_scores` 用 `to_dense()` 取单行，虽然 toy 测试能过，但 24k 商品时等于把整张教师稠密化（约 4.6 GB），已换成本轮的 `SparseRows`，不保留稠密回退路径。

## 真实 TRAIN 图上的教师（本轮实测）

按与训练同源的 `seen_history[-1] → target` 取边（`TransitionPairs` 的口径），无向去重后建教师：

| 量 | 实测 |
|---|---|
| 原始 pair 数 | **339,519** |
| 去重无向边 | **323,633** |
| 商品数 | **24,587** |
| AF3 q05 / q95 | **−1189.0 / −18.0**（度数很大时 AF3 = 4−deg−deg+3 自然为大负数） |
| `T` 非零元（`T=(P+P²)/2`） | **51,144,592**（约为 24.6k² 的 8.5%） |
| 构建耗时 | **7.13 s** |
| 单次行式 gather（5 anchor × 5 候选） | **0.001 s** |

判读：
- `P²` 把教师从 32 万条边扩到 5100 万个非零元，float64 下约 **1.2 GB**（值 409 MB + 两路 int64 索引 818 MB）。当前主机可承受，但这是本臂最重的常驻开销；若后续 OOM，按 Issue 的"效率报告"条款记录归因，不改公式。
- q05/q95 都在**大负数**区间，说明 `u = clip((AF3−q05)/(q95−q05),0,1)` 在这张图上主要由度数驱动：绝大多数边落到 0 或 1（权重 1 或 2）。这是**必须报告的结构事实**——Issue 明确禁止把负 AF3 当负样本，本实现也确实只把它当连续分数。

## SD 单步接到真实量化器（本轮，**42 passed**）

`model/sd_step.py`：
- `quantised_readout(module, embeddings)`：走**训练器同一条路径**（`module.encoder` → `module.rq`），返回 `(z, q)`；
- `structural_loss(...)`：对 batch 的 anchors 取候选集合（16/16/32，去重、排除自身）→ 锚点与候选都过同一量化器取 `q` → `ste_readout` → `student_cosines` → 稀疏教师 `SparseRows.gather` 取关系分 → `relational_loss`；**padding 槽位用集合内最小值填充**，保证不进入 softmax。

测试 `tests/test_sd_step.py` 3 项 + 全树 **42 passed**：
- 损失有限、反传后**编码器参数确实收到梯度**（真实 `RQVAE`，不是合成张量）；
- 换掉嵌入集合会改变学生的码、从而改变损失（教师矩阵不变）；
- 教师侧（`SparseRows` 的 values 是普通张量、`requires_grad=False`）**不可能累积梯度**。

> 过程中修掉两处**我的测试**缺陷：小配置缺 `hidden_sizes`；以及对"教师张量"错误地调用 `.parameters()`。断言改得更贴近事实（教师根本没有参数），未放宽任何实现约束。

## 尚待

把 `structural_loss` 接进训练循环（替换 R48 对比、`SD_OFF` 时完全不执行）、λ* 探针实测并写 manifest、`SD_OFF`/`GEOMETRY_OFF` 逐位一致、4-rank DDP 检查、一 epoch 冒烟，然后 Stage2/Stage3 与结构保真报告。

## 训练循环接线 + λ* 探针（本轮）；**42 passed**

- 训练循环里 R48/#179 的**正负样本对比调用已删除**，替换为 `sd_step.structural_loss`，权重 `SD_LAMBDA = 0 if SD_OFF else lambda_schedule(epoch) · λ*`；`SD_OFF` 时结构项完全不执行（`SD_LAMBDA = 0.0`，分支不进入）。
- 教师在建 loader 前**构建一次**（TRAIN 边、`build_sparse`、`SparseRows`），并建一跳/两跳邻接表；`teacher nnz / build 秒数 / items` 打进日志。
- `probe_lambda_star`：在**第一个训练 batch** 上分别对 `L_RQ`（`mse(decoder(q), emb) + quant_loss`）与 `L_SD` 取编码器参数梯度范数，用 Issue 公式算 `λ* = 0.1·‖g_RQ‖/(‖g_SD‖+1e-12)`，随后**冻结**（`_record(rank, "lambda_probe", lambda_star=…)`），全程不再重测。
- 全树 **42 passed**（教师 8、学生 9、SD 单步 3、继承 22）。

## 尚待

`SD_OFF` 与"#180 几何选码开启、对比关闭、结构项为零"的同树对照逐位一致、4-rank DDP 检查、一 epoch 真实入口冒烟，然后 Stage2/Stage3 与结构保真报告。

## 冒烟慢的实测原因（本轮，未通过也未失败）

冒烟已跑约 7 分钟仍在 `smoke_pending`，进程存活、日志停在教师构建后。定位到**计算量设计问题**：`sd_step.structural_loss` 对**每个 anchor 的每个候选**单独过一次量化器，一次 step 最多 `A × 64` 次编码（`A` 为 batch 内 anchor 数，1024 时为 6.5 万次前向）——候选在 batch 内大量重复，属于**可去重的浪费**。

这不是公式问题，而是把"每 anchor 独立候选集合"直接照字面实现成逐行编码的结果。修法是**先把 batch 内所有候选 id 取并集、只编码一次**，再按 anchor 的候选位置 gather 回来；教师侧本来就是稀疏行式查询、不受影响。

## 候选并集编码（本轮，**43 passed**）

`structural_loss` 改为：先把 batch 内所有候选 id **取并集**（`torch.unique(..., return_inverse=True)`），锚点与并集各编码**一次**，再用 `inverse` 把行 gather 回每个 anchor 的候选位置。单 step 编码次数从"最多 anchors×64"降到"anchors + |并集|"；候选在 anchor 之间重复时收益最大。

新增回归测试 `test_repeated_candidates_are_encoded_once`：两个 anchor 共享全部候选时，统计 `quantised_readout` 的**行数总和**必须显著小于逐 anchor 版本的 `2×(1+64)+64` 上界。全树 **43 passed**。

## 冒烟第二轮：仍卡在教师构建（4 rank 各自构建）

并集编码已生效，但冒烟 4 分钟后仍停在 `torch.sparse.mm(p, p)`。单进程离线构建只需 **7.1 s**，因此变慢的原因不是算法而是**每个 rank 都独立构建一次**（4 份 `P²`、4 份 51M 非零元、4 倍内存与 CPU 争用）。修法：只在 rank 0 构建一次，再把稀疏 `T` 的 `indices/values`（以及 `q05/q95`）通过 `distributed.broadcast` 发给其余 rank；或在 rank 0 落盘后由其它 rank 读取，两条都只构建一次。

## 教师改为 rank 0 构建 + 广播（本轮，**44 passed**）

- `af3_teacher.pack/unpack`：把合并后的教师打成一个可传输的 payload（`shape`、`indices[int64]`、`values[float64]`、两个分位数），`unpack` 逐位还原。
- 训练器里：`distributed.is_initialized() and rank != 0` 时**不构建**，按 `nnz → shape/quantiles → indices/values` 的顺序 4 次 `broadcast` 接收；rank 0 构建一次后同样广播。所有 rank 最终持有**同一张表**（逐位一致），`q05/q95` 也随之传递（不再各自计算）。
- 测试 `test_pack_and_unpack_round_trip_the_teacher_bit_for_bit`：还原后的 `to_dense()` 与原件 `torch.equal`、q05/q95 相同。全树 **44 passed**。

## 冒烟失败 → 修复（`distributed` 未定义）

`NameError: name 'distributed' is not defined`：本树把 `torch.distributed` 导入为 **`dist`**（`import torch.distributed as dist`，第 52 行），而我在教师广播块里写了 `distributed.`。已把该块改为 `dist.`，并在项目解释器下确认 `train_rqvae.py` 可导入。

## 冒烟第三轮：rank 0 单构建后仍卡在 `sparse.mm`

改为 rank 0 构建后，冒烟 3 分钟仍停在 `torch.sparse.mm(p, p)`。离线单进程同一张图只要 7.1 s，差别在**内存**：51.1M 非零元的 `P²` 在 float64 下约 **1.2 GB**（值 409 MB + 两路 int64 索引 818 MB），`sparse.mm` 的中间 CSR 与时序还要再叠一层；四卡进程各自还常驻嵌入与模型，合计把主机压到换页区间。

修法（不改公式，只改数值宽度与存储形态）：`P`/`P²` 用 **float32 + int32 索引**（非零元数值 ≈ 0.6 GB，索引减半），并在 `build_sparse` 里先以 CSR 做乘法再转回 COO 存 `T`。这与 Issue 要求的"教师冻结、公式不变"并不冲突——它改变的是 dtype 与存储，不是 `W`、`P`、`P²` 的定义。
