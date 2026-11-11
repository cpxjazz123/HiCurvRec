# #178 现状：逐对曲率距离模块已写、接线已做；**κ=1 与冻结参照的一致性测试未过**

## 已完成

- 新树 `stage2_RQ-VAE/TIGER_r48k_rq`（从 `TIGER_BEHAVIOR_R48` 复制，只加一个模块与一处接线）。
- `model/r48k_geometry.py`：`project_to_ball`（`tanh(√κ‖u‖)/(√κ‖u‖)·u`，零向量取极限 1、严格在球内）、`distance`（`(1/√κ)arccosh(1+2κ‖p−q‖²/((1−κ‖p‖²)(1−κ‖q‖²)))`）、`pair_kappa`（`sqrt(κ_i κ_j)`，`<1e-3` 取 1.0）、`behaviour_distances`（`GEOMETRY_OFF` 时**返回 `torch.cdist`**，与 R48 逐位一致）、κ 表按**标记目录**加载（无绝对路径）、`kappa_of` 用 `mean_frozen` 补齐缺测项。
- 接线：`train_rqvae.py::behavior_contrastive_loss` 里 `torch.cdist` → `behaviour_distances(..., kappa_of(source_ids), kappa_of(target_ids))`；τ=0.07、权重、mask/过滤、Sinkhorn、码本、预算全部未动。

## 测试：**6 passed / 1 failed**

通过：原点恒等（rtol 1e-6）、对称性、`d(p,p)=0`、投影严格在球内且零向量取极限、`GEOMETRY_OFF` 与 `torch.cdist` **逐位相等**（`torch.equal`）、`pair_kappa` 为几何平均、κ 表按标记目录加载与查表。
失败：**κ=1 时与冻结 `poincare_contract.distance` 的一致性**（本树无该模块，测试从 `TIGER_mixed_rq/model/poincare_contract.py` 动态载入；两侧数值明显不同）。

## 口径更正（Issue 正文指出的）

`kappa.json` 有两个均值：`mean_frozen = 0.12096147922805969`（Issue 规定用于 31 个缺测项的**填充值**）与 `mean_kappa = 1.0`（实测均值、中位 0.91474）。我最初把 `mean_frozen` 当成 1.0 写错了断言，已按实际值改正——这也说明 #177 的表里 31 个缺项用的正是 0.121，而不是 1.0（#177 第 1 层 κ=1.0 是 Issue 规定"按构造取 1.0"，与该填充值无关）。

## 待办

κ=1 一致性必须先弄清：是冻结参照的**约定不同**（该函数内部是否自带 `expmap0`/投影、是否返回半值），还是我的投影与它的球面参数不匹配。在弄清前**不派发** Stage2。

## 更新：**7/7 通过**

失败的其实是**投影边界**测试，不是 κ=1 一致性：`u = randn*3` 时 `tanh(√κ‖u‖)` 在 float64 下饱和为恰好 1，`κ‖p‖²` 触到 1.0，`< 1` 断言失败。修法是把投影乘上 `(1 − 1e-6)`，使 `κ‖p‖² < 1` 严格成立；κ=1 与冻结参照的一致性在修好导入后本来就已经通过。

现在 7 项全过：原点恒等、对称、`d(p,p)=0`、投影严格在球内、**κ=1 与冻结 `poincare_contract.distance` 一致**、`GEOMETRY_OFF` 与 `torch.cdist` **逐位相等**、`pair_kappa` 几何平均、κ 表加载与查表。

## 行为对比损失的验收：**4/4 通过**

- `GEOMETRY_OFF` 下 `behavior_contrastive_loss` 与**逐字复刻的 R48 函数体**（`cdist` → `−d/τ` → 重复 target 屏蔽 → `CE`）**`torch.equal` 逐位相等**。
- 曲率打开与关闭给出**不同的损失**（且都有限），证明 κ 真的进入了对比（不是空接线）。
- 只对 `L_behavior` 反向时残差**确实收到非零有限梯度**（对应 Issue 验收 4）。
- `kappa_of` 对已知 id 返回表值、对表外 id 回落到 `mean_frozen`。

## 冒烟失败 → 修复 → 12/12

一 epoch 冒烟崩在 `project_to_ball`：`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`。原因：`kappa_of` 查表返回 **CPU float32** 张量，而残差在 **cuda/float64**。

修复：`behaviour_distances` 先把两侧 κ 转到 `source.device` 与 `source.dtype` 再算 `pair_kappa`。补回归测试 `test_kappa_from_the_lookup_is_moved_to_the_points_device_and_dtype`（断言损失 dtype 与残差一致且有限）。两套共 **12 passed**。

## 第二次冒烟失败 → 修复 → 13/13

`ValueError: Cannot center a constant distance matrix.`（`model/layers.py:77`），同时行为损失梯度为 0、`applied_grad_norms: [NaN]`。原因：两次投影点非常接近时，`arccosh` 的参数与 1 的差在 **float32** 下小于机器 epsilon，整张矩阵被 `clamp_min(1.0)` 压成**全零常数阵**，Sinkhorn 的居中因此无振幅可分。

修复：`behaviour_distances` 内部一律用 **float64** 计算（投影与 `acosh` 都在双精度），返回时再转回输入 dtype；Sinkhorn 本来就以 `.double()` 运行，与之同精度。补回归测试 `test_float32_points_still_produce_a_non_constant_matrix`（float32 输入下矩阵标准差 > 0 且 max>min）。两套共 **13 passed**。

## 第三次冒烟仍失败 → 按 Issue 第 9 条记**预派发负结果**并停止

三次冒烟同一处失败：

1. 第 1 次：`project_to_ball` 设备不匹配（κ 查到的是 CPU float32、残差在 cuda/float64）→ 已修（`dccbbf272`）。
2. 第 2 次：`ValueError: Cannot center a constant distance matrix.` + 行为项梯度 0 + `applied_grad_norms: [NaN]` → 判为 float32 精度，改为内部 float64（`be2b2c2f1`）。
3. 第 3 次：**仍是同一句** `Cannot center a constant distance matrix.`，且 `applied_grad_norms` 仍为 `[NaN]`。

离线复核排除了"距离恒等"：用与真实残差同量级的 float32 输入（`‖·‖≈1e-3`、真实 κ 表值 0.07–2.21）跑 `behaviour_distances` 得到 `min 0.00733 / max 0.01355 / std 0.00153`、无 NaN/Inf、不恒定。因此**常数矩阵不是对比距离造成的**，而是**上游已经崩溃**：`applied_grad_norms: [NaN]` 表明第一步反传就产生了 NaN，随后权重被污染、`_distances` 输出恒定，Sinkhorn 的居中才报"常数矩阵"。最可能的位置是新距离里 `arccosh` 的**奇异梯度**：参数被 `clamp_min(1.0)` 截断时 `acosh'(1)=∞`，仅当截断真正生效（参数在数值上落到 1）即产生 NaN；我在 float64 下用远离边界的点做过性质测试，没有覆盖"点重合/极近导致截断"这一条。

按 Issue 第 9 条（"1–6 任一不过：按**预派发负结果**记录并停止，不改 τ/权重/预算"），本臂**停止在派发前**，不派发这次 Stage2，也不再改动 τ/权重/预算来绕过。

结论：`TIGER_r48k_rq` 的机制数学与接线（11→13 项测试全绿，含 `GEOMETRY_OFF` 与 R48 逐位相等、κ=1 与冻结参照一致、球内投影、梯度通路）成立，但**真实入口的一 epoch 冒烟未通过**（NaN 反传 → 常数距离矩阵），机制因此不可达；这是实现层缺陷而非实验结论，未产生任何 Stage2/Stage3 数字。
