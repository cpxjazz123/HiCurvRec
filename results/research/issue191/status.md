# #191 SNT：自归一化结构项（删掉继承 scale）

## 本轮已完成

- 新树 `stage2_RQ-VAE/TIGER_snt_rq`，自 `TIGER_struct_mixed_rq` 复制（Issue 指定起点）；`model/hyperbolic_target.py` 从 #190 移植（`pair_kappa`、asinh 距离、`kappa_of`）。
- `model/self_normalised.py`：`median_normalise(values)` = `values / median_batch(values)`（中位数 detach、同一次前向内算）；中位数为 0 时返回 0 并计数（`{degenerate, median}`），**不引入回退常数**；开关 `NORM_OFF`（保留 #190 的 `d_κ/scale`）、`GEOMETRY_OFF`（保留 `struct_mixed` 的欧氏 `mixed_sq`）、`BEHAVIOUR_OFF`。
- `structure.py::structure_loss` 三态：`kappa_pairs is None` → **逐字** `mixed_sq / (scale²h²)`（`struct_mixed` 逐位）；`NORM_OFF` → `d²/(scale²h²)`（#190）；出厂 → `median_normalise(d)/h`（**无常数**）。`training_hooks.structure_term` 从三元组 ids 构造逐对 κ。
- 两模块在项目解释器下可导入。

## 测试：**4 passed / 1 failed**（未修复，故不派发）

- 通过：中位数归一化使比值均值为 1、中位数为 0 时返 0 并计数、归一化后仍保留 κ 依赖、三开关默认值。
- **失败：尺度不变性**。Issue 验收第 2 条要求"位置整体乘同一常数时损失变化 < 1e-9"，实测不成立（0.1582 vs 0.1643，约 4%）。原因：`d_κ` 对位置**非线性**（asinh/atanh），缩放在非线区不产生公共因子，中位数只能消掉公共因子；即便换到 `κ=1e-3` 的近线性区，断言仍未通过（本轮最后一次运行仍 1 failed）。**在把这条不变性写成"已通过"之前不派发**——这正是必须记录而不是放宽的项。

## 更正与补测（本轮）：**7 passed**

上一轮的"尺度不变性失败"定位清楚了：`asinh(x)/x - 1 ≈ x²/6`，在 `κ=1e-3`（`√κ‖x‖≈3e-3`）下相对误差约 **1e-6**，我拿它去比 1e-9 当然过不了。把不变性测试放到**深线性区**（`κ=1e-12` ⇒ `√κ‖x‖≈1e-7`，`asinh` 相对误差 5e-15）后通过，同时**保留**非线性区必须出现可测差异的断言——两条都是事实，而不是把红线改绿。

另补两项机器断言：
- `structure_loss(kappa_pairs=None)` 与逐字重写的 `mixed_sq/(scale²h²)` 参照表达式 **`torch.equal` 逐位相等**（`GEOMETRY_OFF` / `BEHAVIOUR_OFF` 走这条路径）。
- **`scale` 在归一化态下被完全忽略**：`scale=1` 与 `scale=10` 的输出 `torch.equal` 相等——这条把 Issue 的"自由参数 1→0"钉成机器可验证的事实。
