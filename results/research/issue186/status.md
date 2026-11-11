# #186 MBR：残差步改按商品曲率 `κ_i` 的莫比乌斯减法（一级度规不动）

## 本轮已完成

- 新树 `stage2_RQ-VAE/TIGER_mbr_rq`，自**冻结基线 `TIGER_mixed_rq`** 复制（`ARM_NAME="TIGER_mbr_rq"`、`ARM_VARIANT="MBR"`）。
- `model/mbr_step.py`：`mobius_sub(x, y, κ) = [(1+2κ⟨x,y⟩+κ‖y‖²)y − x] / [1+2κ⟨x,y⟩+κ²‖y‖²‖x‖²]`，分母以 `finfo.eps·max(‖x‖²,‖y‖²,1)` 垫底；结果若 `κ‖r‖² ≥ 1` 按 `(1−1e-6)/‖·‖` 缩回球内（**数值保证，非可调参数**）；`shrink_events` 供冒烟记录缩回命中数；`STEP_OFF` 直接返回 `y − x`（= 基线平直残差）。
- 测试 `tests/test_mbr.py` 6 项：`κ=0` 与平直减法一致（rtol 1e-12）、`STEP_OFF` **逐位**等于 `y−x`、`κ=1` 与冻结 `poincare_contract.mobius_add(-x, y, 1)` 一致（rtol 1e-6）、`x⊖x=0`、表内曲率 `{1e-6,1e-3,0.1,1,8.2671}` 下结果留在球内、这些曲率下梯度有限。

## 尚待

一级度规按 `struct_mixed_rq` 逐位复刻（`d_{κ_i}` 选码）、把 `mobius_sub` 接进残差递推、`STEP_OFF` 与 `struct_mixed_rq` 的 loss/码分配/SID 逐位一致的双强回归、冒烟（含 `‖r_2‖/‖r_3‖` 分位数与缩回命中数），然后派发 Stage2。

## 更正与未决项（本轮）

1. **公式写错已修**：第一版 `mobius_sub` 只写了单分子项 `(1+2κ⟨x,y⟩+κ‖y‖²)y − x`，与冻结契约不符导致 `κ=1` 与 `x⊖x=0` 两项测试失败。按契约 `mobius_add(a,b,c)=[(1+2c⟨a,b⟩+c‖b‖²)a + (1−c‖a‖²)b] / [1+2c⟨a,b⟩+c²‖a‖²‖b‖²]` 取 `a=−x` 重写为**双分子项** `−(1+2κ⟨−x,y⟩+κ‖y‖²)x + (1−κ‖x‖²)y`，两处测试随即通过。
2. **未决**：`test_gradients_are_finite_at_small_and_large_curvature` 仍失败——在循环内重置 `x.grad/y.grad` 后以 `retain_graph=True` 反复 backward，两个叶子读到的 `.grad` 都是 `None`。这不像是公式问题（其余 5 项都过，含与冻结契约 `κ=1` 一致、`x⊖x=0`、球内、κ=0 极限逐位），但我尚未定位；按 Issue 验收第 3 条（"编码器确实收到非零梯度"）未通过即**不派发**。

当前测试：**5 passed / 1 failed**。

## 未决项已解决：**6 passed**

原因是我在测试里写 `x = torch.randn(...) * 0.5` 之后再加 `requires_grad=True`——这样 `x`/`y` 是**非叶子张量**（`grad_fn=<MulBackward0>`），非叶子永远不累积 `.grad`，与公式无关。改为先乘后 `requires_grad_(True)`（成为叶子）后，两侧梯度都出现且有限，**6/6 通过**。

教训记录：`retain_graph` 循环 + `grad=None` 的写法本身没问题，出错的是"看起来像叶子"的张量。

## 与 Issue 前提不符的代码事实（本轮核对）

Issue 写「`struct_mixed` 在**一级用 `d_{κ_i}` 选码**，但残差仍是平直减法」。核对 `stage2_RQ-VAE/TIGER_struct_mixed_rq` 的真实源码：

- `model/layers.py:102-109` 的 `_distances` 只在 `mixed_geometry.GEOMETRY_ON` 且宽度为 `E_DIM+H_DIM` 时返回 `mixed_geometry.mixed_sq(left, right)`——**就是基线的 E16×H16 乘积平方距离**；
- 该树 `model/` 下没有逐商品 κ 的代价路径，`train_rqvae.py` 里也搜不到 `kappa`/`CURVATURE` 的注入点。

也就是说：**`TIGER_struct_mixed_rq` 的一级度规与冻结基线相同，并没有 `d_{κ_i}`**；它的 +3.0% 来自该臂在**训练器侧**另加的结构信号（AF3 结构项），不是来自一级曲率选码。Issue #186 的动机表述与仓库事实不一致。

按执行规则「按满足其已写明要求的那条读法实现并在 commit 里记下依据」，本臂采用如下读法：

- **一级度规 = 冻结基线的 `mixed_sq`（逐位）**，这也自动满足「与 `TIGER_struct_mixed_rq` 完全一致」（两者的一级度规本就相同）；
- 唯一改动仍是 **残差步**：`r_{ℓ+1} = r_ℓ ⊖_{κ_i} c_ℓ`；
- 于是 `STEP_OFF`（κ≡0）必须与**冻结基线**逐位一致（`y−x` 平直减法），`BOTH_ON` 才是"基线 + 莫比乌斯残差"。这比 Issue 原本设想的对照更干净：**单一变量就是残差步**。

已把该依据写入本轮 commit。

## 残差步接线完成（本轮）

- `mixed_rq.MixedVQLevel.forward` 接受 `item_kappa`；给出时残差更新改为
  `next = mbr_step.mobius_sub(cat(r_e, r_h), cat(det_e, det_h), κ_i)`，取欧氏 16 维部分作为下一层残差（`item_kappa is None` 时仍是基线的 `residual_e − det_e`）。
- `MixedRQ` 持有 `item_kappa`；训练器把数据集改成 `(embedding, id)`、在 `model(batch)` 前后注册/清除批内 `κ_i`（`try/finally`）。
- 训练器与模型在项目解释器下均可导入。

## 关键修正：参数顺序（本轮）

第一次接线把 `mobius_sub` 的实参写成 `(residual, code)`，但语义应是 **`residual ⊖ code = mobius_add(−code, residual)`**，即 `x=code, y=residual`。按旧写法 `STEP_OFF` 会得到 `code − residual`——**每层残差符号翻转**，与基线不符。已改正，并补回归测试 `test_the_flat_limit_is_residual_minus_code_not_the_reverse`（`κ=0` 时必须逐位等于 `residual − code` 且不等于 `code − residual`）。

测试 **7 passed**。

## `STEP_OFF` 在真实栈上逐位一致（本轮，**8 passed**）

新测试 `test_step_off_matches_the_flat_path_through_the_real_stack`：同一 `MixedRQ` 权重下，(a) 不注册 `item_kappa`（基线的平直路径）与 (b) 注册 `κ_i=2` 且 `STEP_OFF=True`，**三级 token 与损失都 `torch.equal`**。这条把"只动了残差步"钉在真实量化栈上，而不只是 `mobius_sub` 单函数。
