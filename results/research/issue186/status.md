# #186 MBR：残差步改按商品曲率 `κ_i` 的莫比乌斯减法（一级度规不动）

## 本轮已完成

- 新树 `stage2_RQ-VAE/TIGER_mbr_rq`，自**冻结基线 `TIGER_mixed_rq`** 复制（`ARM_NAME="TIGER_mbr_rq"`、`ARM_VARIANT="MBR"`）。
- `model/mbr_step.py`：`mobius_sub(x, y, κ) = [(1+2κ⟨x,y⟩+κ‖y‖²)y − x] / [1+2κ⟨x,y⟩+κ²‖y‖²‖x‖²]`，分母以 `finfo.eps·max(‖x‖²,‖y‖²,1)` 垫底；结果若 `κ‖r‖² ≥ 1` 按 `(1−1e-6)/‖·‖` 缩回球内（**数值保证，非可调参数**）；`shrink_events` 供冒烟记录缩回命中数；`STEP_OFF` 直接返回 `y − x`（= 基线平直残差）。
- 测试 `tests/test_mbr.py` 6 项：`κ=0` 与平直减法一致（rtol 1e-12）、`STEP_OFF` **逐位**等于 `y−x`、`κ=1` 与冻结 `poincare_contract.mobius_add(-x, y, 1)` 一致（rtol 1e-6）、`x⊖x=0`、表内曲率 `{1e-6,1e-3,0.1,1,8.2671}` 下结果留在球内、这些曲率下梯度有限。

## 尚待

一级度规按 `struct_mixed_rq` 逐位复刻（`d_{κ_i}` 选码）、把 `mobius_sub` 接进残差递推、`STEP_OFF` 与 `struct_mixed_rq` 的 loss/码分配/SID 逐位一致的双强回归、冒烟（含 `‖r_2‖/‖r_3‖` 分位数与缩回命中数），然后派发 Stage2。
