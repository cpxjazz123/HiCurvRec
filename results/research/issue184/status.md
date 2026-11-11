# #184 LKC：一级分配的**按码曲率**（无对比损失）

## 本轮已完成

- 新树 `stage2_RQ-VAE/TIGER_lkc_rq`，自**冻结基线 `TIGER_mixed_rq`** 复制（`ARM_NAME="TIGER_lkc_rq"`、`ARM_VARIANT="LKC"`）。
- `model/r48k_geometry_learned.py`：
  - `initial(n)` → `κ_k(0)=1.0`；
  - `update(κ_k, κ_item, assigned_codes)`：每 epoch 末 `κ_k ← mean{κ_i : 分配到 k}`（`index_add_` 累加、**detach 无梯度**），**空码保持旧值**，`κ<1e-3` 取 1.0 并计数，返回 `{moved, empty, floored}` 记录；
  - `KAPPA_FROZEN`：直接返回旧表（`frozen=True`）且 `level1_cost` 退回**欧氏表达式**（与基线逐位可比）；`BEHAVIOUR_OFF`：按固定种子**置换**新表（同分布、错误对应）；
  - `to_ball`（严格球内、零向量极限 1）、`distance`（asinh 形式、重合点有限）、`level1_cost(latent, codes, κ_codes)`：**用码的曲率** `d_{κ_k}`，不是商品的曲率。
- 测试 `tests/test_lkc.py` 7 项：按码取均值、空码保持、触底替换并计数、`KAPPA_FROZEN` 不动表、**更新不携带梯度**、代价确实随 κ 变化且 frozen 时与欧氏**逐位相等**、`to_ball` 严格球内与重合点距离有限。

## 尚待

把 `level1_cost` 接进一级分配（替换 `mixed_rq` 里的代价），在训练循环 epoch 末调用 `update(...)` 并把 `{moved, empty, floored}` 与 κ 分位数写入 metrics，`KAPPA_FROZEN` 与基线**损失与导出逐位一致**的测试，一 epoch 冒烟，然后派发 Stage2。
