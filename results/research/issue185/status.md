# #185 CMR：相对曲率 `ρ_ik = clip(κ_i/κ_k)` 一级代价

## 本轮已完成

- 新树 `stage2_RQ-VAE/TIGER_cmr_rq`，自**冻结基线 `TIGER_mixed_rq`** 复制（`ARM_NAME="TIGER_cmr_rq"`、`ARM_VARIANT="CMR"`）。
- `model/cmr_geometry.py`：
  - `rho_matrix(κ_i, κ_k)`：`RATIO_OFF` → 全 1；`NUMERATOR_ONLY` → `κ_i`；`DENOMINATOR_ONLY` → `1/κ_k`；出厂 → `clip(κ_i/κ_k, 1e-3, 1e3)`；
  - `level1_cost(latent, codes, κ_i, κ_k)`：`RATIO_OFF` 时**逐位**返回欧氏表达式（与基线可比），否则 `d_{ρ_ik}(to_ball(r;ρ), to_ball(c;ρ))`；
  - `refresh_codes`：每 epoch 末 `κ_k ← mean{κ_i : 分到 k}`，`no_grad` + `detach`、空码保持、触底取 1 并计数；
  - `to_ball`（严格球内、零向量极限 1）、`distance`（asinh 形式、重合点有限）、`kappa_of`+`load_table`（标记目录、`mean_frozen` 补齐、1e-3 下限）。
- 测试 `tests/test_cmr.py` 8 项：比值手算、clip 两端、**`RATIO_OFF` 与欧氏逐位相等**、`NUMERATOR_ONLY` 与商品无关地重复 `κ_i`、`DENOMINATOR_ONLY` 只用 `1/κ_k`、刷新取均值且空码保持、刷新与比值都不携带梯度、球内与重合点有限。

## 尚待

一级代价接进 `MixedVQLevel`（仅第 1 层）、训练器 epoch 末刷新并记录 `ρ` 分位数与 `{moved, empty, floored}`、`NUMERATOR_ONLY` 与 `TIGER_struct_mixed_rq` 的损失逐位一致（双强回归的第二条）、一 epoch 冒烟，然后派发 Stage2。

## 测试：**8 passed**

四项状态与刷新规则的测试全部通过（比值手算、clip 两端、`RATIO_OFF` 与欧氏**逐位相等**、`NUMERATOR_ONLY`/`DENOMINATOR_ONLY` 的退化、刷新均值与空码保持、无梯度、球内与重合点有限）。

## 接线完成（本轮）

- `model/mixed_rq.py`：`MixedVQLevel.distances/forward` 接受 `item_kappa/code_kappa`，给出时该层代价改为 `cmr.level1_cost(...)`；`MixedRQ` 持有 `kappa_codes`（κ_k）与 `item_kappa`（批内 κ_i），**只把这两者传给第 1 层**，其余层保持基线乘积代价（Sinkhorn 未改，§19h）。
- `train_rqvae.py`：数据集改为 `(embedding, id)`、循环注册 `raw_module.rq.item_kappa = kappa_of(batch_ids)`（`try/finally` 清除）、累积 `(κ_i, 一级 token)` 并在 **`epoch_time_s = ...` 之前**刷新 `κ_k` 与记录 `kappa_refresh`（`grep -c` = 1，训练器可导入）。`κ_k` 初值 256 个 1.0。
