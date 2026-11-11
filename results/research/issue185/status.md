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
