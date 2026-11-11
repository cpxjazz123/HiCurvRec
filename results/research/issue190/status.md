# #190 SMH：struct_mixed 三元组目标的双曲化

## 本轮已完成

- 新树 `stage2_RQ-VAE/TIGER_smh_rq`（`ARM_NAME="TIGER_smh_rq"`、`ARM_VARIANT="SMH"`），**自 `TIGER_struct_mixed_rq` 复制**——依据：Issue 正文明确"起点：`stage2_RQ-VAE/TIGER_struct_mixed_rq`（只读…）→ 新树"，且本臂的定义就是"struct_mixed + 一处目标距离替换"，从该树起步才能保证 `GEOMETRY_OFF` 与它逐位一致。
- `model/hyperbolic_target.py`：`pair_kappa = √(κ_a κ_b)`（下限 1e-3）、`distance`（asinh 形式、重合点有限、边界钳位）、`squared = d²`（与 `mixed_sq` 承担同一"平方距离"角色）、`kappa_of`（标记目录读冻结表、`mean_frozen` 补齐）、`GEOMETRY_OFF` 开关（off 时调用方继续用 `mixed_sq`，本模块不参与）。
- `structure.py::structure_loss` 增加可选 `kappa_pairs`：`None` 时**逐字**走 `G.mixed_sq`（保持 `GEOMETRY_OFF`/默认状态的逐位一致），非 `None` 时改用 `HT.squared`。
- `training_hooks.structure_term` 从三元组的 item ids 构造每对的 `κ = √(κ_aκ_b)` 传入；采样器与 seed 未改（三元组集合与 `struct_mixed` 同源）。
- 测试 `tests/test_smh.py` **6 passed**：κ=1 与冻结 `poincare_contract.distance` 一致（rtol 1e-6）、`pair_kappa` 为几何平均、重合/近重合有限、`squared = d²`、κ 表查表与下限、默认几何开。
- 模块与 hooks 在项目解释器下可导入（`PAIR_INDEX = ((0,1),(0,2),(1,2))`）。

## 尚待

`GEOMETRY_OFF` 与 `struct_mixed` 损失的**逐位一致**测试、三元组集合逐位相同的机器断言、一 epoch 冒烟（含目标距离分位数落盘），然后**等待 owner 批准**——Issue 明确：该族属当前主线禁令，**批准前只允许实现 + 预验收，不得派发**。
