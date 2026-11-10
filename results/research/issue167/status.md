# #167 现状：逐商品曲率的几何基元已实现并验证，接线与门待做

## 已完成

- 新树 `TIGER_af3_curvature_rq`（`AF3CURV`），以 #165 的 `TIGER_conditional_quota_final_rq` 为参考复制（Issue 第 1 条允许"以 #165 为参考实现"），配额与指派构造原样保留。
- `model/per_row_curvature.py`：**逐商品曲率的几何基元**。原因是冻结的 `poincare_contract.distance` 以 `float(c) == 0.0` 开头、只接受标量曲率，`float(多元素张量)` 会直接报错——所以逐商品 `κ_i` 无法传进冻结契约。本模块用同一 Möbius 公式实现可按行广播的曲率（`as_column` / `mobius_add_rows` / `distance_rows`），**不修改冻结契约**。
- 测试 `tests/test_per_row_curvature.py`（4 passed）：常数曲率下与冻结 `distance` 数值一致（rtol 1e-9）、`c=0` 退化为欧氏极限（2‖x−y‖）、不同行曲率给出不同距离、曲率列形状约束成立。

## 尚待完成（本轮未做，因此不派发）

1. **κ 表**：`κ_i = CURVATURE·(1−u_i)/mean_frozen(1−u)`，`mean_frozen` 取冻结表全局均值（不用 batch 统计，保证导出与训练同值），写盘并记 SHA。
2. **接线**：候选树的 `mixed_cost` 在给定 κ 时改用 `distance_rows`（每行自己的曲率），且 `ball = expmap0(codes_h, CURVATURE)` 仍用参考曲率映射码字——这一点是几何口径的关键，必须在报告里写明（逐商品曲率只进入**距离**，不进入码字映射）。
3. **移除三项监督**：`L_rank/L_spread/L_tri` 在本变体下不参与损失（目前 trainer 仍按 `CONDQ` 分支挂载，需要按 `AF3CURV` 关闭）。
4. **门**：1（碰撞 ≤0.05）、2（容量恒等式）、3（**结构灵敏度**：置换 `u_i` 后 SID 变化 ≥10% 商品，并报 `mean|κ−1|`、κ 四分位、两套距离分布）、4（单一实现 + `ROUNDING_OFF`/`GEOMETRY_OFF`）、5（护栏）、6（checkpoint 绑 κ 表 SHA）、7（TDD：κ 均值恒等式、`u` 单调性、置换灵敏度、开关恒等、c=0 极限、训练/导出同码）。
