# Issue #207 关闭依据

机制：L3K —— 冻结 AF3 曲率只作用于**三级**代价，一二级 `κ≡1`；起点 `TIGER_mixed_rq_F0`（同族最好，0.057278156）。

## 验收对照

| # | 要求 | 结果 |
|---|---|---|
| 1 | `BEHAVIOUR_OFF` 与 F0 逐位一致 | `BEHAVIOUR_OFF` ⇒ `kappa_rows` 全 1、`level_uses_kappa` 全 False（逐项断言）；几何代码见 2 |
| 2 | 一二级未被改动；机器断言一二级码分配相同 | **通过**：`level_uses_kappa` = [F,F,T]，`LEVEL12_ON` = [T,T,T]；`get_indices` 的 **L1 与 L2 码在两种配置下逐值相同** |
| 3 | `d_κ` 性质（rtol 1e-6、`p=q ⇒ 0`、边界/κ→0 有限）、gradcheck、编码器梯度、κ detach | **通过**（`tests/test_l3k_metric.py`：κ=1 与冻结合同 rtol 1e-6、重合点 <1e-5、κ∈{0.001,1,8.2671} 与边界梯度有限、gradcheck、编码器非零梯度、无 `kappa` 参数） |
| 4 | 一 epoch 真实入口冒烟：损失有限、导出 SID 有效 | **`SMOKE_PASS`**（第 2 次派发，24 步梯度全正；冒烟内导出 24,587 行） |
| 5 | 导出统计 + §89 + 可触发的闸门用例 | 闸门**在真实训练 epoch 50 触发**（`collision 0.5044129011`），未产生导出；闸门谓词用例覆盖 0.9 / 0.5122 / 0.8234 触发、0.3999 / 0.257168 不触发 |
| 6 | §63 指标；Stage3 后 §6b 归档 | **不适用**（闸门在导出前终止，无 SID、无 Stage3、无归档） |
| 7 | TDD | 13 项用例（几何未动、作用层、`d_κ`、gradcheck、闸门）全过；其中"几何未动"分两部分：`layers.py`/`poincare_contract.py` 与 F0 SHA 相同，`mixed_rq.py` 内 F0 的 `mixed_cost` 函数体逐字不变且新增定义只有 `mixed_cost_with_kappa` |

## 裁决

塌缩闸门在**首个计划评估点**（epoch 50，`collision 0.5044129011266116`）触发 ⇒ **记负结果、收线**；`test_R@10 = NOT_RUN`。
允许类"三级"格记为 **gated**（被闸门终止），不是测得的负值；四层位（一级 LKC/CMR/MBR/SIGN、二级 L2K、三层 HC-3、三级本臂）与损失侧（F0V）**均未给出正的 `test_R@10`**。

## 依据 commit

`63ea8715b`（建树/闸门）· `c108ab8b2`（13 项验收用例）· `da64e6d32`（闸门限定为真实训练）· 本次提交（实测记录、裁决与关闭依据）。
