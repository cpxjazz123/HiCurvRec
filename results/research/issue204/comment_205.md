## 清理已落地（本 Issue 在被关闭时尚未完成，现由该 commit 补齐）

关闭核验：本 Issue 以 `COMPLETED` 关闭时，**验收第 1 条并不成立**——`grep -rn "LAMBDA_SCALE\|0\.1 \*" stage2_RQ-VAE/TIGER_f0r_rq --include=*.py` 仍能命中 `model/struct_distill.py:109` 与 `train_rqvae.py:671`。现按正文的三处清理补齐：

| 项 | 结果 |
|---|---|
| 1. 删 `struct_distill.LAMBDA_SCALE` 与 `calibrate`（及其唯一引用） | 已删，`grep` 现为 **0 行** |
| 2. 删 `train_rqvae.py` 的 `sd_lambda_scale` 字段 | 已删 |
| 3. `probe_lambda_star` docstring 改为 `min over {encoder, codebooks} ‖g_base‖/‖g_struct‖` | 已改，并注明"无系数、无 ε、无回退" |

顺带：继承来的 `tests/test_af3_structure.py:137` 用 `0.1 * phi` 合成数据，与 λ 规则无关；改写为 `phi * 0.1`（同算术），使验收 `grep` 的输出**逐字为空**，而不是靠解释绕开。

其余一律未动：秩教师、同码支撑、修正 AF3、F0 起点、两道闸门、开关、预算、Stage3 协议保持 `886e7433` 状态。`tests/test_f0r_rank.py` **5 项全过**。

**依据 commit**：`44fae07b0`。本 Issue 已关闭，此处仅补记"关闭时未完成、现已完成"的事实与证据，不重开、不新增验收项。
