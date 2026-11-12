# Issue #196 — #195 撤销决定与本轮核实（不训练、不派发、不改源码）

## 1. #195 作废

`#195`（同一曲率选码路径下的四臂对照：无辅助 / 仅结构KL / 仅行为对比 / 两者结合）**不得实现、不得建树、不得派发、不得登记结果**，以 owner 指令取消关闭，编号不复用。

## 2. 全仓核实（本轮实测，2026-11-12）

| 检查 | 命令 | 结果 |
|---|---|---|
| 消融树 | `ls -d stage2_RQ-VAE/TIGER_abl*` | **空** |
| 消融结果 | `ls -d results/stage2_RQ-VAE/TIGER_abl* results/stage3_T5Train/TIGER_abl*` | **空** |
| 消融进程 | `pgrep -af TIGER_abl` | **空** |

开关只存在于代码中，且各自被廉价接线检查覆盖（未训练任何关闭态臂）：

- `stage2_RQ-VAE/TIGER_sdh_rq/model/sdh_kernel.py`：`SUPPORT_OFF = False`、`GEOMETRY_OFF = False`
- `stage2_RQ-VAE/TIGER_sdh_rq/model/struct_distill.py`：`BEHAVIOUR_OFF = False`（`SD_OFF` 经 `__getattr__` 别名读取同一标志）
- `stage2_RQ-VAE/TIGER_sdh_rq/model/r48k_geometry.py`：`GEOMETRY_OFF` / `BEHAVIOUR_OFF`（#180 沿用的同名开关）
- 接线检查：`tests/test_sdh_off_state.py`（off 时机制路径不可达、`_assign_kappa` 返回 `None`、开关在调用时读取）与 `tests/test_sdh_kernel.py`（`SUPPORT_OFF` 放开支撑），共 65 项测试通过。

## 3. 一处必须如实交代的边界情形

`#194` 验收 1 要求证明 `BEHAVIOUR_OFF` 与冻结基线一致，为此**跑过一次一 epoch 的 CPU 单进程对照**（`results/research/issue194/off_equivalence.py`，各 3 次重复）。它：

- 不是 Stage2 臂：未走四卡 DDP 冻结配方、未进 3000 epoch、未产生用于比较的 SID、未登记为结果；
- 只为验收"关闭态与基线等价"这一条，且结论是**步数逐位一致、数值在噪声内、逐位相等不可测**。

即：不存在"跑过的关闭态臂"，§19g 的意图未被绕过；此处仅公开该一次对照运行的存在，避免"未运行关闭态"的说法与事实不符。

## 4. #193 遗留的"分离问题"不再用实验回答

按本 Issue：只以既有单臂读数 + 相关性证据留档，不新开实验。

- `#193` P0-3：Pearson(教师权重, 共同前缀层数) **0.10530**、Spearman **0.10746**、mean KL(P_teacher‖Q_sid) **2.03845**、零共享对占比 **94.95%**；分层均值 0.000396 → 0.000868 → 0.002221 → 0.020023。
- `#194` 单臂：`test_recall@10 = 0.05598983269207333`（预注册 `≤ 0.056216` 收线）。
- `#181`（缺陷版 + 4× 批量）：0.054266。

即"行为对比 vs 结构 KL 哪一侧有害"这一问题**留作未回答**，不以下一次四臂实验回答。

## 5. 后续 Issue 的形状要求（记录，供后续轮次遵守）

新 Issue 必须是**单臂**、自带预注册指标与关闭条件；不得以"先做个对照看看"的形式提出。
