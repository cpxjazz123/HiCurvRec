# #175 DUL 复现：实现与测试（派发前状态）

## 实现

- 新树 `TIGER_dul_rq`（`DUL`），从**冻结基线**复制：量化、残差递推、Sinkhorn（ε=0.003、50 sweeps、centring）、读出、导出逐字节未改；#171–#174 的双容量/行优先/增广/软容量代码**均未带入**。
- `model/unique_loss.py`：`L_unique = mean over P of relu(cos(z_i,z_j) − m)`，`P` = 同一 batch 内三级码完全相同且商品不同的行对（`collision_pairs` 按三元组键分组），`z` 为**量化前**连续表示，三级码 `detach`；`BETA_UNIQUE=2.0`、`MARGIN=0.9`（论文值，不调）；`UNIQUE_OFF` / `PAIR_SOURCE_OFF` / 出厂三态。
- 接线：`MixedRQVAE.forward` 在 `self.rq(encoded)` 之后计算并加到 `quant_loss` 上，因此 `L = L_RQ + 2.0·L_unique`，无需改 trainer 的损失装配。
- 每 batch 记录 `|P|`、`cos>m` 对数、`L_unique`（`LAST_STATS`），并在评估 epoch 写进 `training_metrics.jsonl` 的 `unique_pairs` / `unique_above_margin` / `unique_loss` 与打印行。
- 方向性矛盾已写进模块 docstring：按公式 `m` 越大越**宽松**（`cos ≤ m` 即零损失），与论文正文表述相反；**照公式实现**。

## 范围说明（owner 授权，写进代码与报告）

§19 要求"结构信号 → 显式双曲运算 → 选码"同一条通路；本阶段是**欧氏余弦排斥、无双曲运算**。该偏差由 owner 2026-11-11 指令显式豁免，**仅覆盖本阶段复现**；几何化/图加权版本须另开 Issue。

## 测试（`tests/test_unique_loss.py`，7 passed）

对集合恰为"三级码全同"的组（3 个同码行 → 一个 3 元组、无单例组）；同码不同行仍是合法对；**零碰撞 batch 返回 0 且有限**；梯度只经余弦回到 `z`（首版测试因 `relu` 饱和导致零梯度——已按"必须在 margin 之上"修正测试）；`m` 方向按公式（0.5 的损失 > 0.99）；`UNIQUE_OFF` 归零；`PAIR_SOURCE_OFF` 用随机行对仍产出有限损失。

## 尚未完成（因此本轮未派发）

- 一 epoch 真实入口冒烟（需派发时由 MCP 执行）；
- 全量导出统计（三级碰撞率对 0.162444、码使用量、最大份额、二级前缀数）；
- §89 入场判断（`collision < 0.40`）。
