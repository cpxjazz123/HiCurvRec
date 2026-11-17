# Uniform-KNN post-review audit and code fixes

日期：2026-11-17  
范围：只读审计 + 源码修复；**没有启动 Stage2 训练**。

## 1. 已修改的 P0/P1

### P0 q memory 同步

`losses/knn.py` 新增 `QuantizedMemory`：

- 全部商品在每个 epoch 开始时从**当前模型参数**刷新；
- 每个 batch 在 `optimizer.step()` 之后重新计算该 batch 的 q，再写回；
- DDP 多卡通过 `all_gather_object` 合并各 rank 的 batch q，四张卡的 memory 不再各自漂移；
- 每个商品记录 age，训练日志写 `memory_mean_age` / `memory_max_age` / `memory_stale_items`；
- 不使用 pre-step 的 q_hard 写回。

### P0 训练/导出分配不一致

本臂新增 `EXPORT_CONSISTENT_ASSIGNMENT=True`：

- 训练、推理、导出统一使用 **argmin**；
- 不再在训练 L3 使用 Sinkhorn、导出却用 argmin；
- `soft_quantized_readout` 默认 `assignment="export"`；
- 这样 KL、q memory 和最终 SID 使用同一个硬分配路径。

### P1 温度 detach

- `knn_neighbourhood_kl` 的 `tau` 已 `.detach()`；
- `soft_quantized_readout` 的 softmax 温度也已 `.detach()`；
- 温度只作为归一化常数，不再进入梯度。

### P1 KL 权重

新增审计测得 final-checkpoint 上 KL/base 参数梯度范数比约 **6206**；旧的 `KNN_WEIGHT=1e-3` 使加权 KL 梯度约为基础项的 **6.2 倍**，虽然标量值看似同量级。

因此固定权重改为 **`1e-4`**（由实测梯度比推导，不是扫描），加权后的梯度约为基础的 **0.45 倍**。

### 审计工具

- `results/research/uniform_knn_evaluation/audit_assignments.py`
- `assignment_audit.json`

## 2. 审计结果（在旧 checkpoint 上，只读；没有重新训练）

### 2.1 token 一致性

旧训练代码（Sinkhorn L3）对同一个 checkpoint：

| 比较 | joint mismatch | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| training vs export | **34.693%** | 0.126% | 0.655% | **34.636%** |
| inference vs export | 2.892% | 0.126% | 0.655% | 2.835% |

修复后的本臂强制 argmin 后：

| 比较 | joint mismatch | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| training vs inference | **0%** | 0% | 0% | 0% |
| training/inference vs export | **2.892%** | 0.126% | 0.655% | 2.835% |

结论：

- **训练期 Sinkhorn 导致的 34.6% L3 不一致已被消除**；
- 仍然存在 **2.89% 的离线重编码/导出差异**；这不是本次 KL 新增代码造成的，来源需要单独定位（argmin 并列、CPU/GPU 取舍或导出时状态）；
- 不能把 2.89% 当成零，也不能把它归因于 Sinkhorn。

### 2.2 软代理与硬 q 的梯度方向

在旧 checkpoint 上执行 **100 次很小的 KL-only SGD 更新**（lr=1e-6）：

- soft KL：0.3899337 → 0.3899828
- hard export-q KL：0.3899337 → 0.3899828

两者方向一致，但 **KL 不降反升**。这说明：

- soft 与 hard 的数值路径目前一致；
- 但当前软代理的更新方向**没有证明能改善硬 q KL**；
- 不能把 P1 软代理问题标记为已解决。

### 2.3 KL/基础项梯度强度

在同一 batch 上：

- 基础 RQ-VAE 参数梯度范数：0.000407
- 原始 KL 参数梯度范数：1.831
- 原始比值：**4498×**
- 加权 `KNN_WEIGHT=1e-4` 后：约 **0.45×**

这说明仅比较 loss 标量（KL 0.39 vs base 0.00035）会严重低估 KL 的梯度支配程度；新的 1e-4 是有依据的修正。

## 3. 当前判断

- q memory 独立/陈旧问题：**已修**；
- 训练/导出 Sinkhorn 不一致：**已修**（本臂统一 argmin）；
- tau 梯度泄漏：**已修**；
- KL/基础梯度失配：**已修权重，但软代理能否改善硬 q 仍未通过审计**；
- 2.89% 的离线导出差异：**未解决，需单独定位**；
- 未运行 Stage2 训练，未改旧实验结果。

## 4. 明确不做

- 不因本次修复自动重跑 3000 epoch；
- 不在软代理梯度方向未确认前启动 Graphlet-KNN 或曲率实验；
- 不用旧 `TIGER_Uniform_KNN_RQ` 结果证明 Uniform-KNN 方向可行或不可行。
