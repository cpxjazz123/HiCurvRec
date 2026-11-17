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

新增审计在**初始 checkpoint、尚未更新**时测得 KL/base 参数梯度范数比约 **5992**；旧的 `KNN_WEIGHT=1e-3` 使加权 KL 梯度约为基础项的 **6.0 倍**，虽然标量值看似同量级。

因此固定权重改为 **`1e-4`**（由实测梯度比推导，不是扫描），加权后的梯度约为基础的 **0.60 倍**。

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

第一次审计把 candidate q memory 固定在初始 checkpoint；用户指出的口径问题成立。重新审计时每一步都用当前参数重算全部 q：

- 10 次 KL-only SGD（lr=1e-6）；
- soft KL：0.3899337 → **0.3899181**（最低点）
- hard export-q KL：同值，最终 0.3899217；
- 10 步中有 9 步下降，最后一步小幅回升；
- soft 与 hard 的数值路径一致。

结论：

- 在**同步更新的 q memory** 下，软代理的更新方向**能够**降低重算硬 q KL；
- 但幅度很小，且并非每一步单调，仍需更严格的 gradient alignment 测试；
- 不能把 P1 标记为完全解决，也不能再用“固定 memory 的 100 步”作为否定证据。

### 2.3 KL/基础项梯度强度

初态同批量、同模型、尚未更新时：

- 基础 RQ-VAE 参数梯度范数：**0.000308**
- 原始 KL 参数梯度范数：**1.8446**
- 原始比值：**5992×**
- 加权 `KNN_WEIGHT=1e-4` 后：约 **0.60×**

之前出现的 **6206**（20 步后）和 **4498**（100 步后）是模型已被 KL-only 诊断更新后的状态，不是初始 checkpoint；**5992** 是当前应引用的初态比值。

### 2.4 2.89% 导出差异的定位

按层统计重编码 token 与已保存 SID 的差异：

| 层 | mismatch | 比例 | mismatch gap median | mismatch gap ≤1e-4 | 全部 gap ≤1e-4 |
|---|---:|---:|---:|---:|---:|
| L1 | 31 | 0.126% | 6.20e-4 | 16.1% | 0.03% |
| L2 | 161 | 0.655% | 1.27e-5 | 84.5% | 10.0% |
| L3 | 697 | 2.835% | 8.09e-6 | 85.4% | 37.6% |

CPU 和 GPU 重编码结果完全相同；`sids_for_hgrec.npy` 与 `item_sids.json` 的前三码完全一致。  
因此该差异不是 Sinkhorn，也不是 CPU/GPU 取舍；它集中在 **较小 margin 的 argmin 边界**，但仍有约 15% 的 mismatch 不在 1e-4 内，不能仅用“并列”解释。当前判断是**导出时的模型状态与该 checkpoint 的重编码状态不完全 bit-identical**，需要在下一次导出时记录 token manifest/checkpoint hash 才能最终定位。

### 2.5 checkpoint-bound export guard

`export_guard.py` now makes the saved checkpoint, not the live training module, the
export authority:

1. save the final checkpoint;
2. load it into a fresh model;
3. compute argmin tokens with the export assignment and write SID;
4. load the checkpoint **again** into a second fresh model and recompute tokens;
5. fail the run if the two token arrays differ;
6. write `export_manifest.json` with:
   - checkpoint SHA-256,
   - export and re-encoded token SHA-256,
   - `tokens_match`,
   - SID and `item_sids.json` SHA-256,
   - assignment/model mode/items.

Dry run on the current checkpoint (no training):

- `checkpoint_sha256 = b00e4765...`
- `export_tokens_sha256 = 2b79a5f9...`
- `reencoded_tokens_sha256 = 2b79a5f9...`
- `tokens_match = true`

This does **not** rewrite the old SID; the old 2.89% mismatch is from the pre-guard export
and remains diagnostic. Future exports cannot publish a SID unless the same checkpoint
re-encodes to exactly the same tokens.

## 3. 当前判断

> 2026-11-17 正式重训后补充：训练出口改为 **CPU canonical**；同一 checkpoint 的外部 CPU 重编码与导出 SID mismatch 为 **0**，`build_vectors.tokens_match_export=true`。旧 2.89% 差异是 GPU 导出与 CPU 重编码的 canonical 设备差异；新训练结果见 `REPORT_UNIFORM_KNN_FIXED.md`。

- q memory 独立/陈旧问题：**已修**；
- 训练/导出 Sinkhorn 不一致：**已修**（本臂统一 argmin）；
- tau 梯度泄漏：**已修**；
- KL/基础梯度失配：**已修权重**；同步 memory 下软代理的最小方向审计**通过**（硬 q KL 从 0.3899337 降到 0.3899181），但仍需更严格验证；
- 2.89% 的离线导出差异：**未解决**；已排除 Sinkhorn、CPU/GPU 取舍和 SID 文件不一致，集中在 argmin 边界/导出状态；
- 未运行 Stage2 训练，未改旧实验结果。

## 4. 明确不做

- 不因本次修复自动重跑 3000 epoch；
- 不在软代理梯度方向未确认前启动 Graphlet-KNN 或曲率实验；
- 不用旧 `TIGER_Uniform_KNN_RQ` 结果证明 Uniform-KNN 方向可行或不可行。
