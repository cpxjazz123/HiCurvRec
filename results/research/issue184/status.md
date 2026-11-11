# #184 LKC：一级分配的**按码曲率**（无对比损失）

## 本轮已完成

- 新树 `stage2_RQ-VAE/TIGER_lkc_rq`，自**冻结基线 `TIGER_mixed_rq`** 复制（`ARM_NAME="TIGER_lkc_rq"`、`ARM_VARIANT="LKC"`）。
- `model/r48k_geometry_learned.py`：
  - `initial(n)` → `κ_k(0)=1.0`；
  - `update(κ_k, κ_item, assigned_codes)`：每 epoch 末 `κ_k ← mean{κ_i : 分配到 k}`（`index_add_` 累加、**detach 无梯度**），**空码保持旧值**，`κ<1e-3` 取 1.0 并计数，返回 `{moved, empty, floored}` 记录；
  - `KAPPA_FROZEN`：直接返回旧表（`frozen=True`）且 `level1_cost` 退回**欧氏表达式**（与基线逐位可比）；`BEHAVIOUR_OFF`：按固定种子**置换**新表（同分布、错误对应）；
  - `to_ball`（严格球内、零向量极限 1）、`distance`（asinh 形式、重合点有限）、`level1_cost(latent, codes, κ_codes)`：**用码的曲率** `d_{κ_k}`，不是商品的曲率。
- 测试 `tests/test_lkc.py` 7 项：按码取均值、空码保持、触底替换并计数、`KAPPA_FROZEN` 不动表、**更新不携带梯度**、代价确实随 κ 变化且 frozen 时与欧氏**逐位相等**、`to_ball` 严格球内与重合点距离有限。

## 尚待

把 `level1_cost` 接进一级分配（替换 `mixed_rq` 里的代价），在训练循环 epoch 末调用 `update(...)` 并把 `{moved, empty, floored}` 与 κ 分位数写入 metrics，`KAPPA_FROZEN` 与基线**损失与导出逐位一致**的测试，一 epoch 冒烟，然后派发 Stage2。

## 一级代价接线 + 刷新去梯度（本轮）

- `model/mixed_rq.py`：`MixedVQLevel.distances/forward` 接受可选 `kappa_codes`，给出时该层代价改为 `lk.level1_cost(cat(r_e, r_h), codes, κ_codes)`（**只有第 1 层**由 `MixedRQ.forward` 传入，其余层仍用基线的乘积代价）；`MixedRQ` 持有 `self.kappa_codes` 供训练器刷新。
- `r48k_geometry_learned.update` 的均值赋值改为在 `torch.no_grad()` 内完成并对结果 `detach()`——**刷新是分配的统计量、不是可微路径**（Issue 验收 4）。
- 测试修正一处：表项是 float32，`0.7` 的比较容差从 1e-12 放宽到 1e-6（存储精度，不是实现松弛）；`update` 无梯度断言此前失败，正是缺 `detach` 的真实缺陷。

## 继承测试失败的定性（重要）

在**未改动的冻结基线** `TIGER_mixed_rq` 上跑同一套 `tests/`：**10 failed / 109 passed**；本树 **10 failed / 116 passed**——**同样的 10 个失败在基线上就存在**（`test_support_regressions.py` 的配额用例、`test_prefix_eligibility.py` 的陈旧前缀用例等），与本机制无关；本树多出的 7 个通过用例正是 `test_lkc.py`。

## epoch 末刷新已接线

- 数据集改为 `(embedding, id)`；循环解包 `batch, batch_ids`；每 batch 把 `(κ_of(batch_ids), 一级 token)` 累积下来。
- epoch 末调用 `r48k_geometry_learned.update(...)`，新表写回 `raw_module.rq.kappa_codes`，并把 `{moved, empty, floored}` 与 κ 的 5%/50%/95% 分位数写入 metrics（`kappa_refresh` 事件）。
- 初始 `κ_k(0) = 1.0`（256 个码）。

## 重要缺陷：刷新块曾**静默未插入**（冒烟 PASS 但臂等于基线）

第一次冒烟 PASS（`collision=0.847765`），但 metrics 里 **`kappa_refresh` 计数 = 0**。核查发现：我插入刷新块用的锚点 `_epoch_t1 = time.time()` 在本树中不存在（本树是 `epoch_time_s = round(time.time() - _epoch_t0, 3)`），那次 `str.replace` **静默失败**——于是累积的 `(κ_i, token)` 从未被消费，`κ_k ≡ 1`，**该次运行实际就是冻结基线**。

已按本树的真实锚点（`epoch_time_s = ...` 之前）插入刷新块，`grep -c kappa_refresh` = 1，训练器可导入。教训：`str.replace` 的锚点必须回读确认，不能只信打印。

## 冒烟 **PASS**，刷新确实生效（本轮实测）

- `SMOKE_PASS`；metrics 中 `kappa_refresh` 事件**逐 epoch 记录**（到第 401 epoch 已 401 条），不再是静默 no-op。
- 实测表在动：epoch 1 `moved=153 / empty=103 / floored=0`，`q05=0.556027、median=1.0、q95=1.455871`；epoch 401 `moved=66 / empty=190`、`q05=0.296023、q95=1.861915`——**κ_k 从常数 1 分化出来**，空码数随分组集中而上升（256 个码里 190 个在该 epoch 没分到商品）。
- 训练进行中（6 进程），epoch 401/3000、loss 1.51e-4。
