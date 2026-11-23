## 节点 2（C5）：诊断无副作用 —— 开关前后逐位等价，三种运行模式全绿

脚本 `results/research/issue240/equivalence_test.py`，结果 `equivalence/equivalence.json`。
做法：同一棵树的三个副本（诊断开 / 关 / 再关一次），同种子跑完整真实入口，比对
`item_sids.json` 与 `rqvae_best.pth` 的 sha256。

| 模式 | 关 vs 关（确定性） | 关 vs 开（等价） |
|---|---|---|
| 单进程直跑 | 一致 | **一致** |
| **ddp1（历史配方）** | 一致 | **一致** |
| ddp4（四卡 smoke） | 一致 | **一致** |

### 反向校验：加权项梯度之和 = 实际 backward 梯度

记录器对每个 loss 项单独做 `autograd.grad`（不建 `AccumulateGrad` 节点，因此不触发 DDP reducer，
也不写 `.grad`），再把各项按实现系数加总，与实际 `.grad` 比：

| 模式 | `‖pred − actual‖ / ‖actual‖` 最大值 |
|---|---|
| 单进程 | **6.6e-05** |
| **ddp1** | **2.3e-04** |
| ddp4 | 1.06 |

ddp4 的 1.06 不是 bug：DDP 的 `.grad` 是各 rank 的**均值**，而探针算的是本 rank 的梯度，
两者之差就是 rank 间离散度。这正是 issue 要求「不可将单 rank 未同步范数冒充全局梯度」的实测依据；
本轮正式运行是 `world_size=1`，全局即本地。

### 过程中修掉的两个真 bug（都会让数字失真，已写进代码注释）

1. **STE 陷阱**：`VQLayer.forward` 返回的是 `x + (x_q − x).detach()`，其数值等于 `x`。
   直接对 hook 拿到的输出重算 codebook/commitment 项会得到恒零梯度。改为用输出 token
   和同一码本参数重建 **STE 之前**的 `x_q`（同索引、同权重、同图边），梯度才与实际一致。
2. **系数是 1/3 不是 1**：`RQLayer.forward` 返回 `sum_quant_loss / codebook_num`。
   逐层量化项进入 loss 的系数是 **1/3**（commitment 部分为 β/3）。1.4 的 `rel_error`
   就是漏掉这个 1/3 造成的。

修正后 `rel_error` 从 3.1 → 6.6e-5（单进程）。

### 同时确认的配方事实（C2）

issue 要求核实每卡/全局 batch。历史 `train_start` 记录为
`single_gpu=False, mode=ddp1, batch_size_per_rank=4096, item_batch=4096, global_item_batch=4096`。
源码存在**单卡分支 `batch_scale=REFERENCE_WORLD_SIZE(=4)`**：若用 `python train_rqvae.py` 直跑，
`item_batch` 会变成 16384 —— **与历史配方不同**。因此正式复跑用
`torchrun --nproc_per_node=1`（`mode=ddp1`, `batch_scale=1`），启动后 `train_start` 记录已核对为
`item_batch=4096, pair_batch=2048, world_size=1`，与历史逐键相同。

*（`results/research/issue240/COMMENT_02.md`）*
