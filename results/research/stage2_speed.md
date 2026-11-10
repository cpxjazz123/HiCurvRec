Stage2 执行速度：实测账目、已落地改动、余量与禁止项

所有数字都来自本仓库的真实运行记录（`results/stage2_RQ-VAE/*/logs/training_metrics.jsonl`）
或本机在 `genrec_env_v2` 下直接跑出的测量；没有估计值。

一、一臂 Stage2 的时间花在哪

以最近三个臂的实际记录拆分（startup = `train_start` 到第 1 个 `train`，
export = 最后一个 `train` 到 `train_end`）：

    arm                             startup   3000 epochs   export     total
    TIGER_struct_prior_marginal_rq     3.0 s      1083.4 s    0.7 s    1090.0 s
    TIGER_af3_gate_mixed_rq            3.3 s      1404.5 s    0.9 s    1412.7 s
    TIGER_struct_mixed_rq_ste_fix     10.0 s      1023.4 s    0.6 s    1038.1 s

结论：**99.4% 的时间在 epoch 循环**（启动 0.3%、导出 0.1%）。
启动里包含 KMeans 码本初始化；`kmeans(256, max_iter=10)` 在 768 维上要 24.9 s，
但训练器喂的是 32 维 encoder 输出，三层合计 < 3 s，所以它不是优化目标。

二、epoch 循环的构成

3000 epoch × `ceil(24556 / 4096) = 6` 步 = 18000 步。实测每步 ~60 ms（#150）。

逐块测量（CPU，batch 1024，中位数；`SetWindow` 为本次改造后的实现）：

    SetWindow.batch_sets            9.2 ms   （改造前 21.6-35.7 ms）
    triangle_loss / fork_loss       0.8 / 1.9 ms
    prior_kl                        0.7 ms
    三项 + 反传合计                 14.5 ms
    模型单步（不含结构项）           307.9 ms   ← CPU 值，仅用于比较，不代表 GPU
    模型单步 + 结构项                661.6 ms   ← 同上

要点：CPU 上"加结构项"多出的 354 ms 是**设备侧**代理反传（[1024,256]×3 层的
softmax/one-hot/centring），在 GPU 上是微秒到毫秒级；真正 host 绑定、无论如何都
要付的只有 `SetWindow` 那一块。

三、同配方下各类臂的实测每 epoch（3000 epoch、1024/rank、全局 4096、4 卡、seed 42）

    _reference_baseline_deterministic        0.055 s/epoch   165 s
    TIGER_transition-geodesic-prior          0.073 s/epoch   219 s
    TIGER_BRad-RQ                            0.088 s/epoch   264 s
    TIGER_struct_prior_marginal_rq           0.319 s/epoch  1090 s
    TIGER_struct_mixed_rq_ste_fix            0.327 s/epoch  1038 s
    TIGER_cooccur_w30                        0.449 s/epoch  1406 s
    TIGER_curvature-prefix-ranking-forman    0.681 s/epoch  2098 s

36 个有 metrics 的臂：中位 571 s，累计 7.4 h。0.055 s/epoch 是这批模型的地板，
差距全部来自各臂自带的每步 host/设备工作。

四、已落地（main 上，均已推送）

1. `db4cd4b1`  host 侧集合构造 2.3 倍
   `batch_sets` 每步把 546438 三角 + 315391 边整体过 lookup：
   21.6-35.7 ms/步，约占一臂 epoch 的 62%。改用 `SetWindow`（加载时建一次
   item→incident-row CSR，0.20 s），每步只取本 batch 512 个商品的邻域：
   **9.2 ms/步**，返回的 `triangles/within/cross/coverage` 逐位相同
   （`tests/test_struct_sets_fast.py` 在合成池与真实 #150 池上断言）。
   同时 `centred_cost` 的分支判断改用 `spread.detach()`：原来在带梯度张量上取标量，
   每层每步触发告警与 device sync；返回式的分母仍保留可微 `spread`，梯度不变。

2. `9952e032`  耗时投影错 35 倍
   `stage2_status.cost_projection_hint` 曾把 #150 报成 10.5 h，实际 1090 s。
   两处叠加：步数用了单卡的 24（ddp4 真实 6，放 4 倍）；`per_step_s` 是冒烟进程
   **总墙钟**除以步数（含启动/加载/DDP 建组，0.525 s/步 vs 稳态 60.5 ms/步，放 8.7 倍）。
   现在步数按启动布局推导、结果标注 `upper_bound: true`，且 `stage2_status` 读取时
   重算历史记录（`tools/research/stage2/tests/test_cost_projection.py` 6 例）。

3. `c34db9e2`  每 epoch 分块计时
   `PROFILE_EPOCHS = (1,)`：第 1 个 epoch 记录 `profile` 事件，
   字段 `data_s / forward_s / terms_s / backward_s / total_s / steps`。
   同步只在该 epoch 发生，被测管线保持异步。

五、仍可尝试的两条（都未落地）

1. GPU 侧掩码：**已实现**（`SetWindowTorch`，训练器检测到 CUDA 时启用，
   非 CUDA 保持 numpy 路径）。三角/边以 int64 常驻显存，每步在卡上做两次
   gather 与掩码，host 只交出 1024 个 id。
   - 正确性：与 `SetWindow` 及 `batch_sets` 逐位一致（合成池与真实 #150 池，
     `tests/test_struct_sets_fast.py`，10 例通过）。
   - 速度：**未在本机测量**（两个项目解释器均 `torch.cuda.is_available() = False`）。
     CPU 上它反而更慢（19.1 → 41.2 ms/步），所以路径按 `device.type == "cuda"`
     门控；卡上的收益是推断值（全池 gather 861k 元素，预计亚毫秒），不是实测值。
2. 探针前向频率：3072 行/步 → 每 k 步一次。这是**改目标**（每 epoch 结构项数量变化），
   按 CLAUDE.md §19g 必须在派发前写进 `math_spec.json`，不属于"优化"。

六、已否决的做法（记录理由，避免重复试）

1. 布尔 stamp + `flatnonzero` 去重：**16.1 ms/步**，比当前的 9.0 ms 更慢
   （去重本身快了，但每步要清空并扫描 546k 的标记数组）。结果逐位相同。
2. 复刻 `DistributedSampler` 的每 epoch 打乱顺序以"每 epoch 只算一次集合"：
   理论上可再快 5.5 倍，但一旦复刻与 DataLoader 实际批次有偏差，集合会静默错位
   ——训练照跑、指标照出、结论错。不做。
3. 减少 epoch、放大全局 batch、降精度：改变冻结配方（§18/§19），结果与其它臂不可比。

七、怎么读耗时数字

- `cost_projection_hint` 是**上界**（含冒烟启动），不要用它判断快慢。
- 判断真实速度看 `latest_train.epoch_time_s` 或 `train_end.total_wall_time_s`。
- 想知道下一步该优化什么，看 `profile` 事件那一行。
- Stage2 已经不是整轮的大头：Stage3 单臂 150 epoch ≈ 30-38 分钟（冻结协议，不可动）。
