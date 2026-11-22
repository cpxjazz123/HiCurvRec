## 四卡 DDP 串行已生效:速度 2.6×,锁与卡位都可查

### 运行状态(启动后 45 秒)

| 项 | 值 |
|---|---|
| 卡的占用 | GPU 0/1/2/3 **全部占用**(1281 / 961 / 961 / 961 MiB,30–36% 利用率) |
| 串行锁 | `results/research/stage2_serial.lock` = pid 3633043 |
| 速度 | **0.17 s/epoch**(单卡时 0.45 s) |
| 预计 | 每臂约 **8.5 分钟**,两臂串行约 17 分钟 |

### 派发方式

通过框架唯一入口 `launcher.launch(entry, result_dir, env_extra={"CUDA_VISIBLE_DEVICES":"0,1,2,3"})`,
先治疗臂、后打乱对照臂,由同一个驱动进程串行调用。`launch` 内部先 `assert_four_cards` 再
`acquire_serial`,`finally` 里 `release_serial`。

### 与之前单卡运行的差别(为什么重跑)

| 项 | 单卡(已终止) | 四卡 DDP(现在) |
|---|---|---|
| 每步 batch | 4096(单进程) | 4096 = 1024×4 |
| DDP 梯度 | 无 all_reduce | 每步 `all_reduce` |
| 速度 | 0.45 s/epoch | 0.17 s/epoch |

梯度同步顺序不同,数值上**不是同一次运行**;因此单卡那两个臂(epoch 375/315 处终止)的数字不进入结论。

### 规则已固化的证据

`pytest tools/research/stage2/tests/` **15 passed**,其中 6 个新测覆盖:
四卡常量、单卡拒绝、四卡接受、并发拒绝、stale 锁替换、释放后可再启。
另做了一次真实 `launch()` 调用:单卡时抛 `LaunchRefused` 且**结果目录未被创建**。
