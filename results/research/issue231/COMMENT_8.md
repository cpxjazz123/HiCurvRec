## 变更(owner 指令):Stage2 固定四卡 DDP,**串行**

### 规则

1. 可见卡为 4 时,**必须**四卡 DDP;**不允许单卡跑一个 Stage2**。
2. Stage2 **串行**:一次只允许一个在跑。

### 落地位置:`tools/research/stage2/launcher.py`(唯一启动路径)

| 函数 | 作用 |
|---|---|
| `assert_four_cards(visible)` | 可见卡 < 4 即 `LaunchRefused`,在**写任何结果目录之前**抛出 |
| `acquire_serial(lock, pid)` | 已有存活 pid 持锁即拒绝;pid 已死的锁视为 stale 可替换 |
| `release_serial(lock, pid)` | 运行结束(含异常)在 `finally` 里释放 |

`launch()` 里新增两道前置检查:`world_size != 4` 拒绝;`_visible_cards(env_extra)` 按
`CUDA_VISIBLE_DEVICES` 实际计数后 `assert_four_cards`。

### 候选树里的单卡退路已删除

`train_rqvae.py` 原来在 `_visible_card_count() <= 1` 时设置 `GENREC_SINGLE_GPU=1` 并单进程训练。
现在改为**直接 `SystemExit`**,并把 smoke 的 `world_size=1` 也去掉 —— smoke 同样四卡。
`_single_card_mode()` 恒返回 `False`,保留签名以免动到 DDP 块。

### 验证

| 检查 | 结果 |
|---|---|
| `pytest tools/research/stage2/tests/` | **15 passed**(含 6 个新测) |
| 单卡 `assert_four_cards(1)` | 抛 `LaunchRefused` |
| 单卡真实 `launch(...)` | 抛 `LaunchRefused`,**结果目录未创建**(无部分状态) |
| 四卡 `assert_four_cards(4)` | 通过 |
| 第二个并发 run | 拒绝 |
| stale lock(pid 已死) | 可替换 |

### 影响:round 1 的两臂已被我终止

之前两个臂是**单卡**(GPU 0 / GPU 1 各一个),按新规则不合格。已在 epoch 375 / 315 处停止,
其数字不进入结论。重新派发时改为**四卡串行**:治疗臂先跑,跑完再跑对照臂。

这与 CLAUDE.md §19d0b(允许 Stage2 并行)冲突,按 owner 指令以本条为准。

证据:`tools/research/stage2/tests/test_four_card_serial.py`、`tools/research/stage2/launcher.py`。
