# "Bus error (core dumped)": 找到的原因

结论：**不是训练代码的缺陷**。排查过程中把三类现象分开记账之后，实际原因和 GPU 报错
无关，而是**进程被工具的生命周期回收**；而真正能被称作 GPU 故障的 Xid 事件是历史记录，
不是反复出现的那一类。

## 一、核心里程碑证据

| 现象 | 证据 | 时间 | 判定 |
|---|---|---|---|
| `Xid 31` MMU fault（卡 0，`name=python`）+ 同刻 `python segfault in libcuda.so.580.126.20` | `dmesg` | **21252 分钟前（14.7 天）** | 历史事件，与当前无关 |
| `Xid 109` CTX SWITCH TIMEOUT ×6（卡 1，`name=pt_main_thread`） | `dmesg` | **11455 / 8817 分钟前（8 / 6 天）** | 历史事件 |
| `Memory cgroup out of memory: Killed process … (pt_main_thread)`，RSS 14–19 GB | `dmesg`，共 14 条 | **9067–30252 分钟前（6–21 天）** | 历史事件 |
| 四张卡的最小 kernel 测试 | `CUDA_VISIBLE_DEVICES=i` 做 512×512 矩阵乘 | 刚才 | **四张卡全部正常** |
| 训练进程在 ~2 分钟内以 exit=1 退出且**日志里没有 traceback** | `proc://` 服务状态 + 服务日志 | 刚才 | 进程被外部终止，不是崩溃 |

⇒ "Bus error" 那一次的真实机理仍然是 GPU 侧：Xid 31 打掉的是 **CUDA 上下文**而不是进程，
进程随后访问失效的 device 映射就收到 SIGBUS，glibc 打印 `Bus error (core dumped)`。但这条
记录是 14 天前的，**不能用来解释这两天的反复退出**。

## 二、这两天反复退出的真实原因（三个叠加因素）

1. **会话中期换卡**（A100 → L40S）。旧上下文在设备消失的瞬间失效，这是第一次三臂在
   epoch 620 / 2321 / 2491 处突然 `end=True` 的原因。
2. **同一张卡上跑多个独立 CUDA 上下文**。我并行脚本与 service 重复启动时一度出现
   **6 个进程 / 3 张卡**（每卡 2 个上下文），这正是历史上 Xid 109 出现的条件，必须避免。
3. **进程挂在工具自己的生命周期上**。用 `bash` 的 service（`persist=false`）或普通
   `nohup &` 启动的训练，会在会话工具调用切换时被回收器一并杀掉，表现为"跑了 2 分钟就
   没了、日志无 traceback"。这条是我这边的操作错误，与集群无关。

## 三、已执行的处置

- 启动脚本 `/tmp/run_arm.sh` 用 `setsid nohup` 把训练**脱离调用者的进程组**，每臂
  独立日志 `logs/train_card<N>.log`；一臂一卡（M3→2、M1→3、M4→0），**每卡只有一个
  CUDA 上下文**；卡 1 因有 Xid 109 历史暂不使用。
- 全局 batch 仍为 4096（单 rank 单卡分支），与冻结配方一致。
- 训练期间不再并发跑 `checks.run_all`（它的 `four_rank` 探针会额外开上下文，且会与
  正在跑的 admission 副本互相删除，之前两次假 `NOT_RUN`/`refused` 就是这么来的）。

## 四、需要 root 的部分（我做不到）

- 若再次出现 Xid 109 / Xid 31，需要 `nvidia-smi --gpu-reset` 或重启节点；用户态无权限。
- cgroup 内存上限我没有读到（`/sys/fs/cgroup/memory.max` 不可读），但节点 `free` 显示
  1511 GB 总内存、可用 1343 GB，所以并发 3 臂（每臂 ~19 GB 峰值）不是内存问题。
