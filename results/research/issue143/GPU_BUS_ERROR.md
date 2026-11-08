# "Bus error (core dumped)": 已经查到的原因

结论先说：这些 `Bus error` **不是 Python/训练代码的缺陷**，是 NVIDIA 驱动在 GPU 上
打错之后把 CUDA 上下文销毁，进程随后访问已失效的映射就收到 SIGBUS。证据全部在
`dmesg` 里，进程名就是我们的训练进程。

## 证据（`dmesg` 里 404 条 NVRM 记录，逐条对应到卡）

| 卡 | PCI | 驱动报错 | 进程 | 含义 |
|---|---|---|---|---|
| 0 | `0000:02:00` | **Xid 31**：`MMU Fault: ENGINE GRAPHICS GPC5 GPCCLIENT_GCC faulted @ 0x… Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ` | `name=python` | GPU 的 MMU 在一次虚拟读上 fault，上下文被销毁；同一时刻用户态出现 `python … segfault … in libcuda.so.580.126.20` |
| 1 | `0000:64:00` | **Xid 109** ×6：`CTX SWITCH TIMEOUT` | `name=pt_main_thread` | 上下文切换超时（每一条都是我们的 PyTorch 训练主线程），驱动随后丢弃这些进程 |

另外还有重复的：

```
NVRM: Going over RM unhandled interrupt threshold for irq 445
```

这是驱动在丢中断 —— 典型的卡住/驱动层故障前兆，不是应用能触发的东西。

硬件侧核对：四张 L40S 的 ECC 都是 `Enabled`，`retired_pages.pending` 都是 `N/A`，
没有退化页 ⇒ 不是显存坏块。

## 为什么是 SIGBUS 而不是普通异常

Xid 31（MMU fault）打掉的是 **CUDA 上下文**，不是进程。进程继续跑，下一次对已失效
的 device 映射的访问就由内核投递 SIGBUS；glibc 打印 `Bus error (core dumped)`。
Xid 109（context switch timeout）则让驱动直接终止持有上下文的进程。

## 我们这边能控制、也确实会放大的因素

1. **同一张卡上跑多个独立 CUDA 上下文**。第一次是 4 个臂挤在 1 张卡上（5-6 个进程），
   第二次是我并行脚本和 service 重复启动，一度出现 **6 个进程 / 3 张卡**（每卡 2 个
   上下文）。Xid 109 的 6 条全部出现在这种"多上下文同卡"的时段。
2. **会话中期换了显卡**（A100 → L40S）。旧上下文在换卡瞬间失效，这是第一波
   `end=True` 在 epoch 620 / 2321 / 2491 处突然结束的直接原因。

## 处置（已执行 / 约束）

- 现在改成 **一臂一卡**，每卡只允许 **一个** CUDA 上下文：
  M3 → card 2、M1 → card 3、M4 → card 0（card 1 有 Xid 109 历史，先不用）。
- 启动器在 `run_parallel.py` 里记录每臂的 `pid`、`card`、源码摘要；同一卡出现第二个
  进程即视为配置错误。
- 训练不再与任何 CPU/GPU 检查并发执行：`checks.run_all` 的 `four_rank` 探针会额外开
  上下文，之前的假 `NOT_RUN`/`refused` 就是它和别人抢同一份 admission 副本造成的。

## 仍需人（有 root）处理的部分

- 若 Xid 109 / Xid 31 再次出现，卡需要 `nvidia-smi --gpu-reset` 或整机重启；
  用户态没有权限做这件事，也无法用代码规避。
- 如果只是某一张卡反复出 Xid 109，可以把这张卡从训练里排除（这正是当前的做法：
  用 card 2/3 跑，card 0 观察，card 1 停用）。
