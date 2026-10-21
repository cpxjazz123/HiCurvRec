---
name: stage2-start-mcp
description: Use when starting or watching a Stage2 RQ-VAE run in this repo. Stage2 is split into two MCP tools — stage2_start only dispatches, stage2_status is the read-only monitor — and this skill gives the call contract, the minute-by-minute health check, and what each refusal or terminal status means.
---

# 经 MCP 启动并监控 Stage2

两个工具，职责不重叠：

| 工具 | 干什么 | 不干什么 |
|---|---|---|
| `stage2_start(source_dir)` | 解析树 → 校验入口 → 活进程守卫 → 归档旧产物 → 派 detached worker | 不跑 smoke、不启动训练、不等待 |
| `stage2_status(source_dir)` | 纯只读：状态、smoke 结论、存活进程、最新 train 记录、run/native/worker 日志尾巴、产物、pid/launched_at | 不启动、不停止、不改任何东西 |

**禁止**手工复制底层命令（`nohup ... train_rqvae.py`）绕过 MCP；MCP 返回 `refused` 时按它指出的前置条件修，不绕开。

## 0. 前置：MCP 进程必须加载当前代码

服务器在挂载/重连时把 `tools/genrec-mcp/server.py` 读进内存。改过该文件后，挂载中的进程仍是旧代码——表现为回包字段是旧结构，或没有 `stage2_status` 这个工具。请用户执行 `/mcp reconnect genrec`，不要改用进程内调用，也不要反复重试旧进程。

## 0.5 环境检查（start 之前必做）

MCP 在 `stage2_start` 里会探测本机可见的 CUDA 卡数并据此选布局，但**布局能不能兑现取决于那棵树自己的 trainer**，所以先读一次环境再决定要不要派发：

```
write xd://mcp__genrec_stage2_status
{"source_dir": "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/<树目录>"}
```

看两个字段，它们是不同的事实：

| 字段 | 含义 |
|---|---|
| `current_gpu_probe` | **现在**这台机器的布局：`physical` / `cuda_visible_devices` / `visible_count` / `required` / `mode` |
| `gpu_probe` | 那棵树**上次派发时**记录的布局（没跑过就是 `null`），不要拿它当今天的环境 |

判读：

| `mode` | 含义 | 该怎么做 |
|---|---|---|
| `ddp`（`visible_count >= 4`） | 4 卡 DDP | 正常派发 |
| `single-gpu`（1–3 卡） | 只有单卡布局 | 只有 `entrypoint_supports_single_gpu` 为 true（trainer 里有 `SINGLE_GPU_ENV = "GENREC_SINGLE_GPU"` 契约）才能派发；否则 `stage2_start` 会拒绝，**不要绕过**，要换 4 卡分配或先给该树加契约 |
| `none`（0 卡） | 没有可用 GPU | 不要派发；两种布局都至少需要一张卡 |

派发后用同一棵树 `stage2_status` 复核两处是否一致：状态里的 `gpu_probe.mode` 与 `smoke.launch_mode`（smoke 实际用的布局）应当一致；不一致说明模式被树里的常量覆盖了，先查那棵树的 `SINGLE_GPU_MODE`。

## 1. 启动

```
write xd://mcp__genrec_stage2_start
{"source_dir": "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/<树目录>"}
```

唯一参数 `source_dir`；超参与协议值全部硬编码在树里，不得经 MCP 传入。

回包毫秒级返回：

```json
{"started": false, "status": "smoke_pending", "worker_pid": 123, "worker_log": "...",
 "archived_artifacts": ["..."], "poll": {"tool": "stage2_status", "arguments": {"source_dir": "..."}}}
```

`smoke_pending` 既不是拒绝也不是成功——它表示 worker 正在跑单 epoch smoke。**必须查一次状态**：

```
write xd://mcp__genrec_stage2_status
{"source_dir": "同一个树目录"}
```

## 2. 服务器这一侧发生了什么

1. 从 trainer 源码的 `_OUT` 解析机制名，推出 `results/stage2_RQ-VAE/<树>/<机制>/` 与 `logs/<机制>/`；入口必须是该树的 `train_rqvae.py` 且含 `__main__` + `_launch_via_torchrun`。
2. **活进程守卫**：真实命令行里已有 trainer，或上一轮的 worker 还活着（`status ∈ {starting, smoke_pending}` 且 `/proc/<pid>/cmdline` 确实是本 worker），都拒绝再启一次。
3. **旧产物不阻断运行**：先 `copy2` 到 `/home/wlia0047/ar57/wenyu/artifact_archive/<树>_<unixts>/` 并逐个校验大小；原文件留到 smoke 通过之后才删（删不掉即拒绝，避免新旧混写）。
4. 派 detached worker `/tmp/genrec-mcp/stage2_worker.py`（import 服务器模块，`start_new_session=True`）。它跑 smoke：patch 树内 `EPOCHS=1`、输出落 scratch、逐项记录 loss 与梯度、崩溃输出 traceback；smoke 的 stdout+stderr **实时**写进 `/tmp/genrec-mcp/smoke_<树>.log`（`python -u`，不再等结束）；真实 4 卡 DDP；另用**真实 ckpt + 真实 DataLoader batch** 对 `probe_model.parameters()` 求梯度。探针的 ckpt 优先用 `result_dir/rqvae_best.pth`；**若目录里没有（例如上一轮被中途结束、产物已按流程删除），回退到归档里最新的那份**，并在 `smoke.checkpoint_source` 里说明来源；归档里也没有才 fail-closed。超时会整组杀掉（`os.killpg`），不留 torchrun 孤儿 rank。
5. smoke 通过 → worker 删原产物 → `Popen` 训练（stdout/stderr → `logs/<机制>/train_run.log`）→ 写回 `status:running / pid / launched_at / smoke`；未通过 → 写回 `status:refused` + smoke 明细 + traceback。状态文件原子写（临时文件 + `os.replace`），监控端不会读到半截 JSON。

探针也会包裹树内返回张量的公开 helper（例如 `project_deep_supervision`），不只标量 loss。对多元素返回值，`probed_terms` 中的 `loss` 是绝对值均值，并附 `reduction: "mean_abs"` 与 `shape`；`grad` 来自返回张量之和对输入的求导，用于检查梯度连接，**不代表训练目标或生产反向传播的梯度**。标量返回值保持原探针语义，空张量不记项。修改该逻辑仍需重连 MCP，再经真实 smoke 验证。

## 3. 每分钟监控

启动成功后**每 60 秒调一次** `stage2_status`，只用这一个工具判断（它已经带 `latest_train`、日志尾巴、存活进程和 `gpu_processes`，不必再开第二个工具或 shell）。正常就什么都不做；异常立刻定位并修。

健康判据（全部满足才算正常）：

| 看什么 | 正常 |
|---|---|
| `status` | `smoke_pending` → `running`（终态 `finished` 也算正常结束） |
| `latest_train.epoch` / `.timestamp` | **仅当 `status == running` 时**要求比上一次采样在涨；`smoke_pending` 阶段还没有新 train 记录，不作要求 |
| `latest_train.loss` | 有限（不是 `NaN`/无穷） |
| `native_log_tail` | 无 `Traceback` / `CUDA out of memory` / `NCCL` 报错 |
| `smoke_log_tail` | **`smoke_pending` 阶段**看它：应有进度（import、epoch、探针输出），且无 `Traceback`；worker 把 smoke 的 stdout+stderr 实时写进这个文件，不会让你等到结束才知道 |
| `live_processes` | `running` 时非空（launcher + torchrun + 4 ranks = 6） |
| `gpu_processes` | `running` 时 4 张卡都被本树 rank 占住 |
| `worker_alive` | `smoke_pending` 时为 true |

注意：`training_metrics.jsonl` **跨 run 追加**，`latest_train` 只给最后一条，所以要么用 `launched_at` 过滤，要么记住起点 epoch——别把上一轮的记录当本轮证据。

## 4. status 取值与处置

| status | 含义 | 动作 |
|---|---|---|
| `smoke_pending` | worker 正在跑 smoke | 等 60 s 再查；不要重复 start |
| `running` | smoke 通过、训练已启动 | 按第 3 节每分钟监控 |
| `refused` | smoke 有死项/非有限/ckpt 梯度为 0，或删不掉原产物 | 读 `refused`、`smoke.dead_terms`、`smoke.nonfinite_terms`、`smoke.checkpoint_probe`、`traceback`，修实现后重试 |
| `worker_gone` | worker 死了且没写状态 | 读 `worker_log_tail` 的 traceback |
| `finished` | 本次 `train_end` 事件已记录 **且** 必需三件产物齐备（`rqvae_best.pth` / `sids_for_hgrec.npy` / `item_sids.json`；`sids_raw.npy` 只有部分树写，不作为完成条件） | 收工，可进 Stage3 |
| `exited_incomplete` | 进程退了但没写 `train_end` 或产物缺失 | 读 `native_log_tail` 定位，再决定重跑 |
| `idle` | 从没启动过 | 直接 start |

## 5. 其它拒绝

| 回包 | 动作 |
|---|---|
| `refused: ...artifacts...` | 按报错修权限/路径，不要手工删产物 |
| `refused: ...already live` | 查 `stage2_status` 确认是真进程，别再 start |
| `refused: ...smoke worker ... already live` | 等 worker 跑完，`stage2_status` 会转 `running` 或 `refused` |
| `unavailable` / 工具不可见 | 请用户 `/mcp reconnect genrec` |

同一任务禁止重复启动；被拒绝时不要用 `nohup` 手工补一刀。
