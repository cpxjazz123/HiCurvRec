---
name: stage3-start-mcp
description: Use when starting or watching a Stage3 HG-Rec run in this repo. Stage3 is split into two MCP tools — stage3_start only dispatches, stage3_status is the read-only monitor — and this skill gives the call contract, the protocol check the server performs, the minute-by-minute health check, and what each status means.
---

# 经 MCP 启动并监控 Stage3

两个工具，职责不重叠：

| 工具 | 干什么 | 不干什么 |
|---|---|---|
| `stage3_start(item_sids)` | 校验 SID 路径 → 协议核对 → 生成 harness → 活进程守卫 → `Popen` | 不跑训练、不等待、不改任何东西 |
| `stage3_status(item_sids)` | 纯只读：状态、存活进程、GPU、最新 train 记录、`test_final.json`、ckpt、run/native 日志尾巴 | 不启动、不停止、不改任何东西 |

**禁止**手工执行 trainer 或复制底层命令绕过 MCP；返回 `refused` 时按它指出的前置条件修。

## 0. 前置：MCP 进程必须加载当前代码

harness 由服务器内存里的模板生成，状态工具也是服务器内的实现。**改过 `tools/genrec-mcp/server.py` 后必须请用户 `/mcp reconnect genrec`**，否则生成出来的仍是旧 harness、或没有 `stage3_status` 这个工具。不要用进程内调用或手工启动替代。

## 0.5 环境检查（start 之前必做）

Stage3 有**两种布局，都由 trainer 自己实现**，MCP 按探测到的可见卡数选，**不能经 `stage3_start` 传参**：

| `mode` | trainer 做什么 | batch 语义 |
|---|---|---|
| `ddp`（≥4 卡） | 照旧 fork `torchrun --nproc_per_node=4` | 每 rank train batch 4096，全局 16384；每 rank infer batch 1024 |
| `single-gpu`（1–3 卡） | 不 fork，本进程单卡直跑 `main()` 原有的 `world_size=1` 分支 | train batch 仍 4096/进程，**全局 4096**；infer batch 缩到 `SINGLE_GPU_INFER_SIZE`（512） |

所以先确认环境：

```
write xd://mcp__genrec_stage3_status
{"item_sids": "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/<树>/<机制>/item_sids.json"}
```

看 `current_gpu_probe`（**现在**这台机器的布局：`visible_count` / `required` / `mode`），不要用 `gpu_probe`（上次派发时的记录）代替：

| `mode` | 含义 | 该怎么做 |
|---|---|---|
| `ddp` | 4 卡 DDP，冻结协议逐值不变 | 可以派发 |
| `single-gpu` | 跑单卡分支：超参、epoch 数、beam 都不变，只是全局 batch 与步数不同 | 可以派发，但**结果不能与 4 卡基线直接比**（见下） |
| `none`（0 卡） | 没有卡 | `stage3_start` 在写 harness、写状态文件之前拒绝；换一个分配 |

**单卡结果的比较口径**（引用数字前务必核对 `launch` 块）：

- 单卡每步全局 batch 4096，4 卡是 16384；单卡每 epoch 的 optimizer 步数是 4 倍，LR 曲线按 epoch 走。
- 因此单卡数字**不能**当作 4 卡基线的替代品，只在“同一机制单卡 vs 单卡”之间可比。
- 冻结协议字段（`NUM_EPOCHS / NO_EVAL / SKIP_TEST / SEED / BEAM_SIZE`）两种布局都核对，都不改。
- infer batch 不进指标（每行独立 beam search，按行 reduce），所以单卡把 infer batch 缩小只影响耗时，不改变 `test_final.json` 里的数字。

判断本次到底是哪种布局，看 `stage3_status` 的 `launch`：`dispatch_mode`（派发时探测）、`reported_mode` + `world_size` + `global_train_batch` + `infer_batch_per_process`（trainer 自己写进 metrics 的 `train_start`），两者应一致；不一致说明 trainer 与服务器看到的布局不同，先查清楚再引用数字。

## 1. 启动

```
write xd://mcp__genrec_stage3_start
{"item_sids": "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/<树>/<机制>/item_sids.json"}
```

- 唯一参数 `item_sids`，必须是 `results/stage2_RQ-VAE/` 之下的路径（也可给包含它的目录）。
- 变体名由该路径推导，写成 `results/stage3_T5Train/<树>__<机制>/`；超参与协议值一律不得经 MCP 传入。
- 回包含 `started`、`pid`、`variant`、`sid`、`harness`、`run_log`、`native_log`、`launch_mode` 与 `protocol`（expected/found/mismatched/ok）。

## 2. 服务器内部按序做什么

1. 读 trainer 的那几个字段核对（当前实现看 `NUM_EPOCHS / NO_EVAL / SKIP_TEST / SEED / BEAM_SIZE` 以及是否还存在 early-stop 分支）。不一致 → `refused: frozen Stage3 protocol mismatch`，回包直接给出 `protocol.expected` 与 `protocol.found`。这是 MCP 自带的检查行为，不是项目规则；被拒时按回包报告差异，不要绕过 MCP。
2. 在 `/tmp/genrec-mcp/run_stage3_param_<变体>.py` 生成 harness（仓库内不落任何东西、trainer 仍无 CLI 参数）；harness 把 `stage3_T5Train` 插入 `sys.path`、覆盖 `CODE_PATH / RQVAE_VARIANT / LOG_PATH / SAVE_PATH / PER_USER_DUMP`、注入 NCCL 环境变量，再按 `RANK` 或 `GENREC_STAGE3_LAUNCHED` 决定 `main()` 还是 `_launch_via_torchrun()`——两者都缺省时进入 launcher，由 trainer 按 `GENREC_SINGLE_GPU` 选 4 卡或单卡分支。
3. 探测可见卡数 → 写子进程环境（`GENREC_SINGLE_GPU=1` 走单卡）→ 活进程守卫（cmdline 含 `run_stage3_` 前缀的也算）→ 拒绝重复启动 → `Popen`（`start_new_session=True`，stdout/stderr → `logs/_stage3_run.log`）→ 落 state（`pid / launched_at / variant / sid / harness / mode / gpu_probe / 日志路径`）。

## 3. 每分钟监控

启动成功后**每 60 秒调一次** `stage3_status`，只用这一个工具判断（它已经带 `latest_train`、`progress`、`error_markers`、日志尾巴、存活进程和 `gpu_processes`）。正常就什么都不做；异常立刻定位并修。

健康判据（全部满足才算正常）：

| 看什么 | 正常 |
|---|---|
| `status` | `running`（终态 `finished` 也算正常结束） |
| `progress` | 形如 `37/150`，比上一次采样**在涨** |
| `latest_train.loss` | 有限（不是 `NaN`/无穷） |
| `error_markers` | 空（只标 `cuda out of memory` / `nccl error` / `runtimeerror` / `cuda error`；被 SIGTERM 结束留下的 traceback 不算） |
| `native_log_tail` | 每 epoch 有 `[train] epoch=N/150 done loss=...`，且无上述错误字样；开头有一行 `[layout] mode=...`（4 卡 `ddp4` / 单卡 `single-gpu`） |
| `run_log_tail` | launcher 的 banner（`[RecBole HG-Rec] launching ...`）。**fork 之前的崩溃只出现在这里**（`native_log_tail` 是空的），所以这一项为空但 `status` 已 `exited_incomplete` 时先读它 |
| `live_processes` | `running` 时非空（4 卡：harness + torchrun + 4 ranks + dataloader workers；单卡：harness + 单进程 + workers） |
| `gpu_processes` | `running` 时 `launch.reported_mode` 说 `ddp4` 就 4 张卡都被本变体的 rank 占住（每卡约 28 GB），说 `single-gpu` 就 1 张卡被本进程占住 |

注意：metrics 流在 `logs/<数据集>/<Mon-DD-YYYY_HH-MM-SS>/training_metrics.jsonl`，**每次启动新建一个时间戳目录**，所以 `metrics` 字段指向的永远是本次那一份，不会混到历史 run。

启动初期 1–3 分钟内进程只有 harness、GPU 为空、日志为空是正常的（torch import + 数据加载）；判定卡死前先看两三个采样周期，必要时用 `ps -o stat,etime,time -p <pid>` 看是否在耗 CPU。

## 4. status 取值与处置

| status | 含义 | 动作 |
|---|---|---|
| `running` | 有本变体的存活进程 | 按第 3 节每分钟监控 |
| `finished` | 进程已退出，**且本次 launch 之后写出** `test_final.json` | 读 `test_result` 与 Stage2 基线对比；这是唯一裁决依据 |
| `exited_incomplete` | 进程退出但本次没写 `test_final.json`（例如被中途结束） | 读 `native_log_tail` 与 `error_markers` 定位 |
| `idle` | 该变体从没启动过 | 直接 start |

**证据归属**：变体目录是跨 run 复用的，所以 `stage3_status` 只认 `launched_at` 之后写出的 `test_final.json` 与 metrics 流——
- 旧 run 残留的 `test_final.json` 会让 `status` 停在 `exited_incomplete`，同时 `test_final_stale` 指向那个文件，不会假报 `finished`；
- `live_processes` 按 harness 名（含变体名）过滤，别的变体的进程不会被算作本变体在跑；
- metrics 也按 `launched_at` 过滤，`latest_train` / `progress` 不会混入历史 run。

## 5. 常见失败

| 现象 | 原因 | 修法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'data'` | harness 未把 `stage3_T5Train` 加入 `sys.path` | 修 `server.py` 的 harness 模板后重连 MCP |
| `refused: frozen Stage3 protocol mismatch` | trainer 里被核对的那几个字段与预期不同 | 比对回包 `protocol.expected` vs `protocol.found` 的差异并据实报告 |
| `refused: ...already live` | 已有 stage3 进程 | 先 `stage3_status` 确认；不并行起第二个 |
| `refused: ...no single-card launch path` | trainer 里的 `GENREC_SINGLE_GPU` 分支被删了（服务器按 marker 判定能力） | 恢复 trainer 的单卡分支后重连 MCP；不要绕过这条拒绝 |
| `refused: no CUDA device is visible` | 本机 0 卡 | 换分配；两种布局都至少要 1 张卡 |
| `error_markers` 非空 / 日志有 CUDA OOM | 与别的训练抢卡（4 卡布局每卡约 28 GB） | 先腾出 GPU 再重试 |
| 单卡跑到 test 阶段 OOM（`[test]` 之前就 `cuda out of memory`） | 单卡显存小于 4 卡的 46 GB 卡，`INFER_SIZE=1024` 的 beam 生成装不下 | 调小 trainer 的 `SINGLE_GPU_INFER_SIZE` 后重连 MCP 再跑；infer batch 不影响指标，调它不改 `test_final.json` 的数字 |
| 单卡 train 阶段就 OOM | 单卡显存小于 4 卡每 rank 的 train 峰值 | 同样下调 `SINGLE_GPU_INFER_SIZE` 之外的 batch 需要改 `BATCH_SIZE`，那会改全局 batch 语义——先报给用户再动 |
