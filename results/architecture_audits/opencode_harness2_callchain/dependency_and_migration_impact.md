# 依赖事实与「移除原版 Harness² 运行依赖」的影响面

**Issue #113 · 只读审计 · `main` = `f6d78b68` · 2026-10-23T14:04:31Z**

**本文件只分析，不实施。** 不写迁移代码、不删模块、不改接口。

---

## 1. 依赖矩阵

`✔` = 代码 import 或运行时依赖；`—` = 无依赖；`△` = 间接（经本仓库其它模块）。

| 本仓库模块 | 原版 `harness2` | 自研 OpenCode CLI | GenRec MCP | Stage2 训练器 | 角色 |
|---|:--:|:--:|:--:|:--:|---|
| `cli.py` | **✔** `core.library`/`core.schedule`/`core.spec` | ✔（经 proposer） | — | — | 唯一进入框架调度器的入口 |
| `proposer.py` | **✔** `core.proposer.Proposer/Reply` | **✔** `run_agent` | ✔（`mcp_server_block`） | — | 框架 ↔ CLI 的接缝 1 |
| `substrate.py` | **✔** `core.tree`/`substrate.base` | **✔** `run_agent`(:416) | ✔ | — | 框架 ↔ CLI 的接缝 2 |
| `adapter.py` | **✔** `core.tree`/`core.spec` | — | — | — | benchmark 契约 |
| `surface.py` | **✔** `core.tree.SurfaceSpec` | — | — | — | 可迭代 surface 声明 |
| `prompts_ext.py` | **✔** `core.prompts` | — | — | — | prompt 注册 |
| `opencode.py` | — | **✔** 唯一封装 | △（提供 config 块） | — | CLI 封装层 |
| `campaign.py` | — | **✔** ×4 | — | △（经 lifecycle） | 自研编排 |
| `lifecycle.py` | — | **✔** ×2 | ✔（in-process import） | △（经 MCP） | 自研生命周期 |
| `dag.py` | — | — | — | — | 纯记账/租约 |
| `controller.py` | — | — | — | ✔（`run_candidate`） | 另一套状态机 |
| `tools/cgv/*` | — | — | — | △（只读被测树） | 独立验证器 |
| `tools/genrec-mcp/server.py` | — | — | 本体 | ✔（启动） | 工具 + 进程启动 |
| Stage2 训练器 | — | — | — | 本体 | 唯一 GPU 消费者 |

**统计**：import 原版框架的模块 **6** 个；调用 OpenCode CLI 的模块 **11** 个（24 个调用点）；
调用 GenRec MCP 的模块 **3** 个（`lifecycle` 直接 import 为模块、`opencode` 写进 config、
`campaign` 间接）；**训练器与以上全部无关**。

---

## 2. 只依赖原版 Harness² 的功能（若移除框架运行依赖会受影响）

| 功能 | 位置 | 受影响程度 | 说明 |
|---|---|---|---|
| `cli sweep`（整个 sweep 入口） | `cli.py:90` | **完全不可用** | `run_sweep` 直接调 `SCHED.run_sweep`；`cmd_sweep` 还会 import `GenrecAdapter`/`OpenCodeProposer`/`ScriptedProposer`，后者继承 `core.proposer.Proposer` |
| `cmd_tasks` | `cli.py:51` | 低 | 只用本仓库 `tasks`，但模块 import 期就要求 `harness2.core` |
| `GenrecAdapter` / `surface` / `substrate` / `prompts_ext` | 各自模块 | **不可 import** | 它们是框架契约的实现（`SurfaceSpec`、`BenchmarkAdapter` 协议、`Install/RunRequest`），不能只删 import |
| `OpenCodeProposer` 的基类 | `proposer.py:144,183` | 中 | `GenrecProposer`/`OpenCodeProposer`/`ScriptedProposer` 都继承 `Proposer`；`Reply` 是框架的返回类型（`proposer.py:245`） |
| `substrate.run_task` 的 verifier 沙箱 | `substrate.py:416` | 中 | 该函数既实现框架的 `Substrate` 协议，又拉起 OpenCode 会话 |
| **离线 P0 演示链**（`cli sweep --offline`） | `cli.py:97` | 中 | 用 `ScriptedProposer` 不碰模型，但仍在框架的调度器里跑 |

## 3. 只依赖自研 OpenCode CLI 的功能（移除框架后**不受影响**）

| 功能 | 位置 |
|---|---|
| `campaign` 的 ideation / 实现 / 修复 / 评估 | `campaign.py:208,316,360,897` |
| `lifecycle` 的实现 / 修复 | `lifecycle.py:233,396` |
| `hypotheses` 的假设生成 | `hypotheses.py:253` |
| `learning` 的修复臂 / 推导策略 / 策略比较 | `learning.py:248,283,328,561,641` |
| `redesign` 的提案 / 实现 / 修复 / 从账本提案 | `redesign.py:214,329,406,557` |
| `candidates.propose_candidates` | `candidates.py:435` |
| `policy_comparison` | `policy_comparison.py:123,203` |
| `repair_demo` | `repair_demo.py:176` |
| `mcp_check`（证明 MCP 可达） | `mcp_check.py:44` |
| `opencode.preflight`（真实一跳探活） | `opencode.py:513` |
| Stage2 启动（CGV 复跑 → 冒烟 → detached worker → torchrun） | `tools/genrec-mcp/server.py:1189` |
| Stage3 启动 | `tools/genrec-mcp/server.py` 的 `tool_stage3_start` |
| 冻结 ER/D2 评测 | `tools/harness2_genrec/analysis.py`；`tools/mvg/er_eval.py`（lifecycle 用） |
| CGV 六门独立验证 | `tools/cgv/run_cgv.py` |
| 全部 GPU 执行 | Stage2/Stage3 训练器 |

## 4. 两者共享的点（移除任一方都会波及）

| 共享点 | 说明 |
|---|---|
| **OpenCode CLI 二进制** | 框架路径（`proposer.py:229`、`substrate.py:416`）与自研路径（`campaign`/`lifecycle`/…）调用的是同一个 CLI、同一种 argv 形态 |
| **genrec MCP server** | 自研路径把它当 in-process 模块用（`lifecycle.py:739`）；框架路径把它写进 OpenCode 的 `mcp.servers`（`opencode.py:234`）⇒ 两种消费方式共用一份实现 |
| **`opencode.json` 工作区契约** | `build_config`（`opencode.py:249`）定义的 permission/MCP/provider 结构，是 agent 隔离与凭证处理的关键，且与框架无关 ⇒ 应视为自研资产 |
| **模型与凭证配置** | `provider_configuration` / `HARNESS2_OPENCODE_*` 环境变量（`opencode.py:40–52`）与框架的 provider 抽象是两套；共用同一个 CLI 的 `--model` |
| **Stage2/Stage3 的准入证据** | `admission.json`（`lifecycle.py:502`）与 CGV 七组声明是**同一套**门槛的两种形态：前者由框架侧控制器签发，后者由 `stage2_start` 直接校验（`server.py:1189`） |

---

## 5. 若「完全移除原版 Harness² 运行依赖、仅保留自研 DAG + OpenCode CLI」，会被触及的接口

**A. 立即失效（需要替换或下线）**

| 接口 | 原因 |
|---|---|
| `tools.harness2_genrec.cli sweep` | 唯一进入 `SCHED.run_sweep` 的入口 |
| `tools.harness2_genrec.cli tasks` | 模块 import 期依赖 `harness2.core` |
| `harness2.core.proposer.Proposer` / `Reply` | `proposer.py` 的基类与返回类型 |
| `harness2.core.spec.*` / `core.tree.SurfaceSpec` / `core.prompts` | `adapter.py`/`surface.py`/`prompts_ext.py`/`substrate.py` 的类型来源 |
| `harness2.substrate.base.Install/RunRequest` | `substrate.py` 的契约 |
| `harness2.core.library.root_for` | `cli.py:42` 的 library 根目录计算 |
| `harness2.core.schedule.MODES` / `mode_token` | `cli.py:37–39` 的模式解析 |

**B. 不受影响，但需要重新接线（调用方在框架侧）**

| 接口 | 谁在用 | 备注 |
|---|---|---|
| `opencode.run_agent` | 24 个调用点 | 只有 `proposer.py:229` 与 `substrate.py:416` 在框架侧；其余 22 个调用点本来就在自研侧 |
| `opencode.mcp_server_block` | `proposer.py:228` | 改为自研侧直接调用即可 |
| `opencode.build_config` / `parse_events` / `redact` / `preflight` | 自研 | 不依赖框架 |
| `genrec-mcp` 的 `tool_stage2_start` / `tool_stage2_status` / `tool_stage3_*` | 自研 + 框架的 MCP 通道 | 一份实现、两种消费方式 |
| `tools/cgv/*` | 自研 | 完全独立 |
| `dag.py` / `controller.py` | 自研 | 与框架零关系；两者本身还需先解决「重复的调度面」问题（见 §6） |

**C. 需要先决策的既有事实（不是本次迁移的一部分，但会叠加）**

1. **两个调度面并存**：`dag.py`（记账/租约，无执行体、无生产调用者）与 `controller.py`
   （状态机 + `run_candidate` 真启动训练）功能重叠；若只保留其一，需要先定哪一套是权威。
2. **两套台账并存**：`dag/state.json`（29 任务）与 `campaign_dag.json`（32 任务，**无任何代码读取**）。
3. **`serve(run)` 缺实现**：DAG 的执行回调从未被写出来，于是 #112 的执行体是交互式协调者。
4. **文档与代码不一致**：README 的 venv 路径与「stdlib-only」声明均已被本次审计证伪。

---

## 6. 影响面小结（不改代码）

| 维度 | 结论 |
|---|---|
| **训练/评测主链** | **不受影响**。Stage2/Stage3 由 `tools/genrec-mcp/server.py` 启动，CGV 由 `tools/cgv/run_cgv.py` 独立验证，训练器不 import 任何 harness 代码 ⇒ 移除框架**不会**触碰任何科学产出路径 |
| **研究自动化** | **大部分不受影响**：22/24 个 OpenCode 调用点在自研侧；受影响的是 `proposer.py` 与 `substrate.py` 这两个框架接缝，以及 `cli sweep` 入口 |
| **框架独有能力** | `sweep` 的并行调度、`library`（种子库）、`surface`（可迭代 harness 面）、`substrate`（隔离 verifier 沙箱）、框架自带的 prompt 注册与任务池抽象 —— 这些在自研侧**没有等价物**（自研侧只有 `dag.py` 的租约记账），下线它们等于放弃「把 harness 当作可迭代 surface」这条路 |
| **运行时约束** | 移除框架后，仓库**不再需要** py3.12 venv，`tools/harness2_genrec/` 全部模块可在项目解释器下 import；这也是把「框架依赖」与「生产依赖」解耦的最大收益 |
| **风险提示** | `substrate.py` 与 `proposer.py` 是框架与自研世界之间的**双向桥**：直接删 import 会同时切断「框架调度器 → Proposer → OpenCode」与「框架 Substrate → OpenCode」两条真实运行过的路径 |

---

## 7. 本文件**没有**回答的问题（按本 Issue 边界刻意不答）

- 不给迁移方案、不给替代架构、不给改动清单。
- 不判断「是否应该移除」——那是用户决定。
- 不评估把 `dag.serve` 的回调写出来会引入什么新行为。
- 不触碰 #110/#111/#112 的任何状态或证据。

**证据等级**：本文件全部结论来自 `code_verified`（路径 + 行号）与 `runtime_observed`
（会话记录、状态文件、事件日志）；文档类主张已单独标注并在 `evidence_index.json` 中
带 `documentation_only` 标记。
