# 调用链审计：原版 Harness² vs 自研 Harness2-GenRec vs OpenCode CLI

**Issue #113 · 只读审计 · 观测范围冻结于 `main` = `f6d78b68`，2026-10-23T14:04:31Z**

本文件只陈述能在仓库里指出路径与行号、或在现存运行工件里指出记录的事实。每一节末尾的
`证据:` 行给出可核查的位置；四级证据标记的含义与 `evidence_index.json` 一致：

| 标记 | 含义 |
|---|---|
| `runtime_observed` | 有现存运行工件（会话 JSON、事件日志、进程状态文件、指标流） |
| `code_verified` | 有源码路径 + 行号，可逐跳跟踪；未在本次审计中执行 |
| `documentation_only` | 只出现在文档/README/注释里，代码未证实 |
| `unknown` | 无证据，本审计不下结论 |

---

## Executive Summary（一页）

1. **原版 Google Research Harness² 在本仓库里只被 6 个模块导入**：`surface.py`、`adapter.py`、
   `prompts_ext.py`、`substrate.py`、`proposer.py`、`cli.py`。其中**只有 `cli.run_sweep`
   （cli.py:60–87）真正调用框架的调度器** `harness2.core.schedule.run_sweep`（cli.py:74）。
   其余 5 个模块只是借用框架的**类型/接口**（`SurfaceSpec`、`BenchmarkAdapter` 契约、`Proposer`
   基类、prompt 注册表）。
2. **这些模块在项目解释器里根本无法 import**：`genrec_env_v2`(py3.9) 与 `genrec_env`(py3.10) 都
   报 `ModuleNotFoundError: No module named 'harness2'`；原版框架只装在
   `/home/wlia0047/ar57_scratch/wenyu/harness2/.venv`(py3.12)。所以「框架路径」与「生产路径」
   **不在同一个进程空间**，只能通过 JSON/子进程/MCP 交接。
3. **子 Agent 的真实底层只有一条**：`opencode.run_agent`（`opencode.py:353`）→
   `subprocess.run([opencode, "run", "--standalone", "--format", "json", "--auto", "--model", …,
   prompt])`（`opencode.py:406–431`），工作目录是隔离的 workspace，配置文件是 workspace 内的
   `opencode.json`（`build_config`，`opencode.py:249`），事件流按行解析成 tool_calls / cost /
   tokens（`parse_events`，`opencode.py:293`）。
4. **`campaign` / `lifecycle` 完全绕过原版框架**：它们不导入 `harness2`，直接
   `opencode.run_agent`（`campaign.py:208,316,360,897`；`lifecycle.py:233,396`）。原版框架的
   `Proposer` 抽象在真实研究里**只被 `cli sweep` 使用一次构造**（`cli.py:104`）。
5. **`dag.py` 是一个纯状态机 + 租约调度器**：不 import `harness2`、不 import `subprocess`、
   **不创建任何 Agent 或进程**（`serve(run)` 收调用方传入的 `run(task)` 回调，
   `dag.py:549`）。**仓库内没有任何生产代码构造 `DAG(` 或调用 `.serve(`**——只有
   `tests/test_dag.py`。`#112` 的 DAG 事件日志里 19 条 `dispatched` 的 `owner` **全是
   `coordinator`**，即人工/交互式驱动，不是服务。
6. **#110–#112 的 Agent L / A / M / I 没有任何 OpenCode 会话证据**。全仓库有 **230 份**
   `*session*.json`（全部 `openai/gpt-6-luna`，228 份 exit 0，累计 6.64M input tokens、
   8,897 s），时间跨度 2026-10-21T08:32Z → 2026-10-23T00:21Z，标题/工作目录中
   **没有一处**出现 `product_l1` / `busemann` / `transported_residual` / `role_assigned`。
   因此：**L/A/M/I 是角色化的工件署名，不是可核实的独立 OpenCode 会话** → `unknown`。
7. **真正跑过 GPU 的那一步有完整进程级证据**：`~/.cache/genrec-mcp/stage2__TIGER_hyp_product_l1_quantiser.json`
   记录 pid 3755994、`cmd=python3.9 train_rqvae.py`、`smoke.passed=true`（24 步 DDP）、
   4 张卡；`/tmp/genrec-mcp/smoke_*.log` 有 `SMOKE_PASS`；训练指标流 3003 条、739 s 跑完
   3000 epoch。**Stage3 从未运行**。
8. **结论（是否使用原版 Harness²）**：作为**研究调度的运行时依赖——没有**；作为**6 个模块的
   类型/契约/调度入口——有**，且这条路径在 py3.12 的框架 venv 里才能运行，与生产（py3.9）
   物理隔离。

---

## 1. 六个实体的准确身份

| # | 实体 | 位置 | 本次审计掌握的事实 | 证据级别 |
|---|---|---|---|---|
| 1 | 原版 Google Research **Harness²** | `/home/wlia0047/ar57_scratch/wenyu/harness2`（= `/fs04/scratch2/...`），包 `harness2==0.1.0`，venv CPython 3.12 | 只装在这一处；项目两个 env 都 import 不到；本仓库只用它的 7 个 seam | `runtime_observed` |
| 2 | 自研 `tools/harness2_genrec/` | 仓库内 | 64 个模块；6 个 import 框架，5 个核心执行模块不 import | `code_verified` |
| 3 | 自研 DAG | `tools/harness2_genrec/dag.py` | 纯状态机；零进程；零 Agent；无生产调用者 | `code_verified` |
| 4 | 自研 OpenCode 封装 | `tools/harness2_genrec/opencode.py` | 唯一的 CLI 调用点 `run_agent` | `code_verified` |
| 5 | **OpenCode CLI** 二进制 | 由 `HARNESS2_OPENCODE_BIN` 或 PATH 解析（`opencode.py:93`） | argv 形态由 230 份会话记录反证 | `runtime_observed` |
| 6 | GenRec MCP | `tools/genrec-mcp/server.py`（local stdio） | Stage2/Stage3 的**唯一**受控启动口 + 工具访问；不编排 Agent | `code_verified` |

**不要把它们一概叫 "harness2"**：`dag.py`、`controller.py`、`campaign.py`、`lifecycle.py`、
`opencode.py` 与实体 1 没有任何 import 关系（`grep -c 'harness2\.core\|from harness2'` 全为 0）。

---

## 2. 原版 Harness² 到底被谁用、用来做什么

### 2.1 六个 import 点与它们的作用

| 模块 | import | 作用 | 是否真的把控制权交给框架 |
|---|---|---|---|
| `cli.py:19–21` | `core.library`、`core.schedule`、`core.spec.ChannelSpec` | `run_sweep` 包一层 `SCHED.run_sweep`（:74）；`cmd_sweep` 构造 `OpenCodeProposer`（:104）→ 调用（:119） | **是**（唯一一处） |
| `proposer.py:21` | `core.proposer.Proposer, Reply` | `GenrecProposer`/`OpenCodeProposer`/`ScriptedProposer` 的基类与返回类型 | 否（基类契约） |
| `adapter.py:24–25` | `core.tree`、`core.spec.BenchPrompts/BenchSpec/ChannelSpec` | 把「一个 Stage-2 机制臂」表达成 benchmark task pool | 否（数据契约） |
| `surface.py:16` | `core.tree.SurfaceSpec` | 把 `CLAUDE.md` + `.agentskills/**/SKILL.md` + `scripts/*.py` 声明为可迭代的 harness surface | 否（声明） |
| `substrate.py:28–29` | `core.tree.SurfaceSpec`、`substrate.base.Install/RunRequest` | 在隔离工作区里跑候选 verifier | 否（接口实现） |
| `prompts_ext.py:16` | `core.prompts` | 注册本仓库的 prompt 文本 | 否（文本注册） |

### 2.2 只有一条控制流把控制权交给框架

```
cli.main  (cli.py:723)
  └─ cmd_sweep (cli.py:90)
       ├─ OpenCodeProposer(...)            (cli.py:104)      ← 全仓库唯一构造
       ├─ GenrecAdapter(...)               (cli.py:112)
       └─ run_sweep(...)                   (cli.py:119) → cli.py:60 → SCHED.run_sweep (cli.py:74)
            └─ harness2.core.schedule.run_sweep(...)        ← 框架调度器（第三方）
                 └─ prop.call(mission, context_dir, label)  ← Proposer 协议
                      └─ OpenCodeProposer.call              (proposer.py:216)
                           └─ opencode.mcp_server_block     (proposer.py:228 → opencode.py:234)
                           └─ opencode.run_agent            (proposer.py:229)
                                └─ subprocess.run(opencode run …)   (opencode.py:425)
```

**除此之外没有任何模块调用 `run_sweep`**（`grep -rn run_sweep` 只命中 `cli.py` 的 3 处与
`tests/test_p0_pipeline.py`）。

### 2.3 三个必须分开的「campaign / lifecycle / DAG / controller」

它们**都不是**同一条链，也**都不经过**原版框架：

| 入口 | 谁调用 | 它自己做什么 | 是否 import 框架 |
|---|---|---|---|
| `cli campaign` (cli.py:259) | 人 | `campaign.execute(config)` → `Campaign.ideate`(:184) / `run_arm`(:272) / `evaluate_harness`(:845) → **直接** `opencode.run_agent`(:208/:316/:360/:897) | **否** |
| `cli lifecycle` (cli.py:194) | 人 | `MechanismRun` 顺序推进 `freeze_proposal → implement(:224) → verify(:270) → repair(:387) → probe(:440) → admit(:502) → stage2(:546) → evaluate(:628)` | **否** |
| `#112 DAG` (dag.py) | **无生产调用者**；本次由交互式 coordinator 用临时脚本驱动 | `tick` 派发租约、`complete/fail` 记账；`serve(run)` 才执行回调 | 否 |
| `cli controller` (cli.py:462) | 人 | `Controller.load()`(:478) → phase 状态机 → `run_candidate`(candidate_run) 启动训练 | 否 |

**四个入口并存、互不同步**。这是本次审计最重要的结构事实：讨论「唯一的调度器」在代码层面
不成立。

`证据:` `evidence_index.json` E02–E07、E11、E22、E23；`call_graph.md` 的四条链。

---

## 3. 自研 OpenCode CLI 封装：真实调用方式

### 3.1 `run_agent` 的完整事实（`opencode.py:353–458`）

| 项 | 事实 | 行号 |
|---|---|---|
| 二进制 | `binary_path()` 读 `HARNESS2_OPENCODE_BIN`，否则 PATH | 93–99 |
| 模型 | `model_name()` 读 `HARNESS2_OPENCODE_MODEL`，否则 `DEFAULT_MODEL="openai/gpt-6-luna"` | 40、136 |
| 工作区 | `Path(workspace).resolve()`，`mkdir(parents=True, exist_ok=True)`；**agent 唯一可写目录** | 375–377 |
| 配置 | 写 `workspace/opencode.json`：`model`、`permission:[{action:"*",resource:workspace,effect:"allow"}]`、可选 `deny_paths`（`effect:"deny"`）、`provider`、`mcp.servers` | 249–282 |
| MCP | `mcp_server_block(repo_root)` → `{"genrec":{"type":"local","command":[python, tools/genrec-mcp/server.py],"cwd":repo_root,"disabled":False,"timeout":{"startup":120000}}}` | 234–247 |
| argv | `[bin, "run", "--standalone", "--format", "json", "--auto", "--model", MODEL] (+["--title",T]) (+["--agent",A]) + [PROMPT]` | 406–420 |
| 子进程 | `subprocess.run(argv, cwd=workspace, env=child_env, capture_output=True, text=True, timeout=timeout)` | 425–431 |
| 环境 | 只传 `PATH/HOME/LANG/TMPDIR/XDG_DATA_HOME/XDG_CACHE_HOME/XDG_STATE_HOME` + provider env + 调用方 env | 380–400 |
| 超时 | 默认 900 s；`TimeoutExpired` → `returncode=-1, timed_out=True`，stderr 追加标记 | 358、439–443 |
| 事件 | `parse_events` 逐行 JSON：`tool_use`→tool_calls（tool/id/status/input/output≤4000/title/start/end）、`text`、`error`、`step_finish`→cost+tokens；**非 JSON 行保留为 `unparsed`** | 293–351 |
| 凭证 | `provider_configuration(model)` 读 `HARNESS2_OPENCODE_CONFIG` / `HARNESS2_OPENCODE_PROVIDER` / `HARNESS2_OPENCODE_KEY`；写盘前 `redact()` 把所有 secret 替换为 `<redacted>` | 174–232、284–291 |
| 返回 | `{events, n_events, tool_calls, n_tool_calls, tools, text, cost, tokens, errors, argv, model, workspace, exit_code, timed_out, elapsed_s, stdout, stderr, config_path}` | 445–458 |

### 3.2 运行证据（不是文档）

全仓库共 **230** 份 `*session*.json`，全部是上面这个结构：

| 指标 | 值 |
|---|---|
| 会话数 | 230 |
| 模型 | 230/230 `openai/gpt-6-luna` |
| 退出码 | 228 × 0，2 × 1 |
| 标题前缀 | implement 153、candidate 65、mechanism 5、repair 3、derive 3、hypotheses 1 |
| 累计耗时 | 8,897.5 s |
| 累计 token | input 6,643,365 / output 221,084 |
| 时间跨度 | 2026-10-21T08:32:09Z → 2026-10-23T00:21:54Z |
| 代表性记录 | `results/harness3_runs/candidates/reference_codebook_geometry_preservation/repair_session.json`（22.25 s，6 次工具调用：shell, read, read, shell, patch, shell） |
| | `results/harness3_runs/candidates/candidates_session.json`（29.05 s，0 次工具调用，2 个事件——纯文本回复） |

`证据:` `evidence_index.json` E08 与 `agent_invocation_inventory.json`（含全部 230 条记录的可核字段）。

---

## 4. 两条路径在哪里连接、哪里独立、是否同一轮同时参与

| 关系 | 判定 | 证据 |
|---|---|---|
| 框架路径 ↔ OpenCode | **连接**，唯一交点 = `Proposer` 协议（`OpenCodeProposer.call`） | `proposer.py:216` / `cli.py:104` |
| `campaign`/`lifecycle` ↔ 框架 | **完全独立**（零 import） | E02 |
| `campaign`/`lifecycle` ↔ OpenCode | **连接**（直接 `run_agent`） | `campaign.py:208,316,360,897`；`lifecycle.py:233,396` |
| DAG ↔ 任何执行体 | **无连接**：`serve(run)` 需要调用方传入；无生产调用者 | `dag.py:549`、E11 |
| DAG ↔ OpenCode | **无直接关系**（dag 不 import opencode） | `grep` 为空 |
| Controller ↔ DAG | **无关系**（两套独立状态机） | E23 |
| 训练器 ↔ 以上全部 | **无关**（trainer 不 import 任何 harness 代码） | E19 |
| 同一轮是否同时参与 | **否**。`#112` 的一轮里：DAG 只记账（19 条 dispatched，全 coordinator）；真正执行的是交互式协调者（写代码、跑 CGV、调 MCP 启训练）；OpenCode 会话参与数为 **0** | E09、E10 |

### 4.1 Agent L / A / M / I 的真实状态

| 角色 | 声明的产物 | 调用位置 | OpenCode 会话 | 结论 |
|---|---|---|---|---|
| L（文献） | `subspace_literature_matrix.json`、`source_search_log.jsonl` | 无（无函数调用点） | 0 | **NOT VERIFIED** |
| A（失败分析） | `subspace_failure_analysis.md` | 无 | 0 | **NOT VERIFIED** |
| M（提案） | `subspace_mechanism_proposals.json`、`proposals/*.json` | 无 | 0 | **NOT VERIFIED** |
| I（实现契约） | `proposal_*_v2.json` 的七组契约 | `cli lifecycle` 的 `implement()` 可产生同类产物，但 #112 未走该入口 | 0 | **NOT VERIFIED** |

它们对应的文件确实存在于 `results/hyperbolic_research_runs/subspace_111/` 与
`issue112/`，并随提交 `1484671d` 落库；但**没有任何 argv、workspace、session id、cost 或
token 记录**能证明是某个独立 Agent 会话产生的。按本 Issue 的验收规则，只能记
`unknown`，不得假设「四个 Agent 确实并行运行」。

`证据:` E09、E25。

---

## 5. 一次机制探索的真实链路（以 #112 为例）

```
提出机制            ← 无 Agent 会话证据（工件：subspace_mechanism_proposals.json）
  ↓
数学审核 / H1–H7    ← tools/harness2_genrec/hypothesis.py + oracles（code_verified）
  ↓
实现                ← 交互式协调者改 sealed tree（无 OpenCode 会话；E09）
  ↓
CGV G1–G6           ← tools/cgv/run_cgv.py（子进程，genrec_env_v2 py3.9，CPU）
  ↓
JOIN R / JOIN E     ← DAG 记账（19 条 dispatched，owner=coordinator）+ 协调者判定
  ↓
原子预约            ← dag.reserve() → reservations.json（1 条 key）
  ↓
Cold Stage2         ← 协调者 → omp MCP client → genrec MCP server(stdio)
                       → tool_stage2_start → CGV 复跑 → detached worker(/tmp/genrec-mcp/stage2_worker.py)
                       → 1-epoch 24-step DDP smoke → torchrun 4 ranks → 3000 epoch
  ↓
冻结评测 ER/D2      ← tools/harness2_genrec/analysis.py（协调者运行）
  ↓
ER 门               ← ER 1.0 < 16.38953942232631 → FAILED
  ↓
Stage3              ← **NOT RUN**（协议禁止；DAG 拒绝用失败测量 complete 门）
  ↓
下一机制            ← role-assigned layout：已实现 + 24 步冒烟通过；未正式训练
```

**真实触发者**（逐步骤）见 `mechanism_lifecycle_trace.md` 的表；核心区分是：
`process_observed`（有 pid/日志/worker 状态）、`artifact_observed`（有产物但无调用证据）、
`manual_mark`（协调者事后 `complete()`）。

---

## 6. 文档与代码不一致之处（本次审计发现的 4 处）

| 文档声明 | 代码事实 | 影响 |
|---|---|---|
| README:17 venv 位置 `<repo>/.venv` | 仓库无 `.venv`；实际在 `/home/wlia0047/ar57_scratch/wenyu/harness2/.venv` | 按文档无法复现 |
| README:24–25「`tools/harness2_genrec/` 全部 stdlib-only」 | 6 个模块 import 第三方 `harness2`，在项目 py3.9 下直接 `ModuleNotFoundError` | 会误判可移植性 |
| README:206「trainer 的 env 从不 import framework」 | **成立**（E19） | — |
| `stage2_start` 返回的 `result_dir` 含 `/hyp_factor_nce_c1` 子目录 | 训练器实际写到 `results/stage2_RQ-VAE/TIGER_hyp_product_l1_quantiser/`（无子目录） | 任何信任该字段的工具会找不到 SID |

`证据:` E20、E21、E24。

---

## 7. 本次审计的边界与证据缺口

- **未执行**：任何 GPU 作业、Stage2/Stage3、probe、CGV 训练入口、Agent 会话、测试用例、
  排队或控制器恢复。全部动作是 `ls/find/grep/git/python 读 JSON`。
- **未修改**：任何生产代码、Prompt、Skill、DAG、CGV/H、数据集、checkpoint、旧 Issue 状态。
  唯一新增 = 本目录 6 个文件。
- **缺口**：
  1. L/A/M/I 的会话证据缺失（`unknown`，E25）；
  2. 第三方框架内部实现未读（只读本仓库 import 的 7 个 seam）；
  3. 交互式协调者的 shell 历史不可得，只能由工件与 DAG 事件反推；
  4. `/tmp/genrec-mcp/` 与 `~/.cache/genrec-mcp/` 属临时目录，可能被清理——本审计已把关键
     字段抄入 `evidence_index.json`，避免结论随临时文件消失。

**采集建议（仅建议，本次未实施）**：把 `run_agent` 的返回结构在每次调用后原样落盘到
`results/<run>/session.json`（现有 230 份已经是这个形态，只是 #110 之后未再使用）；把
`dag.serve` 的回调实现登记为唯一的执行体，并把 `campaign_dag.json` 这类手工台账改为由
`dag.state()` 导出，避免两套账。
