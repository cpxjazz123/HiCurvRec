# Harness² × GeneRec

Google Research 的 **Harness²**（<https://github.com/google-research/harness2>）在本仓库的落地状态。
本文件只记录**已验证的事实**与**已定的计划**；运行记录（runs / library）一律落在仓库外的 scratch，不入库。

## 0. 定位

Harness² 迭代的不是模型，而是**给 agent 的指令集（harness）**。对本项目：

| Harness² 概念 | 本项目对应物 |
|---|---|
| harness（c1..c8 可编辑树） | `CLAUDE.md` + `.agent/skills/` + `scripts/` —— **我们要被迭代的迭代协议本身** |
| task | 一次 Stage-2 机制臂迭代（复制臂树 → 改机制 → CGV → Stage2 → Stage3） |
| domain | 机制族（hyperbolic / sampling / graph / curriculum …） |
| executor（substrate） | Claude Code CLI + 本仓库的 genrec MCP（stdio） |
| probe | 真跑一遍上面的链路，看行为差异 |
| judge | `test_final.json` 的 `test_recall@10`（CLAUDE.md §2 的硬目标 > 0.065） |
| improvement harness | per-domain `EXPERIENCE.md` 课库（cue / lesson / evidence）—— **目前完全缺失**，教训只以散文躺在 CLAUDE.md 里 |

模型本身仍是本仓库的 RQ-VAE；Harness² 提供的是**证据树 + 盲选 + 反思 + 课库 + 可校验的 harness 面**。

## 1. 安装状态（已验证）

| 组件 | 位置 | 版本 / 校验 |
|---|---|---|
| 源码 | `/home/wlia0047/ar57_scratch/wenyu/harness2`（= `/fs04/scratch2/ar57/wenyu/harness2`） | HEAD `1cfe254e8d25b15f37048c3730134686abe60335`（2026-10-13 "Fix arXiv link"），Apache-2.0 |
| 虚拟环境 | `<repo>/.venv` | CPython 3.12.15（框架要求 `requires-python >=3.12,<3.14`） |
| 包 | editable `harness2==0.1.0` + `[render]`（openpyxl / python-pptx） | `uv pip install -e '.[render]'` |
| uv | `/home/wlia0047/.local/bin/uv` | 0.13.3 |
| 入口 | `harness2` `-judge` `-report` `-setup` `-check` `-demo` `-bridge` | 7 个 console script |

**验证**：`harness2-demo --output runs/demo --k 2` → exit 0，`2 tasks completed; 0 failed`
（`execution_count=28`、`editor_calls=32`，平行 + 串行两种递归模式都跑通，不需要任何模型凭据）；
`pytest tests/ -q` → `5 passed, 2 subtests passed`。

**注意**：仓库的 `genrec_env`(3.10) / `genrec_env_v2`(3.9) **装不了框架**。因此本目录下的集成代码必须
纯 stdlib，重活一律经 genrec MCP（stdio）外包给仓库自己的解释器。

## 2. 框架接缝（已读源码；行号相对框架仓库根 `harness2/`）

| 概念 | 接口 | 位置 |
|---|---|---|
| 可编辑面（c1..c8） | `SurfaceSpec` | `core/tree.py:41` |
| 面校验器 | `tree.validate()` | `core/tree.py:336` |
| 基准静态描述 | `BenchSpec` / `BenchPrompts` | `core/spec.py:133` / `core/spec.py:90` |
| **适配器接缝** | `BenchmarkAdapter`（12 个方法） | `core/spec.py:201` |
| 执行器接缝 | `Substrate`（5 个方法） | `substrate/base.py:48` |
| LLM 编辑通道 | `Proposer`（`claude_cli` / `opencode`） | `core/proposer.py:91` |
| 证据树 | `context.build()` | `core/context.py:105` |
| 盲选 / 反思 | `selection.run_selection()` / `reflect_adoption()` | `core/selection.py:268` / `core/selection.py:402` |
| 课库 | `Library` | `core/library.py:58` |
| 调度 | `schedule.run_task()` / `run_sweep()` | `core/schedule.py:566` / `core/schedule.py:780` |
| 成本模型 | `R = 4k` rollouts，`P = 3k` 会话 / 任务 | `core/ledger.py:26`、`core/ledger.py:31` |
| 离线参考实现 | `DemoAdapter` / `DemoProposer` | `demo.py:41` / `demo.py:120` |
| 基准注册表 | `BENCHES = ("jb","wb","lab")` | `cli/_common.py:30` |

## 3. 计划

| 步 | 内容 | 验证物 |
|---|---|---|
| P0 | 写 `surface.py` / `adapter.py` / `proposer.py` / `cli.py`；用假 executor + 脚本化 proposer 跑通离线管道 | `summary.json`、`selection.json`、证据树、`EXPERIENCE.md`、library `INDEX.md` 全落盘；**零 GPU、零模型** |
| P1 | 接真 executor：仓库根 `.mcp.json` + `substrate_claude.py`（Claude Code CLI 跑在候选 harness 的 workspace，MCP `cwd` 仍为仓库） | 单任务 k=1 的 trajectory 里能看到 MCP 调用与 CGV 判定 |
| P2 | 接真 probe：候选用 `stage2_stability_probe`（有界步数）+ SID 质量指标；**只有终选**跑完整 Stage2 + Stage3 | 被选候选对应真 `test_final.json` 的 `test_recall@10`，`EXPERIENCE.md` 的 lesson 引用真实观测数字 |
| P3 | 被采纳的 harness 改动回写 `CLAUDE.md` / `.agent/skills/`，按 CLAUDE.md §13 走一次 `git push origin main` | main 上一次干净迭代 |

## 4. 已识别的阻塞与风险

1. **模型接入**：`Proposer._DROP_ENV`（`core/proposer.py:46`）丢掉 `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`，
   且 `env()`（`core/proposer.py:105`）强设 `CLAUDE_CODE_USE_VERTEX=1` → Vertex 我们没有。
   解法：子类覆盖 `env()`，指向 `~/.claude/settings.json` 里已配好的 MiniMax Anthropic 兼容端点；
   `claude` CLI 2.1.292 已装（`/home/wlia0047/.local/bin/claude`）。
2. **Probe 成本**：实测 Stage2 3000 步 1227–1537 s、Stage3 150 epoch + test ≈ 3096 s。
   `R = 4k` ⇒ k=1 单任务 ≈ 4 × 78 min ≈ **5.2 GPU 小时**。解法：两级探针（见 P2）。
3. **仓库纪律**：§13 每轮 commit+push、§8 只维护 main、§9 禁 worktree、§11 冻结 Stage3 协议。
   ⇒ sweep 产物不入库；只有被采纳的 harness 改动作为一次正式迭代提交；
   harness 改动**不得**触碰冻结超参与 §17 的 CGV 语义。
4. **路径漂移**：CLAUDE.md §10 要求的 `results/stage2_RQ-VAE/curvature_RQ-VAE/` **当前不存在**，
   实测产物在 `results/stage2_RQ-VAE/<arm>/` 与 `results/stage3_T5Train/<arm>/logs/**/test_final.json`。
   ⇒ 适配器按实际路径解析，**不擅自搬目录**。
5. **CGV spec 没有文件形态**：`tool_stage2_start(source_dir, spec)` 的 spec 是调用参数
   （`tools/genrec-mcp/server.py:1188`），树上没有机器可读声明 ⇒ 任务池里每个机制简报必须自带
   spec 骨架，否则 `stage2_start` 直接拒。
6. **可编辑面的路径是固定的**：c7 脚本正则 `scripts/<name>.py`、技能正则 `<skill_dir>/<name>/SKILL.md`
   ⇒ 本仓库 `.agent/skills/<name>/SKILL.md` 与 `scripts/*.py` 天然命中，`tools/*.py` **不在**可编辑面内。

## 5. 本目录不做什么

- 不改框架 core（`recursion` / `selection` / `library` / `prompts` / `substrate` 一律不动）。
- 不改 Stage2 / Stage3 训练超参（§11 冻结协议）。
- 不改 CLAUDE.md 现有条款（被采纳前只读）。
- 不用 git worktree（§9）。
- runs / library 不写进仓库（§12 的 zero-bookkeeping 精神）。
