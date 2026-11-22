---
name: complete-open-issue
description: Use when the task is an open GitHub issue that must be finished end to end. Reads the issue, freezes a machine-checkable acceptance checklist, prepares data and infrastructure, implements with TDD, runs a check loop that re-reads the issue every round, trains when the issue needs it, judges the result against the issue's own criteria, and closes the issue only when every criterion passes.
---

# 按 open issue 完成任务的闭环

一个 issue 只有在**它的验收条件逐条被证据满足**时才算完成。本 skill 把这个过程固定为
`读 issue → infra → 实现 → 检查循环 →(训练,仅当 issue 要求)→ 结论对账 → 关闭`;任何一步不满足,回到它
应该在的那一步,不允许跳步,也不允许"差不多"。

**训练不是固定步骤**:跑不跑由 issue 的验收条件决定(见 §4),不由"skill 有第四步"决定。

**issue 关闭之前不允许暂停。** 一旦开始,就必须在同一个 turn 内推进到 `gh issue view <N>` 返回 `CLOSED`;
不许中途停下、不许把"下一步计划"当交付、不许等确认。这不是"尽量做完",而是本 skill 的完成定义。

| 会被误当成"该停了"的瞬间 | 必须改成 |
|---|---|
| 一轮检查有 FAIL | 立刻回对应步骤修,再开 `round + 1` |
| 某条 `NOT_RUN`,看起来工作量大 | 那就是当前要做的活,做完再判 |
| 训练在跑 | 轮询(≤60 秒/次)直到出结果,然后进 §5 |
| 训练失败/指标低 | 按 §4 分类回对应步骤,或按 §5 出负结论并写进 `VERDICT.md` |
| 遇到需要授权的例外(如 `git push --force`) | 该例外单独处理,但 issue 的其余部分继续推进 |
| 想先汇报进度再继续 | 只在 `out://` 里贴进度,然后**继续同一个 turn 干活** |

唯一合法的终止点是 §6 关闭成功。确实被外部条件卡死(缺数据且无法补齐、需要 owner 改 issue 判据)时,
不得静默停止:在 issue 下评论写明阻塞点与本轮证据,保持 OPEN,并把阻塞原因写进 `VERDICT.md`——
这是"报告阻塞",不是"暂停"。

## 0. 读 issue 并冻结验收清单

```
read issue://<N>                 # 或 gh issue view <N> --json number,title,body,state
```

把 issue 正文里的验收条件**逐条原文抄录**成清单 `C1..Cn`,每条补齐四件事:

| 字段 | 要求 |
|---|---|
| 判据 | 指标名/公式/文件位置,能机械复算 |
| 阈值 | `>`、`<`、`==` 与数值(例如 Stage3 `test_R@10 > 0.065`) |
| 数据 | 用哪份训练/验证/测试输入,路径 + sha256 |
| 口径 | 谁算、在哪算、输出写哪里 |

同时把 `issue #N`、标题、正文的 sha256 记进 `results/research/issue<N>/RUN_LOG.md`,并写下**本轮是否需要训练及依据**
(判定规则见 §4:验收指标能否由已有产物复算)。之后每轮检查都以这份清单为准;issue 正文在本轮中途被改过时,
重新冻结并记录差异。

**issue 缺少可判定条件时不许开工**:先在 issue 下评论列出缺口,等 owner 补齐。

## 1. infra 准备(所有这次任务需要的数据)

逐项准备并留下证据,写 `INFRA.md`:

| 项 | 检查方式 |
|---|---|
| 数据 | 需要的 `results/stage0_build_parquet/*.parquet`、`stage1_GeneEmbedding/output/sentence_t5.npy`、`item_sids.json` 是否存在,sha256 是否记录 |
| 基线/对照 | 对照臂的 SID、recall、stage2_metrics 路径与 sha256 |
| 解释器 | Stage0/1/3 用 `genrec_env/bin/python3.10`,Stage2 用 `genrec_env_v2/bin/python3.9`(CLAUDE.md §6–§8) |
| 硬件 | `nvidia-smi` 可见卡数、端口 50202/50201 空闲 |
| 树 | 冻结基线 `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` 只读;候选树按 §19b/§90 从它复制并只改 `ARM_NAME` |
| 依赖 | `sympy`/`sympytorch` 等只在项目解释器里;装完复查 GPU |

`INFRA.md` 每条写:`需求 → 命令 → 输出摘要 → sha256 → PASS/FAIL`。存在 FAIL 就停在本节修到 PASS。

## 2. 实现(写这次的代码)

1. **先写会失败的测试(红)**:针对本次改动写数学恒等式、开关两侧行为、梯度/参数更新、路径与导出、
   DDP 一致性等测试,确认当前代码确实失败(CLAUDE.md §32b)。
2. 写实现直到测试变绿;`ARM_NAME` 之外的源码改动必须能逐文件解释。
3. 只改机制层;路径、结果目录、日志、SID 导出、GPU/DDP 启动、端口、冻结超参归框架(§66–§68)。
4. 跑全量:候选树自己的测试 + 相关框架测试。

## 3. 检查循环(**必须按这个循环反复**,上次没问题才允许结束)

每一轮都从第 1 步开始,禁止凭记忆:

1. **重新打开 issue 原文**,逐条对照 `C1..Cn`,把本轮"issue 怎么写的"抄进 `CHECK_<round>.md`。
2. 对每条跑对应检查:单元测试、真实入口 1 epoch smoke、SID 技术有效、行为图契约、指标复算。
3. 记录证据:命令、关键输出、指标值、文件 sha256。
4. 判定:全部 PASS,且**没有把 issue 的表述解释性放宽**,才允许结束本节;任一 FAIL 或
   `NOT_RUN`/缺失,修完后 `round + 1`,回到第 1 步。

| 禁止 | 原因 |
|---|---|
| 只看测试绿就结束 | 测试可能测的不是 issue 要的东西 |
| 用同义反复的检查 | 等于没查 |
| 把 `NOT_RUN`、缺数据、超时当通过 | 未执行即不通过 |
| 一轮之内修改 issue 判据 | 判据只能来自 issue,不能来自实现 |

## 4. 训练(**仅当 issue 要求时**)

**先判定,再决定跑不跑。** 在 `RUN_LOG.md` 写下结论与依据(引 issue 原文的哪一段),再二选一:

| issue 的形态 | 判定 | 做法 |
|---|---|---|
| 验收指标只能由训练产生(例如 Stage3 `test_R@10`、碰撞率、对齐率) | **必须训练** | 走下面的入口表 |
| 分析型 / 复现型 / 数据与契约型 / 纯 infra 或文档 | **不训练** | 跳过本节,直接 §5,并在 `VERDICT.md` 标明"无训练:依据 issue 第 N 段" |

"issue 没明确写要不要训练"本身也是一个判定点:按验收指标反推——若指标可以被**已有产物**复算出来,就不训练;
若必须由新权重产生,就必须训练。两种都不允许含糊过去。

| 阶段 | 入口 | 约束 |
|---|---|---|
| Stage2 | `stage2_start` → `stage2_status` 轮询 | 3000 epoch、四卡、端口 50202、seed 42;单次运行;不早停;不调参;不做多种子 |
| Stage3 | `stage3_start` → `stage3_status` 轮询 | 150 epoch 冻结协议、端口 50201;碰撞率硬门槛见 CLAUDE.md §89 |

轮询单次等待 ≤ 60 秒,每次报告 epoch / 产物 / 进程;禁止 `sleep N; 检查` 丢后台。
训练完成后按 CLAUDE.md §6b 归档 R2(四件齐全才上传)。

失败先分类再动手:`infra`(数据/环境)→ 回第 1 步;`implementation`(代码/公式)→ 回第 2 步;
`resource`(预算外推不可行)→ 记录并停下讨论;`experiment`(训练成功但指标低)→ 进入第 5 步。

## 5. 结论对账

写 `VERDICT.md`,逐条列表:

| 条目 | issue 要求 | 实际值 | 证据(路径/sha) | 判定 |
|---|---|---|---|---|

- **全部 PASS** → 第 6 步。
- 任一 FAIL → 写清失败原因与下一步假设,回到第 1 步(需要改机制就改,但不得改判据)。
- 不许把"指标低"写成"机制正确、只是噪声",除非 issue 允许这种判定。

## 6. 关闭 issue

```bash
gh issue close <N> --comment "结论 + 证据路径 + commit sha"
gh issue view <N> --json state      # 必须 CLOSED
```

关闭前确认:代码已推送(§43)、本地与远端 `main` 同 SHA(§45)、R2 归档完成(若 issue 要求)、
`VERDICT.md` 里没有遗留 FAIL。issue 未满足时**保持 open**,在评论里贴本轮证据与阻塞点。

## 反模式

- 不看 issue 原文,按记忆或按上一个 issue 的判据做。
- 不读 issue 就默认要训练,白烧一次 3000 epoch 预算;或反过来,以"这是分析型 issue"为借口跳过
  issue 明确要求的训练。判定必须写进 `RUN_LOG.md` 并引 issue 原文。
- 检查"看起来没问题"就结束循环;一轮没做完就宣布通过。
- 训练前跳过 §3,训练后才想起判据。
- 在 `/tmp` 放脚本或中间数据(§86),跳过 `commit_guard`(§91),不推送就宣称完成。
- issue 没满足就关闭,或把关闭当作完成。
- 在 issue 仍 OPEN 时停下来:贴一份"下一步计划"当交付、等 owner 确认、或因为某条检查 FAIL 就收工。
  除 §6 关闭与"报告外部阻塞"两种情况外,任何形式的暂停都算本轮失败。
