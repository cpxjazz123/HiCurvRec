---
name: complete-open-issue
description: Use when the task is an open GitHub issue that must be finished end to end. Reads the issue, freezes a machine-checkable acceptance checklist, prepares data and infrastructure, implements with TDD, runs a check loop that re-reads the issue every round, trains when the issue needs it, judges the result against the issue's own criteria, closes the issue either when every criterion passes or when a blockage is reported with evidence, and enforces seven mandatory mechanisms (frozen maths contract, paper cross-check, independent reference implementation, known-answer tests, mutation tests, per-module verification, conclusion re-check) before any result may be claimed.
---

# 按 open issue 完成任务的闭环

一个 issue 只有在**它的验收条件逐条被证据满足**时才算完成。本 skill 把这个过程固定为
`读 issue → infra → 实现 → 检查循环 →(训练,仅当 issue 要求)→ 结论对账 → 关闭`;任何一步不满足,回到它
应该在的那一步,不允许跳步,也不允许"差不多"。

**训练不是固定步骤**:跑不跑由 issue 的验收条件决定(见 §4),不由"skill 有第四步"决定。

**issue 关闭之前不允许暂停,但每个关键节点必须立刻评论到 issue。** 一旦开始,就必须推进到
`gh issue view <N>` 返回 `CLOSED`;不许停下、不许把"下一步计划"当交付、不许等确认。
同时,进度不是留在本地的:每完成一个关键节点或拿到一条关键结论,先 `gh issue comment` 再继续
(节点清单与格式见下节)。这不是"尽量做完",而是本 skill 的完成定义。

| 会被误当成"该停了"的瞬间 | 必须改成 |
|---|---|
| 一轮检查有 FAIL | 立刻回对应步骤修,再开 `round + 1` |
| 某条 `NOT_RUN`,看起来工作量大 | 那就是当前要做的活,做完再判 |
| 训练在跑 | 轮询(≤60 秒/次)直到出结果,然后进 §5 |
| 训练失败/指标低 | 按 §4 分类回对应步骤,或按 §5 出负结论并写进 `VERDICT.md` |
| 遇到需要授权的例外(如 `git push --force`) | 该例外单独处理,但 issue 的其余部分继续推进 |
| 想先汇报进度再继续 | 先 `gh issue comment` 贴该节点证据,再用 `out://` 说明,然后**继续干活** |
| **遇到阻塞、失败、或无法在不违反冻结协议下修复的问题** | 立刻 `gh issue comment` 贴证据(命令、输出、文件路径、已排除的解释),然后按 §6.1 **直接关闭 issue** |

唯一合法的终止点有两个:**§6 全部验收 PASS 后关闭**,或 **§6.1 遇阻塞时的"评论 + 关闭"**。
遇到失败先按 §3/§4 修;修不动了(要改判据、要改冻结协议、外部条件缺失且无法补齐)就不要再保持 OPEN 空等 ——
把阻塞点、已尝试的修复、已排除的解释写进评论,写进 `VERDICT.md`,然后关闭。

## 强制机制(七项,单 Agent 执行时全部必做)

单 Agent 没有人可以互相复核,所以把"复核"固化成七项有证据落点的机制。每一项都必须留下文件,
只写在对话里不算做过;任何一项缺失,§3 的检查循环不得判 PASS。

| 机制 | 单 Agent 执行要求 | 证据落点 | 缺了算什么 |
|---|---|---|---|
| **数学契约** | 编码前冻结公式、输入输出、张量形状与 dtype、边界条件(零范数、空集合、不可达、除零、重合点)、数值容差。实现只能服从契约;确需偏离时先改契约并记录日期与理由 | `CONTRACT.md` + 逐条断言 | 视为未开工 |
| **原论文对照** | 逐模块核对论文的章节与公式编号,以及官方实现;不一致的地方写明"论文怎么写 / 官方实现怎么写 / 我们怎么选 / 为什么" | `CONTRACT.md` 的对照表(列:模块 / 论文位置 / 官方实现 / 本项目取舍) | 视为未核对 |
| **独立参考实现** | 用可信库或**不同计算方法**写第二实现,禁止 import 生产模块:如 numpy 对 torch、暴力枚举对向量化、闭式解对迭代解。同一输入下两者在声明容差内一致 | `reference.py` + `tests/test_reference.py` | 视为自证 |
| **确定性测试** | 用手算得出正确答案的极小输入:2–3 个节点的图、单层码本、退化边界。期望值**写死在测试里**,不是运行时算出来的 | `tests/test_known_answers.py` | 视为无判据 |
| **变异测试** | 故意注入典型错误:off-by-one、符号翻转、错误的 reduction、漏 mask、错误的系数、把 STE 换成直通。每个变异必须让至少一个测试失败,并记录"变异 → 被哪个测试抓到" | `tests/test_mutation.py` + `MUTATIONS.md` | 视为测试无检测力 |
| **分模块验证** | 用真实数据逐模块检查:前向输出、梯度是否存在/是否为 None/是否为零、参数是否真的更新。只看总 loss 不算 | `MODULE_CHECKS.json`(每模块一行) | 视为未验证 |
| **结论复核** | 训练结束后重新读源码、从原始产物**重算**头条指标,确认 `REPORT.md` 里的数字与重算一致;确认结论没有超出数据(如把样本内说成泛化) | `VERDICT.md` 的"复核"段 + 重算脚本 | 视为结论未成立 |

变异测试的最低要求:不少于 5 个变异,覆盖数值(系数/符号)、结构(漏 mask/错 reduction)、
路由(梯度是否到该到的模块)三类;少于 5 个或全部被抓到时只说明测试太平,需补强后再判。

## 执行顺序

1. 阅读 issue,冻结验收清单与**数学与数据契约**(§0、§1;`CONTRACT.md`)。
2. 先写**独立参考实现**与**确定性测试**(含手算期望值),再写正式代码(§2)。
3. 实现生产代码,跑确定性测试与参考实现一致性测试。
4. **故意引入典型错误**,验证测试的检测能力(变异测试);不通过先补测试,不要补实现。
5. 用真实数据**逐模块验证**:输出、梯度、参数更新(§3)。
6. 全部准入检查通过后才启动正式训练(§3、§4)。
7. 训练结束后**重新审查源码、指标计算与实验结论**,从原始产物重算头条数字(§5)。
8. 归档证据并关闭 issue(§6 或 §6.1)。

顺序不可交换:先参考实现后生产代码,先变异测试后正式训练,先结论复核后关闭。

## 进度必须回写到 issue(不是只写本地)

每完成一个关键节点、或得到一条关键结论,**立刻**在 issue 下 `gh issue comment <N> --body-file <文件>`,
然后才继续。本地文件(`RUN_LOG.md`/`CHECK_k.json`/`VERDICT.md`)是证据,issue 评论是给别人看的结论;
只写本地而不评论,视为这一节点没交付。

必须评论的节点:

| 节点 | 评论里至少写 |
|---|---|
| 冻结验收清单后 | 清单条数、正文 sha256、**是否需要训练及依据** |
| infra 完成 / 出现 infra FAIL | 数据与基线哈希、缺什么、修复到哪一步 |
| 每条 `C` 从 FAIL→PASS,或出现新的 FAIL | 编号、命令、关键输出、证据文件路径 |
| 每轮检查结束(`CHECK_k`) | 本轮 PASS/FAIL/NOT_RUN 计数、FAIL 的具体证据、下一轮要修什么 |
| 训练启动 / 训练结束 | run 名、pid、epoch 范围、最终碰撞率与产物路径 |
| §5 结论对账 | 逐条对照表、整体判定、`VERDICT.md` 路径 + commit sha |
| §7 不可满足判定 | 互斥规模、计算脚本路径、为什么该主张在预算内不可满足 |
| 关闭 issue | 结果、证据路径、commit sha(就是 `gh issue close` 的 comment) |

评论用 ``gh issue comment <N> --body-file results/research/issue<N>/COMMENT_<序号>.md``,文件本身入库,
保证"issue 上说了什么"和"仓库里有什么证据"一一对应。一次评论只讲一件事,附上路径与数字,不要复述猜测。

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

冻结清单的同时写 `CONTRACT.md`:每个待实现模块的公式、输入输出形状与 dtype、边界条件(零范数、空邻域、不可达、除零、重合点)、数值容差,以及该模块的论文/官方实现对照行。契约先于代码;实现与契约冲突时改契约并记录日期,不许让代码悄悄定义语义。

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

1. **先写独立参考实现** `reference.py`:不 import 生产模块,换一种计算方法(numpy 对 torch、
   暴力枚举对向量化、闭式解对迭代解),按 `CONTRACT.md` 的公式回答同一问题。
2. **再写会失败的测试(红)**:确定性测试用手算的极小输入并把期望值写死;参考一致性测试比较两个实现;
   另加开关两侧行为、梯度/参数更新、路径与导出、DDP 一致性等测试。先确认它们确实失败(CLAUDE.md §32b)。
3. 写实现直到测试变绿;`ARM_NAME` 之外的源码改动必须能逐文件解释。
4. **变异测试**:注入不少于 5 个典型错误(系数/符号/漏 mask/错 reduction/错误路由),每个都至少要有一个
   测试失败;记录 `MUTATIONS.md`。一个都抓不到的变异说明测试没有检测力,先补测试。
5. 只改机制层;路径、结果目录、日志、SID 导出、GPU/DDP 启动、端口、冻结超参归框架(§66–§68)。
6. 跑全量:候选树自己的测试 + 相关框架测试。

## 3. 检查循环(**必须按这个循环反复**,上次没问题才允许结束)

每一轮都从第 1 步开始,禁止凭记忆:

1. **重新打开 issue 原文**,逐条对照 `C1..Cn`,把本轮"issue 怎么写的"抄进 `CHECK_<round>.md`。
2. 对每条跑对应检查:确定性测试、参考实现一致性、变异测试结果、分模块验证(输出/梯度/参数更新)、真实入口 1 epoch smoke、SID 技术有效、行为图契约、指标复算。
3. 记录证据:命令、关键输出、指标值、文件 sha256。
4. 判定:全部 PASS,且**没有把 issue 的表述解释性放宽**,才允许结束本节;任一 FAIL 或
   `NOT_RUN`/缺失,修完后 `round + 1`,回到第 1 步。

| 禁止 | 原因 |
|---|---|
| 只看测试绿就结束 | 测试可能测的不是 issue 要的东西 |
| 用同义反复的检查 | 等于没查 |
| 把 `NOT_RUN`、缺数据、超时当通过 | 未执行即不通过 |
| 一轮之内修改 issue 判据 | 判据只能来自 issue,不能来自实现 |
| 只看总 loss 就判模块正常 | 分模块验证缺失,等于没查 |
| 变异测试一个都没抓到还判 PASS | 说明测试没有检测力 |

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
`resource`(预算外推不可行)→ 写进 `VERDICT.md` 并**按 §6.1 评论后关闭**;
`experiment`(训练成功但指标低)→ 进入第 5 步。
同一失败按对应步骤修过一轮仍复现、且已排除的解释都有实测 → 不再重试,按 §6.1 关闭。

## 5. 结论对账

写 `VERDICT.md`,逐条列表:

| 条目 | issue 要求 | 实际值 | 证据(路径/sha) | 判定 |
|---|---|---|---|---|

- **全部 PASS** → 第 6 步。
- 任一 FAIL → 写清失败原因与下一步假设,回到第 1 步(需要改机制就改,但不得改判据)。
- 不许把"指标低"写成"机制正确、只是噪声",除非 issue 允许这种判定。
- **结论复核**:头条指标必须用一个独立的重算脚本从原始产物(日志、`item_sids.json`、`test_final.json`)
  重新算一遍,与 `REPORT.md` 写的数字按声明容差比对;对不上以重算为准并说明原因。
  复核段同时回答:这个结论依赖哪些前提,数据是否支持这些前提;样本内结果与泛化结果分开写。

## 6. 关闭 issue

```bash
gh issue close <N> --comment "结论 + 证据路径 + commit sha"
gh issue view <N> --json state      # 必须 CLOSED
```

关闭前确认:代码已推送(§43)、本地与远端 `main` 同 SHA(§45)、R2 归档完成(若 issue 要求)、
`VERDICT.md` 里没有遗留 FAIL、**每个 C 的 PASS 都已评论到 issue**,且七项强制机制各有证据文件:
`CONTRACT.md`(含论文对照)、`reference.py` + `tests/test_reference.py`、`tests/test_known_answers.py`、
`tests/test_mutation.py` + `MUTATIONS.md`、`MODULE_CHECKS.json`、`VERDICT.md` 的复核段。

## 6.1 遇阻塞时的关闭(与 §6 并列的合法终止点)

修不动了就关,不要保持 OPEN 等确认。条件(任一成立):

- 要修就必须改 issue 判据、改冻结协议/护栏,或 `[INFERENCE]` 之外的证据缺口无法补齐;
- 同一失败在**按步骤修过之后**仍然复现,且已排除的候选解释都有实测(不是猜测);
- 训练入口的机器闸门(smoke/准入)失败,而绕过它需要违反 §19h/§22/§23 之类的硬规则。

动作顺序固定:

```bash
# 1) 把阻塞写成评论(文件同时入库,保证 issue 与仓库一一对应)
gh issue comment <N> --body-file results/research/issue<N>/COMMENT_<序号>.md
# 2) 写 VERDICT.md:已完成项 + 阻塞点 + 已尝试的修复 + 已排除的解释 + 需要 owner 决定的最小放宽
# 3) 关闭
gh issue close <N> --comment "阻塞: <一句话>; 证据 <文件路径>; commit <sha>"
gh issue view <N> --json state      # 必须 CLOSED
```

评论里**必须**包含:失败命令与输出原文、`VERDICT.md` 路径、已排除的解释及其实测依据、
以及"若要继续,最小需要放宽什么"(选项 A/B/C 形式最好)。缺这些就是"没说清的关闭",视为未完成。

关闭不是把问题藏起来:`VERDICT.md` 保留全部证据,续跑时从 §6.1 的"最小放宽"直接接上。

## 7. 合同不能证明 issue 的机制主张时

§1–§3 全绿只说明"测量与隔离是对的"。如果测量给出的分布让 issue 的机制主张**在数学上不可满足**,
再训练就是把预算花在一个已知为空的目标上。判据:

- 主张要求"某些对更近";实测这些对同时又必须更远(互斥质量 > 0) → 在该预算下不可满足。
- 主张依赖若干结构同时靠拢,而这些结构共享的单元对彼此施加相反要求 → 单一监督项无法同时满足。

这时不得把"未达成"写进 `VERDICT.md` 当结论,也不得硬训:
①在 issue 下贴出测量、互斥规模与计算脚本,说明为什么该主张在此预算下不可满足;
②提出最小可判定的替代主张(通常是"换成能同时满足这些约束的显式表示"),并写成选项供 owner 决定;
③把证据与数学判据写进 `VERDICT.md`,然后按 §6.1 关闭 —— 数学上不可满足属于阻塞,不保持 OPEN。

## 反模式

- 不看 issue 原文,按记忆或按上一个 issue 的判据做。
- 不读 issue 就默认要训练,白烧一次 3000 epoch 预算;或反过来,以"这是分析型 issue"为借口跳过
  issue 明确要求的训练。判定必须写进 `RUN_LOG.md` 并引 issue 原文。
- 检查"看起来没问题"就结束循环;一轮没做完就宣布通过。
- 训练前跳过 §3,训练后才想起判据。
- 在 `/tmp` 放脚本或中间数据(§86),跳过 `commit_guard`(§91),不推送就宣称完成。
- 修不动了还保持 OPEN 空等确认:那属于 §6.1,必须先评论再关闭。
- 关闭时没有评论阻塞证据,或 `VERDICT.md` 里没有"已排除的解释"与"最小放宽"——这是"没说清的关闭",视为未完成。
- 把关闭当作完成而不写 `VERDICT.md`:证据必须留在仓库里,续跑从最小放宽接上。
- 只写本地 RUN_LOG/CHECK/VERDICT 而不同步评论 issue:进度没到达读者手里。关键节点必须落到 issue 评论。
- 不写 `CONTRACT.md` 就开始编码,或实现之后才回头补公式——那是让代码定义语义。
- 把生产代码自己的输出当正确答案,或参考实现直接 import 生产模块:这是自证,不是验证。
- 训练跑完直接下结论,没有从原始产物重算头条指标:数字没人复核过。
