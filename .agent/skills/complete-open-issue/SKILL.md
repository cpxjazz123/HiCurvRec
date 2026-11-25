---
name: complete-open-issue
description: Use when the task is an open GitHub issue that must be finished end to end. Reads the issue, freezes a machine-checkable acceptance checklist, prepares data and infrastructure, implements with TDD, runs a check loop that re-reads the issue every round, trains when the issue needs it, judges the result against the issue's own criteria, and keeps working until `gh issue view` reports CLOSED. Only two terminal states are allowed: completed (every criterion passes, including negative results) or blocked (a genuine external blockage proven with evidence). Workload, context or round budget, long training, failing tests and implementation errors are explicitly not reasons to stop; PROGRESS.json is only a crash-recovery checkpoint, never an exit. Enforces seven mandatory mechanisms (frozen maths contract, paper cross-check, independent reference implementation, known-answer tests, mutation tests, per-module verification, conclusion re-check) before any result may be claimed. The infra step always begins by reading the shared asset tree /home/wlia0047/ar57/wenyu/GeneRec/assets: assets that already exist there are reused by immutable hash and never rebuilt, and anything genuinely new is created there.
---

# 按 open issue 完成任务的闭环

## 铁律:开始之后必须推进到 CLOSED

一旦开始执行 Issue,Agent 必须持续执行,直到 `gh issue view <N> --json state` 返回 `CLOSED`。

**不得以下列任一理由主动结束任务、发送最终回复或关闭 Issue:**

- 任务量大、剩余步骤多;
- 上下文或当前轮次预算不足;
- 训练耗时长,或训练仍在运行(应当轮询等待,而不是收尾);
- 测试失败;
- 实现错误或存在 bug;
- 任何「这一轮做不完」的判断。

**只要还存在可执行任务**(写代码、修 bug、加测试、跑检查、等训练、写证据、评论 issue),
就必须继续做。不得以「已保存进度」作为退出理由。

`PROGRESS.json` **只是断点文件**:用于外部强制中断之后的恢复,**不构成主动退出条件**;
写完之后必须继续执行,不得把它当作收尾动作。

只有两种情况允许停止:

1. `gh issue view <N>` 返回 `CLOSED`(COMPLETED 或 BLOCKED,见 §6.2);
2. 用户明确取消,或外部系统强制中断(进程被杀、机器故障等不可控事件) ——
   这两种不属于 skill 的主动退出;恢复后仍然继续。

遇到真正无法解决的阻塞:按 §6.1 记录完整证据、走阻塞关闭流程、**确认 `CLOSED` 之后才允许退出**。

一个 issue 只有在**它的验收条件逐条被证据满足**时才算完成。本 skill 把这个过程固定为
`读 issue → infra → 实现 → 检查循环 →(训练,仅当 issue 要求)→ 结论对账 → 关闭`;任何一步不满足,回到它
应该在的那一步,不允许跳步,也不允许"差不多"。

**训练不是固定步骤**:跑不跑由 issue 的验收条件决定(见 §4),不由"skill 有第四步"决定。

**每个关键节点必须立刻评论到 issue,然后继续干活。** 评论与断点都不是收尾:
它们必须落到仓库和 issue 上,但做完就继续下一步,直到 §6.2 的 COMPLETED 或 BLOCKED。

不许把「下一步计划」当交付;同样不许把「已评论 / 已写断点」当交付 —— 交付是 §6.4 四问的答案。

同时,进度不是留在本地的:每完成一个关键节点或拿到一条关键结论,先 `gh issue comment` 再继续
(节点清单与格式见下节)。

| 会被误当成"该停了"的瞬间 | 必须改成 |
|---|---|
| 一轮检查有 FAIL | 立刻回对应步骤修,再开 `round + 1` |
| 某条 `NOT_RUN`,看起来工作量大 | 那就是当前要做的活,做完再判 |
| 训练在跑 | 轮询(≤60 秒/次)直到出结果,然后进 §5 |
| 训练失败/指标低 | 按 §4 分类回对应步骤,或按 §5 出负结论并写进 `VERDICT.md` |
| 遇到需要授权的例外(如 `git push --force`) | 该例外单独处理,但 issue 的其余部分继续推进 |
| 想先汇报进度再继续 | 先 `gh issue comment` 贴该节点证据,再用 `out://` 说明,然后**继续干活** |
| **遇到阻塞或失败** | 先按 §4 修;确实不可继续时按 §6.1 记录证据并关闭为 BLOCKED |
| 本轮预算/上下文不足、任务还剩很多 | **不是结束理由**:继续做,必要时先写断点再继续,不得收尾 |
| 测试失败 / 实现报错 | 那是当前要做的活:定位、修、重跑,不要结束 |
| 训练还在跑 | 轮询等待,或先推进不依赖它的部分,不得收尾 |
| 参考实现或单模块测试通过 | 不等于生产代码修好,也不等于训练入口正确(§6.5) |

终止状态只有两种(§6.2):**COMPLETED**(全部验收 PASS)或 **BLOCKED**(§6.1 允许的外部阻塞),
二者都必须让 `gh issue view <N>` 返回 `CLOSED`。

**工作尚未完成,不等于工作无法完成。** 任务量大、训练耗时长、上下文或单轮预算不足、代码还没写完、
可修复的 bug —— 都不是阻塞,也都不是结束理由。把这些写成研究任务的客观阻塞,本 skill 明令禁止。

**不要发送最终回复来结束一轮工作。** 只有 CLOSED,或用户明确取消/外部强制中断,才允许停止。

## 强制机制(七项,单 Agent 执行时全部必做)

单 Agent 没有人可以互相复核,所以把"复核"固化成七项有证据落点的机制。每一项都必须留下文件,
只写在对话里不算做过;任何一项缺失,§3 的检查循环不得判 PASS。

| 机制 | 单 Agent 执行要求 | 证据落点 | 缺了算什么 |
|---|---|---|---|
| **数学契约** | 编码前冻结公式、输入输出、张量形状与 dtype、边界条件(零范数、空集合、不可达、除零、重合点)、数值容差。实现只能服从契约;确需偏离时先改契约并记录日期与理由 | `CONTRACT.md` + 逐条断言 | 视为未开工 |
| **原论文对照** | 逐模块核对论文的章节与公式编号,以及官方实现;不一致的地方写明"论文怎么写 / 官方实现怎么写 / 我们怎么选 / 为什么" | `CONTRACT.md` 的对照表(列:模块 / 论文位置 / 官方实现 / 本项目取舍) | 视为未核对 |
| **独立参考实现** | 换**数学路径**或换可信库,禁止 import 生产模块。把 PyTorch 公式逐行翻译成 NumPy **不算独立**:必须走另一条推导,如闭式解 vs 迭代、暴力枚举 vs 向量化、另一种恒等式、`geoopt`/`scipy` 的成熟实现;并在契约里写明二者为何独立、在什么容差内一致 | `reference.py` + `tests/test_reference.py` | 视为自证 |
| **确定性测试** | 用手算得出正确答案的极小输入:2–3 个节点的图、单层码本、退化边界。期望值**写死在测试里**,不是运行时算出来的 | `tests/test_known_answers.py` | 视为无判据 |
| **变异测试** | 故意注入典型错误:off-by-one、符号翻转、错误的 reduction、漏 mask、错误的系数、把 STE 换成直通。每个变异必须被**指定的断言**抓到,并记录"变异 → 哪条断言失败";崩溃、导入错误、超时都不算 | `tests/test_mutation.py` + `MUTATIONS.md` | 视为测试无检测力 |
| **分模块验证** | 用**实际训练用的 loss** 与真实数据逐模块检查:前向输出、梯度是否存在/是否为 None/是否为零、参数是否真的更新。用临时构造的简单 loss 证明"参数能动"不算验证;只看总 loss 也不算 | `MODULE_CHECKS.json`(每模块一行) | 视为未验证 |
| **结论复核** | 训练结束后重新读源码、从原始产物**重算**头条指标,确认 `REPORT.md` 里的数字与重算一致;确认结论没有超出数据(如把样本内说成泛化) | `VERDICT.md` 的"复核"段 + 重算脚本 | 视为结论未成立 |

变异测试的最低要求:不少于 5 个变异,覆盖数值(系数/符号)、结构(漏 mask/错 reduction)、
路由(梯度是否到该到的模块)三类。**每个有效变异都必须被抓到,并且是被指定的断言抓到**;
程序崩溃、导入错误、异常退出都不算捕获 —— 那只说明变异写坏了,不说明测试有检测力。
**全部被抓到是目标,不是可疑信号。** 需要补强的是这两种情况:某个变异只能靠"崩溃"暴露,
或没有任何断言能定位到它。

## 测试本身的正确性(六条)

1. **参考实现要有数学独立性**:换推导或换库,不是把 torch 公式抄成 numpy(见上表)。
2. **变异必须被指定断言抓到**:崩溃不算捕获;记录"变异 → 失败的那条断言原文"。
3. **优先测生产路径上的函数**:训练实际调用的那个函数才是被测对象;只测新写的参考函数等于没测生产代码。
4. **分模块梯度必须用实际训练 loss**:用临时 loss 证明参数可更新,不证明训练时梯度会到该到的地方。
5. **覆盖三类输入**:正常输入、边界输入(球边界、空邻域、单元素)、退化输入(零向量、常数、重复边)。
6. **训练前后各检查一次接线**:开训前确认实际调用链与冻结契约一致(哪层用哪个曲率、梯度到哪个模块、
   哪个系数进 loss);训练后再核对一次,防止训练期改动让接线漂移。

## 执行顺序

1. 阅读 issue,冻结验收清单(§0;`RUN_LOG.md`)。
2. **先读 `/home/wlia0047/ar57/wenyu/GeneRec/assets`**:需要的 Infra 资产已存在就直接读取(按哈希引用),
   **禁止重复创建**;确实缺失的才创建,并且**只创建到 `assets/`**(§1.0)。
3. **若开始新 Stage2 机制,先归档全部旧机制,再复制冻结基线、开展新机制探索**(§1.0a);
   之后冻结**数学与数据契约**(`CONTRACT.md`)。非新机制任务不搬动当前机制。
4. 先写**独立参考实现**与**确定性测试**(含手算期望值),再写正式代码(§2)。
5. 实现生产代码,跑确定性测试与参考实现一致性测试。
6. **故意引入典型错误**,验证测试的检测能力(变异测试);不通过先补测试,不要补实现。
7. 用真实数据**逐模块验证**:输出、梯度、参数更新(§3)。
8. 全部准入检查通过后才启动正式训练(§3、§4)。
9. 训练结束后**重新审查源码、指标计算与实验结论**,从原始产物重算头条指标(§5)。
10. 归档证据并关闭 issue(§6 或 §6.1)。

顺序不可交换:先查 `assets/` 再建资产,先归档旧机制再复制基线与探索新机制,先参考实现后生产代码,
先变异测试后正式训练,先结论复核后关闭。

## 进度必须回写到 issue(不是只写本地)

每完成一个关键节点、或得到一条关键结论,**立刻**在 issue 下 `gh issue comment <N> --body-file <文件>`,
然后才继续。本地文件(`RUN_LOG.md`/`CHECK_k.json`/`VERDICT.md`)是证据,issue 评论是给别人看的结论;
只写本地而不评论,视为这一节点没交付。

必须评论的节点:

| 节点 | 评论里至少写 |
|---|---|
| 冻结验收清单后 | 清单条数、正文 sha256、**是否需要训练及依据** |
| infra 完成 / 出现 infra FAIL | **`assets/` 复用了哪些资产(`path:sha256`)、哪些是新建的、新建在哪个路径**;数据与基线哈希、缺什么、修复到哪一步 |
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

新 Stage2 机制任务先完成 §1.0a 的旧机制归档,再写 `CONTRACT.md`;其他任务在冻结清单时写。
契约包括每个待实现模块的公式、输入输出形状与 dtype、边界条件(零范数、空邻域、不可达、除零、重合点)、
数值容差,以及该模块的论文/官方实现对照行。契约先于代码。**实现与契约冲突时默认改实现**;
只有当契约本身写错、或研究定义确需变更时才改契约,
并且必须由 owner 确认、记录日期与理由。**禁止为了让实现通过而放宽契约** —— 那是在改正确答案。

## 1. infra 准备(所有这次任务需要的数据)

### 1.0 **先查 `assets/`,禁止重复创建**(第一步,先于本节任何其他动作)

所有 Infra 资产都放在仓库根的 `assets/`(CLAUDE.md §3a)。**本节的第一步永远是读取它**,
而不是重新计算已经存在的东西:

```bash
ls -la /home/wlia0047/ar57/wenyu/GeneRec/assets
cat  /home/wlia0047/ar57/wenyu/GeneRec/assets/manifest.json   # 若存在
```

判定与动作:

| `assets/manifest.json` 的状态 | 动作 |
|---|---|
| 存在,且覆盖本次 issue 需要的全部资产 | **直接读取使用**。逐项核对 `sha256`、`coverage`、`use`(training/evaluation/description)。**禁止重新计算、重新生成、复制或改名**;缺什么就在 `INFRA.md` 里写明"引用 `assets/<name>`(sha256 …)",而不是重建一份 |
| 存在,但只覆盖一部分 | 已有部分**直接读取**;只对**确实缺失**的项新建,新文件也写进 `assets/`,并在自己的 manifest 里登记 |
| 不存在,或完全覆盖不到 | 才允许创建,并**必须直接创建到 `/home/wlia0047/ar57/wenyu/GeneRec/assets/`**,不得写在 `results/research/<issue>/` 或任何临时目录里 |

新建时必须同时产出:`assets/manifest.json`(路径、版本、schema、shape、dtype、行数、字节、sha256、输入哈希、
代码提交、生成命令、覆盖率、训练/评估用途)、`assets/coverage.json`(覆盖率与抽样声明)、`assets/README.md`
(内容与读取方式),并在 `INFRA.md` 里写明每个文件的 `path:sha256`。

`assets/` 下已存在的二进制**禁止覆盖或手工改写**(CLAUDE.md §17d);需要新版本时写新目录并更新 manifest。
复用既有资产时**按不可变哈希引用**,不复制、不重复上传。

### 1.0a 新机制探索前必须先归档全部旧机制

**每次开始一个新 Stage2 机制都必须执行,不是首次执行一次。** 在复制基线代码或开展新机制设计、
参考实现、生产实现之前,先把 `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/` 顶层
所有已有的旧机制目录,完整移动到 `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/_archive/`。

1. **保留 `baseline/` 与 `_archive/` 原位**:不移动、不改写冻结基线,不递归搬动已有归档。
2. 先确认旧树没有被运行中的训练或验证依赖;若仍在使用,等待其结束后再归档,不得擅自中断作业或边训练边搬源码。
3. 已跟踪机制目录用 `git mv`;完全未跟踪的机制目录用 `mv`,随树保留所有未跟踪文件。
   归档目标重名时使用带时间戳的独立目录,禁止覆盖、合并或删除既有归档。
4. 同步仍需使用的路径常量、提示词与文档引用(§90);不改写历史运行的冻结证据。
   在 `INFRA.md` 记录旧路径 → 归档路径、执行命令及结果,并评论到 issue。
5. 核对顶层不再残留任何旧机制目录后,才可从唯一来源 `baseline/TIGER_RQ-VAE`
   复制新候选并设置 `ARM_NAME`,保持顶层仅一个活跃机制。归档未完成,禁止开始新机制探索或复制基线。

普通状态查询、当前机制的修复或续跑不触发此步骤;不得把正在推进的当前机制误归档。

### 1.1 逐项准备并留下证据,写 `INFRA.md`

| 项 | 检查方式 |
|---|---|
| **既有 Infra 资产** | **先查 `assets/`**(见 1.0):需要的资产是否已在 `assets/manifest.json` 中登记;命中就引用其 `path:sha256`,缺失才新建 |
| 数据 | 需要的 `results/stage0_build_parquet/*.parquet`、`stage1_GeneEmbedding/output/sentence_t5.npy`、`item_sids.json` 是否存在,sha256 是否记录 |
| 基线/对照 | 对照臂的 SID、recall、stage2_metrics 路径与 sha256 |
| 解释器 | Stage0/1/3 用 `genrec_env/bin/python3.10`,Stage2 用 `genrec_env_v2/bin/python3.9`(CLAUDE.md §6–§8) |
| 硬件 | `nvidia-smi` 可见卡数、端口 50202/50201 空闲 |
| 树 | 开始新机制前必须先按 §1.0a 归档全部旧机制,保留 `baseline/` 与 `_archive/`;确认后才从只读的 `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` 复制候选树并只改 `ARM_NAME`(§19b/§90) |
| 依赖 | `sympy`/`sympytorch` 等只在项目解释器里;装完复查 GPU |

`INFRA.md` 每条写:`需求 → 命令 → 输出摘要 → sha256 → PASS/FAIL`。存在 FAIL 就停在本节修到 PASS。
"资产已存在于 `assets/`"是一条合法且优先的 PASS 路径,前提是哈希核对通过且没有重复创建。

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

失败先分类再动手:`infra`(数据/环境)→ 回第 1 步;`implementation`(代码/公式)→ 回 §2;
`experiment`(训练成功但指标低)→ 进入第 5 步。

**`resource` 不是失败分类。** 工作量大、训练耗时长、上下文或轮次预算不足、代码尚未实现,
都不是结束或关闭的理由(§6.2);遇到它们,继续做就是了。

可修复的实现错误必须继续定位。**同一种修复方式失败,要更换诊断方法**,例如:换成独立参考实现对照、
逐位比对中间张量、构造最小复现、加变异测试定位未被覆盖的分支、缩小到单 batch 单步。
禁止用「已经修过一轮」当作任务不可完成的结论。

## 5. 结论对账

写 `VERDICT.md`,逐条列表:

| 条目 | issue 要求 | 实际值 | 证据(路径/sha) | 判定 |
|---|---|---|---|---|

- **全部 PASS** → 第 6 步(COMPLETED)。
- **实现/数值失败**(代码错误、NaN、指标算错、试验没有按冻结协议执行)→ 回到对应步骤修,修完重跑。
- **实验/科学结果**(实现与协议都正确,指标没有朝假设的方向走)→ **这是有效结果,不是要修的 bug**:
  写进 `VERDICT.md`,作为 COMPLETED 关闭。**禁止**回头调参、换种子、改判据、加 loss 去把它变成正面结果。
  区分二者的顺序固定:先用七项机制在冻结协议下确认实现正确(§6.5 的 `path:sha256` 绑定),
  再看指标;实现没被证明正确之前,不许把"指标低"解释成机制不行。
- 不许把"指标低"写成"机制正确、只是噪声",除非 issue 允许这种判定;同样不许把"结果不如预期"
  说成"实现有 bug",用来给继续训练或调参找理由。
- **结论复核**:头条指标必须用一个独立的重算脚本从原始产物(日志、`item_sids.json`、`test_final.json`)
  重新算一遍,与 `REPORT.md` 写的数字按声明容差比对;对不上以重算为准并说明原因。
  复核段同时回答:这个结论依赖哪些前提,数据是否支持这些前提;样本内结果与泛化结果分开写。

## 6. 关闭 issue

```bash
gh issue close <N> --comment "结论 + 证据路径 + commit sha"
gh issue view <N> --json state      # 必须 CLOSED
```

关闭前必须通过 §6.4 的四问,并确认:代码已推送(§43)、本地与远端 `main` 同 SHA(§45)、
R2 归档完成(若 issue 要求)、`VERDICT.md` 里没有遗留 FAIL、
**每个 C 的 PASS 都已评论到 issue 且绑定 `path:sha256`(§6.5)**,且七项强制机制各有证据文件:
`CONTRACT.md`(含论文对照)、`reference.py` + `tests/test_reference.py`、`tests/test_known_answers.py`、
`tests/test_mutation.py` + `MUTATIONS.md`、`MODULE_CHECKS.json`、`VERDICT.md` 的复核段。

## 6.1 遇阻塞时的关闭(BLOCKED)

只有 §6.2 里判为「是」的三类情形可以关闭:

- **必要的外部资源确认不可用** —— 数据、模型、凭证或硬件缺失,且有命令输出为证;
- **继续就必须违反冻结协议或硬规则** —— 例如绕过 §19h/§22/§23 的机器闸门;
- **owner 明确要求终止**。

「工作量大」「训练久」「上下文或轮次预算不足」「代码还没写完」「可修复的 bug」**一律不得**据此关闭。

动作顺序固定:

```bash
# 1) 把阻塞写成评论(文件同时入库,保证 issue 与仓库一一对应)
gh issue comment <N> --body-file results/research/issue<N>/COMMENT_<序号>.md
# 2) 写 VERDICT.md:已完成项 + 阻塞点 + 已尝试的修复 + 已排除的解释 + 需要 owner 决定的最小放宽
# 3) 关闭
gh issue close <N> --comment "BLOCKED: <一句话>; 证据 <文件路径>; commit <sha>"
gh issue view <N> --json state      # 必须 CLOSED
```

评论里**必须**包含:失败命令与输出原文、`VERDICT.md` 路径、已排除的解释及其实测依据、
以及"若要继续,最小需要放宽什么"(选项 A/B/C 形式最好)。

## 6.2 终止状态与资源阻塞的判定

| 状态 | 含义 | Issue 状态 | 必须留下的东西 |
|---|---|---|---|
| **COMPLETED** | 全部验收 PASS,结果可以正面也可以负面 | CLOSED | `VERDICT.md` + 逐条证据 + 关闭评论 |
| **BLOCKED** | 存在可验证、不可自行解决的外部阻塞 | CLOSED | §6.1 的三类证据 + `VERDICT.md` 的阻塞段 |

**没有第三种终止状态。** 「还有任务可执行」不是一种状态,而是「继续做」的指令。

|情况|是否允许结束任务|
|---|---|
| 剩余任务工作量大 | **否** |
| 预计训练耗时较长 | **否** |
| Agent 上下文即将耗尽 | **否** |
| 当前执行轮次预算不足 | **否** |
| 代码尚未实现 | **否** |
| 可修复的代码错误 | **否** |
| 必要外部资源确认不可用 | 是,需提供证据 |
| 必须违反冻结协议才能继续 | 是,需提供证据 |
| 用户明确要求终止 | 是 |

判定的顺序固定:先问「还有没有我能执行的下一步」。只要答案是「有」,就必须继续执行 ——
既不得关闭,也不得用断点收尾;不得把「这一轮做不完」翻译成「这个任务做不了」。

## 6.3 断点文件(`PROGRESS.json`,仅用于外部中断)

`PROGRESS.json` 是**恢复用的断点**,不是退出机制。在每个关键节点顺手更新它,
这样进程被杀、机器故障之后可以从这里接着做;写完必须继续执行。

```json
{
  "issue": 243,
  "commit": "<当前 HEAD sha256/short>",
  "state": "WORKING",
  "done": ["C1", "C2", "C4", "C5"],
  "not_run": ["C17", "C21", "C26"],
  "failed": [],
  "running": [{"name": "teacher", "pid": 1234, "checkpoint": "...", "log": "..."}],
  "next_commands": ["python3 results/research/issue243/teacher.py", "..."],
  "blocked_by": null,
  "updated_at": "2026-11-23T18:00:00Z"
}
```

规则:

- `commit` 必须与工作区一致;文件入库并推送(§43/§45)。
- **写完继续干活**,不得以「已保存进度」结束这一轮。
- 恢复时**从这个文件继续**,不得重复 `done` 里的实验,不得重跑已完成的训练。
- 只有 §6.2 判为「是」的情形才能关闭,且必须走 §6.1 的 BLOCKED 流程。

## 6.4 关闭前强制检查

关闭前必须逐条回答。**前四项必须为"是",第五项必须为"否"**,任何一项不符都不允许关闭:

1. **所有生产代码是否已经实现?**(应为**是**)—— 不是参考实现、不是骨架、不是打桩。
2. **所有必要测试是否实际执行?**(应为**是**)—— 有输出与退出码,不是"应该能过"。
3. **所有规定的训练是否已经完成?**(应为**是**)—— 有 run_id、日志、checkpoint、指标。
4. **Infra 资产是否都来自 `assets/` 且没有重复创建?**(应为**是**)—— 复用的资产引用了
   `assets/manifest.json` 里的 `path:sha256`;新建的资产**只创建在 `assets/`** 并登记进 manifest;
   没有把已有资产重算一份、改名或复制到别处(§1.0)。
5. **是否仍然存在可执行的未完成任务?**(应为**否**)—— 若为"是",**不得关闭,继续执行**(断点只用来防外部中断)。

五条答案连同证据写进 `VERDICT.md` 的关闭段。

## 6.5 证据必须绑定最终生产代码

以下四件事**互不替代**:

| 已通过 | **不能**说明 |
|---|---|
| 参考实现通过测试 | 生产代码已修复 |
| 单模块测试通过 | 实际训练入口正确 |
| smoke 通过 | 正式训练完成 |
| 训练 loss 下降 | 机制成功、指标达标 |

因此每条验收证据都要绑定**最终实际执行的生产代码**:文件路径 + `sha256sum`,
以及产生该证据的命令与退出码。改了生产代码之后,同一份证据自动失效,必须重跑。
`VERDICT.md` 的每条 PASS 都要写出对应的 `path:sha256`。

## 7. 合同不能证明 issue 的机制主张时

§1–§3 全绿只说明"测量与隔离是对的"。如果测量给出的分布让 issue 的机制主张**在数学上不可满足**,
再训练就是把预算花在一个已知为空的目标上。但判定"不可满足"必须给出**不可行性证明**,
而不是"发现了一些冲突":

- 构造互斥约束,证明它们的交集为空(同一对商品被要求同时最近与最远,且二者由同一组参数决定);
- 或证明在冻结预算下(参数量、监督项个数、每项一个标量)不存在同时满足的赋值;
- 证明要附可执行脚本、互斥规模与为什么该证明覆盖整个预算。

**只有"部分样本冲突""某些对被同时拉扯"不足以判定不可满足** —— 那通常只是需要更强的表示或更多容量,
属于实现/建模层面的下一步,不是阻塞。

这时不得把"未达成"写进 `VERDICT.md` 当结论,也不得硬训:
①在 issue 下贴出测量、互斥规模与计算脚本,说明为什么该主张在此预算下不可满足;
②提出最小可判定的替代主张(通常是"换成能同时满足这些约束的显式表示"),并写成选项供 owner 决定;
③把证据与数学判据写进 `VERDICT.md`,然后按 §6.1 关闭 —— 数学上不可满足属于阻塞,不保持 OPEN。

## 反模式

- 不看 issue 原文,按记忆或按上一个 issue 的判据做。
| 不读 issue 就默认要训练,白烧一次 3000 epoch 预算;或反过来,以"这是分析型 issue"为借口跳过
issue 明确要求的训练。判定必须写进 `RUN_LOG.md` 并引 issue 原文。
| 不先读 `/home/wlia0047/ar57/wenyu/GeneRec/assets` 就动手造数据,或把已有资产重算一份、改名、
复制到别处 —— 这既浪费算力也让两个版本对不上。命中 `assets/manifest.json` 就直接引用其
`path:sha256`;确实缺失的才创建,且只创建到 `assets/`。
- 检查"看起来没问题"就结束循环;一轮没做完就宣布通过。
- 训练前跳过 §3,训练后才想起判据。
- 在 `/tmp` 放脚本或中间数据(§86),跳过 `commit_guard`(§91),不推送就宣称完成。
- 遇到能修的问题却停在 OPEN 空等确认:要么继续修,要么按 §6.1 证明是外部阻塞并关闭。
- 关闭时没有评论阻塞证据,或 `VERDICT.md` 里没有"已排除的解释"与"最小放宽"——这是"没说清的关闭",视为未完成。
- 把关闭当作完成而不写 `VERDICT.md`:证据必须留在仓库里,续跑从最小放宽接上。
- 只写本地 RUN_LOG/CHECK/VERDICT 而不同步评论 issue:进度没到达读者手里。关键节点必须落到 issue 评论。
- 不写 `CONTRACT.md` 就开始编码,或实现之后才回头补公式——那是让代码定义语义。
- 把生产代码自己的输出当正确答案,或参考实现直接 import 生产模块:这是自证,不是验证。
- 训练跑完直接下结论,没有从原始产物重算头条指标:数字没人复核过。
- 把"这一轮预算/上下文不够"写成阻塞并关闭 issue:工作没做完不等于做不到(§6.2)。
- 写完 `PROGRESS.json` 或发一条进度评论就结束这一轮:那只是断点/节点,必须继续执行。
- 因为测试失败、实现报错或训练还没跑完就发送最终回复:这些都是当前要做的活。
- 参考实现通过就宣称生产代码已修复;单模块测试通过就宣称训练入口正确;smoke 通过就宣称正式训练完成(§6.5)。
- 同一种修复方式失败一次就认定任务不可完成:应当更换诊断方法(§4)。
- 关闭时仍有可执行的未完成任务:必须 OPEN + `PROGRESS.json`(§6.3/§6.4)。

## 8. 公共流水线接口(仓库内实际实现，Issue #257)

本 skill 的 §1–§6 步骤在仓库里有可直接调用的实现，禁止另写一套：

| skill 步骤 | 实现 |
|---|---|
| 冻结验收清单 / 断点 | `results/research/<issue>/RUN_LOG.md`、`PROGRESS.json` |
| infra 与候选边界 | `tools/research/stage2/{launcher,contract}.py`、`tools/issue252/run_workspace.py` |
| 依赖感知验证 | `tools/issue257/depgraph.py`(`Spec(...).fingerprint()` / `plan(specs, cache=...)` / `Cache.record`) |
| 轮次状态机 | `tools/issue257/registry.py`(`Round.register` / `.advance` / `.evidence`) |
| 计时 | `tools/issue257/timeline.py`(`Timeline.start/stop/summary`，区间合并) |
| 自动化分析 | `tools/issue257/analysis.py`(`analyse(identity)`，按 run 身份，无 issue/arm 硬编码) |
| 结论对账 | `tools/issue257/replay.py`(`replay(path)` 只读复算，逐项 PASS/FAIL/NA) |
| 报告集合 | `tools/issue257/report.py`(`write_all` → RUN_MANIFEST/TIMELINE/VALIDATION_PLAN/RECEIPTS/REPORT/VERDICT/ROUND_RESULT) |
| 归档校验 | `tools/issue257/archive.py`(`verify(identity)`；上传仍调 `results/research/issue139/upload_stage3.py`) |

规则：

* 复用 PASS 的依据是**指纹**（导入闭包 + 资产 + 配置 + 测试自身 + 运行时），不是路径、时间或
  候选名；FAIL/NOT_RUN/缺收据/陈旧收据不得复用。
* 机制数学、真实入口/四卡、性能、最终 run 身份仍按 issue 要求每轮实跑；公共层可按指纹复用。
* 训练仍只经 `stage2_start` / `stage3_start`（或 `launcher.launch`），不新写启动器。
* 复算一律从原始产物（`item_sids.json`、`test_final.json`、predictions）重算，容差与期望值写死。
* 测试：`python3.9 -m pytest tools/issue257/tests -q`（已知答案、依赖失效四场景、闭包突变、复算侦错）。
