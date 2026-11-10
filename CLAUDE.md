# CLAUDE.md — 当前仓库执行规则

> 以下每个编号项均是一条独立规则，以当前 `main` 的实际源码和路径为准。

## 1. 仓库边界

1. Stage2 仅使用两棵冻结基线树或由其派生、单独命名且通过独立验证的候选源码树。
2. Stage3 只使用 `stage3_T5Train/train_HG-Rec.py` 作为正式训练入口。
3. 机制专属测试由 `tools/research/verification/` 执行，正式训练与状态查询仍通过 `tools/genrec-mcp/server.py`。
4. 新科研工具统一放在 `tools/research/`，模块职责以该目录 `README.md` 为准。
5. 禁止引用或恢复已清理的研究控制器、旧机制树、旧结果路径和历史任务队列。

## 2. 解释器与配置

6. Stage0、Stage1 和 Stage3 使用 `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10`。
7. Stage2 和机制专属测试使用 `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9`。
8. MCP Server 使用仅依赖标准库的系统 `python3`。
9. 项目超参数、路径和运行模式只能在源码常量中修改，禁止通过命令行参数或环境变量覆写。
10. 项目训练与研究入口默认无参运行，只有内部测试执行器及 `/tmp/` 参考脚本可使用必要的执行参数。

## 3. 数据与产物

11. Stage0 仅向 `results/stage0_build_parquet/` 写入 `items.parquet`、`train.parquet`、`valid.parquet` 和 `test.parquet`。
12. Stage1 产物固定放在 `stage1_GeneEmbedding/output/`。
13. Stage2 模型、SID 与导出文件固定放在 `results/stage2_RQ-VAE/<ARM>/`，不得写入 Stage2 源码树。
14. Stage3 日志和模型产物固定放在 `results/stage3_T5Train/<RQVAE_VARIANT>/`。
15. #117 的研究状态与证据仅写入 `results/research/issue117/`，不得污染正式训练目录。
16. 构建商品 SID 的行为监督只允许使用训练划分中合法可见的交互，禁止读取验证或测试阶段的未来行为。
17. 同一商品在不同用户、批次或推理请求中必须使用相同的全局 Semantic ID。

## 4. 训练与评估

18. Stage2 保持 3000 epochs 的完整训练预算，禁止重新加入基于训练损失或验证指标的提前停止。
19. **Issue #121 的联合耦合范围门**：任何新机制必须是**训练只行为信号**与**显式曲线运算**在**同一条连通计算通路**上的耦合，去掉任一侧核心计算即改变；仅行为后缀、仅把欧氏距离换成双曲距离、两个互不相干的损失、以及继承但未使用的双曲代码都判 OUT_OF_SCOPE。范围判定检查的是 Agent 写下的**内容**（行为信号与算法、曲线运算与作用位置、耦合通路与方程、三个开关的行为关/曲线关/both-on **定义**、最小可实现规格；按 §19g 这些开关只要求在代码中存在且实现正确，**不训练任何关闭开关的臂**），不是布尔标记；范围裁决以提案自身内容为准，单个评分者的主观标签不得单独决定去留，分歧必须留痕。契约硬编码在 `tools/research/rounds/agents.py` 的 `RESEARCH_OBJECTIVE` 与 `SCOPE_CONTRACT`，并传播到全部三个构思、三个评分、实现规格、三个验证与修复提示。
19b0. **源码中禁止绝对路径**：`stage2_RQ-VAE/` 下的所有 trainer（含冻结基线）不得出现 `/home/...`、`/fs04/...`、`/tmp/...` 这类绝对路径字面量，也不得用 `Path(__file__).parents[N]` 数父目录；仓库根必须由「向上查找含 `stage2_RQ-VAE/` 的目录」推导，产物写在 `REPO_ROOT/"results"/"stage2_RQ-VAE"/ARM_NAME`，`ARM_NAME` 为单一常量。准入预检在**真正运行的位置**（`stage2_RQ-VAE/TIGER_<idea>/`）求值，出现绝对路径或深度依赖即拒绝。（Issue #121）
19b. **Issue #121 的基线固定为冻结树**：**每一轮**新机制的**唯一**允许起点都是 `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_RQ-VAE`，**不继承上一轮的实现**（解析实现见 `tools/research/rounds/baseline.py::baseline_path`：任何 `root` 参数都被忽略，源缺失或不可读即失败关闭）。每个选中想法必须从该冻结基线**新建独立副本**（禁止从别的未完成候选、变异工作树、历史实验树或上一轮候选起步），并记录基线解析路径与基线/副本源码摘要。副本只允许由框架改写 `ARM_NAME` 这一个常量，其余源码必须与基线逐文件一致（`_same_source_except_arm` 校验）。基线只读：源缺失或不可读即失败关闭（不得静默换基线），候选的输出路径若与基线结果目录或其他候选冲突则在训练前被拒绝。行为关/曲线关/both-on 三个开关写在**候选树自身**里（不另派生对照树），且按 §19g 不训练关闭开关的臂。
19c. **Issue #121 的四类停止必须分开记账**：`AGENT_INFRA_ERROR`（Agent 崩溃/超时/答案不可用）只重试、**不得**因此 BLOCK 机制；`IMPLEMENTATION_NEEDS_FIX` 是真实缺陷，交专职修复 Agent；`RESOURCE_BLOCKED` 是实测不可行（冻结预算外推超时），保留想法走优化路线；`STAGE2_3_FAILED` 是训练或评估失败。单个候选的修复轮次上限为 5 轮，到顶即带证据记录，不再消耗 Agent。
19c0. **一次探索只产出一个机制**：每轮构思（5 个 Agent、3-9 个想法）经 3 个评分者全评后，选择规则**只保留唯一最好的合格机制**（`max_selected = 1`、`keep_single_best`），其余想法以拒绝理由留档；范围不合格的想法永不入选，也不能作为 fallback。该机制必须走完 **实现 → 审查/修复 → Stage2（冻结四卡配方）→ Stage3** 的完整链路，产出一个真实的 `test_R@10`，才算这一轮的产出。不得把一轮当成候选批，也不得在未训练前宣称机制成功。该机制只跑一次正式训练（§19g）：行为关/曲线关/both-on 三个开关必须存在于代码中并有廉价的接线检查，但**不训练任何关闭开关的臂**，也不做调参扫描或多种子重复。（Issue #121）
19c1. **Issue #122 的排序与选题**：子 Agent **不打分**，只**排序**。一轮 = 冻结证据包（缺证据写 `UNKNOWN`、未检索写 `NOT_CHECKED`）→ 3 个独立评审把**每个**提案按优劣排成序，每条给出依据证据的理由 → 一次有界反驳（重新评审）→ **名次相加最小**（并列则比第一名次数、再比 idea_id）者为本轮机制，其余按序进 backups。策略（`max_selected=1`、`min_reviewers`、`tie_break_order`）只写在 `tools/research/rounds/rubric_v1.json`（带版本与 sha256，轮次 pin 住），只能在**新轮开始前**改；提示词、解析器、矩阵必须用同一形状（`RANK: <idea_id> | position=<n> | reason=<...>`），漏排/重复/并列/无理由一律 schema 错误，不做部分平均。没有自动 fallback：无人合格即 `NO_QUALIFIED_IDEA` 并回到构思；事后重算只写 `rescored_audit_<ts>.json`，不得改写历史决策。
19c2. **Issue #123 的评分前证据探针**：构思之后、评分之前必须先跑证据探针（train-only 行为统计 + 小张量双曲自检 + 可核验文献），三份评审读到**同一份冻结证据包**。探针只读、只在 CPU 上跑、有硬上限（每提案 120 秒、文献 ≤3 次查询 / ≤15 篇、证据修订 ≤1 次、探针内 0 次 Stage2/3）；缺失一律写 `DATA_UNAVAILABLE` / `UNKNOWN` / `NOT_CHECKED` / `NOT_RUN` / `NOT_IMPLEMENTED`，**不得**把「没测到」写成「已否定」，**不得**编造证据或文献。探针不得抬高评分：与假设矛盾的证据应当降低排序。`log_0(exp_0(v))=v` 只是映射自洽，**不构成**曲率有价值的证据；小规模代理实验**不是**候选自身的通路；非欧效应**不构成** Stage3 Recall 会提升的证据。（Issue #123）
19d0. **Stage2 派发必须用冻结配方形态**：轮次派发默认 `STAGE2_LAUNCH_STRATEGY="ddp4"`——一个候选独占 4 张卡（`world_size=4`、全局 batch 4096、0.16 秒/epoch），同一时刻只跑一个候选。**禁止**为了并发把候选钉在单卡上（全局 batch 变成 1024，是另一套优化配方，与冻结基线不可比）；需要并发只能在明确标注「不可比」的前提下作为探索性诊断使用。（Issue #121）
19d0c. **多个待训 Stage2 机制必须并行,不得串行**(owner 2026-11-08 明确要求):当有多个已过准入、等待训练的 Stage2 机制时,**必须同时启动它们的训练**,禁止排队一个一个跑。**单张卡可同时容纳最多 10 个 Stage2 机制运行**(实测 A100-40GB 上每臂约 0.5-2 GB 显存,四个机制并发无压力)。并行不改变优化配方:每臂仍为单进程单 rank、**全局 batch 4096**、冻结的 3000 epoch 与 seed=42;并发只影响墙钟时间,不构成"钉卡降 batch"的例外。启动顺序按准入通过顺序,一次把当前所有已过闸的臂全部派发。
19d0b. **Stage2 可以并行运行（2026-11-08 新增本地规则，owner 于本会话明确要求）**：多个 Stage2 候选**不需要串行**——即使当前已有 Stage2 正在运行，也可以再启动别的 Stage2 候选，按显存/内存/CPU 与主机 GPU 数量自行决定并发度，不必等前一个结束。该规则**只覆盖 Stage2**：Stage3 仍按框架现有的全局单例守卫串行（每臂独占整卡），第 19d0 条里“同一时刻只跑一个候选”的措辞不再适用于 Stage2 的派发节奏，但“不得为了并发把候选钉在单卡上、从而改变全局 batch”这条仍然有效——单卡主机上的单卡分支保持全局 batch 4096，属于同一优化配方，不算钉卡。
19d. **Issue #121 的轮次工具面**：每个 phase 都是独立 MCP 工具（`research_round_ideate` / `_score` / `_implement` / `_verify`，加 `_start` / `_run` / `_status` / `_results` / `_launch` / `_collect`），phase 幂等、每轮一把运行锁；轮次按 issue 归档（`results/research/issue121/rounds/<round_id>/`），候选树在 `results/research/issue121/trees/<round_id>/<idea_id>/`。并发 Stage2 每卡最多 2 个作业、每作业独立 GPU 与输出目录，技术有效即自动进 Stage3。
19g. **实验政策（硬性）**：一个候选**只跑一次**正式训练（冻结预算：3000 epoch 四卡 + 冻结的 Stage 3 协议，无早停）。**不做消融实验**——behaviour-off / geometry-off / both-on 三个开关必须存在于代码中且实现正确，但**不训练任何关闭开关的臂**；**不做调参实验**——耦合权重、clip 界、温度、用量权重等常量必须**可推导**（相对尺度的界、实测量、有据可依的默认值），不允许靠试值挑选；**不做多种子实验**——候选不换种子重跑，单次运行即定论，种子噪声计入不确定性。行为类诊断指标（共现一致性、邻接率、转移可预测性）只记录、**不作准入闸门**，Stage 3 `test_recall@10` 是最终判断。需要消融/扫描/多种子才能成立的提案，不属于本计划。  **例外（Issue #140–#143，2026-11-08，owner 于本会话明确要求）**：#140（M1 分层可学曲率 κ-stereographic）、#141（M2 异质径向流形 E15×H16×R1）、#142（M3 局部曲率 profile 结构角色监督）、#143（M4 量化后 AF3 可微码本优化）四个机制**各只跑一次 Stage2**（seed=42、3000 epoch、冻结全局 batch），**不进入 Stage3**，不因 Stage2 指标选择、不做 early stop、不做多种子、不做调参扫描；behavior-off / geometry-off / both-on 三个开关必须存在于代码并有廉价接线检查，但不训练关闭开关的臂。四个臂可并发（每臂独占一张卡、全局 batch 保持 4096，即同一优化配方）。该例外只覆盖这四个 Issue 列出的机制，不对其他候选生效。  **例外（Issue #144–#147，2026-11-09，owner 于本会话明确要求）**：#144（hard-SID 结构保真度的逐层几何外部选择，**仅分析、不训练**）、#145（AF3 符号分布的静态容量分配）、#146（embedding 半径的径向可变曲率）、#147（E16×H16 因子化联合码本）四项研究，**允许 AF3 只在 RQ-VAE 外部做配置选择或导出后评估**，不要求行为信号进入训练通路（本例外是对 §19 范围契约的显式更新）。#145–#147 每机制**只跑一次** Stage2（seed=42、3000 epoch、冻结全局 batch），**Stage2-only、不进入 Stage3**，不因 Stage2 指标选择、不 early stop、不做多种子、不做调参扫描；三 token SID、每层 256 码、Sinkhorn 均衡分配规则不变。几何起点为仓库冻结的 32 维混合几何基线 `stage2_RQ-VAE/TIGER_mixed_rq`（E16×H16 + Sinkhorn；`TIGER_RQ-VAE` 的 E32 无法表达本组的逐层维度分配），起点逐文件 SHA 记入 manifest，每个候选只允许框架改写 `ARM_NAME` 并与其它候选隔离。该例外只覆盖这四个 Issue，不对其他候选生效。
  **例外（Issue #138，2026-11-04）**：#138 明确要求的匹配臂——冻结基线、AF3+欧氏（behavior ON / geometry OFF）、曲线+乱序 AF3（geometry ON / behavior OFF）、both-on、普通 Forman+曲线——**允许训练**，每个臂只跑一次冻结预算，不因 Stage3 结果选择、不early stop；该例外只覆盖 #138 列出的这五个臂，其余提案仍按本节执行。**例外（Issue #139，2026-11-07，owner 于本会话明确批准）**：#139 要求的五个匹配臂——F0（E16×H16、α 固定 1/2）、G（全局 α_ℓ）、L（按 prefix 的局部 α_ℓ(p)）、E（E16×E16、c=0、同一局部外循环）、A（E16×H16 + 既有 AF3 辅助监督、α 固定 1/2）——**允许训练**，每臂只跑一次 seed=42 冻结预算 3000 epoch，**仅 Stage2、不自动进入 Stage3**，不因 Stage2 指标选择、不做 early stop、不做多种子、不做独立超参扫描；该例外只覆盖 #139 列出的这五个臂。**补充（Issue #139，2026-11-08，owner 于本会话明确要求）**：该 Issue 的目标是完成五个机制的 Stage2、记录五个 SID 的失真度并**为每一个完成对应的 Stage3**，因此 `tools/genrec-mcp/stage2_quality_gate.py` 的 `ISSUE139_STAGE3_EXCEPTION` 为这五个臂名显式放行学习碰撞门槛（L3 碰撞仍测量并记录为诊断，不静默跳过），例外按臂名限定、写进证据，不对其他任何候选放宽。
19h. **Sinkhorn 平衡分配不是机制可以改的东西**：基线的一级分配是 Sinkhorn 均衡分配，**均衡正是码本均匀的原因**。机制的创新**不允许**把均衡分配替换成普通 `argmin`：机制的打分必须**进入均衡分配内部**（Sinkhorn 的代价矩阵带上曲率距离与行为偏移），**分配规则本身仍是基线的**。**实测代价**：`round_20261105_144909_130` 的 `pmi-geomargin-l1-rq` 把一级分配写成 `k1 = argmin_k(D_k - off_k)`，一级码本随即坍缩——256 个码只用了 42 个（参照臂 241 个）、最大码占 31% 的商品（参照 1%）、二级前缀 1027（参照 16145）、L3 碰撞率 0.703（参照 0.065），被所有者的学习门槛拒绝，Stage 3 未启动。**三处强制**：(1) 范围闸门（`tools/research/rounds/scope.py`）把“用普通 argmin 取代 Sinkhorn 均衡分配”的提案判为 OUT_OF_SCOPE，范围契约第 6 条写明；(2) 规格闸门（`tools/research/rounds/evals.py` 的 `assignment-keeps-the-balancing`）在冻结前拒绝这种表面；(3) 机器准入（`tools/research/stage2/checks.py` 的 `check_assignment_rule`）拒绝源码里对机制打分取 `argmin` 的候选树。**要改分配规则，本身就得是一个被显式批准的研究问题**。
19f. **准入闸门是机器检查的结论**:派发前必须通过 `checks.py` 的四类一致性与四套测试的真实执行;任何一项 `FAIL`、缺失或执行零用例都禁止启动 Stage 2。不存在"带记录驳回"的通道,也没有 Agent 判决可被手工绕过。
19e. **轮次里的代理分工与主 Agent 的决定权**:每轮由 **5 个**独立构思 Agent 提案(`IDEA_AGENTS`)、3 个评分 Agent 独立打分;**没有实现 Agent,也没有验证 Agent**——入选提案的候选树由流程从基线复制好,**由主 Agent 直接写实现**,写完用 `research_round_register_implementation` 登记(记录改动文件、冒烟结果与源码哈希),再由主 Agent 用 `research_round_accept` 接受。**验证是机器验证,不是 Agent 评审**:准入由框架在派发时执行的四类一致性检查(`tools/research/stage2/checks.py`:冻结常量与启动器归属、真实入口跑通一 epoch、导出的 SID 技术有效、四卡不挂死)以及 `verification_check` 真实执行的四套独立测试承担,`NOT_RUN` 不是通过。旧的验证/修复 Agent、`research_round_verify`/`research_round_plan_repair`/`research_round_apply_repair` 三个工具及其提示词、解析器与测试已删除,不得恢复。
19. **Stage3 启动不作任何准入检查（owner 2026-11-09 明确要求）**：只要开始 Stage3，就不再检查 SID 技术有效性、Hard-L1 ER、二级前缀数、L3 Collision、码使用率、冗余度、分层记录等任何准入条件；这些指标只作为诊断被测量与记录，不阻断启动。Resample 的 ER=16.38953942232631 与 L2=16145 仅为**历史参照**值，不是分界值。Stage3 test_R@10 > 0.065 仍是最终成功条件；实验登记保留所有候选（含低召回与失败分支），不得只留最优。
20. 所有新 Stage2 机制在正式训练前必须通过独立 Agent 编写的数学、梯度、DDP 和真实入口测试。
21. 梯度验证必须使用实际模型与训练 batch 检查参与训练的损失、梯度及参数更新，不得只检查合成张量。
22. Stage3 固定使用 `NUM_EPOCHS=150`、`EVAL_INTERVAL=5`、`EVAL_START_EPOCH=75`、`EARLY_EVAL_EPOCHS=(25,50,75,100,125)`、`SEED=42`、`BEAM_SIZE=20`、`NO_EVAL=True` 和 `SKIP_TEST=False`。
23. Stage3 只能更换合法的 SID 输入路径、结果路径和变体标识，禁止修改冻结训练超参数或恢复 early-stop。
24. 正式四卡 DDP 的 Stage2 使用端口 `50202`，Stage3 使用端口 `50201`。
25. 当前实验的最终采纳标准为 Stage3 的 `test_R@10 > 0.065`，不得自动套用已关闭研究 Issue 的旧指标门槛。

## 5. 独立 Agent 验证与训练准入

26. 每个新候选必须提供独立数学、梯度、DDP 和真实入口四套可执行测试，不得使用仅返回成功的占位用例。
27. MATH、GRADIENT、RUNTIME 和 REVIEW 验证角色必须保留相互独立的真实 OpenCode 执行记录。
28. 验证凭据必须绑定冻结数学提案、实际 Stage2 源码哈希、四套测试文件哈希和独立审查结论。
29. MCP 必须在每次正式训练启动前重新执行四套测试，任一测试失败、缺失或执行零用例均禁止启动。
30. 正式 Stage2 仅允许调用 `stage2_start(source_dir, verification_id)`，并在测试通过后执行实际训练入口的一轮梯度 Smoke Test。
31. 旧研究框架的历史 admission 记录不得作为新机制的训练准入凭据。
32. `tools/research/` 不得擅自启动 Stage2、Stage3 或 GPU 作业。
32b. **TDD 是实现与修复的强制形式**：主 Agent 在实现新机制、修改机制或修复机制缺陷时，必须**先写测试、再写实现、最后跑测试**：先在候选树中写出针对本次改动的、当前代码会失败的测试（数学恒等式与定义域、开关两侧行为、梯度与参数更新、路径与接口、DDP/集合通信、训练与导出一致性等，按改动性质选择），**确认其确实失败（红）**；再写实现使其通过（绿）；最后连同候选树既有测试与框架测试一起跑全量。禁止先写实现再补测试，也禁止写与实现同义反复的测试（例如用同一函数两遍互相验证）。测试文件与实现同属候选树，`research_round_register_implementation` 登记时一并记录其摘要。修复缺陷时同样先补一个能重现该缺陷的失败测试，再修。

## 6. 原生科研工具

33. `tools/research/core/` 只管理唯一持久 DAG、原子租约和不可变证据。
34. `tools/research/agents/` 由 **OpenAI Agents SDK** 创建与运行 Agent（`agents.Agent` + `agents.Runner`，工具用 `agents.function_tool` 包装）：边界由工具保证（读只限工作区与策略允许的根，写只限工作区，无 shell，被拒的读入收据），进度仍走 `append_progress` 通道，答案文件与 `execution.json` 收据（含边界指纹）不变；模型由 `settings.AGENT_CHAT_MODEL` 给出，未配置时拒绝运行；**已删除 OpenCode CLI 路径与 LangChain 后端**，禁止恢复旧代理调度框架。
35. `tools/research/science/` 必须以训练用户行为数据验证业务问题、数学候选和审查身份。
36. `tools/research/workflows/` 必须先冻结 DATA 与 LITERATURE 证据，再允许 GEOMETRY 正式提案。
37. Reviewer 的 Prompt、当前候选 ID、公式与文件哈希不一致时，审查必须被判为无效。
38. 真实 Agent 必须保留可核验的进程、输入、输出和日志证据，不能将模拟执行当成已完成研究。
39. `tools/research/integrations/` 只使用现有 MCP 训练接口，不得自行创建第二套训练调度器。
40. `tools/research/agents/` 的真实进程默认禁用，必须在隔离和测试验收后才允许开启。

## 6b. Stage3 完成后必须归档到 R2（2026-11-08 owner 指令）

每次 **Stage3 跑完**，必须把该次结果所需的一切上传到 Cloudflare R2，缺一不可：

1. **该 Stage3 所用的 Stage2 代码**（`stage2_RQ-VAE/TIGER_mixed_rq_<ARM>/` 的 `train_rqvae.py`、`model/`、`math_spec.json`、`reference_contract.json`，打包为 `stage2_source.tar.gz`）；
2. **SID 文件**（`item_sids.json`）；
3. **导出该 SID 的 Stage2 checkpoint**（`rqvae_best.pth`）；
4. **完整的 Stage3 recall 记录**——不止 `test_recall@10`，`test_final.json` 的全部键值原样上传，并附 `HG_Rec.log` 与 `training_metrics.jsonl`。

实现：`results/research/issue139/upload_stage3.py <ARM>`，凭证从仓库根的 `r2_apikey` 读取（**该文件已加入 `.gitignore`，禁止入库**），rclone 通过环境变量配置、不在主机留持久配置。目标位置是**专用桶 `genrec`**（2026-11-08 新建）。桶内**第一层目录是 `01`、`02`、`03`……编号**，每个编号目录下**只有四项**（owner 2026-11-08 指定，不出现 issue/臂名）：

```
genrec/<NN>/<机制名>/       ← 该机制的 Stage2 源码：train_rqvae.py 与 model/（若候选树另有 math_spec.json、reference_contract.json，则一并附带；它们不是归档必需项）
genrec/<NN>/item_sids.json   ← SID 表
genrec/<NN>/rqvae_best.pth   ← 导出该 SID 的 Stage2 checkpoint
genrec/<NN>/recall.json      ← Stage3 结果的全部键值
```

编号与身份、以及每个上传文件的 SHA-256 记在仓库内 `results/research/issue139/r2_ledger.json`（桶里不放假名，来源仍可审计）。`run_stage3.py` 在每臂产出 `test_final.json` 后自动调用它。

**只归档四件齐全的**：SID 表、码本、Stage2 源码树、完整 Stage3 recall 块——缺任一件就**不上传**，并在脚本输出的缺口统计里注明原因（这是 owner 2026-11-08 的明确要求）。历史 118 个 VERIFIED 配对中齐全的只有 **8 个**；`upload_pairs.py` 默认只传这 8 个（`--include-incomplete` 才放宽），`upload_stage3.py` 默认要求 `test_final.json` 存在（`--allow-incomplete` 才放宽）。

`r2_apikey` 只用于上传；任何日志、manifest、提交都不得包含其内容。

## 7. Git 与交付

41. Git 远端只允许使用 `https://github.com/cpxjazz123/HiCurvRec.git`。
42. 当前开发只向 `main` 提交，不建立长期研究分支。
43. 每次迭代必须提交并成功推送到 GitHub 后才可声明完成。
44. 禁止未经明确授权执行 `git push --force` 或 `git push --force-with-lease`。
45. 每次推送前必须核对远端地址，推送后必须确认本地 `main` 与远端 `main` 的 Commit SHA 一致。
46. **二进制产物不入库**:Stage2/Stage3 的 `.pth`/`.pt`/`.npy` 不再提交到 git(2026-11-01 起,原因是 LFS 对象不可变、每次重跑永久占用配额,2.9 GB 已超出免费 1 GiB)。它们由入库的 `item_sids.json` 与一次约 22 分钟的 Stage2 重跑重建;`.gitattributes` 的 LFS 规则保留,以免将来若确需入库时误把二进制写进普通对象。历史已重写一次以释放配额,不得再引入新的二进制提交。

## 8. MCP 科研调度

47. 所有新科研任务必须通过现有 GenRec MCP 的 `research_start` 和 `research_resume` 进入唯一持久 DAG。
48. MCP 查询与取消研究任务必须使用真实 `run_id`，不得将 Issue Open 或任务已创建视为 Agent 正在运行。
49. DATA 与 LITERATURE 允许并行执行，GEOMETRY 和 REVIEW 必须等待各自实际证据依赖完成。
50. MCP 的 `research_verify_start` 和 `research_verify_review` 必须使用当前候选的冻结提案及真实独立 OpenCode 会话。
51. 研究 MCP 只能执行授权的 Agent 任务，正式 GPU 训练仍必须通过独立的 `verification_check` 和 `stage2_start` 准入。

## 9. MCP 重载（改了 `tools/` 源码后必须做）

52. MCP Server 是 harness（`omp`）拉起的 stdio 子进程：它只 import 一次代码，改了 `tools/` 下的源码后，**已在运行的进程仍是旧代码**。
53. 重载方式：只对 server 子进程发 SIGTERM，harness 会按原命令行把它重新拉起（实测 3 秒内完成），新进程即载入当前磁盘上的代码。
54. 定位与重载命令：

    ```bash
    OLD=$(pgrep -f 'genrec-mcp/server.py' | head -1)
    ps -o pid,ppid,cmd -p "$OLD"          # 确认父进程是 omp，且只有这一个 server
    kill -TERM "$OLD"
    for i in $(seq 10); do sleep 3; NEW=$(pgrep -f 'genrec-mcp/server.py' | head -1); \
      [ -n "$NEW" ] && [ "$NEW" != "$OLD" ] && break; done
    pgrep -f 'genrec-mcp/server.py'        # 必须有新的 pid，且 tools/ 里的工具可正常调用
    ```

55. 禁止杀 `omp` 本体或它的 worker：那会连同本会话的上下文与工具挂载一起丢掉。
56. 重载后必须实测一次 MCP 工具调用（而不是只看进程号变了）：调用失败或返回旧行为即视为重载未生效。
57. 只有确认重载生效后，才可用 MCP 触发依赖新代码的行为（例如按源码漂移重新起草验证套件）；禁止绕过 MCP 直接调用内部函数来推进流程。
58. 若重载后 MCP 仍拒绝某项请求，那是规则本身在 fail-closed（例如源码已变、证据哈希对不上），应先修规则或重新生成证据，不得靠直接调用内部函数绕过。

## 10. GPU 任务授权（常设授权）

59. 仓库所有者已给出**常设授权**：任何 GPU 任务（Stage2 训练、Stage3 训练、四套机制测试里的真实 DDP 检验）可自行启动，**不必逐次询问授权**。
60. 该常设授权只替代"人工逐次批准"这一步，**不放宽任何机器强制的闸门**：四套独立测试必须由 `verification_check` 真实执行并全部通过，source/test SHA 必须与冻结 manifest 一致，正式训练前必须完成一轮真实 Trainer smoke，Stage-3 冻结协议不得改动。
61. 因此 issue #117 §1「当前资源权限」与 §4 第 5 步「HPC GPU/权限有效」由本节满足；`tools/research/integrations/preflight.py` 错误文案中的 "explicit authorization" 指的也是这一层。
62. 常设授权不等于免除记录：每次 GPU 运行仍必须留下真实 run_id、Worker PID、实际模式、checkpoint 与评估工件；不得以"已获授权"为由伪造运行结果，也不得绕过 `verification_check`/`stage2_start` 直接跑 trainer。


## 11. Stage2 分层指标永久记录


63. **每个新 Stage2 候选在完成训练并导出 item_sids.json 后、报告质量结论或尝试 Stage3 前，必须对该 SID 单独计算并持久化一份完整的分层指标记录**（使用 `tools/research/science/sid_layer_analysis.py::record_single_sid`）。必记的实际数值包括：L1 Hard ER、L1 正/负样本前缀命中率（HitRate，仅诊断）、不同的二级 SID 前缀数、不同的三级 SID 组合数、L3 Collision（三级 SID 重复商品占比）、L3 Refinement Gain（bits）、Conditional G3（平均 log2 概率比），并记录 Resample 历史基线与诊断对比（无晋级门槛）。必须同时记录实际 SID 路径和 SHA-256、训练/验证输入路径及 SHA-256、指标口径、有效验证样本数、实验/候选身份和输出文件位置。记录必须在读取该候选的 Stage3 测试成绩前产生，固定使用训练行为与冻结独立验证样本，禁止借测试结果调参或选最优运行；任一指标无法计算时明确标记缺失/失败及原因，不得虚构、填写 0 或称作 PASS。所有 L3 指标为**强制记录的诊断值**；按第 19 条 Stage3 启动不再作任何准入检查（owner 2026-11-09），该记录仍写，但失败只记录、不拒绝启动。分层报告保存在该 Stage2 结果目录下 `stage2_metrics/<sid_sha256>.json`，不得覆盖其他 SID 的历史记录。


64. **科研审查的准入标准**：独立 REVIEW 只审查冻结提案的数学类型与可定义域、同容量对照是否只差度量、是否使用验证/测试/未来行为或预测时用户上下文、损失与目标分布能否唯一复算、以及反证是否可机械判定；提案必须同时注册 Stage3 行为检验（指标、方向与 mixed/null 分类）；按 §19g **不做多种子实验**，行为检验以单次运行的诊断记录，不做种子合并。Stage2 只构建 SID、从不排序，因此**不得**以「Stage2 侧指标无法自证 next-item 收益」为由阻断晋级；行为收益由 Stage3 `test_R@10` 判定，Stage2 指标（Hard-L1 ER、二级前缀、L3 三统计）只作为前导指标。

64b. **SID–recall 配对的准入判据（2026-11-02 修订）**：任何"某个 SID 对应某个 `test_R@10`"的结论，只在**目录同名**时成立；不满足即标注 `REJECTED_NAME` 并从统计中排除，禁止与准入配对混报。
(a) **准入条件（目录同名）**：`results/stage2_RQ-VAE/<X>` 与 `results/stage3_T5Train/<X>` 必须同名；唯一允许的等价形式是嵌套路径把 `/` 写成 `__`（如 `A/r8_control` ↔ `A__r8_control`），大小写必须一致。
(b) **诊断项（不作准入）**：配对表必须记录 SID 表与 Stage 3 结果文件各自的**引入 commit 与时间戳**（`git log --diff-filter=A -1 --format=%H -- <原始路径>`，在仓库原始路径上判定；`analysis_sid_recall/sids/` 的收集副本不算路径证据），并给出 `SAME_COMMIT` / `COMMITS_DIFFER` / `NOT_TRACKED` 分层；recall 的引入早于 SID 时标注 `SID_AFTER_RECALL`。该列只作风险提示：强制"同 commit"会把可用集合压到单一系列（2026-11-02 实测：同 commit + 同名 = 73 对，其中 99% 来自 `curvature_RQ-VAE`；仅同名 = 110 对、34 个系列、最大 69%），不再作为准入条件。
补充约束：
(i) 关联键只能是 **`sid_sha256`**，**禁止**按 variant 名关联——182 行只有 168 个 variant 名（`curvature_RQ-VAE` 一名下 11 个不同 SID），按名关联会静默配错表。
(ii) 每个 Stage2/Stage3 机制臂必须让 Stage 3 变体目录名与 Stage 2 臂目录名完全一致，新臂不得另起名字；同名要求在派发前检查，不符即视为配对不可用。
(iii) 配对表必须由 `analysis_sid_recall/pair_audit.py` 生成（`pair_audit.csv` 的 `verdict` 与 `commit_tier` 列），任何 SID–recall 统计必须同时在 `VERIFIED` 子集与全表上重算并并列报告；全表保留在 `analysis_sid_recall/archive/sid_recall_full_182.csv`。单报全表数值视为不合规。
(iv) **必须报告集合的来源构成**：按 Stage 2 父系列统计配对数量（`_iterNN` 折叠为同一系列）；若单一系列占比 ≥90%，该集合只能支持「同系列内」结论，**不得**用于跨机制结论。逐层构成与统计见 `analysis_sid_recall/PAIR_TIERS.md`。截至 2026-11-02：182 对中 **110 对**通过同名判据（34 个系列，最大系列 69%），其中 73 对同时满足同 commit、20 对 commit 不同、17 对 recall 文件未入库；72 对因目录名不符被排除。

## 12. 机器失败持续修复

65. **Issue #117 的可修复机器错误不允许用“尝试达到 3 次/5 次”等固定次数淘汰研究任务。** Agent 输出格式/JSON、候选 schema、数学符号实现、代码与独立测试失败必须保留每次真实执行凭据和错误，反馈具体问题，在同一 `run_id` 中持续修正、复验直到合规或用户取消；不得伪造测试成功。科学审查中的 `MATERIAL` 可按可复核证据退回 GEOMETRY，以新版本继续审查，不设固定轮次上限；`FATAL`/`PARKED` 属于真实科学否决/暂停，应记录并阻止该候选晋级，不能重复要求同一 Reviewer 改票。任何循环仍必须遵守用户取消、算力成本、Worker 租约、隔离、正式训练准入和原 Stage2/Stage3 冻结预算；不保证客观上不成立的机制必然能够成功。

## 13. 工程不变量归框架（候选只改机制）

66. **候选只拥有机制。** 一个 Stage2 候选可修改的只有：行为统计、双曲操作、分配规则、残差更新与新增损失，以及针对这些计算的测试。**路径、结果目录、日志、SID 导出、GPU/DDP 启动、端口、冻结超参**由框架拥有（`tools/research/stage2/`），候选不得自行实现或覆盖。
67. **路径与结果由 `tools/research/stage2/contract.py` 唯一决定。** 仓库根用**无歧义标记**（同时含 `stage2_RQ-VAE`、`results`、`stage1_GeneEmbedding`）；一次运行绑定 `results/stage2_RQ-VAE/_runs/<run_id>/<arm>/{src,result}` 的不可变源码快照与独立结果目录，新修订用新目录，永不覆盖正在运行的版本。
68. **启动只有一条路径。** 四卡训练必须调用 `tools/research/stage2/launcher.py::launch`（端口 50202、world_size 4、cwd 为结果目录）。候选源码中出现 `--nproc_per_node=` 或 `--master_port=` 即判不合格。cwd 必须是结果目录而不能是候选树：框架重新提升候选树会替换该目录，进程在 torch 导入调用 `os.getcwd()` 时死亡。
69. **Agent 输出契约。** 框架负责校验、序列化与保存 agent 的答案；只读角色返回内容而不写文件。磁盘上已有的答案只有在 `execution.json` 的 `verified` 为真时才复用；超时/取消留下的部分答案必须移入 `partial/` 并重跑，不得被当作答案。
70. **失败分三类，只恢复出错的那一步。** `auth`/`quota`/`timeout`/`crash` 属执行故障：只重试失败的 agent，认证与配额故障**暂停**该服务而非重试；维度/梯度/公式与提案不符属机制实现错误：改代码并复审受影响部分；训练完成但 `test_R@10` 低属实验负结果：记录后进入下一轮。同一阶段的多个 agent 必须逐个持久化，任一失败不得丢弃其余结果，候选保持“等待审查”而不是被判无效。
71. **准入前执行四类一致性检查**（`tools/research/stage2/checks.py`，按源码摘要缓存）：冻结常量与启动器归属、真实入口跑通一 epoch、导出的 SID 技术有效、四卡不挂死。`NOT_RUN` 不是通过。**每修好一种工程故障，必须留下一个能重现它的回归测试**（`tools/research/stage2/tests/`）；底座变更须用未修改的基线跑通短流程后再发布。
72. **状态轮询的间隔上限。** 监督任何长任务(Stage2/Stage3 训练、评估、构建)的进度时,单次等待**不得超过 60 秒**,禁止一次性 sleep 很久才去看状态。确实需要更长的等待时,把它拆成多次、每次不超过 60 秒的检查,每次都报告当前进度(已完成的 epoch、产物是否已生成、进程是否存活)。把 `sleep N; 检查` 放到后台作业里再事后取回输出,不算轮询,同样禁止;轮询用工具自带的 status 接口(如 `stage2_status`、`stage3_status`)或直接读日志/产物文件。
73. **每个工程问题都必须留下对应的测试文件**：凡出现过的工程故障（集合通信、路径与导出、常量引用、DDP 启动、缓存与状态、MCP 重载等），必须在 `tools/research/stage2/tests/` 或候选树 `tests/` 中新增一个能**重现该故障形态**的独立测试文件（含对检测器本身的自测），并与修复在同一次提交里入库；没有对应测试文件的工程修复不得声明完成。已有测试若覆盖同一形态，可扩展该文件，不必重复建文件。

## 14. 公式参考实现：实现阶段的强制两写对照

81. **数学规格先于实现：Idea / Formula / Coding 三责分离**。一轮在评分选出机制之后、进入实现之前，必须先走**公式阶段**：**Formula Agent** 只负责把**已选中的** Idea 写成完整数学定义，必须覆盖该机制涉及的**全部**关键计算（度量、行为如何进入代价、分配、梯度如何回到编码器、重构、每个开关的作用、以及明确排除的表面），并给出**可微表面**的 SymPy 语句；它**不得**改变 Idea 的机制，若 Idea 缺少机制必需的��义，必须返回 `SPECIFICATION_INCOMPLETE` 并列出缺什么，**不得**自行补假设。**两个 Math Review Agent** 独立检查该规格**是否忠实表达了 Idea**（不是“能不能跑”），并检查数学是否正确、完整、无歧义（符号、量纲、极限、梯度通路、开关语义）；任一评审不通过即退回 Formula Agent 修改一次。**冻结**由确定性程序完成：写出 `math_spec.json` 及其 sha256。**冻结之后 Coding Agent 只能按该规格实现**，不得重新解释或修改数学定义。
82. **实现的数学必须就是冻结的数学**：`register_implementation` 在没有冻结规格时直接拒绝；候选树里的 `reference_contract.json` 的语句**必须逐条等于**冻结规格 `continuous_surface.statements`，否则拒绝并说明“实现的是重述而不是规格”。框架的参考契约检查因而校验的是「代码 vs 冻结规格」，不再依赖实现者手写的契约。

74. **每个 Stage2 机制必须在候选树根部提交 `reference_contract.json`**，把自身**连续可微面**（距离、核、罚项、损失项、残差更新）写成 SymPy 语句：`symbols`（符号顺序）、`args`（参数名/宽度/采样域或标量）、`statements`（有序赋值）、`output`（输出表达式），并声明被比较的函数 `module:callable`。提案里的 `spec_equations` 与契约里的语句必须指向同一个公式。
75. **参考实现由框架独立重建**：`tools/research/stage2/reference.py` 用 `sympytorch` 从这些语句生成参考模块，在双精度下比较候选实现与参考实现的**取值**（`torch.testing.assert_close`，rtol 1e-6 / atol 1e-8）、**梯度**（rtol 1e-5 / atol 1e-7），并在**候选实现与参考实现两侧**分别执行 `torch.autograd.gradcheck`。任一项不通过即 `FAIL`，禁止启动 Stage2。
76. **准入接入**：`tools/research/stage2/checks.py::check_reference_agreement` 已并入 `run_all`；无契约文件的树记为 `NOT_APPLICABLE` 并保留理由，**不得**视为通过；契约存在但不一致即为 `FAIL`。检查在 Stage2 解释器下以子进程运行，超时 15 分钟。
77. **符号对照的诚实边界**：argmin 分配、straight-through 代理梯度等离散/不可微部分**不可能**被符号比较覆盖，必须在契约的 `excluded` 中逐条列出「表面」与「理由」，并继续由四套独立测试与一 epoch 冒烟承担。禁止把离散部分塞进符号对照来制造「全绿」。
78. **工具角色**：`sympytorch` 是参考实现的唯一裁决者；`wrenfold` 只作为可选的**第二代码生成器**（生成可读 torch 代码供人工对照），不进入准入裁决；`torch.autograd.gradcheck` 证明**两侧**的前向-反向自洽，但公式本身写错时它也可能通过，因此三者必须同时使用、不能互相替代。
79. **依赖归属**：`sympy`、`sympytorch`、`wrenfold` 只装在 Stage2/Stage3 的项目解释器中（`genrec_env_v2` / `genrec_env`），不装进系统 `python3`；安装后必须复查 4 卡可用与 GPU 运算正常，发现 CUDA 依赖被改动即回退。
80b. **差分验证的两种结论必须分开记账**:`differential` 的判定分两类——**矛盾**(两份同实例推导在取值/梯度/符号上不一致)与**不可比**(实例、签名或参数布局不同,无法对齐)。**矛盾**阻断登记与训练;**不可比**记为该提案「规格未被钉到可独立复现」的**局限**,随实现记录留档,准入由**机器检查过的代码证据**(公式参考契约、耦合面、八项一致性)裁决。不得把「不可比」当成「通过」,也不得用它当矛盾来阻断。
80. **N 版本独立推导与差分验证**：入选机制的公式必须由 **3 个互相不可见的推导 Agent** 各自独立给出（SymPy 语句 + 自身 torch 实现），由确定性验证器 `tools/research/rounds/differential.py` 检查：①两两**符号等价**（SymPy，`x**2` 与 `x**2+10` 梯度相同但必须被判为不同函数）；②前向取值一致；③梯度一致；④每个实现与**自身语句**重建的参考一致并双侧 `gradcheck`。三者不 `AGREE` 时，`register_implementation` 直接拒绝并提示先定位分歧；`AGREE` 之后仍要求候选契约与推导等价。**共同模式失效**必须写明：三者读错同一规格会互相一致却仍然是错的，冻结的提案文本才是权威，本验证器是证据不是证明；STE/argmin 仍在 `excluded` 中列理由。
81. **回归测试归属**：参考检查的自测在 `tools/research/stage2/tests/test_reference.py`（含两个反例：符号写错必须靠取值抓出，取值对而梯度错必须靠梯度对照与 gradcheck 抓出），反例实现在同目录 `fake_mechanisms.py`；任何绕过参考检查的「全部通过」结论都不成立。

## 15. 统一验证：两个阶段，不再有独立的 CGV

83. **CGV 不再作为独立框架**，但它负责的**全部检查**保留，并入两条阶段化验证：
   - **阶段一 · 数学验证（注册时执行，廉价）**：冻结规格（磁盘未变 + 两位 Math Reviewer 通过）、三个独立实现一致（前向、梯度、**梯度方向**、各自与自身语句重建的参考 + 双侧 `gradcheck`）、候选契约语句**逐条等于**冻结语句；
   - **阶段二 · 训练集成验证（派发时执行，昂贵但按源码摘要缓存）**：框架八项一致性（冻结常量与启动器、真实入口跑通一 epoch 并导出标识符、标识符本身、四卡不挂死、参考契约、编码器梯度）+ 新增的 `encoder_gradient`：在候选**自己的模型**上只对**重构损失**做一次反向，要求编码器参数真的收到梯度。
84. **`encoder_gradient` 的存在理由**：三份独立的距离函数实现可以完全一致，而真实接线里一个 `detach` 就让编码器收不到任何梯度——数学检查全绿、训练却学不到东西。因此该检查**只**用重构损失（若掺入 commitment 项，即使丢了 straight-through 编码器也会拿到梯度，检查就失效）；负对照已实测：去掉 STE 的树 8/8 ⇒ **0/8** 张量收到梯度，检查判 `FAIL`。
85. **未执行即不通过**：`FAIL`、`INCOMPARABLE`、`NOT_RUN`、`TIMEOUT`、`SKIPPED` 一律阻断；阶段一在 `register_implementation` 生效，阶段二在 `round_admission`（派发前）生效，两者合并为 `compiler.verdict(deep=True)` 的单一判决并留档。

## 16. 临时文件位置（2026-11-10 owner 指令）

86. **禁止在 `/tmp` 下放置任何脚本、数据或文件。** 所有临时脚本、日志、测试产物与中间数据只能放在当前仓库目录内（例如 `results/research/` 下的专门目录），并在用完后自行清理；已在 `/tmp` 留下的历史文件必须删除。任何命令、日志或提交中都不得再出现新的 `/tmp/<name>` 产物路径。

87. **申报公式与代码的自动比对已从准入移除（2026-11-10 owner 指令）。** `check_reference_agreement` 不再进入 `run_all`，round 登记的 `reference_verdict` 门槛与 `stage2_start` 的 `admissible` 分支已删除；其余机器检查（冻结常量、真实入口、SID、四卡、分配规则、编码器梯度）不变。`reference_contract.json` 仍保留为文档与差分证据，但不再阻断训练。

88. **Stage2 机器一致性准入只保留真实入口检查（2026-11-10 owner 指令）。** `run_all` 只执行 `check_entry`（真实入口跑通 1 epoch 并导出 SID）；`check_names`、`check_collectives`、`check_constants`、`check_geometry_contract`、`check_identifier`、`check_four_rank`、`check_assignment_rule`、`check_encoder_gradient` 不再作为准入条件，仅保留为独立诊断函数，不阻断训练。

89. **Stage3 入场门槛：三层码字碰撞率必须 < 0.40（2026-11-10 owner 指令）。** 该门槛由 `tools/genrec-mcp/server.py::tool_stage3_start` 强制执行：读 `item_sids.json` 前三码，`collision = 1 − 不同三元组数 / 商品数`，`collision ≥ 0.40` 直接拒绝并返回实测值，不启动 Stage3。这一条**覆盖**第 19 条里"Stage3 启动不作任何准入检查"的旧措辞；其余诊断（L1/L2 使用率、Hard ER、二级前缀数、码使用率等）仍只记录、不阻断。第 63 条的 `l3_collision`（三码 item-weighted 重复占比）继续照记，报告时必须与门槛所用的 `collision` 并列、带名字，不得互相替代。实测：`TIGER_af3_cost_side_rq` 的 `collision=0.9700248098588685`，已被拒绝进入 Stage3。
