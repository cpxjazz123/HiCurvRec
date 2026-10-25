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
19. **Issue #121 的联合耦合范围门**：任何新机制必须是**训练只行为信号**与**显式曲线运算**在**同一条连通计算通路**上的耦合，去掉任一侧核心计算即改变；仅行为后缀、仅把欧氏距离换成双曲距离、两个互不相干的损失、以及继承但未使用的双曲代码都判 OUT_OF_SCOPE。范围判定检查的是 Agent 写下的**内容**（行为信号与算法、曲线运算与作用位置、耦合通路与方程、三个消融、最小可实现规格），不是布尔标记；范围裁决以提案自身内容为准，单个评分者的主观标签不得单独决定去留，分歧必须留痕。契约硬编码在 `tools/research/rounds/agents.py` 的 `RESEARCH_OBJECTIVE` 与 `SCOPE_CONTRACT`，并传播到全部三个构思、三个评分、实现规格、三个验证与修复提示。
19b0. **源码中禁止绝对路径**：`stage2_RQ-VAE/` 下的所有 trainer（含冻结基线）不得出现 `/home/...`、`/fs04/...`、`/tmp/...` 这类绝对路径字面量，也不得用 `Path(__file__).parents[N]` 数父目录；仓库根必须由「向上查找含 `stage2_RQ-VAE/` 的目录」推导，产物写在 `REPO_ROOT/"results"/"stage2_RQ-VAE"/ARM_NAME`，`ARM_NAME` 为单一常量。准入预检在**真正运行的位置**（`stage2_RQ-VAE/TIGER_<idea>/`）求值，出现绝对路径或深度依赖即拒绝。（Issue #121）
19b. **Issue #121 的唯一基线**：新机制的**唯一**允许起点是 `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_RQ-VAE`；每个选中想法必须从它**新建独立副本**（禁止从别的候选、变异工作树或历史实验树起步），并记录基线解析路径与基线/副本源码摘要。基线只读：源缺失或不可读即失败关闭（不得静默换基线），候选的输出路径若与基线结果目录或其他候选冲突则在训练前被拒绝。消融对照同样从该基线派生。
19c. **Issue #121 的四类停止必须分开记账**：`AGENT_INFRA_ERROR`（Agent 崩溃/超时/答案不可用）只重试、**不得**因此 BLOCK 机制；`IMPLEMENTATION_NEEDS_FIX` 是真实缺陷，交专职修复 Agent；`RESOURCE_BLOCKED` 是实测不可行（冻结预算外推超时），保留想法走优化路线；`STAGE2_3_FAILED` 是训练或评估失败。单个候选的修复轮次上限为 5 轮，到顶即带证据记录，不再消耗 Agent。
19c0. **一次探索只产出一个机制**：每轮构思（3 个 Agent、3-9 个想法）经 3 个评分者全评后，选择规则**只保留唯一最好的合格机制**（`max_selected = 1`、`keep_single_best`），其余想法以拒绝理由留档；范围不合格的想法永不入选，也不能作为 fallback。该机制必须走完 **实现 → 审查/修复 → Stage2（冻结四卡配方）→ Stage3** 的完整链路，产出一个真实的 `test_R@10`，才算这一轮的产出。不得把一轮当成候选批，也不得在未训练前宣称机制成功。该机制的消融对照（行为关、曲线关）是可选控制实验，单独记账、不计作机制产出。（Issue #121）
19d0. **Stage2 派发必须用冻结配方形态**：轮次派发默认 `STAGE2_LAUNCH_STRATEGY="ddp4"`——一个候选独占 4 张卡（`world_size=4`、全局 batch 4096、0.16 秒/epoch），同一时刻只跑一个候选。**禁止**为了并发把候选钉在单卡上（全局 batch 变成 1024，是另一套优化配方，与冻结基线不可比）；需要并发只能在明确标注「不可比」的前提下作为探索性诊断使用。（Issue #121）
19d. **Issue #121 的轮次工具面**：每个 phase 都是独立 MCP 工具（`research_round_ideate` / `_score` / `_implement` / `_verify`，加 `_start` / `_run` / `_status` / `_results` / `_launch` / `_collect`），phase 幂等、每轮一把运行锁；轮次按 issue 归档（`results/research/issue121/rounds/<round_id>/`），候选树在 `results/research/issue121/trees/<round_id>/<idea_id>/`。并发 Stage2 每卡最多 2 个作业、每作业独立 GPU 与输出目录，技术有效即自动进 Stage3。
19. Stage2→Stage3 的准入只做**技术有效性**判定（Issue #120）：候选 SID 文件可解析、商品映射为规范十进制 id 空间、token 宽度与取值合法、完整 SID（含确定性消歧 token）唯一，即自动进入 Stage 3。**不得**把 Hard-L1 ER、二级前缀数、L3 Collision、码使用率、冗余度等任何质量指标作为准入门槛；这些只作为诊断被测量与记录。Resample 的 ER=16.38953942232631 与 L2=16145 仅为**历史参照**值，不是分界值。Stage3 test_R@10 > 0.065 仍是最终成功条件；实验登记保留所有候选（含低召回与失败分支），不得只留最优。
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

## 6. 原生科研工具

33. `tools/research/core/` 只管理唯一持久 DAG、原子租约和不可变证据。
34. `tools/research/agents/` 只通过真正的 OpenCode CLI 子进程执行 Agent，禁止恢复旧代理调度框架。
35. `tools/research/science/` 必须以训练用户行为数据验证业务问题、数学候选和审查身份。
36. `tools/research/workflows/` 必须先冻结 DATA 与 LITERATURE 证据，再允许 GEOMETRY 正式提案。
37. Reviewer 的 Prompt、当前候选 ID、公式与文件哈希不一致时，审查必须被判为无效。
38. 真实 Agent 必须保留可核验的进程、输入、输出和日志证据，不能将模拟执行当成已完成研究。
39. `tools/research/integrations/` 只使用现有 MCP 训练接口，不得自行创建第二套训练调度器。
40. `tools/research/agents/` 的真实进程默认禁用，必须在隔离和测试验收后才允许开启。

## 7. Git 与交付

41. Git 远端只允许使用 `https://github.com/cpxjazz123/HiCurvRec.git`。
42. 当前开发只向 `main` 提交，不建立长期研究分支。
43. 每次迭代必须提交并成功推送到 GitHub 后才可声明完成。
44. 禁止未经明确授权执行 `git push --force` 或 `git push --force-with-lease`。
45. 每次推送前必须核对远端地址，推送后必须确认本地 `main` 与远端 `main` 的 Commit SHA 一致。
46. `.pth` 与 `.npy` 文件必须按仓库 `.gitattributes` 使用 Git LFS。

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


63. **每个新 Stage2 候选在完成训练并导出 item_sids.json 后、报告质量结论或尝试 Stage3 前，必须对该 SID 单独计算并持久化一份完整的分层指标记录**（使用 `tools/research/science/sid_layer_analysis.py::record_single_sid`）。必记的实际数值包括：L1 Hard ER、L1 正/负样本前缀命中率（HitRate，仅诊断）、不同的二级 SID 前缀数、不同的三级 SID 组合数、L3 Collision（三级 SID 重复商品占比）、L3 Refinement Gain（bits）、Conditional G3（平均 log2 概率比），并记录 Resample 历史基线与诊断对比（无晋级门槛）。必须同时记录实际 SID 路径和 SHA-256、训练/验证输入路径及 SHA-256、指标口径、有效验证样本数、实验/候选身份和输出文件位置。记录必须在读取该候选的 Stage3 测试成绩前产生，固定使用训练行为与冻结独立验证样本，禁止借测试结果调参或选最优运行；任一指标无法计算时明确标记缺失/失败及原因，不得虚构、填写 0 或称作 PASS。所有 L3 指标为**强制记录的诊断值**，不新增 L3 或 HitRate 的 Stage3 准入硬门槛；准入仍只按第 19 条的放宽后 ER、L2 双硬门槛及现有完整性校验执行。分层报告保存在该 Stage2 结果目录下 `stage2_metrics/<sid_sha256>.json`，不得覆盖其他 SID 的历史记录。


64. **科研审查的准入标准**：独立 REVIEW 只审查冻结提案的数学类型与可定义域、同容量对照是否只差度量、是否使用验证/测试/未来行为或预测时用户上下文、损失与目标分布能否唯一复算、以及反证是否可机械判定；提案必须同时注册 Stage3 行为检验（指标、方向、最小效应、种子合并与 mixed/null 分类）。Stage2 只构建 SID、从不排序，因此**不得**以「Stage2 侧指标无法自证 next-item 收益」为由阻断晋级；行为收益由 Stage3 `test_R@10` 判定，Stage2 指标（Hard-L1 ER、二级前缀、L3 三统计）只作为前导指标。

## 12. 机器失败持续修复

65. **Issue #117 的可修复机器错误不允许用“尝试达到 3 次/5 次”等固定次数淘汰研究任务。** Agent 输出格式/JSON、候选 schema、数学符号实现、代码与独立测试失败必须保留每次真实执行凭据和错误，反馈具体问题，在同一 `run_id` 中持续修正、复验直到合规或用户取消；不得伪造测试成功。科学审查中的 `MATERIAL` 可按可复核证据退回 GEOMETRY，以新版本继续审查，不设固定轮次上限；`FATAL`/`PARKED` 属于真实科学否决/暂停，应记录并阻止该候选晋级，不能重复要求同一 Reviewer 改票。任何循环仍必须遵守用户取消、算力成本、Worker 租约、隔离、正式训练准入和原 Stage2/Stage3 冻结预算；不保证客观上不成立的机制必然能够成功。
