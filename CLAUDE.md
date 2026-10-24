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
19. Stage2 的 ER、HitRate 和稳定性探针等指标仅作诊断记录，不得新增外部指标硬门槛。
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

