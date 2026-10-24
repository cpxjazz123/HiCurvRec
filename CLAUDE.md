# CLAUDE.md — 当前仓库执行规则

> 以下每个编号项均是一条独立规则，以当前 `main` 的实际源码和路径为准。

## 1. 仓库边界

1. Stage2 只使用 `stage2_RQ-VAE/TIGER_RQ-VAE/` 和 `stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/` 两棵现存源码树。
2. Stage3 只使用 `stage3_T5Train/train_HG-Rec.py` 作为正式训练入口。
3. 机制专属测试由 `tools/research/verification/` 执行，正式训练与状态查询仍通过 `tools/genrec-mcp/server.py`。
4. 新科研工具统一放在 `tools/research/`，模块职责以该目录 `README.md` 为准。
5. 禁止引用或恢复已清理的研究控制器、旧机制树、旧结果路径和历史任务队列。

## 2. 解释器与配置

6. Stage0、Stage1 和 Stage3 使用 `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10`。
7. Stage2 和 CGV 使用 `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9`。
8. MCP Server 使用仅依赖标准库的系统 `python3`。
9. 项目超参数、路径和运行模式只能在源码常量中修改，禁止通过命令行参数或环境变量覆写。
10. 项目训练与研究入口默认无参运行，只有内部测试执行器及 `/tmp/` 参考脚本可使用必要的执行参数。

## 3. 数据与产物

11. Stage0 仅向 `results/stage0_build_parquet/` 写入 `items.parquet`、`train.parquet`、`valid.parquet` 和 `test.parquet`。
12. Stage1 产物固定放在 `stage1_GeneEmbedding/output/`。
13. Stage2 模型、SID 与导出文件固定放在 `results/stage2_RQ-VAE/<ARM>/`，不得写入 Stage2 源码树。
14. Stage3 日志和模型产物固定放在 `results/stage3_T5Train/<RQVAE_VARIANT>/`。
15. #116 的研究状态与证据仅写入 `results/research/issue116/`，不得污染正式训练目录。
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
31. 旧 CGV 声明和历史 admission 记录不得作为新机制的训练准入凭据。
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
