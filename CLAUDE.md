# CLAUDE.md — 当前仓库执行规则

> 以下每个编号项均是一条独立规则，以当前 `main` 的实际源码和路径为准。

## 1. 仓库边界

1. Stage2 只使用 `stage2_RQ-VAE/TIGER_RQ-VAE/` 和 `stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/` 两棵现存源码树。
2. Stage3 只使用 `stage3_T5Train/train_HG-Rec.py` 作为正式训练入口。
3. 机制验证使用 `tools/cgv/`，训练与状态查询使用 `tools/genrec-mcp/server.py`。
4. 新科研工具统一放在 `tools/research/`，模块职责以该目录 `README.md` 为准。
5. 禁止引用或恢复已清理的研究控制器、旧机制树、旧结果路径和历史任务队列。

## 2. 解释器与配置

6. Stage0、Stage1 和 Stage3 使用 `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10`。
7. Stage2 和 CGV 使用 `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9`。
8. MCP Server 使用仅依赖标准库的系统 `python3`。
9. 项目超参数、路径和运行模式只能在源码常量中修改，禁止通过命令行参数或环境变量覆写。
10. 项目训练与研究入口默认无参运行，只有现存 CGV 校验工具的契约输入及 `/tmp/` 参考脚本可使用规定的参数。

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
20. Stage2 机制在正式训练前必须通过真实梯度通路检查和 CGV 验证。
21. 梯度检查至少使用一个真实 checkpoint 和 batch 验证总损失的计算图以及所有声明为非零的机制梯度路径。
22. Stage3 固定使用 `NUM_EPOCHS=150`、`EVAL_INTERVAL=5`、`EVAL_START_EPOCH=75`、`EARLY_EVAL_EPOCHS=(25,50,75,100,125)`、`SEED=42`、`BEAM_SIZE=20`、`NO_EVAL=True` 和 `SKIP_TEST=False`。
23. Stage3 只能更换合法的 SID 输入路径、结果路径和变体标识，禁止修改冻结训练超参数或恢复 early-stop。
24. 正式四卡 DDP 的 Stage2 使用端口 `50202`，Stage3 使用端口 `50201`。
25. 当前实验的最终采纳标准为 Stage3 的 `test_R@10 > 0.065`，不得自动套用已关闭研究 Issue 的旧指标门槛。

## 5. CGV 与训练权限

26. 每次 `cgv_check` 或 `stage2_start` 都必须传入真实、完整的七组机制声明：`mechanism/model/forward_contract/gradient_contract/update_contract/distributed_contract/runtime_contract`。
27. CGV 的 G1–G6 分别验证前向契约、梯度传播、梯度数学、参数更新、分布式一致性和真实训练入口。
28. CGV 状态只允许使用 `passed`、`failed`、`not_applicable` 和 `insufficient_evidence`，后两者不得伪装为通过。
29. 数学 Oracle 必须独立于被测机制实现，G3 公式模式必须绑定独立 Oracle。
30. 正式 Stage2 必须通过现存 MCP 的 `stage2_start` 入口重新执行 CGV，禁止直接启动训练器绕过验证。
31. 禁止通过改写机制声明、验证阈值或门禁代码掩盖实测失败。
32. `tools/research/` 不得擅自启动 Stage2、Stage3 或 GPU 作业。

## 6. 原生科研工具

33. `tools/research/core/` 只管理唯一持久 DAG、原子租约和不可变证据。
34. `tools/research/agents/` 只通过真正的 OpenCode CLI 子进程执行 Agent，禁止恢复旧代理调度框架。
35. `tools/research/science/` 必须以训练用户行为数据验证业务问题、数学候选和审查身份。
36. `tools/research/workflows/` 必须先冻结 DATA 与 LITERATURE 证据，再允许 GEOMETRY 正式提案。
37. Reviewer 的 Prompt、当前候选 ID、公式与文件哈希不一致时，审查必须被判为无效。
38. 真实 Agent 必须保留可核验的进程、输入、输出和日志证据，不能将模拟执行当成已完成研究。
39. `tools/research/integrations/` 只对接现有接口，不得创建第二套 CGV 或 GPU 调度器。
40. `tools/research/` 的真实 Agent 默认禁用，只有完成离线测试、权限隔离和验证器验收后才可启用。

## 7. Git 与交付

41. Git 远端只允许使用 `https://github.com/cpxjazz123/HiCurvRec.git`。
42. 当前开发只向 `main` 提交，不建立长期研究分支。
43. 每次迭代必须提交并成功推送到 GitHub 后才可声明完成。
44. 禁止未经明确授权执行 `git push --force` 或 `git push --force-with-lease`。
45. 每次推送前必须核对远端地址，推送后必须确认本地 `main` 与远端 `main` 的 Commit SHA 一致。
46. `.pth` 与 `.npy` 文件必须按仓库 `.gitattributes` 使用 Git LFS。
