# CLAUDE.md — 当前仓库执行规则

> 本文件是仓库唯一的执行规则。**规则编号保持稳定**：源码注释与报告会引用 `§19b0`、`§68`、`§89` 这类编号，
> 因此不重排、不重新编号。已失去代码支撑的条款不进正文，统一列在 **附录 A**,并注明废止原因与可追溯的
> git 提交。正文描述的是 **2026-11-22** 的代码现状。
>
> 目录:§0 现状 · §1 仓库边界 · §2 解释器 · §3 数据与产物 · §3a 基础设施资产位置 · §4 训练预算与机制范围 ·
> §5 准入与验证 · §6 R2 归档 · §7 Git 与交付 · §8 MCP 运维 · §9 GPU 授权 · §10 指标与配对 · §11 工程不变量 ·
> §12 临时文件 · §13 提交预检 · 附录 A 已废止条款

## 0. 现状一览(2026-11-22)

| 项 | 现状 |
|---|---|
| 正式入口 | 训练:`stage2_RQ-VAE/<ARM>/train_rqvae.py` → `tools/research/stage2/launcher.py::launch`(四卡、端口 50202);Stage3:`stage3_T5Train/train_HG-Rec.py`(端口 50201) |
| MCP 服务 | 三个:`stage2`(`tools/stage2/server.py` → `stage2_start/status/stop`)、`stage3`(`tools/stage3/server.py` → `stage3_start/status`)、`genrec`(`tools/genrec-mcp/server.py` → `verification_check`) |
| `tools/research/` | 只保留启动与准入内核(§94),没有轮次、Agent、差分与科学分析模块 |
| 数据 | `results/stage0_build_parquet/`(items 24,587;train 396,958)+ `stage1_GeneEmbedding/output/sentence_t5.npy` |
| 活跃臂 | `stage2_RQ-VAE/TIGER_graphlet_wide_l1_rq`(3000 epoch 完成,collision 0.0521),Stage3 尚未启动 |
| 冻结基线 | `stage2_RQ-VAE/baseline/TIGER_RQ-VAE`(只读,新臂唯一复制来源) |
| **Infra 资产** | **`assets/`**(仓库根):全量截面曲率 `v4-full-sectional-248` = 28,391,948 行三元组表(13 shard)+ 逐商品曲率分布 + 逐节点精确距离直方图 + 节点矩 + 查询索引 + `manifest.json`(62 项)+ `coverage.json`。详见 §3a |

## 1. 仓库边界

1. Stage2 仅使用冻结基线树(`stage2_RQ-VAE/baseline/TIGER_RQ-VAE`)或由其派生、单独命名且通过本轮准入的候选树。
2. Stage3 只使用 `stage3_T5Train/train_HG-Rec.py` 作为正式训练入口。
3. 机制专属测试放在候选树的 `tests/` 与 `tools/research/verification/`(后者保留 `admission.py` 与
   `run_suites.py`);**训练与状态查询**走 `tools/stage2/server.py`、`tools/stage3/server.py`,验证走
   `tools/genrec-mcp/server.py`。
4. 新工具放在 `tools/` 下独立子目录。`tools/research/` 不再接收新模块,它的保留清单见 §94。
5. 禁止引用或恢复已清理的研究控制器、旧机制树、旧结果路径和历史任务队列。

## 2. 解释器与配置

6. Stage0、Stage1 和 Stage3 使用 `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10`。
7. Stage2 和机制专属测试使用 `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9`。
8. MCP Server 使用仅依赖标准库的系统 `python3`。
9. 项目超参数、路径和运行模式只能在源码常量中修改,禁止通过命令行参数或环境变量覆写。
10. 项目训练与研究入口默认无参运行;只有内部测试执行器可以使用必要的执行参数(临时文件位置见 §86)。

## 3. 数据与产物

11. Stage0 仅向 `results/stage0_build_parquet/` 写入 `items.parquet`、`train.parquet`、`valid.parquet` 和 `test.parquet`。
12. Stage1 产物固定放在 `stage1_GeneEmbedding/output/`。
13. Stage2 模型、SID 与导出文件固定放在 `results/stage2_RQ-VAE/<ARM>/`,不得写入 Stage2 源码树。
14. Stage3 日志和模型产物固定放在 `results/stage3_T5Train/<RQVAE_VARIANT>/`,且目录名必须与 Stage2 臂同名。
15. #117 的研究状态与证据仅写入 `results/research/issue117/`,不得污染正式训练目录。
16. 构建商品 SID 的行为监督只允许使用训练划分中合法可见的交互,禁止读取验证或测试阶段的未来行为。
17. 同一商品在不同用户、批次或推理请求中必须使用相同的全局 Semantic ID。

## 3a. 基础设施资产位置(2026-11-24 owner 指令)

17a. **所有 Infra(基础设施)资产统一存放在仓库根的 `assets/`**,不再按 issue 号分目录。当前 `assets/` 的内容:

| 文件/目录 | 内容 |
|---|---|
| `assets/shards/shard_000..012.npz` | **全量截面曲率三元组表**:28,391,948 行(中心 m、邻居对 b<c、`g(b,c)²`、曲率、有效参考数、有效掩码、无效原因、有向 3 节点结构类) |
| `assets/item_summary.npz` | 逐商品曲率分布:三元组总数/有效无效数/正零负计数与比例/均值/方差/极值/分位数/缺失原因 |
| `assets/distance_histogram.npz` | 逐节点**精确距离直方图**(无损中间量;`S1`/`S2`/偏心距/分量大小全部由它重算) |
| `assets/node_moments.npz` | 逐节点 `S1`/`S2`/偏心距/分量 id 与大小/入出互惠度 |
| `assets/index.npz` | 查询索引:item → (shard, start, count)、shard 名、总行数、逐结构类计数、逐中心期望行数 |
| `assets/manifest.json` | **唯一索引**:62 项,每项含路径、用途(`training`/`evaluation`/`description`)、schema、字节、sha256、来源、生成命令;另含输入哈希、源码哈希、commit、耗时与存储成本 |
| `assets/coverage.json` | 覆盖率、采样声明(4 节点 graphlet 仍为抽样)、shard 清单、运行时与存储成本 |
| `assets/train/` | 可见有向图 forward/reverse/undirected + item-id 映射 |
| `assets/README.md` | 内容说明与读取示例 |

17b. `assets/` 与 `results/research/issue248/assets/` 的文件**逐字节一致**(同一 sha256);后者是 canonical 构建位置,
`assets/` 是对外读取位置,manifest 用 `path` 指向 `assets/`、`copy_of` 指向 canonical、`canonical_root`/`copy_root` 标明两者。
17c. 读取入口:`tools/graph_infra/full_assets.py::FullAssets`(按商品 / 按三元组 / 按结构类查询)与
`tools/graph_infra/interfaces_v4.py`(`TrainingView` 拒读 `evaluation` 资产,`EvaluationView` 只服务 strict bank)。
17d. **禁止重建、覆盖或手工改写 `assets/` 中的二进制**;需要新版本时写入新目录并更新 manifest。禁止把资产
只放在临时目录或机器缓存(§86)。
17e. `assets/` 下的二进制(`.npz`/`.npy`)按 §92 不入库,只提交 `README.md`、`manifest.json`、`coverage.json`;
资产实体保存在共享卷上,`assets/manifest.json` 记录每个文件的 sha256 供核验。
17f. 评估资产(strict bank、held-out support、用户划分)留在冻结的 `results/research/infra246/assets/` 中并按哈希引用,
**不复制进 `assets/`**,以保持训练侧与评估侧隔离。

## 4. 训练预算与机制范围

18. Stage2 保持 3000 epochs 的完整训练预算,禁止重新加入基于训练损失或验证指标的提前停止。
19. **机制范围门(Issue #121,文本仍有效)**:新机制必须是**训练只行为信号**与**显式曲线运算**在同一条
    连通计算通路上的耦合,去掉任一侧核心计算即改变;仅行为后缀、仅把欧氏距离换成双曲距离、两个互不相干的
    损失、继承但未使用的双曲代码都判 OUT_OF_SCOPE。提案必须写清:行为信号与算法、曲线运算与作用位置、
    耦合通路与方程、三个开关(行为关/曲线关/both-on)的定义、最小可实现规格。自动范围闸门
    (`rounds/scope.py`)已随轮次删除,改由人工按本节文本判定。
19b0. **源码中禁止绝对路径**:`stage2_RQ-VAE/` 下所有 trainer(含冻结基线)不得出现 `/home/...`、`/fs04/...`、
    `/tmp/...` 字面量,也不得用 `Path(__file__).parents[N]` 数父目录;仓库根由"向上查找含 `stage2_RQ-VAE/`
    的目录"推导,产物写在 `REPO_ROOT/"results"/"stage2_RQ-VAE"/ARM_NAME`,`ARM_NAME` 为单一常量。
19b. **基线固定为冻结树**:每次新机制的**唯一**起点是 `stage2_RQ-VAE/baseline/TIGER_RQ-VAE`,不继承上一轮实现;
    每个候选新建独立副本,只允许改写 `ARM_NAME` 一个常量,其余源码逐文件一致;副本的输出目录不得与基线或
    其他候选冲突。解析实现见 `tools/research/rounds/baseline.py::baseline_path`(源缺失即失败关闭)。
19d0. **Stage2 派发用冻结配方**:一个候选独占 4 张卡(`world_size=4`、全局 batch 4096、seed 42、3000 epoch);
    禁止为了并发把候选钉在单卡上改变全局 batch。并发只在显式标注"不可比"时作为诊断使用。
19d0b. **Stage2 可并行**:多个 Stage2 候选不必串行,按显存与主机 GPU 数决定并发度;该规则只覆盖 Stage2——
    Stage3 仍按全局单例守卫串行,每臂独占整卡。
19d0c. **多个待训机制必须并行**:已过闸的臂一次全部派发,禁止排队一个一个跑;并行不改变上述优化配方。
19g. **实验政策(硬性)**:一个候选**只跑一次**正式训练(冻结预算:3000 epoch 四卡 + 冻结 Stage3 协议,无早停)。
    **不做消融**——三个开关必须存在于代码且有廉价接线检查,但不训练关闭开关的臂;**不做调参**——权重、clip 界、
    温度等常量必须可推导,不允许试值挑选;**不做多种子**——单次运行即定论。行为类诊断指标只记录、不作准入门槛;
    Stage3 `test_R@10` 是最终判断。
19h. **Sinkhorn 均衡分配不可替换**:基线一级分配是 Sinkhorn 均衡分配,机制的创新必须**进入其代价矩阵**,
    分配规则本身仍是基线的;用普通 `argmin` 取代均衡分配即 OUT_OF_SCOPE。实测代价:一轮 `argmin` 变体把
    256 个码用到 42 个、L3 碰撞 0.703(参照 0.065)。三处强制:范围文本、规格闸门
    (`evals.py::assignment-keeps-the-balancing`)、机器检查(`checks.py::check_assignment_rule`,现为诊断)。
20. 全新 Stage2 机制在正式训练前必须通过数学、梯度、DDP 和真实入口四类检查(实现方式见 §32b、§71)。
21. 梯度验证必须使用实际模型与训练 batch 检查参与训练的损失、梯度及参数更新,不得只检查合成张量。

## 5. Stage3、准入与验证

19f. **准入是机器检查的结论**:`stage2_start` 派发前执行 `tools/research/stage2/checks.py::run_all`,当前包含
    `check_entry`(真实入口跑通 1 epoch 并导出 SID)与 `check_behavior_graph`(§89b 行为图契约);`FAIL`、
    缺失或 `NOT_RUN` 一律禁止启动。其余检查(`check_constants`/`check_four_rank`/`check_encoder_gradient`/
    `check_assignment_rule`/`check_reference_agreement` 等)在 §87/§88 之后只作独立诊断。
22. Stage3 固定使用 `NUM_EPOCHS=150`、`EVAL_INTERVAL=5`、`EVAL_START_EPOCH=75`、`EARLY_EVAL_EPOCHS=(25,50,75,100,125)`、`SEED=42`、`BEAM_SIZE=20`、`NO_EVAL=True` 和 `SKIP_TEST=False`。
23. Stage3 只能更换合法的 SID 输入路径、结果路径和变体标识,禁止修改冻结训练超参数或恢复 early-stop。
24. 正式四卡 DDP 的 Stage2 使用端口 `50202`,Stage3 使用端口 `50201`。
25. 当前实验的最终采纳标准为 Stage3 的 `test_R@10 > 0.065`,不得自动套用已关闭研究 Issue 的旧指标门槛。
    (原 §19 的第二条"Stage3 启动不作任何准入检查"已并入 §89:现在只有一个硬门槛,即三码碰撞率。)
26. 每个新候选必须提供数学、梯度、DDP 和真实入口四套可执行测试,不得使用仅返回成功的占位用例。
28. 验证凭据必须绑定冻结数学提案、实际 Stage2 源码哈希、四套测试文件哈希与独立审查结论。
29. 正式训练启动前应重跑四套测试,任一失败、缺失或执行零用例即禁止启动;当前 `stage2_start` 只机器执行
    §19f 的两项,四套测试由人工或 `verification_check` 执行。
30. 正式 Stage2 仅允许调用 `stage2_start`(`tools/stage2/server.py`)或 `train_rqvae.py`(经 `launcher`),
    并在启动后完成一轮真实 Trainer smoke。
31. 旧研究框架的历史 admission 记录不得作为新机制的训练准入凭据。
32. 科研工具不得自行启动 GPU 作业;唯一启动路径是 `launcher.launch`(§68)。
32b. **TDD 是实现与修复的强制形式**:先写一个当前代码会失败的测试(红),再写实现使其通过(绿),最后连同
    候选树既有测试一起跑全量。禁止先写实现再补测试,也禁止同义反复的测试。修复缺陷同样先补复现测试。

## 6b. Stage3 完成后必须归档到 R2(2026-11-08 owner 指令)

每次 Stage3 跑完,必须把该次结果所需的一切上传到 Cloudflare R2,缺一不可:①该次 Stage3 所用的 Stage2 源码
(`train_rqvae.py`、`model/`,打包为 `stage2_source.tar.gz`);②`item_sids.json`;③导出该 SID 的
`rqvae_best.pth`;④完整 Stage3 recall 记录(`test_final.json` 全部键值 + `HG_Rec.log` + `training_metrics.jsonl`)。

实现:`results/research/issue139/upload_stage3.py <ARM>`;凭证从仓库根 `r2_apikey` 读取(已 gitignore,禁止入库);
rclone 通过环境变量配置。目标桶 `genrec`,第一层是 `01`、`02`……编号,每个编号目录下只有四项:
`<机制名>/`(Stage2 源码)、`item_sids.json`、`rqvae_best.pth`、`recall.json`。编号与每个文件的 SHA-256 记在
`results/research/issue139/r2_ledger.json`。**只归档四件齐全的**,缺任一件不上传并在缺口统计里注明原因;
`upload_stage3.py` 默认要求 `test_final.json` 存在(`--allow-incomplete` 才放宽)。`r2_apikey` 的内容不得出现在
任何日志、manifest 或提交中。

## 7. Git 与交付

41. Git 远端只允许使用 `https://github.com/cpxjazz123/HiCurvRec.git`。
42. 当前开发只向 `main` 提交,不建立长期研究分支。
43. 每次迭代必须提交并成功推送到 GitHub 后才可声明完成。
44. 禁止未经明确授权执行 `git push --force` 或 `git push --force-with-lease`。
45. 每次推送前必须核对远端地址,推送后必须确认本地 `main` 与远端 `main` 的 Commit SHA 一致。
46. **二进制产物不入库**:Stage2/Stage3 的 `.pth`/`.pt`/`.npy` 不提交(2026-11-01 起,LFS 配额原因);
    它们由入库的 `item_sids.json` 与一次 Stage2 重跑重建。`.gitattributes` 的 LFS 规则保留,不得再引入新的
    二进制提交。

## 8. MCP 运维(改了 `tools/` 源码后必须做)

52. MCP Server 是 harness(`omp`)拉起的 stdio 子进程,只 import 一次代码;改了 `tools/` 下的源码后,
    已在运行的进程仍是旧代码。当前有三个服务:`genrec`、`stage2`、`stage3`。
53. 重载方式:只对目标 server 子进程发 SIGTERM,harness 会按原命令行重新拉起(实测 3 秒内),新进程载入磁盘代码。
54. 定位与重载:

    ```bash
    OLD=$(pgrep -f 'genrec-mcp/server.py' | head -1)   # 或 tools/stage2|stage3/server.py
    ps -o pid,ppid,cmd -p "$OLD"                        # 父进程必须是 omp
    kill -TERM "$OLD"
    for i in $(seq 10); do sleep 3; NEW=$(pgrep -f 'genrec-mcp/server.py' | head -1); \
      [ -n "$NEW" ] && [ "$NEW" != "$OLD" ] && break; done
    pgrep -f 'genrec-mcp/server.py'
    ```

    **新增或删除 server** 时,单个进程重启不会重新发现 `.omp/mcp.json`;需要在终端执行 `/mcp reload`
    (disconnectAll → 重新发现 → 重绑工具),它不重启 omp、不丢上下文。
55. 禁止杀 `omp` 本体或它的 worker:那会连同本会话的上下文与工具挂载一起丢掉。
56. 重载后必须实测一次 MCP 工具调用(不是只看进程号变了),失败或旧行为即视为未生效。
57. 只有确认重载生效后,才可用 MCP 触发依赖新代码的行为;禁止绕过 MCP 直接调用内部函数推进流程。
58. 重载后 MCP 仍拒绝某请求时,那是规则在 fail-closed(例如源码已变、证据哈希对不上);应先修规则或重新
    生成证据,不得靠直接调用内部函数绕过。

## 9. GPU 任务授权(常设授权)

59. 所有者已给出**常设授权**:任何 GPU 任务(Stage2/Stage3 训练、四套测试里的真实 DDP 检验)可自行启动,
    不必逐次询问。
60. 该授权只替代"人工逐次批准",**不放宽任何机器闸门**:§19f 的机器检查必须通过,四套独立测试应真实执行并
    全部通过,source/test SHA 必须与冻结 manifest 一致,Stage3 冻结协议不得改动。
61. `tools/research/integrations/preflight.py` 错误文案中的 "explicit authorization" 指的也是这一层
    (该文件已随 §94 删除,语义保留)。
62. 授权不等于免除记录:每次 GPU 运行仍必须留下真实 run_id、Worker PID、实际模式、checkpoint 与评估工件;
    不得以"已获授权"为由伪造运行结果。

## 10. 指标记录与配对

63. **每个新 Stage2 候选在导出 `item_sids.json` 后、报告质量结论或启动 Stage3 前,必须持久化一份分层指标
    记录**:L1 Hard ER、L1 正/负前缀命中率(仅诊断)、不同二级前缀数、不同三码组合数、L3 Collision、
    L3 Refinement Gain(bits)、Conditional G3,并记录 SID 路径与 SHA-256、训练/验证输入路径与 SHA-256、口径、
    有效样本数、候选身份与输出位置;报告写在 `results/stage2_RQ-VAE/<ARM>/stage2_metrics/<sid_sha256>.json`,
    不得覆盖历史。记录必须在读取 Stage3 成绩前产生。**注意**:自动记录脚本
    (`tools/research/science/sid_layer_analysis.py`)已随 §94 删除,当前需手工复算或重写脚本;失败时标记缺失,
    不得虚构或填 0。
64. **科研审查的准入标准**:只审查冻结提案的数学类型与可定义域、同容量对照是否只差度量、是否使用验证/测试/
    未来行为或预测时用户上下文、损失与目标分布能否唯一复算、反证能否机械判定;提案必须同时注册 Stage3 行为
    检验(指标、方向、mixed/null 分类)。按 §19g 不做多种子实验。Stage2 只构建 SID、从不排序,不得以"Stage2
    指标无法自证 next-item 收益"阻断晋级;行为收益由 Stage3 `test_R@10` 判定。
64b. **SID–recall 配对的准入判据**:结论只在目录同名时成立(唯一等价形式是把 `/` 写成 `__`,大小写一致),
    否则标 `REJECTED_NAME` 并排除。配对表必须由可复算的审计脚本生成(历史脚本与表已不在仓库,新统计需重写并保留产物);关联键只能是
    `sid_sha256`,禁止按 variant 名关联;统计必须同时在 `VERIFIED` 子集与全表上重算并并列报告,并按 Stage2
    父系列报告来源构成(单系列 ≥90% 时只能支持同系列结论)。recall 引入 commit 与时间戳只作风险提示分层
    (`SAME_COMMIT`/`COMMITS_DIFFER`/`NOT_TRACKED`/`SID_AFTER_RECALL`),不是准入条件。

## 11. 工程不变量归框架(候选只改机制)

66. **候选只拥有机制**:可修改行为统计、双曲操作、分配规则、残差更新与新增损失,以及针对这些计算的测试;
    路径、结果目录、日志、SID 导出、GPU/DDP 启动、端口、冻结超参由框架拥有,候选不得自行实现或覆盖。
67. **路径与结果由 `tools/research/stage2/contract.py` 唯一决定**:仓库根用无歧义标记;一次运行绑定
    `results/stage2_RQ-VAE/_runs/<run_id>/<arm>/{src,result}` 的不可变源码快照与独立结果目录,新修订用新目录。
68. **启动只有一条路径**:四卡训练必须调用 `tools/research/stage2/launcher.py::launch`(端口 50202、
    `world_size=4`、cwd 为结果目录);候选源码出现 `--nproc_per_node=` 或 `--master_port=` 即不合格。
71. **准入前执行机器一致性检查**(`tools/research/stage2/checks.py`,按源码摘要缓存)。当前闸门只有
    `check_entry` 与 `check_behavior_graph`(§19f/§88/§89b),其余为诊断;`NOT_RUN` 不是通过。
    **每修好一种工程故障,必须留下一个能重现它的回归测试**(`tools/research/stage2/tests/`)。
72. **状态轮询的间隔上限**:监督长任务时单次等待不得超过 60 秒,拆成多次并每次报告进度;
    禁止把 `sleep N; 检查` 放后台作业再取回。轮询用 status 接口或直接读日志/产物文件。
73. **每个工程问题都必须留下对应的测试文件**:凡出现过的工程故障(集合通信、路径与导出、常量引用、DDP 启动、
    缓存与状态、MCP 重载等),必须新增一个能重现该故障形态的独立测试(含对检测器本身的自测),与修复同一次
    提交入库;已有测试覆盖同一形态时可扩展该文件。

## 12. 临时文件位置(2026-11-10 owner 指令)

86. **禁止在 `/tmp` 下放置任何脚本、数据或文件。** 临时脚本、日志、测试产物与中间数据只能放在仓库目录内
    (例如 `results/research/` 下的专门目录),用完自行清理;任何命令、日志或提交中都不得出现新的
    `/tmp/<name>` 产物路径。

## 13. 提交预检与大文件(2026-11-18)

91. **任何 `commit.draft()` 之前必须先运行 `python3 tools/commit_guard.py`。** draft 会构造 `HEAD → 工作区`
    的完整 diff(含未跟踪文件),`take()` 在 draft 之后才生效。guard 返回非零时,必须先把大文件加入
    `.gitignore`、移出工作区,或对确属证据的路径显式使用 `--max-paths/--max-lines` 并说明理由;不得忽略失败。
92. **`results/research/` 下的生成数据默认不入库**:`*.csv`、`*.tsv`、`*.micro`、`*.npy`、`*.npz`、`*.pth`、
    `*.pt`、`*.bin`、`*.h5`、`*.out`、`*.in` 已被整体忽略;需要入库的小型证据用 `!` 显式放行,
    大体积产物按 §6b 去 R2。
93. **不要用 `d.show()` 查看大 draft**:先 `d.print()` 看摘要,确认路径后再对小子集 `show()`。

## 14. Stage3 入场与行为图

87. **申报公式与代码的自动比对已从准入移除(2026-11-10)**:`check_reference_agreement` 不进 `run_all`;
    `reference_contract.json` 保留为文档,不阻断训练。
88. **Stage2 机器准入只保留真实入口与行为图**:`run_all` 只执行 `check_entry` 与 `check_behavior_graph`(§89b);
    其余检查(冻结常量、四卡、编码器梯度、分配规则、命名、集合通信)保留为独立诊断,不阻断训练。

89. **Stage3 入场:硬门槛只有三层码字碰撞率 < 0.4;量化误差与有效秩是强制记录并告警的诊断值**
    (2026-11-19 指令;2026-11-21 owner 把硬门槛从 0.10 放宽到 0.4)。门槛由
    `tools/stage3/server.py` 的 `stage3_start` 强制执行,逻辑在 `tools/research/stage2/sid_quality.py`
    (以 `genrec_env_v2` 子进程运行,stdout 最后一行 JSON 为判决):**硬条件**
    `collision = 1 − 不同三码三元组数 / 商品数 < 0.4`(只用前三码,第四项是导出时的碰撞计数器);缺 checkpoint、
    缺 `stage1_GeneEmbedding/output/sentence_t5.npy`、或码本宽度与编码器输出不符时仍然拒绝。
    **诊断值**(越界只告警):`quant_error = mean ‖z − Σ 选中码字‖`(`z` 为 `encoder.mlp.*` **含 bias** 的前向输出;
    码字取 `rq.vq_layers.*` 或 `rq.levels.*` 前三级),`effective_rank = (Σλ)² / Σλ²`。为什么只硬卡碰撞率:
    实测 `mixed_rq` 家族的 quant_error 0.028–0.455、effective_rank 1.84–22.86,把向量窗口当硬门槛会让混合
    曲率整条线无法进入 Stage3。研究价值由 Stage3 `test_R@10 > 0.065` 判定;§63 的 `l3_collision` 照记,
    与门槛同一口径并列报告。
89b. **统一无向行为图**:只维护一份无向商品行为图 `results/research/behavior_graph/`,四个文件即契约
    (`edges.parquet`、`graph.npz`、`node_features.npy`、`metadata.json`);AF3、graphlet、结构保真度等一切
    结构监督都必须读它,禁止各自重建。方向不是建模变量:有序序列保留在 `train.parquet`,确需有向图时从它重建。
    **强制执行**:`tools/research/stage2/graph_contract.py::check_behavior_graph` 已并入 `checks.run_all`——
    候选树自建邻接/CSR 即 FAIL,读取非规范图即 FAIL,文件与 metadata 的 sha256/size 不符即 FAIL;
    完全不涉及图的树记 `NOT_APPLICABLE`。
90. **`stage2_RQ-VAE/` 顶层只允许保留一个活跃 Stage2 目录**,外加两个容器目录:
    - `stage2_RQ-VAE/baseline/`:冻结基线。新臂唯一复制来源是 `baseline/TIGER_RQ-VAE`;
      `baseline/TIGER_BEHAVIOR_SAMPLE_RQ-VAE` 只作历史样例。二者只读。
    - `stage2_RQ-VAE/_archive/`:历史树归档,不参与训练或准入。
    活跃臂必须是 `stage2_RQ-VAE/<ARM>/` 的直接子目录(`stage2_start`/`stage2_status` 只接受这一层);
    归档用 `git mv`,移动后同步所有引用该树路径的常量、提示词与文档。
## 15. tools/research 的保留范围(2026-11-22 owner 指令)

94. **`tools/research/` 只保留 Stage2/Stage3 启动、状态与准入门槛所需的代码。** owner 要求
    "`tools/research` 可以删掉,但不要动 stage2/stage3 的训练启动"。保留:`stage2/{contract,launcher,checks,
    gpu_alloc,graph_contract,export_policy,export_check,sid_quality,ddp_probe,collapse_watch}.py`、
    `stage2/anticollapse/{floor,artifact,patch_tree}.py`、`rounds/baseline.py`、`stage2/tests/`
    的两个诊断辅助、`verification/{admission,run_suites}.py`、`settings.py`。其余 153 个文件已删除
    (rounds 其余、science、geometry、agents、core、mcp、pipeline_v2、integrations、tests、
    stage2 的诊断模块与其 tests)。**不变**:`stage2_start/status/stop`、`stage3_start/status`、训练入口。
    **随之失效**:全部 `research_round_*` 工具、轮次工具面、自动范围/差分/参考实现检查(见附录 A)。

---

## 附录 A:已废止条款

保留编号以便对照历史引用;**全文见 git 历史**(`9ba91c7b^`、`f47caa01^` 等被引用提交之前的版本)。

| 编号 | 原主题 | 废止原因 |
|---|---|---|
| 19c | Issue #121 四类停止分账 | 轮次与 Agent 框架已删除 |
| 19c0 | 一次探索只产出一个机制 | 同上(选择规则依赖 rounds) |
| 19c1 | Issue #122 排序与选题 | 同上 |
| 19c2 | Issue #123 评分前证据探针 | 同上(probes 已删除) |
| 19d | 轮次 MCP 工具面 | `research_round_*` 实现已删除 |
| 19e | 轮次代理分工 | rounds/agents 已删除 |
| 19g 各 Issue 例外段 | #138–#147 的匹配臂例外 | 相关 Issue 已关闭,臂已归档 |
| 27 | 验证角色必须有 OpenCode 执行记录 | OpenCode 路径已删除 |
| 33 | `core/` 持久 DAG | 模块已删除 |
| 34 | `agents/` OpenAI Agents SDK | 模块已删除 |
| 35 | `science/` 行为验证 | 模块已删除 |
| 36 | `workflows/` 冻结 DATA/LITERATURE | 目录已删除(2026-11-22) |
| 37 | Reviewer prompt 一致性 | 审查框架已删除 |
| 38 | 真实 Agent 进程证据 | Agent 框架已删除 |
| 39 | `integrations/` 只用 MCP 接口 | 模块已删除 |
| 40 | Agent 真实进程默认禁用 | 同上 |
| 47·48·49·50·51 | MCP 科研 DAG 调度(`research_*`) | 工具已删除 |
| 65 | Issue #117 机器错误持续修复 | run_id/Worker/DAG 已删除 |
| 69 | Agent 输出契约 | Agent 框架已删除 |
| 70 | 三类失败只恢复出错步骤 | 同上 |
| 74·75·76·77·78·79·80·80b·81·82 | 公式参考实现与差分验证 | `reference.py`、`differential.py`、`compiler.py` 已删除;`reference_contract.json` 仅作文档(§87) |
| 83·84·85 | 统一验证两阶段与 `compiler.verdict` | 同上,无执行器 |

**保留但需人工**:§63(分层指标,脚本已删)、§29(四套测试重跑)、§26/§28(候选自带测试与哈希绑定)。
