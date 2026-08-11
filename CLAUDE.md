# CLAUDE.md

> HG-Rec (Hyperbolic RQ-VAE + Differential-Length Codebook + T5) 流水线复现工作目录, 基线 Task #84 valid R@10=0.1267 / test R@10=0.1024 (Musical_Instruments 9922 items).

---

## R 规则

**R1** — 默认 `genrec_env`; KG 任务用 `deepke`; base anaconda 仅 zero-dep grep + MiniMax API。

**R2** — 禁止 fallback 逻辑 (默认值/回退/降级); 预期内缺失返回 None/空值, 预期外失败直接 raise。

**R4** — 修改 Python 脚本后必须立即 `python3 -m py_compile` 验证语法 (文档例外)。

**R5** — 任务硬约束 = 基线 HG-Rec Task #84 (valid R@10=0.1267, test R@10=0.1024), 仅 RQ-VAE 量化, Musical_Instruments (9922 items), 4 阶段流水线, seed=42。

**R7** — 启动新实验前必须 `nvidia-smi` 核对 (util<10%, mem<5GB), 选完全空闲 GPU。

**R11** — AI 自主决策, 禁"等用户拍板"/"是否启动?"阻塞话术; 兜底顺序: CLAUDE.md > 上游 default > 论文 > 简单实用。

**R12** — 训练固定阶段强制存 checkpoint (epoch 末), 删旧 ckpt, 写 `_TRAINING_PID`。

**R13** — 禁 `EnterWorktree` + git worktree, 代码改共享 checkout, 临时文件用 `$CLAUDE_JOB_DIR/tmp`。

**R17** — commit message 必含 `Gate <N> FAIL/PASS` + 失败原因; 前 Gate FAIL → 后 Gate STOP。

**R18** — 新 issue 必须跟历史做 4 维度对比 (D1 spec 摘录 / D2 实施核心 / D3 Gate 1 失败机制 / D4 引用文献), 任一不同 → 必须实验, 禁"路径同构" NO-GO。

**R19** — AI = 激进 owner, precheck PASS 立即启动 GPU 训练, 跨 issue 并行 (一卡一实验)。

**R20** — commit + issue comment 必须详细回答 4 Gate (≥3-5 行/Gate); close issue 前必发 comment。

**R21** — commit hash 必须明示 (禁 "pending"/"TBD"/"TODO"); comment 必须在 commit + push 之后发。

**R22** — tick 必须实际推进 (启动 precheck/Gate1/训练/评估/修复/验证 之一); 4 卡全占 → 换 GPU/nohup/缩减规模。

**R23** — tick 扫活跃训练, 7 信号任一触发立即终止 (val_R@10=0 跨 ≥2 ckpt / loss 不下降 / val loss 反向 / wrapper broken / ckpt 不存 / NaN-Inf / GPU 100% loss 不变), kill -9 + NO-GO + commit + push + close。

**R24** — tick 检查 in_progress 任务是否真在执行 (有 PID + file mtime 更新), 无活跃行为 → 立即决策 + 执行。

**R25** — issue 里的"方向A"/"方向B"指的就是 taskA/ + taskB/ (代码位置/入口/产物路径), 禁解读为 TIGER/LETTER/RecBole/phonism 等其他 lineage。

**R26** — tick 必须实际推进 issue (启动 precheck/Gate1/训练/评估/修复/验证 之一), 禁仅规划/状态/commit + close。

**R27** — tick 必须 `python3 <existing_script>.py` 启动脚本产生 PID 活跃, 禁仅规划/读/写脚本/改 verdict。

**R28** — 任何决策/修复方案/启动判断禁"等用户拍板"/"是否启动?"/A-B 选项/"你选"/"请告诉我"/"要不要"话术, 必须直接给出推荐方案 + 立即执行 (唯一例外: 不可逆操作)。

**R29** — tick 输出必须 (a) ≥1 个 R26 动作 + (b) ≥1 个 R27 动作, 末尾明示"已执行 X" + 实际产物 (verdict 路径/commit hash/PID)。

**R30** — 脚本超参硬编码进模块, 禁 `os.environ.get` 读超参, 禁 wrapper 传 num_epochs/batch_size/lr/seed 等数值超参 (路径参数可传)。

**R31** — 每个 stage 目录只允许一个主脚本, 禁 fork `_v2.py/_v8.py` 多版本并存, 历史实验变体从 git 历史恢复。

**R32** — 运行脚本必须直接 `python3` 执行, 禁写 `.sh` 包装启动, GPU 选择用 `CUDA_VISIBLE_DEVICES=0 python3 -u ...` 内联 (唯一例外: DDP 多卡 `torchrun`)。

**R33** — 任务完成 verdict 写到 `tasks/<task_dir>/issue<NN>_verdict.json` (本任务自己的目录), 不集中放 `verdicts/`; 文件名格式 `issue<NN>_verdict.json`, NN = gitlab issue 编号; 区分 stage3/verdict.json (产物级) 与 issue 闭环 verdict (任务级)。

**R34** — 每次迭代新版本前, 必须在 `tasks/` 下新建 `Issue<NN>_<任务名>/` 目录 (NN = gitlab issue 编号, 禁止 `vN_xxx_from_vN-1` 命名; 历史 v* 目录不追溯), 必须有且仅有 stage1/2/3/4_beam20.py 四个脚本。

**R35** — 评估强约束: 只使用单 checkpoint + `beam_search=20`, 禁 Borda Rank Fusion / 任何 ensemble 多 ckpt 融合。

**R36** — 方法路径强约束: 禁止通过调参形式 (LR/dropout/label_smoothing/weight_decay sweep) 提升指标; 必须通过改善曲率框架 (Stage 2 κ 学习 / Stage 3 κ frozen→learnable / 新曲率正则项 / Poincaré-Minkowski-Lorentz 曲率机制变更)。

**R37** — 版本回滚硬约束: 新版本 (vN) test_R@10 < 上一版本 (vN-1) → 立即终止 lineage, 必须回到 vN-1 重新创新, 禁止在比旧版本差的版本上进行二次创新。

**R38** — 训练期早停回退硬约束: vN 训练期 valid_R@10 / loss / 收敛速度明确比 vN-1 差 (连续 ≥30 epoch 平台 / loss 高 ≥0.05 / NaN-Inf / loss 反向) → 立即 kill + 写 R38 决策行 + 触发 R37 回退。

**R39** — Open Issue 立即实现硬约束: 任何 open issue 一旦被 loop tick 发现, 必须立即实施 (R19+R26+R27 联动), 严禁等待用户评论授权。

**R40** — 四 Stage 全量运行硬约束: 每个任务目录必须完整实时运行 stage1 → stage2 → stage3 → stage4, 无论创新点位于哪个 stage; stage n+1 输入必须唯一来自 stage n 实时运行产物 (落 tasks/<task_dir>/), 禁引用任何外部脚本/外部产物; stage n 产物缺失 → stage n+1 raise FileNotFoundError 禁启动 (issue 显式豁免某 stage 除外)。

**R41** — 所有 Stage3 / Stage2 训练脚本的 `EARLY_STOP` 统一硬编码为 20 (历史 v121 用 30 不追溯, 仅本规则生效后新任务生效)。

**R42** — Stage2 / Stage3 必须用 `torchrun --nproc_per_node=4` DDP 4 卡运行, 禁单卡 (world_size=1); 唯一例外: nvidia-smi 显示 GPU 1/2/3 都被占时允许单卡, 但 verdict_r37.json 必须显式记录 (R7+R42 联动)。

**R43** — 禁止脚本使用 CLI 传入数值超参; 脚本超参必须硬编码为模块级常量, 调用方只能改代码, 不能传 `--xxx`; 唯一允许传参: `--sid_npy` / `--product_dir` / `--tag` 路径参数 (R30 强化)。

**R44** — 数据集与依赖库位置硬约束:
- 数据集: 每个新任务的 4 stage 脚本必须显式从 `/home/wlia0047/ar57/wenyu/GeneRec/dataset/` 读取 (Instruments.item.json, Instruments.inter.json, train.parquet, valid.parquet, test.parquet 等), 禁引用任何外部数据集路径 (HG-Rec/dataset/, 用户家目录其他位置等);
- 依赖库: stage 需要的 baseline 模型/工具库 (如 HRQVAE、quantizer、utils 等) 必须从 `/home/wlia0047/ar57/wenyu/GeneRec/_lib/` 复制到对应任务目录的 `_lib/` 子目录, 任务脚本 `sys.path.insert(0, str(<task_dir>/_lib))`; 禁止从外部路径 import baseline 代码;
- 两者共同强化 R40 自包含, 确保任务目录可独立运行、可重现。

**R45** — Git remote 强约束: 不使用 GitHub, 只使用 GitLab。`git remote` 必须仅包含 `origin` 指向 `git@gitlab.com:wlia0047/generec.git`; 禁止添加任何指向 github.com / WENYULIANG123/GeneRec.git 的 remote; 所有 commit + push 一律走 gitlab (`git push origin main`)。若当前 repo 已残留 github remote, 立即执行 `git remote remove github`。issue 编号 / 评论 / verdict `issue<NN>_verdict.json` 一律按 gitlab issue 编号 (与 R33+R34 联动)。
