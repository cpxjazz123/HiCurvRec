# CLAUDE.md

> HG-Rec (Hyperbolic RQ-VAE + Differential-Length Codebook + T5) 流水线复现工作目录, 基线 Task #84 valid R@10=0.1267 / test R@10=0.1024 (Musical_Instruments 9922 items).

---

## 最新测试结果 (2026-08-23 baseline 切换 v87 + 历史 baseline, beam=20)

| 流水线 | 状态 | valid R@10 | valid NDCG@20 | **test R@10** | test R@20 | test NDCG@20 | ckpt |
|---|---|---|---|---|---|---|---|
| **v87 F3 Spread Loss Revisit** (新 baseline, 2026-08-23) | ✅ | — | 0.0976 (best valid ndcg@10) | **0.11171069587628867** | 0.1409 | 0.0898 | `curvature_base/out/decoder/instruments_hgrec_configs/hgrec_f3_spread_loss_revisit_v87/best_ckpt.pt` |
| **v69 G5 Codebook-aware Manifold Contrastive** (已 R50 删除, 2026-08-23 切换) | ref | — | 0.0964 | **0.11042203608247422** | 0.1407 | 0.0882 | (旧 `curvature_base/out/.../hgrec_v69_g5_codebook_manifold_contrastive/best_ckpt.pt`) |
| **HG-Rec 重跑** (Aug-14-2026_20-15-45) | ✅ | 0.1312 (valid) | **0.1049** | **0.1074** | 0.1369 | 0.0879 | `HG_Rec_epoch_63.pth` |
| **v19 R51+ baseline** (已 R50 删除, 2026-08-22 切换) | ref | 0.1267 | 0.0961 | **0.1090931056701031** | — | — | (旧 `curvature_base/out/.../hgrec_v19/best_ckpt.pt`) |
| **RQ-VAE-Recommender 新跑** (DDP 4 卡, 200 epoch) | ✅ | 0.8841 (best_ckpt epoch 199) | **0.8313** | **0.0926** | 0.1163 | 0.0740 | `out/decoder/instruments/best_ckpt.pt` |
| HG-Rec 原 baseline (R5 硬约束, 已过期) | ref | 0.1267 | — | 0.1024 | — | — | — |

**关键观察 (R36 valid 偏置)**:
- RQ-VAE-Recommender valid NDCG@20=0.8313 (比 HG-Rec 高 8 倍),但 test R@10=0.0926 (反而比 HG-Rec 0.1074 低 14%) — 强烈 valid overfitting。
- HG-Rec 重跑 valid NDCG@20=0.1049 → test R@10=0.1074,valid-test 方向一致 (提升 +0.005 vs 原 0.1024)。
- 走 R36 严格化 v2 (几何变换, 避免 valid 偏置) 是后续改进方向。

**测试结果物路径**:
- HG-Rec 重跑日志: `/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/hg_rec_eval_test.log`
- RQ-VAE-Recommender test JSON: `RQ-VAE-Recommender/out/decoder/instruments/test_final.json`

---

## R 规则

**R1** — 默认 `genrec_env`; KG 任务用 `deepke`; base anaconda 仅 zero-dep grep + MiniMax API。

**R2** — 禁止 fallback 逻辑 (默认值/回退/降级); 预期内缺失返回 None/空值, 预期外失败直接 raise。

**R4** — 修改 Python 脚本后必须立即 `python3 -m py_compile` 验证语法 (文档例外)。

**R5** — 任务硬约束 = **v87 F3 Spread Loss Revisit baseline (R51+ locked, test_R@10=0.11171069587628867, +1.16% vs v69 G5 baseline)**, 仅 RQ-VAE 量化, Musical_Instruments (9922 items), 4 阶段流水线, seed=42。原 HG-Rec Task #84 test_R@10=0.1024 / v19 R51+ test_R@10=0.1090931056701031 / v69 G5 R51+ test_R@10=0.11042203608247422 数值均已过期, R37 当前阈值以本条 v87 baseline 为准。**(2026-08-23 baseline 切换)** 原 curvature_base/ (v69 G5 R51+ locked) 已 R50 删除, 改用 curvature_experiment_f3_spread_loss_revisit_v87/ 重命名为 curvature_base/ 作为新 baseline (Stage 1 RQ-VAE 端 F3 Spread Loss 微调 SPREAD_LOSS_WEIGHT=0.005, SPREAD_LOSS_MARGIN=2.5, R36o v3.7 阶段 C 微调空间探索成功, 打破 R36h ceiling 19 次 +1.16%)。

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
> **R50 联动例外**: 回滚后 (R37/R38 触发) 的 tick 不强制满足 R26/R27 (禁止重跑同任务), 但 R19 触发新任务的 tick 仍需满足 R26/R27。R50 与 R19+R26+R27 区分明确: 那三条是 tick 启动新任务时强制实操, R50 是回滚后强制停手。

**R30** — 脚本超参硬编码进模块, 禁 `os.environ.get` 读超参, 禁 wrapper 传 num_epochs/batch_size/lr/seed 等数值超参 (路径参数可传)。
> **R51 联动例外**: seed (42 / 42+process_index) 作为数值超参, 因 R51 DDP 4 卡噪声消除硬约束必要, 仍硬编码为模块级常量而非 CLI 参数, 不暴露为 `--seed` CLI 参数, 不走 R30/R43 禁 CLI 数值超参路径。

**R31** — 每个 stage 目录只允许一个主脚本, 禁 fork `_v2.py/_v8.py` 多版本并存, 历史实验变体从 git 历史恢复。

**R32** — 运行脚本必须直接 `python3` 执行, 禁写 `.sh` 包装启动, GPU 选择用 `CUDA_VISIBLE_DEVICES=0 python3 -u ...` 内联 (唯一例外: DDP 多卡 `torchrun`)。

**R35** — 评估强约束: 只使用单 checkpoint + `beam_search=20`, 禁 Borda Rank Fusion / 任何 ensemble 多 ckpt 融合。

**R35b** — Stage3 / Stage4 DDP 同口径硬约束 (已合并原 R35c):
- **Stage 4 评估**: 4 卡各自评估**不重复**的数据分片 (DistributedSampler / 按 rank 切片, 每样本只被一个 rank 处理), 各 rank 在 **本地** 汇总其分片的命中数 (hits) 与 NDCG 总和, 最后 `all_reduce SUM` 全部样本的命中数与 NDCG 总和, 统一除以总样本数 N — 保证 R@K / NDCG@K 与单卡评估完全同口径 (每个样本恰好计数一次, 无重复、无遗漏、无按 rank 平均的错误口径)。禁直接对 4 个 rank 的均值取平均。
- **Stage 3 训练期 Valid 评估**: 每次 valid 评估时, 4 卡各自评估**不重复**的 Valid 数据分片 (DistributedSampler / 按 rank 切片, 每样本只被一个 rank 处理), 各 rank 在**本地**汇总其分片的**逐样本**命中结果 (hits) 与 NDCG 总和 (每样本一行), 通过 `all_gather` / `all_reduce SUM` 汇总 4 卡对**完整 Valid 集**的全部逐样本结果, 由 **rank 0** 依据**全量 Valid R@10** (每样本恰好计数一次, 与单卡评估完全同口径) 选择 best checkpoint 并触发早停 (EARLY_STOP=20)。禁各 rank 用自己分片的 valid R@10 独立选 ckpt / 禁按 rank 均值选 ckpt; rank 0 选定的 best ckpt 必须与单卡评估口径下的 best ckpt 一致。

**R36** — 方法路径强约束: 禁止通过调参形式 (LR/dropout/label_smoothing/weight_decay sweep) 提升指标; 必须通过改善曲率框架 (Stage 2 κ 学习 / Stage 3 κ frozen→learnable / 新曲率正则项 / Poincaré-Minkowski-Lorentz 曲率机制变更)。

**R37** — 版本回滚硬约束: 新版本 (vN) test_R@10 < 上一版本 (vN-1) → 立即终止 lineage, 必须回到 vN-1 重新创新, 禁止在比旧版本差的版本上进行二次创新。

**R38** — 训练期早停回退硬约束: vN 训练期 valid_R@10 / loss / 收敛速度明确比 vN-1 差 (连续 ≥30 epoch 平台 / loss 高 ≥0.05 / NaN-Inf / loss 反向) → 立即 kill + 写 R38 决策行 + 触发 R37 回退。

**R39** — Open Issue 立即实现硬约束: 任何 open issue 一旦被 loop tick 发现, 必须立即实施 (R19+R26+R27 联动), 严禁等待用户评论授权。

**R40** — 四 Stage 全量运行硬约束: 每个任务目录必须完整实时运行 stage1 → stage2 → stage3 → stage4, 无论创新点位于哪个 stage; stage n+1 输入必须唯一来自 stage n 实时运行产物 (落 tasks/<task_dir>/), 禁引用任何外部脚本/外部产物; stage n 产物缺失 → stage n+1 raise FileNotFoundError 禁启动 (issue 显式豁免某 stage 除外)。

**R41** — 所有 Stage3 / Stage2 训练脚本的 `EARLY_STOP` 统一硬编码为 20 (历史 v121 用 30 不追溯, 仅本规则生效后新任务生效)。

**R41b** — Stage3 训练期 Valid 评估频率硬约束: 每个 epoch 都必须执行一次 valid 评估 (EVAL_INTERVAL 恒等于 1), 禁止每 5 个 epoch 才 eval 一次 (历史 EVAL_INTERVAL=5 不追溯, 仅本规则生效后新任务生效); 每次 eval 仍需遵循 R35c (4 卡分片不重复评估完整 Valid 集, all_reduce SUM, rank 0 按全量 Valid R@10 选 best ckpt + 触发早停), `EARLY_STOP` 保持 20 (R41)。

**R41c** — Stage2 A/B 对照初始一致性硬约束: 任何 Stage2 曲率机制对照实验 (Control vs Treatment) 必须满足以下因果链: 同一 Stage1 embedding → 同一 KMeans 初始中心 (sklearn KMeans 必须固定 `random_state`, 禁止默认随机初始化) → 初始 codebook 数值一致 (逐元素差 < 1e-7) → 初始 distance / assignment / SID 一致 (argmin 逐元素相同, SID 差异 < 1%) → 之后只开启或关闭一个曲率机制。若两套 Stage2 因 KMeans 未固定等原因初始就产生不同 SID, 判定 `STAGE2_NONDETERMINISM`, 不得把差异称为曲率机制收益, 必须先修复初始化再实验。

**R42** — Stage2 / Stage3 必须用 `torchrun --nproc_per_node=4` DDP 4 卡运行, 禁单卡 (world_size=1); 唯一例外: nvidia-smi 显示 GPU 1/2/3 都被占时允许单卡, 但 verdict_r37.json 必须显式记录 (R7+R42 联动)。

**R43** — 禁止脚本使用 CLI 传入数值超参; 脚本超参必须硬编码为模块级常量, 调用方只能改代码, 不能传 `--xxx`; 唯一允许传参: `--sid_npy` / `--product_dir` / `--tag` 路径参数 (R30 强化)。

**R44** — 数据集与依赖库位置硬约束:
- 数据集: 每个新任务的 4 stage 脚本必须显式从 `/home/wlia0047/ar57/wenyu/GeneRec/dataset/` 读取 (Instruments.item.json, Instruments.inter.json, train.parquet, valid.parquet, test.parquet 等), 禁引用任何外部数据集路径 (HG-Rec/dataset/, 用户家目录其他位置等);
- 依赖库: stage 需要的 baseline 模型/工具库 (如 HRQVAE、quantizer、utils 等) 直接放在 `/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/_lib/` 主目录下, 任务目录脚本 `sys.path.insert(0, str(<main_dir>/_lib))` 直接 import; 不再为每个任务目录复制 _lib/ 子目录;
- 数据集自包含 (R40), 依赖库主目录共用 (避免重复维护), 两者共同确保任务目录可独立运行、可重现。

**R44b** — 下载位置与磁盘硬约束: 禁止向 `/home/wlia0047/` 写入任何数据 (模型权重、数据集、缓存等) — /home 挂载仅 20G 且曾 100% 满导致 HuggingFace 模型下载截断损坏; 所有下载/HF 缓存必须指向 `/home/wlia0047/ar57_scratch/wenyu/` (大磁盘); 运行下载类任务前必须 `df -h /home/wlia0047/` 检查剩余空间, 不足 5G 时先清理或改路径; HuggingFace 相关必须显式设 `HF_HOME=/home/wlia0047/ar57_scratch/wenyu/.cache/huggingface` (或对应 scratch 路径)。

**R44c** — pip 安装位置硬约束: 禁止任何 pip 安装落到 `/home/wlia0047/` 下的用户 site (即 `~/.local`, 含 `/home/wlia0047/.local` 与 `/home/wlia0047/ar57/wenyu/.local`) — 曾因无 `-t` 的 pip install 把 13G 依赖树写进 `/home/wlia0047/ar57/wenyu/.local` 撑爆 20G /home 挂载; pip 安装必须显式 `pip install <pkg> -t <conda_env>/lib/python3.10/site-packages` (如 `-t /home/wlia0047/ar57_scratch/wenyu/genrec_env/lib/python3.10/site-packages`), 或 `--user` 仅指向 scratch 路径; 安装前先 `df -h /home/wlia0047/` 核对, 安装后检查 `/home/wlia0047/` 下不得出现新的 `.local`/`.cache` 目录。

**R45** — Git remote 强约束: 不使用 GitHub, 只使用 GitLab。`git remote` 必须仅包含 `origin` 指向 `git@gitlab.com:wlia0047/generec.git`; 禁止添加任何指向 github.com / WENYULIANG123/GeneRec.git 的 remote; 所有 commit + push 一律走 gitlab (`git push origin main`)。若当前 repo 已残留 github remote, 立即执行 `git remote remove github`。issue 编号 / 评论 / verdict `issue<NN>_verdict.json` 一律按 gitlab issue 编号。

**R47** — sentence-t5-xxl 模型位置硬约束: `sentence-transformers/sentence-t5-xxl` (RQ-VAE-Recommender 默认 embedding 模型, ~10GB) **必须**存放在 `/home/wlia0047/hj82_scratch2/wenyu/` 下面 (该挂载点 6.6T 总量, 充裕可装下 RQ-VAE-Recommender 全套); HF 相关环境变量必须显式设:
- `HF_HOME=/home/wlia0047/hj82_scratch2/wenyu/.cache/huggingface`
- `HUGGINGFACE_HUB_CACHE=/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub`
- `TRANSFORMERS_CACHE=/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub`

禁止把 xxL 模型缓存写到 `/home/wlia0047/` 下 (20G 撑爆), 或 `/home/wlia0047/ar57_scratch/wenyu/` (用户专属); xxL 与 base 不可混用 (参数量 11B vs 220M, embedding 质量显著差异); 下载前 `df -h /home/wlia0047/hj82_scratch2` 检查剩余空间 (需 ≥15G)。

**R48** — 临时文件位置硬约束: 所有临时文件 (训练日志、调试输出、nohup.out、自建临时目录) 必须放在 `/home/wlia0047/hj82_scratch2/wenyu/` 下, 推荐统一子目录 `/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/` (沿用现有路径); 禁止把临时文件写到 `/fs04/scratch2/...` (Lustre 子配额 100% 满导致 ENOSPC 输出截断) 或 `/home/wlia0047/` (20G 撑爆); Bash 输出若报 `temp filesystem is full`, 应立即 `df -h /fs04` 检查并迁移到 hj82_scratch2; 任务脚本中的 `tempfile.NamedTemporaryFile` 等 Python 临时 API 也应通过 `TMPDIR` 环境变量指向 hj82_scratch2 (如 `TMPDIR=/home/wlia0047/hj82_scratch2/wenyu/tmp`)。

**R49** — 禁 sleep 阻塞等待: 启动长任务 (训练/推理/下载) 后**禁止使用 `sleep` 休眠等待进程执行**, 改为**定期手动轮询检查** (每次轮询: 查日志尾部 / `ps` 进程状态 / 产物 mtime, 轮询间隔由 tick 节奏决定, 单次轮询命令内不 sleep 长于 ~10s)。等待期间应并行推进其他可做事项 (写代码/分析/准备下一阶段), 而不是空转; 任务完成判定基于日志标记 (如 `[train] done` / `Early stopping`) 或产物出现, 非时间估算。

**R50** — 回滚后禁重跑硬约束: 任何代码/产物回滚操作 (R37/R38 触发的 `git checkout <commit> -- <file>` / 手动恢复 best_ckpt.pt / 手动恢复 test_final.json / conda env 升降级 / 脚本超参改回上一版本) 完成之后, **禁止立即自动重新启动训练或评估脚本** (`python3 train_decoder.py` / `python3 test_eval_only.py` / `accelerate launch ...`); 验证手段仅限于 `git diff --stat` / `git status` / `python3 -m py_compile` / 文件 mtime 比比 / `cat <result.json>` 校验数值。回滚操作的产出物以 "已回滚 + 已记录 verdict" 为终态, 不擅自触发新一轮 GPU 作业; 是否重跑、什么时候重跑、用什么配置重跑, 必须由用户明确指令或下一次 tick 主动启动新任务时再决定 (与 R19 + R26 + R27 区分: 那三条是 tick 启动新任务时强制实操, R50 是回滚后强制停手)。

**R51** — DDP 4 卡噪声消除硬约束 (2026-08-19 实测触发, 任何新版本必须遵守):
- **触发证据**: 历史 v19 三次完整重跑 test_R@10 = 0.1102 / 0.1099 / 0.1113 (±0.001 量级噪声, 来自 DistributedSampler seed 不固定 + 训练端无 manual_seed); R51 实施后三次 test_R@10 = 0.11130798969072164 (字符级完全一致, 差异 < 1e-15)。
- **R51+ 升级** (commit ac9cb5f): 在 R51 only 基础上叠加 `torch.use_deterministic_algorithms(True, warn_only=True)` + `cudnn.deterministic=True` + `CUBLAS_WORKSPACE_CONFIG=":4096:8"` + `PYTHONHASHSEED=42` + `torch.set_float32_matmul_precision("high")` + DataLoader `worker_init_fn` 固定 worker RNG。R51+ 实施后 3 RUN 字符级完全一致 test_R@10 = **0.1090931056701031** (差异 < 1e-15, best_ckpt epoch=97)。R51+ 数值 (0.10909) 与 R51 only 数值 (0.11131) 不同是因为叠加约束改变了浮点路径 (use_deterministic_algorithms 替换部分非结合 GEMM 调用), 两者均字符级一致, 不矛盾。Verdict: `/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/curvature_base_rerun/r51_verdict.json` (PASS_R51_PLUS, 2026-08-20 验证)。
- **Stage 3 训练端** (`train_decoder.py` train() 入口): 在 `accelerator = Accelerator(...)` 创建**之前**显式设:
  ```python
  import random as _random
  import numpy as _np
  _random.seed(42 + accelerator.process_index)
  _np.random.seed(42 + accelerator.process_index)
  torch.manual_seed(42 + accelerator.process_index)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(42 + accelerator.process_index)
  ```
  `process_index` 偏移必须存在 (4 个 rank 用 4 个不同 seed, 否则 4 卡同步退化)。**必须在创建 DataLoader 之前**完成。
- **Stage 4 评估端** (`test_eval_only.py` main() / `_test_eval_hgrec()` 入口): 在 DataLoader 创建**之前**显式设:
  ```python
  torch.manual_seed(42)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(42)
  import random as _random
  _random.seed(42)
  import numpy as _np
  _np.random.seed(42)
  ```
- **Stage 1 RQ-VAE 训练端** (`train_rqvae_instruments.py` train() 入口): 同 Stage 3, 加 `manual_seed(42 + process_index)`; Stage 2 SID 推理 (`infer_sids_instruments.py`): 同 Stage 4, 加 `manual_seed(42)` (单进程)。
- **禁手动加 `DistributedSampler(seed=42)`** — 与 `accelerator.prepare()` 双重 wrap 会导致 4×4 = 16 分片, 反而引入更大噪声。修复策略: 在 DataLoader 创建**之前**固定全局 RNG, 让 `accelerator.prepare()` 自动 wrap 出确定分片。
- **版本演进保护**: 任何 vN → vN+1 改动**禁止删除**上述 seed 代码块; `git diff` 必须确认 seed 块未被触碰 (Stage 1/2/3/4 入口各一处)。新文件 / 新模块如需 DataLoader, 必须先 seed 再创建。
- **验证标准**: 同一版本 (相同 git commit + 相同数据集 + 相同 ckpt 起点) 连续 3 次完整 4 阶段重跑, test_R@10 必须**字符级完全一致** (差异 < 1e-15)。若 3 次结果出现 ±0.001 量级偏差, 立即判定 `SEED_NONDETERMINISM`, 必须修复后重新验证。
- **bf16/fp16 非结合性 + cudnn benchmark**: 残留 ±1e-5 量级噪声理论存在, 实测不可观测, 不需额外 fix。
- **配套 R30/R43**: seed (42 / 42+process_index) 是数值超参, 但因 R51 强约束必要, 不走 R30 路径 (不暴露为 CLI 参数, 仍硬编码)。
