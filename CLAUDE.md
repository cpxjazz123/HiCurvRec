# CLAUDE.md

> HG-Rec (Hyperbolic RQ-VAE + Differential-Length Codebook + T5) 流水线复现工作目录, 基线 Task #84 valid R@10=0.1267 / test R@10=0.1024 (Musical_Instruments 9922 items).

---

## R 规则 (28 条)

**R1**: 默认 `genrec_env`, KG 任务用 `deepke`, base anaconda 仅 zero-dep grep + MiniMax API。

**R2**: 禁止 fallback 逻辑 (默认值/回退/降级), 预期内缺失返回 None/空值, 预期外失败直接 raise。

**R4**: 修改 Python 脚本后必须立即 `python3 -m py_compile` 验证语法 (文档例外)。

**R5**: 任务硬约束 = 基线 HG-Rec Task #84 (valid R@10=0.1267, test R@10=0.1024), 仅 RQ-VAE 量化, Musical_Instruments (9922 items), 4 阶段流水线, seed=42。

**R7**: 启动新实验前必须 `nvidia-smi` 核对 (util<10%, mem<5GB), 选完全空闲 GPU, 禁等待已占卡或挤同一张卡。

**R10**: 每次 tick 首步 `glab issue list`, 有 open → 完成 + close, 无 open + §16 空 + 用户未派工 → 允许 idle。

**R11**: AI 自主决策, 子步骤禁"等用户拍板"/"是否启动?"阻塞话术, 兜底顺序: CLAUDE.md > 上游 default > 论文 > 简单实用。

**R12**: 训练固定阶段强制存 checkpoint (epoch 末), 删旧 ckpt, 写 `_TRAINING_PID`。

**R13**: 禁 `EnterWorktree` + git worktree, 代码改共享 checkout, 临时文件用 `$CLAUDE_JOB_DIR/tmp`。

**R15**: issue 闭环 = verdict 落盘 + commit + `git push` + `glab issue close`, 禁本地 commit 不 push。

**R16**: (合并到 R10)

**R17**: commit message 必含 `Gate <N> FAIL/PASS` + 失败原因, 前 Gate FAIL → 后 Gate STOP。

**R18**: 新 issue 必须跟历史做 4 维度对比 (D1 spec 摘录 / D2 实施核心 / D3 Gate 1 失败机制 / D4 引用文献), 任一不同 → 必须实验, 禁"路径同构" NO-GO。

**R19**: AI = 激进 owner, precheck PASS 立即启动 GPU 训练, 跨 issue 并行 (一卡一实验), 禁任何"是否启动?"询问。

**R20**: commit + issue comment 必须详细回答 4 Gate (≥3-5 行/Gate), close issue 前必发 comment。

**R21**: commit hash 必须明示 (禁 "pending"/"TBD"/"TODO"), comment 必须在 commit + push 之后发。

**R22**: OPEN issue → 立即 R10+R17+R18+R20+R21 闭环, owner 派工立即执行, 4 卡全占 → 换 GPU/nohup/缩减规模, 禁"等下一轮"/"等 owner 拍板"。

**R23**: tick 扫活跃训练, 7 信号任一触发立即终止 (val_R@10=0 跨 ≥2 ckpt / loss 不下降 / val loss 反向 / wrapper broken / ckpt 不存 / NaN-Inf / GPU 100% loss 不变), kill -9 + NO-GO + commit + push + close。

**R24**: tick 检查 in_progress 任务是否真在执行 (有 PID + file mtime 更新), 无活跃行为 → 立即 R11.5 决策 + 执行, 禁"等 owner 拍板"/"wrapper 复杂"/"等下一 tick"。

**R25**: issue 里的"方向A"/"方向B"指的就是 taskA/ + taskB/ (代码位置/入口/产物路径), 禁解读为 TIGER/LETTER/RecBole/phonism 等其他 lineage。

**R26**: tick 必须实际推进 issue (启动 precheck/Gate1/训练/评估/修复/验证 之一), 禁仅规划/§16 状态/commit + close。

**R27**: tick 必须 `python3 <existing_script>.py` 启动脚本产生 PID 活跃, 禁仅规划/读/写脚本/改 verdict。

**R28**: 任何决策/修复方案/启动判断禁"等用户拍板"/"是否启动?"/A-B 选项/"你选"/"请告诉我"/"要不要"话术, 必须直接给出推荐方案 + 立即执行 (唯一例外: 不可逆操作)。

**R29**: tick 输出必须 (a) ≥1 个 R26 动作 + (b) ≥1 个 R27 动作, 末尾明示"已执行 X" + 实际产物 (verdict 路径/commit hash/PID)。

**R30**: 脚本参数硬编码, 禁 `os.environ.get` 读超参, 所有超参 (epoch/batch_size/lr/seed/...) 必须硬编码进脚本 (常量或 argparse 默认), launch 脚本仅设 GPU/路径。

**R31**: 每个 stage 目录只允许一个主脚本 (如 `taskA_stage2.py`), 禁 fork `_v2.py/_v8.py` 多版本并存, 历史实验变体从 git 历史恢复。

**R32**: 运行脚本必须直接 `python3` 执行, 禁写 `.sh` 包装启动, GPU 选择用 `CUDA_VISIBLE_DEVICES=0 python3 -u ...` 内联 (唯一例外: DDP 多卡 `torchrun`)。

**R33**: verdict 文件必须放 `verdicts/<gitlab_iid>/<final_verdict>.<ext>` (1:1 映射 iid 1-90), 中间产物不落盘, orphan (internal #N > 90) → `verdicts/_misc/orphan/`。

---

## 项目元数据 (6 条)

**仓库**: HG-Rec 复现 + κ-Stereographic 变体实验, 当前基线 Task #84 (valid R@10=0.1267, test R@10=0.1024), 不用 phonism/Toys。

**GPU**: 4× NVIDIA L40S (sm_89, 46GB/卡), 驱动 580.126.20, CUDA 13.0 (torch 2.11.0+cu130) / 12.x (TF kgat_mckg)。

**目录**: 上游 clone 只读 (HG-Rec/, data/, papers/), 可写 (verdicts/, products/, logs/)。taskA/taskB 只保留 Stage 1 / Stage 2 子目录 (各 1 个主脚本)。**Stage 3 / Stage 4 只允许使用 `common/` 下的脚本** (用户指示 2026-08-06): `common/stage3/stage3_train_pure_t5.py` (T5 训练, R30 硬编码超参 + argparse 路径) / `common/stage4/stage4_eval_pure_t5.py` (pure T5 评估) / `common/stage4/stage4_eval_beam20.py` (beam20 评估) / `common/stage4/stage4_decode.py` (解码工具)。taskA/stage3+stage4 与 taskB/stage3+stage4 目录已删除, 任何 stage3/4 任务必须用 common/。**common/ 目录结构 (用户指示 2026-08-06)**:
  - `common/stage1/stage1_hyperbolic.py` (Stage 1 双曲 sentence-t5-base embedding)
  - `common/stage2/` (空占位 — Stage 2 在 taskA/stage2/ 与 taskB/stage2/, 各自为方向专用变体)
  - `common/stage3/stage3_train_pure_t5.py` (Stage 3 纯 T5 训练)
  - `common/stage4/stage4_{decode,eval_pure_t5,eval_beam20}.py` (Stage 4 解码 + 评估工具)
  **训练产物统一放 `taskX/_history/`** (stage2 ckpt+verdict + stage3 adapter+verdict + stage4 canary; 主脚本 `PRODUCT_DIR` 均指向 `_history/`), stage 目录不放产物与 .pid。

**流水线**: 4 阶段 — Stage 1 sentence-t5-base embedding → Stage 2 Poincaré RQ-VAE SID (3→4 层去重 digit) → Stage 3 T5-mini 训练 → Stage 4 R@K/NDCG 评估。

**评估**: HG-Rec baseline (全量核查 valid.parquet 24772 样本) R@5/10/20 = 0.1029/0.1267/0.1561, test.parquet = 0.0819/0.1024/0.1283; NDCG 未全量核查 (旧记录 0.0690/0.0755/0.0821)。决策阈值 = valid R@10 > 0.1267 GO。注意: N=1000 前序子集系统性偏低 ~25%, 评估必须用全量或随机采样。

**依赖**: torch >= 2.0, pytorch-lightning, hydra-core, transformers, torchmetrics (详见 requirements.txt)。