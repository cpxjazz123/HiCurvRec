# CLAUDE.md

> HG-Rec (Hyperbolic RQ-VAE + Differential-Length Codebook + T5) 流水线复现工作目录, 基线 Task #84 valid R@10=0.1267 / test R@10=0.1024 (Musical_Instruments 9922 items).

---

## R 规则 (28 条)

**R1**: 使用 `genrec_env` (默认所有任务) 或 `deepke` (KG 任务) 两个 conda env, base anaconda Python 3.11.7 仅适用于 zero-dep grep + MiniMax API 调用.

**R2**: 禁止 fallback 逻辑 (默认值/回退/降级), 预期内的缺失返回 None/空值, 预期外的失败直接 raise.

**R4**: 修改 Python 脚本后必须立即 `python3 -m py_compile <file>` 验证语法, 文档文件例外.

**R5**: 任务硬约束 — 基线 HG-Rec Task #84 (valid R@10=0.1267, test R@10=0.1024), 仅 RQ-VAE 量化, Musical_Instruments 数据集 (9922 items), 4 阶段流水线, seed=42.

**R7**: 启动新实验前必须 `nvidia-smi` 核对 GPU 状态 (util<10%, mem<5GB), 选完全空闲 GPU 启动, 禁止等待已占卡或把多实验挤同一张卡.

**R10**: open issue 优先, 0 open issue + §16 空 + 用户未派工允许 idle (R10 v2), 取消原"主动推进"硬规则.

**R11**: AI 自主决策原则, 用户已派工后子步骤不允许抛回用户等决策 (禁止"等用户拍板"/"是否启动?"等阻塞话术), 兜底顺序: CLAUDE.md > 上游 default > 论文原始方案 > 简单实用.

**R12**: 训练必须在固定阶段强制保存 checkpoint (推荐 epoch 末), 每次保存删旧 ckpt (磁盘只保留最新), launcher 写 `_TRAINING_PID`.

**R13**: 禁止使用 `EnterWorktree` 工具或 git worktree 机制, 代码/配置修改直接落在共享 checkout, 临时文件写到 `$CLAUDE_JOB_DIR/tmp`.

**R15**: issue 闭环必须四件套 = verdict 落盘 + commit + `git push` + `glab issue close`, 不允许 commit 在本地不 push.

**R16**: 每次 loop tick 第一步 `glab issue list` 检查, 有 open issue → 必须完成 + close, 没有 → 允许 idle.

**R17**: commit message 必含 `Gate <N> FAIL/PASS` + 失败原因, 4 Gate = Stage 1/2/3/4, 前 Gate FAIL → 后 Gate STOP.

**R18**: 新 issue 必须跟历史做 4 维度对比 (D1 spec 摘录 / D2 实施核心 / D3 Gate 1 失败机制 / D4 引用文献), 任何不同 → 必须做实验, 不能凭"路径同构" NO-GO.

**R19**: AI = 激进 owner, 任何决策不需要等 owner 拍板, precheck PASS 立即启动 GPU 训练, 跨 issue 必须并行 (一张 GPU 一个), 禁止任何"是否启动?"询问.

**R20**: commit + issue comment 都必须详细回答 4 Gate (≥3-5 行/Gate: 状态/关键数据/失败原因/verdict 路径/commit hash), close issue 前必发 comment.

**R21**: commit hash 必须明示 (不允许 "pending"/"TBD"/"TODO" 占位), comment 必须在 commit + push 之后发 (拿到 hash 才能发).

**R22**: 任何 OPEN issue 出现 → 立即 R16+R17+R18+R20+R21 闭环, owner 派工立即执行, 4 卡全占 → 换 GPU/nohup/缩减规模, 禁止"等下一轮"/"等 owner 拍板".

**R23**: 每个 tick 扫一眼活跃训练, 7 信号任一触发立即终止 (val_R@10=0 跨 ≥2 checkpoint / loss 不下降 / val loss 反向 / wrapper broken / ckpt 不存 / NaN/Inf / GPU 100% 但 loss 不变), 立即 kill -9 + NO-GO verdict + commit + push + close issue.

**R24**: 每个 tick 检查 in_progress 任务是否真在执行 (有活跃 PID + file mtime 更新), 若 in_progress 但无活跃行为 → 立即 R11.5 决策 + 立即执行, 禁止任何"等待 owner 拍板"/"wrapper 复杂"/"等下一 tick" 话术.

**R25**: issue 里的 "方向A" / "方向B" 指的就是 taskA/ + taskB/ (代码位置 / 入口 / 产物路径), 禁止把它们解读为其它 lineage (e.g. TIGER / LETTER / RecBole / phonism). 命名演化: 历史 issue 文本可能用 "方向A/方向B" 或 "Issue #55 lineage / Issue #56 lineage" 表达, 实际当前实现是 taskA/ + taskB/ 目录; 检索代码 / 产物 / 数据时用此映射.

**R26**: 每个 tick 必须实际推进 open issue 的工作 (启动 precheck / Gate1 / 训练 / 评估 / 修复 / 验证 之一), 不允许仅停留在规划 / §16 状态描述 / commit + close comment 而未启动实际工作. 判定标准: 该 tick 至少执行一次以下动作 — 读取相关代码并产出诊断结论、运行 precheck 脚本、启动训练 (写 _TRAINING_PID)、产出 Gate 评估物 (verdict / 产物文件). 仅发 kickoff comment + commit §16 = 不算实际工作.

**R27**: 在 R26 基础上强化: 每个 tick 在做 issue 任务时必须 **启动实际脚本运行** (例如 `python3 taskA/stage2/taskA_stage2_kappa_sync.py` / `python3 taskB/stage3/..._train.py` / `python3 taskA/stage4/..._eval.py`), 而不是仅规划脚本 / 读脚本 / 写脚本 / 改 verdict 文件. 判定标准: 该 tick 至少执行一次 `python3 <existing_script>.py` 并产生 PID 活跃运行 (写入 _TRAINING_PID 或 nvidia-smi 显示 util>0%). 仅 hash 计算 / verdict 写入 / comment 发送 = 不算实际脚本运行.

**R28**: 优先级裁决 + 禁止 A-B 选项话术. 任何 OPEN issue 出现时 R10 v2 (idle) 自动失效, 触发 R19 激进 owner 模式. 任何决策 / 修复方案 / 启动判断**不允许**用"等用户拍板" / "是否启动?" / A-B 选项 / "你选" / "请告诉我" / "要不要" 等阻塞话术呈现, 必须直接给出推荐方案 + 立即执行 (R11 兜底顺序: CLAUDE.md > 上游 default > 论文 > 简单实用). **唯一例外**: 不可逆操作 (rm -rf / force push / 替换已发布 commit hash / 删除 issue 关联系) 才需用户确认. 出现 R10 v2 vs R19/R22 矛盾时, R22 (OPEN 立即闭环) 优先.

**R29**: tick 完成硬判定. 每个 tick 输出必须同时满足: (a) ≥1 个 R26 动作 (启动 precheck / Gate 评估 / 修复 commit / 验证运行 / 训练启动 / 落盘 verdict) **且** (b) ≥1 个 R27 动作 (`python3 -m py_compile` / `python3 <existing_script>.py` 启动 / `nvidia-smi` 查询 / `git commit` / `git push` / `glab issue close`). 仅"分析报告 + A-B 选项 + 等用户拍板"形式 = R26+R27 双 FAIL, 即使伴随 4 维度对比 / 修复方案. Tick 末尾必须明确"已执行 X" + 给出实际产物 (verdict 路径 / commit hash / PID), 不允许"下一步待定" / "等你确认" / "请选择 A 或 B".

**R30**: 脚本参数硬编码. 所有 `.py` 脚本 (`taskA/stage2/taskA_stage2.py` / `common/stage3_train_pure_t5.py` / `common/stage4_eval_*.py` 等) **不允许** 使用 `os.environ.get(...)` / `os.environ[...]` 动态读取参数. 所有超参 (epoch / batch_size / lr / seed / bf16 / κ EMA / trust region / anchor / REC_LOSS / REL_STRUCT / INFER_SIZE / MAX_LEN / ...) 必须硬编码进脚本 (常量赋值或 argparse 默认值). launch 脚本 (`.sh`) 仅负责 GPU 选择 (`CUDA_VISIBLE_DEVICES`) + 路径 (`PRODUCT_DIR` / `SID_NPY` / `CKPT_PATH`) + 产物目录设定, 不传任何超参. 历史代码中的 env var 读取是过渡期产物, 后续修改一律去除. **Why:** env var 隐式接口难以审计, 易遗漏 (如本轮 launch 漏设 `INFER_SIZE=256`), 复现性依赖 launcher 上下文而非脚本本身. **How to apply:** 新写脚本 → 顶部常量区列所有超参; 修改脚本 → 若只是改超参, 直接改脚本常量, 不要碰 launcher; 实验变体 → 复制脚本为新文件改常量, 不复用同一脚本 + env toggle.

**R31**: 每个 stage 目录 (如 `taskA/stage2/` / `taskB/stage3/`) 只允许一个主脚本 (例如 `taskA_stage2.py` / `taskB_stage3.py`), 不允许 fork 出 `taskA_stage2_v2.py` / `taskA_stage2_v8.py` 等多版本并存. 修改时直接在主脚本上改, 历史实验变体从 git 历史恢复, 不在 stage 目录保留多份. 历史 issue 已 fork 的 v3-v8 副脚本 (如 `taskA_stage2_v7.py` / `taskA_stage2_v8.py`) 在本规则生效后必须删除, 仅留主脚本作为唯一入口. **Why:** 多版本并存 → 启动时不知道跑哪个 → 容易跑错版本 → 复现性崩溃; 主脚本单一入口 + git 历史 = 任何变体都可追溯, 同时避免误启动. **How to apply:** 新实验变体 → 改主脚本 CONFIG 块 + commit; 旧的 v*-forked.py 文件 → 立即 `rm` (commit + push 一起发); `_history/` 目录的产物文件夹 (如 `taskA_stage2_v7_issue43/`) 仅保留产物, 不影响主脚本选择.

**R32**: 运行脚本必须直接 `python3` 执行, **不允许** 写 `.sh` 包装脚本启动 (如 `launch_xxx.sh`). GPU 选择走 `CUDA_VISIBLE_DEVICES=0 python3 -u ...` 内联环境变量; 路径走 `--product_dir <path>` argparse 参数; 日志走 `tee` 或 `nohup ... > log.txt 2>&1` (`.sh` 仅作内联一次性命令, 不落盘). 历史 `.sh` 启动器 (如 `launch_stage2_v7_issue43.sh`) 一律删除, 启动方式统一为 `CUDA_VISIBLE_DEVICES=0 python3 -u <script>.py --args...`. **Why:** `.sh` 包装层 → 超参容易从 launcher 注入 → 违背 R30 硬编码原则; `.sh` 累积 → 仓库膨胀 / 哪个版本对应哪个 launcher 难追溯; 直接 python 执行 → 单行命令自描述, 复现性直接 grep 命令即可. **How to apply:** 写新实验 → 不写 `.sh` 文件; 启动训练 → 在终端直接 `CUDA_VISIBLE_DEVICES=0 python3 -u <script>.py --args... > log.txt 2>&1 &` (后台) 或 `CUDA_VISIBLE_DEVICES=0 python3 -u <script>.py --args...` (前台) 或 `bash -c 'CUDA_VISIBLE_DEVICES=0 python3 -u <script>.py --args...' | tee log.txt` (前台+日志). **唯一例外**: DDP 多卡 `torchrun` (env 必须由 torchrun wrapper 设, 无法绕开) — 此场景保留 `.sh` 包装.

**R33**: verdict 文件路径规范 (1:1 映射). 所有 verdict / precheck / canary / evidence / diagnostic / probe / audit 文件必须放在 `verdicts/<gitlab_iid>/<final_verdict>.<ext>`, 其中 `<gitlab_iid>` 是 GitLab issue iid (1-90 范围). 具体规范:
  - **路径**: `verdicts/<gitlab_iid>/<filename>.<ext>`, 顶层只保留 `index.md` + `README.md` + `gitkeep` (不分子目录如 `verdicts/_misc/<scope>/`)
  - **文件名**: snake_case, kebab→snake, 去 internal verdict 编号前缀 (e.g. `issue141_v85c_nogo.md` → `v85c_nogo.md`)
  - **每个 iid 一个最终 verdict**: 中间产物 (gate/canary/evidence/diagnostic/probe/audit) 不落盘, 写入最终 verdict 文件作为 section; 仅有最高 rank 的 verdict 文件落盘 (rank 顺序: verdict/nogo/partial_go/go > result/ceiling/saturation > summary/final > fix > rejudge/verify > evidence/diagnostic > precheck/audit/probe)
  - **orphan**: 若 internal verdict #N > 90 (项目内部 verdict 编号, 无对应 GitLab issue 1-90) → `verdicts/_misc/orphan/<original_filename>`, 文件名保留 internal #N 前缀
  - **文件类型**: `.md` 用于 verdict 报告 (含 4-Gate 审计 + 数据 + 失败原因); `.json` 用于结构化 result (机器可读 Gate 状态)
  - **frontmatter**: 每个 verdict `.md` 文件顶部必须有 frontmatter (type/issue/status/created/tags/up), `up: "[[index]]"` 建立 wiki-link

  **Why:** 此前 verdicts/ 是 327 个文件平铺, 历史 issue 中 `issue<N>` scope 命名易与 GitLab iid 混淆 (Issue #90 已修正). 1:1 映射 → 每 issue 一个 verdict 文件 → 路径 grep 即定位, 检索成本 → O(1); 中间产物落盘累积 213 个孤儿文件, 删除后仓库瘦身 21k 行; orphan 隔离 → 内部 verdict #N > 90 的工作仍可追溯但不污染 iid 主索引. **How to apply:** 新写 verdict → 先查 `glab issue list` 确定对应 iid (1-90), 写最终文件到 `verdicts/<iid>/<verdict>.<ext>`, 不写中间产物; orphan 路径 → 内部 verdict #N > 90 直接 `verdicts/_misc/orphan/issue<N>_<rest>.<ext>`; 复盘脚本 `/home/wlia0047/.claude/jobs/4efe348f/tmp/verdict_restructure_exec.py` 是参考实现. **禁止**: 在 `verdicts/_misc/<letter_scope>/` 留中间分类目录 (历史产物已清空); 写 verdict 时带 `issue<N>_` 前缀到 iid 子目录 (前缀要去掉).

---

## 项目元数据 (6 条)

**仓库**: HG-Rec 复现 + κ-Stereographic 变体实验, 当前基线 Task #84 (valid R@10=0.1267, test R@10=0.1024), 不用 phonism/Toys.

**GPU**: 4× NVIDIA L40S (sm_89, 46GB/卡), 驱动 580.126.20, CUDA 13.0 (torch 2.11.0+cu130) / 12.x (TF kgat_mckg).

**目录**: 上游 clone 只读 (HG-Rec/, data/, papers/), 可写 (verdicts/, products/, logs/). taskA/taskB 只保留 Stage 1 / Stage 2 子目录 (各 1 个主脚本). **Stage 3 / Stage 4 只允许使用 `common/` 下的脚本** (用户指示 2026-08-06): `common/stage3/stage3_train_pure_t5.py` (T5 训练, R30 硬编码超参 + argparse 路径) / `common/stage4/stage4_eval_pure_t5.py` (pure T5 评估) / `common/stage4/stage4_eval_beam20.py` (beam20 评估) / `common/stage4/stage4_decode.py` (解码工具). taskA/stage3+stage4 与 taskB/stage3+stage4 目录已删除, 任何 stage3/4 任务必须用 common/. **common/ 目录结构 (用户指示 2026-08-06)**:
  - `common/stage1/stage1_hyperbolic.py` (Stage 1 双曲 sentence-t5-base embedding)
  - `common/stage2/` (空占位 — Stage 2 在 taskA/stage2/ 与 taskB/stage2/, 各自为方向专用变体)
  - `common/stage3/stage3_train_pure_t5.py` (Stage 3 纯 T5 训练)
  - `common/stage4/stage4_{decode,eval_pure_t5,eval_beam20}.py` (Stage 4 解码 + 评估工具)
  **训练产物统一放 `taskX/_history/`** (stage2 ckpt+verdict + stage3 adapter+verdict + stage4 canary; 主脚本 `PRODUCT_DIR` 均指向 `_history/`), stage 目录不放产物与 .pid.

**流水线**: 4 阶段 — Stage 1 sentence-t5-base embedding → Stage 2 Poincaré RQ-VAE SID (3→4 层去重 digit) → Stage 3 T5-mini 训练 → Stage 4 R@K/NDCG 评估.

**评估**: HG-Rec baseline (全量核查 valid.parquet 24772 样本) R@5/10/20 = 0.1029/0.1267/0.1561, test.parquet = 0.0819/0.1024/0.1283; NDCG 未全量核查 (旧记录 0.0690/0.0755/0.0821). 决策阈值 = valid R@10 > 0.1267 GO. 注意: N=1000 前序子集系统性偏低 ~25%, 评估必须用全量或随机采样.

**依赖**: torch >= 2.0, pytorch-lightning, hydra-core, transformers, torchmetrics (详见 requirements.txt).