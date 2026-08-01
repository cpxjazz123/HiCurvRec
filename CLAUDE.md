# CLAUDE.md

> HG-Rec (Hyperbolic RQ-VAE + Differential-Length Codebook + T5) 流水线复现工作目录, 基线 Task #84 R@10=0.1020 (Musical_Instruments 9922 items).

---

## R 规则 (25 条)

**R1**: 使用 `genrec_env` (默认所有任务) 或 `deepke` (KG 任务) 两个 conda env, base anaconda Python 3.11.7 仅适用于 zero-dep grep + MiniMax API 调用.

**R2**: 禁止 fallback 逻辑 (默认值/回退/降级), 预期内的缺失返回 None/空值, 预期外的失败直接 raise.

**R3**: 所有 LLM 调用必须通过 `/home/wlia0047/ar57/wenyu/PersoanlQuery/llm_client.py` 的 `MiniMaxAnthropicClient`, 不直接导入 OpenAI/Anthropic/Qwen 等客户端.

**R4**: 修改 Python 脚本后必须立即 `python3 -m py_compile <file>` 验证语法, 文档文件例外.

**R5**: 任务硬约束 — 基线 HG-Rec Task #84 (R@10=0.1020), 仅 RQ-VAE 量化, Musical_Instruments 数据集 (9922 items), 4 阶段流水线, seed=42.

**R6**: 不要把 Stage 2 SID `merged_predictions_tensor.pt` (N,4) 误用成 Stage 1 embedding (N,2048), 不要把 `num_hierarchies=3` 直接传给 Stage 3, 修改上游 HG-Rec/ 前要意识到是只读 clone.

**R7**: 启动新实验前必须 `nvidia-smi` 核对 GPU 状态 (util<10%, mem<5GB), 选完全空闲 GPU 启动, 禁止等待已占卡或把多实验挤同一张卡.

**R8**: 完成的任务必须从 loop.md §16 表删除该行 (不写"已归档"), verdicts/ 保留 verdict 文件.

**R9**: 新任务编号必须 max+1 连续无空洞, R9-Enforce 三层防护 (创建前必跑 max+1 命令 + 创建后必验证连续 + loop tick 周期审计).

**R10**: open issue 优先, 0 open issue + §16 空 + 用户未派工允许 idle (R10 v2), 取消原"主动推进"硬规则.

**R11**: AI 自主决策原则, 用户已派工后子步骤不允许抛回用户等决策 (禁止"等用户拍板"/"是否启动?"等阻塞话术), 兜底顺序: CLAUDE.md > 上游 default > 论文原始方案 > 简单实用.

**R12**: 训练必须在固定阶段强制保存 checkpoint (推荐 epoch 末), 每次保存删旧 ckpt (磁盘只保留最新), launcher 写 `_TRAINING_PID`.

**R13**: 禁止使用 `EnterWorktree` 工具或 git worktree 机制, 代码/配置修改直接落在共享 checkout, 临时文件写到 `$CLAUDE_JOB_DIR/tmp`.

**R15**: issue 闭环必须四件套 = verdict 落盘 + commit + `git push` + `gh issue close`, 不允许 commit 在本地不 push.

**R16**: 每次 loop tick 第一步 `gh issue list --state open` 检查, 有 open issue → 必须完成 + close, 没有 → 允许 idle.

**R17**: commit message 必含 `Gate <N> FAIL/PASS` + 失败原因, 4 Gate = Stage 1/2/3/4, 前 Gate FAIL → 后 Gate STOP.

**R18**: 新 issue 必须跟历史做 4 维度对比 (D1 spec 摘录 / D2 实施核心 / D3 Gate 1 失败机制 / D4 引用文献), 任何不同 → 必须做实验, 不能凭"路径同构" NO-GO.

**R19**: AI = 激进 owner, 任何决策不需要等 owner 拍板, precheck PASS 立即启动 GPU 训练, 跨 issue 必须并行 (一张 GPU 一个), 禁止任何"是否启动?"询问.

**R20**: commit + issue comment 都必须详细回答 4 Gate (≥3-5 行/Gate: 状态/关键数据/失败原因/verdict 路径/commit hash), close issue 前必发 comment.

**R21**: commit hash 必须明示 (不允许 "pending"/"TBD"/"TODO" 占位), comment 必须在 commit + push 之后发 (拿到 hash 才能发).

**R22**: 任何 OPEN issue 出现 → 立即 R16+R17+R18+R20+R21 闭环, owner 派工立即执行, 4 卡全占 → 换 GPU/nohup/缩减规模, 禁止"等下一轮"/"等 owner 拍板".

**R23**: 每个 tick 扫一眼活跃训练, 7 信号任一触发立即终止 (val_R@10=0 跨 ≥2 checkpoint / loss 不下降 / val loss 反向 / wrapper broken / ckpt 不存 / NaN/Inf / GPU 100% 但 loss 不变), 立即 kill -9 + NO-GO verdict + commit + push + close issue.

**R24**: 每个 tick 检查 in_progress 任务是否真在执行 (有活跃 PID + file mtime 更新), 若 in_progress 但无活跃行为 → 立即 R11.5 决策 + 立即执行, 禁止任何"等待 owner 拍板"/"wrapper 复杂"/"等下一 tick" 话术.

**R25**: issue 里的 "方向A" / "方向B" 指的就是 taskA/ + taskB/ (代码位置 / 入口 / 产物路径), 禁止把它们解读为其它 lineage (e.g. TIGER / LETTER / RecBole / phonism). 命名演化: 历史 issue 文本可能用 "方向A/方向B" 或 "Issue #55 lineage / Issue #56 lineage" 表达, 实际当前实现是 taskA/ + taskB/ 目录; 检索代码 / 产物 / 数据时用此映射.

---

## 项目元数据 (6 条)

**仓库**: HG-Rec 复现 + κ-Stereographic 变体实验, 当前基线 Task #84 (R@10=0.1020), 不用 phonism/Toys.

**GPU**: 4× NVIDIA L40S (sm_89, 46GB/卡), 驱动 580.126.20, CUDA 13.0 (torch 2.11.0+cu130) / 12.x (TF kgat_mckg).

**目录**: 上游 clone 只读 (HG-Rec/, data/, papers/), 可写 (verdicts/, products/, logs/).

**流水线**: 4 阶段 — Stage 1 sentence-t5-base embedding → Stage 2 Poincaré RQ-VAE SID (3→4 层去重 digit) → Stage 3 T5-mini 训练 → Stage 4 R@K/NDCG 评估.

**评估**: HG-Rec baseline R@5/10/20 = 0.0816/0.1020/0.1279, NDCG@5/10/20 = 0.0690/0.0755/0.0821, 决策阈值 R@10 > 0.1020 GO.

**依赖**: torch >= 2.0, pytorch-lightning, hydra-core, transformers, torchmetrics (详见 requirements.txt).