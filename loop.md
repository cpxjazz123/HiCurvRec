# GRID Reproduction Task Plan

> **本文件** = 当前活跃任务的"做什么"清单, 配合 `descriptions/` 给出每个 task 的详细设计.
> **核心目标**: 复现 Snap Research GRID 流水线 (Stage 1 LLM embedding → Stage 2 RQ-VAE SID → Stage 3 TIGER 训练 → Stage 4 推断 → Recall/NDCG 评估)
> **数据集**: 仅 `data/amazon_data/toys/`
> **量化算法**: 仅 RQ-VAE
> **超参约束**: `num_hierarchies=3` (Stage 2 训练) → 推断后追加 1 列 dedup digit → Stage 3/4 用 `num_hierarchies=4`
> **种子**: `seed=42` (跨 run 固定)
> **总阶段数**: 4 个 (Stage 1 / Stage 2 训练 / Stage 2 推断 / Stage 3+4)

---

## §1. RAG 模式提醒

**禁止使用 fallback 逻辑**: 所有代码逻辑必须明确处理预期缺失 (返回 None/空值) 与预期外失败 (直接 raise). 任何默认值/回退/降级策略都禁止.

---

## §2. Loop 执行流程

每个 loop tick 必须按以下顺序推进:

1. **读取 `loop.md` §16 当前活跃任务**, 若有活跃任务则推进其完成度跟踪表中的一项
2. **执行该步骤**: 跑对应命令/分析/写 verdict
3. **更新 §16**: 若完成则归档到 verdicts 并从 §16 删除, 同时登记下一个待办任务

---

## §3. 阶段指标 (硬约束)

GRID 流水线有 4 个 Stage, 评估指标在 Stage 4 推断后:

| 指标 | 目标 (RQ-VAE Toys) | RK-Means Toys (参考) | TIGER 原论文 |
|------|------|------|------|
| Recall@5 | ≥ 0.034 | 0.0376 | 0.0446 |
| Recall@10 | ≥ 0.051 | 0.0577 | 0.0679 |
| NDCG@5 | ≥ 0.022 | 0.0243 | - |
| NDCG@10 | ≥ 0.028 | 0.0308 | - |

评估实现: `src/components/eval_metrics.py` 中的 `SIDRetrievalEvaluator`, `NDCG`, `Recall` 类.

---

## §4. 流水线架构

```
Stage 1 LLM Embedding (flan-t5-xl)
    ↓ (N, 2048)
Stage 2 RQ-VAE SID
    ↓ (4, N) 3 hierarchies + 1 dedup digit
Stage 3 TIGER Training
    ↓ checkpoint
Stage 4 TIGER Inference + Recall/NDCG eval
```

每个 Stage 独立 Stage 训练可重做, Stage 间只传递张量/路径.

---

## §5. 启动命令模板

```bash
# Stage 1: LLM 嵌入
python -m src.inference experiment=sem_embeds_inference_flat \
    data_dir=data/amazon_data/toys \
    embedding_model=google/flan-t5-xl \
    task_name=<task_id>

# Stage 2.1: RQ-VAE 训练
python -m src.train experiment=rqvae_train_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=<stage1_pt> \
    embedding_dim=2048 num_hierarchies=3 codebook_width=256 \
    task_name=<task_id>

# Stage 2.2: RQ-VAE 推断
python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/toys \
    embedding_path=<stage1_pt> \
    ckpt_path=<stage2.1_ckpt> \
    num_hierarchies=3 codebook_width=256 \
    task_name=<task_id>

# Stage 3: TIGER 训练
python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=<stage2.2_cluster_ids_pt> \
    sequence_length=120 num_hierarchies=4 \
    task_name=<task_id>

# Stage 4: TIGER 推断 + 评估
python -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=<stage2.2_cluster_ids_pt> \
    ckpt_path=<stage3_ckpt> \
    sequence_length=120 num_hierarchies=4 \
    task_name=<task_id>
```

---

## §6. 配置文件

GRID 用 Hydra 做配置组合, 每个 run 形如:

```bash
python -m src.train experiment=<yaml_basename> data_dir=<path> [override_key=value ...]
```

`configs/experiment/` 下 7 个 yaml 名称即可传入 `experiment=`. `src/train.py` 和 `src/inference.py` 是仅有的 CLI 入口.

---

## §7. 早停策略

Stage 3 TIGER 训练:
- val_R@5 不再提升 ≥ 3 个 val_check (每 100 steps 一次) → 停止训练
- max_steps=100000 是上限

---

## §8. 子任务论文核验 (强制)

每个 stage 完成后必须核验:
1. **shape**: 输出张量维度是否正确 (e.g. Stage 2 SID `(4, N)`, Stage 4 predictions `(N_users, 10, 4)`)
2. **codebook coverage**: Stage 2 SID 每层使用码本比例 ≥ 80% (256 桶至少用 205)
3. **eval metric**: Stage 4 评估召回率 ≥ 0.034 (RQ-VAE Toys 阈值)

不通过则需要修复, 不能直接进下一 stage.

---

## §9. 结果归档规则

### §9.1 强制 verdict 文件

每个任务必须在最后一个 stage + 评估 JSON 产出后立即写:

`verdicts/task<ID>_result.md`

即使失败/中断也要写, 记录 root cause.

### §9.2 verdict 文件结构

1. **任务目标**: 一句话说明要验证什么
2. **执行时间线**: 各 stage 时间戳
3. **关键指标**: Recall@5/10, NDCG@5/10
4. **分析解读**: 与 baseline/paper 对比, 根因
5. **产物清单**: log paths, ckpt paths, eval json
6. **后续建议**: ROI 排序的下一步

### §9.3 verdict 必须包含 `result:` 行

`result:` 行是任务交付的**唯一**自动识别信号 (任务状态机依赖此行). 必须放在 verdict 文件末尾.

---

## §10. 完成判定

每个任务满足下列任一即视为完成:
1. `verdicts/task<id>_result.md` 已存在且包含 `result:` 行
2. 该任务的 Stage 4 评估已通过决策阈值
3. 任务的 PID 已不存在 (训练已结束)

---

## §11. 任务活跃度原则

- **同时只允许一个活跃任务**: 登记新任务前必须先把 §16 当前活跃任务归档
- **完成立刻归档**: verdict 写完后立刻从 §16 删除该任务 (R8 强制清理)
- **无任务时必须主动推进**: §16 表格为空时禁止空闲等待, 必须按 R11.3 backlog / 自主决策推进 (R10 主动模式, 2026-07-23 修订)

---

## §12. GPU 并行使用规则

按 CLAUDE.md R7:
- 新任务启动前必须 `nvidia-smi` 确认目标 GPU 空闲 (util<10%, mem<5GB)
- 4 张 A40 都可并行使用, 禁止等待已占用 GPU
- 每个独立实验建议绑定单独 GPU

---

## §13. 路径约定

| 用途 | 路径 |
|------|------|
| 任务定义 | `/home/wlia0047/ar57/wenyu/GeneRec/descriptions/task<N>_<slug>.md` |
| 调度文件 | `/home/wlia0047/ar57/wenyu/GeneRec/loop.md` |
| 结论 | `/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task<N>_result.md` |
| 训练产物 | `products/task<N>/train/<id>/checkpoints/` |
| 推断产物 | `products/task<N>/inference/<id>/pickle/` |

---

## §14. 命名规则

- `descriptions/task<N>_<slug>.md` — slug 用 2-4 个 kebab-case 英文单词
- `verdicts/task<N>_result.md` — 一律 result.md
- `verdicts/task<N>_recall_eval.json` — 评估结果 JSON
- `scripts/task<N>_<purpose>.py` — 任务专属脚本

---

## §15. 监控与日志

### §15.1 任务进度监控

每个任务在 `logs/<task_name>/` 下保留:
- `train.log` 或 `infer.log` — stdout/stderr
- `runs/<date>/<time>/` — Lightning/Hydra 训练产物

### §15.2 Checkpoint 路径解析

Lightning 保存的 ckpt 形如 `checkpoint_epoch=000_step=000100.ckpt`, Hydra 严格 parser 不能处理 `=` 符号. 解法: 用 `ln -sf <abs_path> logs/<task_id>_best.ckpt` 建简单符号链接.

### §15.3 任务分析子目标

每个任务如有分析类子目标, 在 §16 表格中以子行追踪.

### §15.4 R12 强制 checkpoint 保存且只保留最新 (2026-07-23 新增)

> **事故背景**: 2026-07-23 Task #83 P5-SID 训练完成 53050 steps (1h 41min), 但 evaluate 时 transformers `ValueError: ...['whole_word_embedding_type']` 崩溃, model 从未保存 → 整场训练浪费. 同类风险在 task82 P5-CID 即将发生.

**强制规则**:
- ✅ **每个训练必须在固定阶段 (推荐 epoch 末) 保存 checkpoint** (即使后续 evaluation / processing 失败也能保住产物)
- ✅ **每次保存新 checkpoint 必须删除旧 checkpoint** (磁盘只保留最新, 防止路径歧义)
- ✅ **launcher 必须写 `products/task<N>/_TRAINING_PID`** (R88 daemon 检测用)
- ❌ **禁止** 只在训练 final 阶段保存 (中途崩溃 = 全部丢失)
- ❌ **禁止** 保留多个 epoch 的 checkpoint (磁盘浪费 + 路径歧义)

**实施 (R11.3 自主决策)**:
- **LLM-RecSys-ID (task82/83)**: main.py:310 后插入 `if rank == 0: os.remove(args.model_dir); torch.save(model.state_dict(), args.model_dir)` (patched 2026-07-23)
- **HuggingFace Trainer**: `save_strategy="epoch", save_total_limit=1` 自动保留最新
- **RecBole (FDSA/S3Rec/LightGCN)**: 默认 `checkpoint_dir` + `stopping_step` 符合 R12 (best metric 自动覆盖)

**配套规则**:
- **task88 daemon**: 检测 `is_pid_alive(PID) AND ckpt_exists(ckpt_path)` 触发 Stage 4 inference
- **task81/82/83 launcher**: 启动时 `echo $TRAIN_PID > _TRAINING_PID` + 训练结束 `rm _TRAINING_PID`

### §15.5 R13 禁止使用 Worktree / EnterWorktree (2026-07-23 新增)

> **事故背景**: 2026-07-23 AI 误用 `EnterWorktree` 创建 worktree, 用户立即撤销并新增 R13 规则.

**强制规则**:
- ❌ **禁止** 使用 `EnterWorktree` 工具
- ❌ **禁止** 使用 git worktree 机制 (`git worktree add` / `ls` 等)
- ❌ **禁止** 创建 `.claude/worktrees/<name>/` 目录下任何文件
- ❌ **禁止** 在对话中建议或提议使用 worktree
- ✅ 代码/配置修改直接落在 cwd 共享 checkout
- ✅ Edit 工具被 cwd 拦截时, 用 `Bash` + `sed/cat/echo` 直接改共享 checkout 文件
- ✅ 临时文件写到 `$CLAUDE_JOB_DIR/tmp`

---

### §15.6 R14 GitHub Issue 自动监听 + 即时处理 (2026-07-29 新增, 用户要求)

> **用户要求**: 每次检查 https://github.com/WENYULIANG123/GeneRec 是否有新的 issue, 如果有, **马上根据 issue 的要求完成并且 commit**, 并且**尽量并行完成 issue**.



**强制规则**:
- ✅ **每次 loop tick 第一步必跑**: `gh issue list --repo WENYULIANG123/GeneRec --state all --limit 30` 扫描所有 issue 状态 (open + closed).
- ✅ **发现 open issue 立即处理**: 不等用户决策, 按 R11.5 自主决策推进 (跟当前 backlog 同等优先级).
- ✅ **commit 粒度**: 每 issue 一个独立 commit (避免混合 commit 难追踪). commit message 格式: `Issue #<N>: <一句话结论> (Task #<task_id>)`.
- ❌ **禁止** 跳过 issue 留到下次 tick 处理 — 必须当 tick 闭环 (除非 GPU/数据不足, 这种情况在 issue comment 注明 + §16 标记).
- ❌ **禁止** 等用户授权 issue 处理 — R11.5 自主决策原则, issue 要求即任务. 用户 override 仅在不可逆/破坏性操作时触发.
- ❌ **禁止** 在 issue 处理流程中用 fallback 掩盖失败 — R2 仍生效, 失败即 raise + issue comment 记录.
- ✅ **autonomous issue triage**: 如 issue 描述模糊, 按 R11.2 选推荐方案 + issue comment 明示选了什么 + 备选方案. 绝不抛回用户. (符合用户 2026-07-29 override "不允许等用户拍板, 必须自行决定").

**实施细节**:
- `gh` CLI 路径: `/usr/bin/gh` (已安装, auth 已配置 WENYULIANG123 账号).
- 监听频率: 每个 loop tick 第一步 (R10 主动推进的前置步骤).
- issue 编号 vs task 编号: issue #N 跟 task #N **不一定对应** (issue 是用户提的, task 是 AI 派的). 一个 issue 可能映射多个 task. 推荐命名: `task<M>_issue<N>_*.md` 让 task 号跟 issue 号双向追踪.
- commit push 策略: 默认只 commit 不 push (避免 AI 误推破坏主分支). 用户授权后用 `git push` 推送. **严禁** `--force` push.
- 关联 commit 跟 issue: `gh issue close` 时 commit 自动关联. 若 GitHub UI 不显示, 在 issue comment 里手动贴 commit hash.

**与现有规则的关系**:
- R11 (autonomous decision) > R14: issue 处理默认走 R11.2 兜底顺序, 不允许等用户拍板
- R7 (并行 GPU) > R14: 多 issue 并行时仍遵守 GPU 不抢卡约束 (4 GPU 各跑 1 issue 主臂)
- R9 (任务编号连续) > R14: issue 处理创建新 task 必须 R9-Enforce 三层防护 (取 max+1 编号)
- R12 (checkpoint 强制保存) > R14: issue 触发的训练仍遵守 R12 强制 ckpt 保存


### §15.7 R15 Issue 完成时 verdict 文件必须 push 到仓库 (2026-07-30 用户新增)

> **用户要求 (2026-07-30)**: 每次完成 issue 的时候, 都需要把对应的完成这个 issue 的 verdict 文件推送到仓库.
> **事故背景**: 早期 issue (#1-#100) verdict 写完但只 commit 不 push, reviewer 看不到完整闭环记录. 后续 R9-Enforce + R14 处理流程仍未强制 push, 历史 verdict 滞留本地.

**强制规则**:
- ✅ **每次 issue 闭环时** (写完 verdict 文件 + commit), **必须** `git push` 把 commit 推到 origin (默认推到主分支 `main`).
- ✅ **commit message 必须含**: issue 编号 + 关键结论 + verdict 路径. 例: `Issue #43 Gate 2a PASS (Task #334): HypPreEncoder 5/5 test, c=0.74. verdicts/task334_issue43_gate2a_hyp_pre_encoder_result.md`
- ✅ **verdict 文件本身必须 tracked in git** (不能只在本地 untracked 状态). R8 + R9-Enforce 已保证 verdict 文件被 `git add`, R15 保证 commit 被 push.
- ✅ **push 之前必跑**: `git status --short` 确认 working tree 干净, 无未跟踪残留.
- ✅ **push 之后必跑**: `git log --oneline -1` 确认 commit hash 已落在 origin (用 `git ls-remote origin main` 或 `git fetch origin` 验证).
- ✅ **push 之后必须关闭 GitHub Issue**: `gh issue close <N> --repo WENYULIANG123/GeneRec --reason completed` (或 `not_planned` 走 D4 收口). 此步在 `git push` 之后、下一个任务启动之前执行.

**例外 / 豁免**:
- ⏸️ **本地调试 / 中间产物**: 未完成的中间 verdict / 调试 log / scratch files **不强制 push** (R2 不允许 fallback 掩盖错误, 但允许中间产物本地滞留).
- ⏸️ **GPU 训练中的中间 ckpt**: `products/task<N>/*.pth` 大型 binary 通常 `.gitignored` 或选择性 push (per R12 R88 daemon 验证需要本地可达).
- ⏸️ **不可逆操作**: 任何 `--force` push / branch 改写 / tag 删除 → 严禁. 必须 owner 显式授权.

**与现有规则的关系**:
- **R14 (issue 自动处理) > R15**: R14 已要求 commit, R15 加强 push (commit 在本地 ≠ reviewer 可见).
- **R8 (§16 清理) + R15**: 完成 issue 时既要从 §16 表格删除行, 又要 push verdict.
- **R9-Enforce (descriptions/ contiguous) + R15**: descriptions + verdicts 都必须 tracked AND pushed.
- **R11.5 自主决策 + R15**: 默认 push. 不允许"等用户授权 push" (push 是 reviewer transparency 的硬要求, 不是 R11.4 critical 决策).

**实施细节**:
- **push 命令**: `git push origin main` (默认无 --force). 若 push 失败 (网络/auth), 在 issue comment 注明 + 标记 §16 backlog 真空状态 + 重试.
- **CI / lint 兼容**: push 前本地跑 `python3 -m py_compile` (R4 强制) + dispatcher 5/5 PASS (R14 配套).
- **频率**: 每个 issue 闭环 push 一次 (跟 commit 同频). 不允许批量 push (合并多 issue 单一 commit 难追踪).

**关键 caveat**:
- ❌ **禁止** 把 R15 误用为"push 一切" — 大型 ckpt / logs / pids 仍按 .gitignore 规则保留本地.
- ❌ **禁止** 在 push 失败时跳过该 issue — 失败即记录 + 重试, 不允许静默 skip.
- ❌ **禁止** 用 fallback 跳过 push (R2 兜底).


## §16. 当前活跃任务 (执行情况 / 进度)

> **🟢 §16 框架 (universal rule)**: 本节是 loop 当前**唯一**允许的"执行情况/进度"记录位置 (CLAUDE.md 不写). 历史任务记录**禁止**入 loop.md, 必须落 verdicts/ + descriptions/ (per R9-Enforce + R15).

### §16.1 表格规范 (universal rule)
- **活跃任务表** (有任务时填, 没任务时空表或注释 "R10 backlog 真空"):
  | Task ID | Issue | 类型 | 当前阶段 | GPU | 进度 | ETA |
  |---------|-------|------|---------|-----|------|-----|
  | task<N> | #X | Stage X/Y Gate Z | Stage 1 / Stage 2 / Stage 3 / Stage 4 | GPU 0/1/2/3 | X% / epN/M | ~Yh |
- **每行 ≤ 100 字** (简洁, 详情在 verdicts/)
- **任务完成**: 立即从 §16 表中删除该行 (R8 强制, 不写 "已归档")
- **行数限制**: 同时活跃 ≤ 5 行 (per R7 并行 GPU 容量)

### §16.2 当前活跃任务 (2026-07-31 14:42)

| Task ID | Issue | 类型 | 当前阶段 | GPU | 进度 | ETA |
|---------|-------|------|---------|-----|------|-----|
| (空) | — | — | — | — | — | — |

> **状态**: R10 backlog **真空** 维持. Issue #47/#48/#49/#50/#43/#51/#52/#53/#54/#57/#61/#62/#63/#64/#65 全部闭环 (verdicts 落盘 + GitHub closed per R16). 唯一 ROI > 0 路径 (#30+#43 联合 / Issue #43 HypPreEncoder 深化) 已穷尽. 4 张 L40S 全空闲. 等 owner 拍板新方向.

### §16.3 历史任务记录位置 (universal rule)

- **历史 task verdicts** → `verdicts/task<N>_*.md` (R9-Enforce + R15 强制)
- **历史 task descriptions** → `descriptions/task<N>_*.md` (R9-Enforce 强制 contiguous)
- **历史 issue 关闭记录** → GitHub issue timeline (R16 强制 `gh issue close --reason completed`)
- **历史 drift-cycle / R-Drop / 4-Gate 协议等详细 task 追踪** → 不写入 loop.md (本节仅记录当前活跃任务, 历史详情查 verdicts/)

---

## §18. 每次 loop tick 强制检查 open issue + 完成 + 关闭 (2026-07-31 新增, 硬规则)

> **背景**: 用户 2026-07-31 反馈, `/loop` tick 偶尔报告 "无 actionable work" 但实际 GitHub 有 OPEN issue 未完成. 当前 §16 (主动推进) + R15 (issue push 闭环) 缺少**强制检查** + **强制关闭**两步. §18 强制每个 loop tick 必须先查 issue, 有就完成, 完成后关闭, 没有才可什么都不做. 对应 CLAUDE.md R16.

### §18.1 核心要求
- ✅ **每次 loop tick 第一步** (无论 §16 是否有 backlog 候选): 调用 `mcp__github__list_issues state=OPEN` 或 `gh issue list --state open` 检查当前 open issue.
- ✅ **有 open issue** → **必须**根据 issue 要求完成任务 (按 issue 的 Gate 顺序执行, 前 Gate 不通过不进下一 Gate, 不一定跑完所有 Gate).
- ✅ **issue 完成后** (无论 GO / NO-GO / PARTIAL) → **必须** `gh issue close --reason completed` 关闭 issue.
- ✅ **没有 open issue** → 可以什么都不做 (no actionable work).
- ❌ **禁止** "loop tick 无 actionable work" 但 GitHub 有 OPEN issue 未处理 (§18 违规).
- ❌ **禁止** 完成 issue 后不调用 `gh issue close` (§18 违规).
- ❌ **禁止** "等用户拍板" / "等用户授权" 拖延 issue 关闭 (R11.5 自主决策已生效, §18 加强).

### §18.2 实施细节
- **检查命令**: 优先 `mcp__github__list_issues` (per owner 工具栈), 备选 `gh issue list --state open --repo WENYULIANG123/GeneRec --limit 30`.
- **完成定义**: verdicts/task<N>_*.md 已落盘 + commit (含 issue 编号 + 关键结论 + verdict 路径) + push (R15) 三件套齐全.
- **关闭命令**: `gh issue close <issue_number> --reason completed --repo WENYULIANG123/GeneRec` (默认 reason=completed; 若 owner 需 "not planned" 或其他 reason, R11.5 自主决策).
- **关闭时机**: commit push 之后立即关闭 (R15 + §18 同频, 不允许 commit 后不 close).
- **关闭 comment 模板**: 贴 commit hash + verdict 路径 + 关键结论 (R15.3 已要求 commit 关联, §18 加强 comment 显式标注).

### §18.3 与现有规则的关系
- **§16 (主动推进) > §18**: §16 真空 + 有 open issue → §16 + §18 联立强制启动 issue 处理, 不允许 "无 actionable work".
- **R15 (push) ⊂ §18 (close)**: R15 只保证 push, §18 加强 issue 状态必须在 GitHub 上 close.
- **R11.5 (自主决策) > §18**: issue 关闭决策由 AI 自主 (基于 verdict 落盘 + commit + push), 不需要等 owner 拍板.
- **R8 (§16 清理) + §18**: 完成 issue 时既要从 §16 表格删除行, 又要 push verdict, 又要 gh issue close.

### §18.4 关键 caveat
- ❌ **禁止** 跳过 §18 强制的"先检查 issue" 步骤 (即使认为"无 issue 可做", 也必须先跑 `list_issues`).
- ❌ **禁止** 用 "drift-cycle 终结" / "backlog 真空" 跳过 issue 检查 (R10 backlog 真空 ≠ GitHub issue 真空).
- ❌ **禁止** 关闭 issue 时不写 reason (默认 --reason completed 强制).
- ❌ **禁止** 用 fallback "issue 不重要先放着" (R2 不允许 fallback, §18 强制关闭).

### §18.5 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-31 §18 新增 | loop tick 报告 "无 actionable work" 但 GitHub 有 OPEN issue 未处理 (#63/#64/#65 三方向 ×× 闭环后 issue 仍 OPEN) | §16 backlog 真空 ≠ GitHub issue 真空, §16 没强制 issue 检查 + 关闭 | §18 强制每个 tick 第一步检查 + 完成 + 关闭 | 每次 loop tick 第一步必须 list_issues |
