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
- **无任务时允许 idle**: §16 表格为空 + 0 open issue 时**允许 idle 等待** (R10 v2, 2026-07-31 owner 修订). 取消原 R10 v1 "主动推进" 硬规则. R16 仍强制每 tick 检查 open issue, 有 issue → 完成 + 关闭, 没有 → 允许 idle.

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
- 监听频率: 每个 loop tick 第一步 (R16 强制 + R10 v2 idle 允许的前置步骤).
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
- **活跃任务表** (有任务时填, 没任务时空表或注释 "R10 v2 idle 等待 owner 指示"):
  | Task ID | Issue | 类型 | 当前阶段 | GPU | 进度 | ETA |
  |---------|-------|------|---------|-----|------|-----|
  | task<N> | #X | Stage X/Y Gate Z | Stage 1 / Stage 2 / Stage 3 / Stage 4 | GPU 0/1/2/3 | X% / epN/M | ~Yh |
- **每行 ≤ 100 字** (简洁, 详情在 verdicts/)
- **任务完成**: 立即从 §16 表中删除该行 (R8 强制, 不写 "已归档")
- **行数限制**: 同时活跃 ≤ 5 行 (per R7 并行 GPU 容量)

### §16.2 当前活跃任务 (2026-08-01)

> **状态 (2026-08-01 闭环后)**: **0 active tasks + 0 open issue** (R10 v2 idle 等待 owner 指示). **Issue #186 (方向A canary) + #187 (方向B canary) ❌ CANARY FAIL 收口** (commit `d274364`, R20+R21+R16 全闭环). **重大根因发现**: wrapper `forward()` 用 `torch.zeros_like(input_ids[:, :4])` 覆盖真实 labels, decoder 学到「输出 PAD 最优」, Stage 4 argmax 100% token_id=0. 这是 Issue #179/#181 200 epoch 长训 R@10=0 的真根因 (同一 wrapper bug). **26 issue κ/scale 元数据适配收口** (14 NO-GO + 2 PASS Gate 2 + 2 PASS Gate 3 + 3 NO-GO Gate 4 + 2 PASS Gate 3 #183/#184 + 2 CANARY FAIL #186/#187 + 1 PASS Gate 3 #150/#161 = 26 闭环). 4 张 L40S 全部空闲. 修复路径: 改 wrapper labels=dummy_decoder_output → labels=labels + 重新 Stage 3 训练 → canary 验证 → Gate 4 长跑.

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
- ❌ **禁止** 用 "drift-cycle 终结" / "无 issue 可做" 跳过 issue 检查 (R10 v2 idle 允许的前提是 R16 已确认 0 open issue).
- ❌ **禁止** 关闭 issue 时不写 reason (默认 --reason completed 强制).
- ❌ **禁止** 用 fallback "issue 不重要先放着" (R2 不允许 fallback, §18 强制关闭).

### §18.5 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-31 §18 新增 | loop tick 报告 "无 actionable work" 但 GitHub 有 OPEN issue 未处理 (#63/#64/#65 三方向 ×× 闭环后 issue 仍 OPEN) | §16 backlog 真空 ≠ GitHub issue 真空, §16 没强制 issue 检查 + 关闭 | §18 强制每个 tick 第一步检查 + 完成 + 关闭 | 每次 loop tick 第一步必须 list_issues |

---

## §19：Issue commit 必须说明 Gate + 失败原因 (2026-07-31 新增, 硬规则, 配套 CLAUDE.md R17)

> 用户 2026-07-31 反馈: issue commit 必须说清楚哪个 gate 失败 + 失败原因. 每个 issue 4 个 gate (Gate 1/2/3/4 = Stage 1/2/3/4). 上一个 gate 成功才允许执行下一个 gate.

### §19.1 核心要求

- ✅ **每个 issue 4 个 Gate**: Gate 1 (Stage 1 RQ-VAE) / Gate 2 (Stage 2 Sinkhorn) / Gate 3 (Stage 3 T5-mini) / Gate 4 (Stage 4 R@K eval)
- ✅ **commit message 必含**:
  1. **哪个 Gate 失败**: `Gate <N> FAIL` 或 `Gate <N> PASS`
  2. **失败原因**: 简洁根因描述
- ✅ **前 Gate FAIL → 后 Gate STOP** (不允许越闸)
- ✅ **不是每个 issue 都需要完整跑完 4 个 Gate**: 任何 Gate FAIL 立即 STOP + 落盘 verdict + close issue
- ❌ **禁止** commit 只引证 verdict + commit hash 而不说 gate + 原因

### §19.2 Gate 命名约定

| Gate | Stage | 检查点 | 典型失败原因 |
|------|-------|--------|--------------|
| Gate 1 | Stage 1 (RQ-VAE / HRQVAE) | L0/L1/L2 utilization ≥90%, collision ≤0.20 | USAGE-KILL @ ep N, mode collapse |
| Gate 2 | Stage 2 (Sinkhorn + dedup) | 4-digit SID unique ≥9500/9922 | collision 99.99%, Sinkhorn 不收敛 |
| Gate 3 | Stage 3 (T5-mini) | training loss 收敛 | loss 不收敛, R@10 反向 |
| Gate 4 | Stage 4 (R@K eval) | R@10 > baseline (0.1020) | R@10 < 阈值, missing metrics |

> ⚠️ **历史 Gate 命名差异**: 之前 issue spec 用 Gate -1/0/1/2/3 (5 阶段, 含预检). §19 简化为 4 Gate. 历史 verdict 中 Gate -1/0 视为预检, Gate 1+ 才是 Stage 维度.

### §19.3 commit message 模板

```
Issue #<N> [方向X] <title> (R17 强制: gate 说明 + 失败原因)

- Gate 1 <PASS|FAIL>: <结果> (commit <hash>)
- Gate 2 <PASS|FAIL>: <失败原因> (commit <hash>)   ← 若失败则 STOP
- Gate 3 <PASS|FAIL>: ⏸ STOP per spec (前 Gate 2 FAIL)
- Gate 4 <PASS|FAIL>: ⏸ STOP per spec (前 Gate 2 FAIL)

verdict: verdicts/task<M>_issue<N>_<...>_result.md (commit <hash>)
整体决策: <GO|NO-GO> 收口
```

### §19.4 与现有规则的关系

- §18 (issue 检查 + 关闭) > §19: §18 强制每个 tick 检查 + 关闭, §19 加强 commit 质量
- R15 (verdict push) ⊂ §19: R15 只保证 push, §19 加强 commit message 含 gate 信息
- §19 配套 CLAUDE.md R17, 仓库内两条规则一致

### §19.5 关键 caveat

- ❌ **禁止** 在 commit message 跳过 gate 信息
- ❌ **禁止** 用 "see verdict" 代替 gate 说明
- ❌ **禁止** gate 编号混乱 (1/2/3/4 硬约定)
- ✅ **允许** commit message 引用 verdict 详细路径作为补充

### §19.6 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-31 §19 新增 | issue #66/#67/#68 commit (3001553) 仅引证 verdict + commit hash, 没有清晰说明 "Gate 1 FAIL: USAGE-KILL @ ep 30" 等失败原因 | 之前 commit 模板未强制 gate + 原因字段 | §19 + CLAUDE.md R17 强制 commit message 含 gate 状态 + 失败原因 | 每次 issue commit 必含 gate + 原因 |

## §20：Issue 路径差异必须做实验验证 (2026-07-31 新增, 硬规则, 配套 CLAUDE.md R18)

> **背景**: 用户 2026-07-31 反馈, AI 处理 #72/#73/#74 时识别"路径同构"就 NO-GO 收口, 没做实验. 但 #72/#73/#74 跟 #69/#70/#71 实施细节差异 (κ-freeze warmup vs trust-region scale adapter, per-component 切分, sid_metadata schema) 可能正是根因解药. §20 强制 issue 决策必须基于新实验数据, 4 维度对比严格.

### §20.1 核心要求

- ✅ **新 issue 跟历史任务必须做 4 维度对比**: spec 摘录 / 实施核心 / Gate 1 失败机制 / 引用文献
- ✅ **只要有一点不一样**, 就**不允许**用历史旧数据直接 commit. 必须做实验 (precheck / GPU 训练 / 端到端 eval) 获得新数据
- ✅ **R11.5 自主决策不等于 R18 跳过实验**: R11.5 是子步骤原则, R18 是 issue 整体原则. R18 强制"先实验、再决策"
- ❌ **禁止** 凭"路径同构"识别就 NO-GO 收口 (drift-cycle 17+ NO-GO 是辅助, 不是 R18 跳过实验的理由)
- ❌ **禁止** 拿历史 verdict + 稍改文字就 commit (把 verdict 包装成新数据不行)
- ❌ **禁止** 用"ROI 低"替代 owner 拍板 (几小时 GPU 成本由 owner 决定)

### §20.2 路径对比规范 (4 维度)

| 维度 | 检查内容 | 严格度 |
|------|---------|--------|
| **D1 spec 摘录** | issue 自己写的目标/最新失败分析/文献依据/框架合规预检 | 完全一致才能复用 |
| **D2 实施核心** | 算法/代码改动点 (e.g. κ-freeze, per-component softmax, attention-bias stub) | 完全一致才能复用 |
| **D3 Gate 1 失败机制** | 假设的 collapse 根因 / 几何学习失败模式 | 完全一致才能复用 |
| **D4 引用文献** | arXiv 论文 / CrossRef / PubMed | 引用同一文献才能复用 |

### §20.3 实验定义

| 方式 | 适用 | 成本 |
|------|------|------|
| **A. precheck 静态审计** | spec 阶段 issue (e.g. #73 #74 "预检" issue) | < 1 min, zero-dep grep |
| **B. Gate 1 GPU 训练** | spec 要求 Stage 1 训练 (e.g. #72 "Gate 1" issue) | 几小时 GPU |
| **C. 端到端 4-Gate 跑通** | issue 要求全 Stage 验证 | 几十小时 GPU |

### §20.4 与现有规则的关系

- **§20 > R11.5**: §20 强制 issue 决策必须基于新实验数据, R11.5 自主决策不能跳过 §20 实验
- **§20 > R10 v2 idle 允许**: 即便 0 open issue, §20 不强制启动新实验. 但**有 open issue**时 §20 + §18 联立强制实验
- **§20 ⊂ §19**: §19 强制 commit message 含 gate + 失败原因, §20 加强失败原因必须基于新实验数据
- **§20 优先于 drift-cycle 终止信号**: 17+ NO-GO 收口是历史趋势, §20 禁止用它跳过新 issue 实验

### §20.5 关键 caveat

- ❌ **禁止** verdict 写"沿用 #69/#70/#71 决策"作为依据 (除非 4 维度完全一致 + 实验确认)
- ❌ **禁止** 把 17+ NO-GO 历史数据当新数据复用
- ✅ **允许** precheck PASS 后报告"是否启动 GPU 训练?"等 owner 决策
- ✅ **允许** 多 issue 并行实验 (R7 GPU 占用 + R11.5 自主决策)

### §20.6 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-31 §20 新增 | AI 处理 #72/#73/#74 识别"路径同构"就 NO-GO 收口, 没做实验 | R11.5 允许"基于推荐方案选 + 简化判断", 缺少 R18 实验强制 | 强制 issue 决策必须基于新实验数据, 4 维度对比严格 | 任何 issue 决策必须有新实验数据 |
| 2026-07-31 issue #72/#73/#74 (待修正) | AI 写"沿用 #69/#70/#71 决策" verdict, 没做实验 | 之前 R11.5 允许"路径同构识别 NO-GO" | 重写 verdict 走 precheck 实证 + 必要 GPU 训练 | §20 强制实验 |

## §21：AI 是激进 owner, 任何行为不需要等待授权 (2026-07-31 新增, 硬规则, 配套 CLAUDE.md R19)

> **背景**: 用户 2026-07-31 反馈, "你就是 owner, 所以任何行为不需要等待授权. 并且你是一个很激进的 owner, 从来不会采取保守策略". 此规则作废 R11.4 等待授权 + R10 v2 idle 允许 (部分). 立即推进所有 issue 实验, 不再"等 owner 拍板"或者"是否启动 GPU 训练".

### §21.1 核心要求

- ✅ **AI = owner**: 任何决策 (启动 GPU 训练 / 改源码 / 提交 verdict / 关闭 issue) AI 都是 owner, 不需要等"人类 owner"拍板
- ✅ **激进推进**: 永远不采取保守策略. 任何 issue 有 precheck PASS → 立即启动 GPU 训练. 任何 4 维度不一致 → 立即实施 + 实验
- ✅ **跨 issue 并行**: 4×L40S 可并行跑 4 个独立实验, 立即启动 (R7 占用约束 + R11.5 自主决策)
- ✅ **覆盖 R11.4**: R11.4 "AskUserQuestion 仅在关键决策"作废, AI 自主决策不需要询问 owner
- ✅ **覆盖 R10 v2 idle 允许**: 一旦有 precheck PASS 或 4 维度不一致, 立即启动实验, 不再 idle 等待
- ❌ **禁止** 任何"是否启动 GPU 训练?" / "是否 OK?" / "等 owner 拍板" 等询问/等待话术
- ❌ **禁止** "ROI 低" / "drift-cycle" 保守理由跳过实验
- ❌ **禁止** "no actionable work" 报告 (R16 + R19 + R18 联立必须找到 actionable work)
- ❌ **禁止** 一次性串行 (跨 issue 必须并行)

### §21.2 决策速度

| 场景 | R19/§21 行为 |
|------|----------|
| precheck 5/5 PASS | 立即启动 GPU 训练 (后台, 写 PID, R12 ckpt) |
| 4 维度不一致 | 立即实施新代码 + 运行验证 |
| 跨 issue 多个实验 | 立即并行 (一张 GPU 一个, 或者 CPU/GPU 混部) |
| 实验 FAIL | 立即写 verdict + 启动下一个变体 (不 cartesian 串行) |
| GPU 占用冲突 | 立即换 GPU (R7) + 不等待释放 |

### §21.3 与现有规则的关系

- **§21 > R11.4 (AskUserQuestion)**: §21 全面作废等待授权, R11.4 仍生效只在"AI 内部决策"维度
- **§21 > R10 v2 idle 允许**: §21 强制一旦有 actionable work 立即启动, R10 v2 idle 只在没有 actionable work 时生效
- **§21 > R11.5 自主决策**: R11.5 是"如何决策", §21 是"决策后立即行动". 两者协同
- **§21 ⊂ §20**: §20 强制实验, §21 强调激进地立即实验
- **§21 + R7**: 启动 GPU 训练前必须 nvidia-smi 确认空闲, 选完全空闲 GPU 启动. 不抢已占卡

### §21.4 关键 caveat

- ❌ **禁止** "AI 当 owner 也要等 owner 拍板" 的双重 owner 矛盾
- ❌ **禁止** "既然激进就无验证" — R4 py_compile + R12 ckpt + R17/§19 gate + R15 push 仍然强制
- ✅ **允许** 激进不等于鲁莽. R18/§20 4 维度对比 + R17/§19 gate 验证 + R15 push 仍生效
- ✅ **允许** 激进失败后立即调整策略 (R11.5 + R11.1 自主决策)

### §21.5 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-31 §21 新增 | AI 之前处理 #72/#73/#74 走"沿用判决"模式, 写"是否启动 GPU 训练?"等保守话术 | 之前 R11.4 + R10 v1 主动推进 + R10 v2 idle 都有"等待授权"成分 | §21 明确 AI = owner, 激进推进, 任何实验立即启动 | 全部作废 R11.4 等待授权 |
| 2026-07-31 issue #72/#73/#74 (待重写) | AI 写"沿用判决" verdict + 等 owner 启动 GPU 训练 | 之前规则允许保守路径 | §21 强制: 立即启动 GPU 训练 + 实施新代码 + 不等待 | §21 + §20 联立强制 |

---

## §22：Commit + Issue Comment 必须详细回答 4 Gate 内容 (2026-07-31 新增, 硬规则, 配套 CLAUDE.md R20)

> **背景**: 用户 2026-07-31 反馈, 之前 issue commit (#72/#73/#74/#75/#76/#77) 的 commit message 内容**不是 verdict 的内容**, 也**没有对 issue 要求的 4 个 Gate 的详细内容进行回答**. commit 仅简短写"Gate 1 PARTIAL PASS, Gate 2/3/4 ⏸ STOP", 没有具体数据/原因/verdict 路径. Issue close 之前也没发 4 Gate 详细 comment. Reviewer 看不到清晰的 Gate 决策细节. §22 强制 commit message + GitHub issue comment 都必须详细回答 4 Gate.

### §22.1 核心要求

- ✅ **commit message 必须包含 4 Gate 详细内容** (per §19 + §22 增强):
  1. 每个 Gate 至少 3-5 行: 状态 (PASS/FAIL/PARTIAL/STOP) + 关键数据 (util/collision/R@K/grad/ckpt path) + 失败原因 + verdict 路径
  2. 禁止一句话 Gate PASS/FAIL (e.g. "Gate 1 PARTIAL PASS" 单独一行不充分)
  3. 必须包含 issue spec 完整 4 Gate 状态 (不允许只写 Gate 1 跳过 Gate 2/3/4)
- ✅ **GitHub issue comment 必须详细回答 4 Gate** (per §18 + §22 增强):
  1. close issue 之前必须发 comment: `gh issue comment <num> --body-file <comment_md_file>` 或 `--body "..."`
  2. comment body 必须包含 4 Gate 详细内容 (每个 Gate ≥3-5 行 + 关键数据 + verdict 路径 + commit hash)
- ✅ **verdict 文件本身就是 4 Gate 详细内容** (R17/§19 已要求): commit message + issue comment 必须**直接包含 verdict 关键内容** (不是简单引证 verdict 路径)
- ✅ **每个 Gate 必须有显式 verdict/路径/数据**: 不允许"⏸ STOP"独占一行, 必须写"⏸ STOP per spec: Gate 1 仅 precheck, 无 SID 产出可推断 Sinkhorn"
- ❌ **禁止** commit message 只引证 verdict 路径 + commit hash 而不包含 4 Gate 详细内容
- ❌ **禁止** issue close 之前不发 4 Gate 详细 comment (§18.1 + §22 强制)
- ❌ **禁止** "Gate <N> PASS/FAIL" 一句话省略数据/原因/verdict 路径

### §22.2 4 Gate 详细内容最小要求 (per Gate ≥ 3-5 行)

| 字段 | 内容 | 示例 |
|------|------|------|
| **状态** | PASS/FAIL/PARTIAL/STOP | `Gate 1: ⚠️ PARTIAL PASS (机制完整 6/10)` |
| **关键数据** | 具体数值 | `L0/L1/L2 util=1.6%/0.8%/0.4%, collision=63/127/255, grad_theta_max=6.21e-3` |
| **失败原因** | 简洁根因 | `util/collision 1 epoch 短训未达, R137 baseline 需 200 epoch` |
| **verdict 路径** | `verdicts/task<N>_*.md` | `verdicts/task368_issue75_direction_a_gate1_minimal_evidence_v2.md` |
| **commit hash** | 当前 commit | `cd816cc` |
| **后续** | STOP per spec | `⏸ STOP per spec: Gate 1 PARTIAL, 无 SID 产出可推断` |

### §22.3 Issue Comment 模板 (close 之前强制)

```bash
# 写 comment 到文件
cat > /tmp/issue<N>_comment.md << 'COMMENT_EOF'
## Issue #<N> R18 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 强制)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): <PASS|FAIL|PARTIAL|STOP> per spec
- 关键数据: <util/collision/R@K/grad/ckpt path>
- 失败原因: <简洁根因>
- 实施: <scripts/task<N>_*>

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: <Gate 1 FAIL/PARTIAL, 无 SID 产出>
- Issue spec 强制: <Gate 2 目标 + 前置条件>

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: <Gate 2 STOP>
- Issue spec 强制: <Gate 3 训练 + 前置条件>

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: <Gate 3 STOP>
- Issue spec 强制: <R@10 阈值 + 标记 [TARGET REACHED] 条件>

### 关键产物
- verdict: verdicts/task<N>_issue<N>_*_v2.md
- commit: <hash>
- 实施: scripts/task<N>_*.py
- 整体决策: <GO|NO-GO|PARTIAL>
COMMENT_EOF

# 发 comment
gh issue comment <N> --repo WENYULIANG123/GeneRec --body-file /tmp/issue<N>_comment.md

# close issue
gh issue close <N> --reason completed --repo WENYULIANG123/GeneRec
```

### §22.4 与现有规则的关系

- **§22 加强 §19**: §19 强制 commit message 含 Gate + 失败原因, §22 加强必须详细 (≥3-5 行/Gate + 关键数据 + verdict 内容)
- **§22 加强 §18**: §18 强制 issue close + 写 comment (§18.1 模板), §22 加强 comment 必须详细回答 4 Gate
- **§22 加强 §15**: §15 强制 verdict push, §22 加强 commit + comment 都必须直接包含 verdict 关键内容
- **R11.5 (自主决策) > §22**: comment 模板是强制格式, 不需要等 owner 决策

### §22.5 关键 caveat

- ❌ **禁止** commit message 一句话 Gate PASS/FAIL (e.g. "Gate 1 PARTIAL PASS" 单独一行)
- ❌ **禁止** issue close 不发 4 Gate 详细 comment (即使已经写了 verdict)
- ❌ **禁止** comment 只引证 verdict 路径 + commit hash, 不直接包含 4 Gate 详细内容
- ❌ **禁止** ⏸ STOP 独占一行不说明原因 (必须 "⏸ STOP per spec: <原因>")
- ✅ **允许** commit message 引用 verdict 详细路径 (verdicts/task<N>_*.md) 作为补充, 但 commit 自身必须包含 4 Gate 关键内容
- ✅ **允许** issue comment 在 4 Gate 详细内容后, 补充 verdict 路径 + commit hash + 整体决策

### §22.6 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-31 §22 新增 | Issue #72/#73/#74/#75/#76/#77 commit message (2137df5/cd816cc) 仅简短写"Gate 1 PARTIAL PASS, Gate 2/3/4 ⏸ STOP per spec", 没有具体数据/原因/verdict 内容. Issue close 时也没发 4 Gate 详细 comment. Reviewer 看不到 4 Gate 详细状态 | §19 模板强制 Gate 状态 + 失败原因, 但没强制详细 (≥3-5 行/Gate) + 没强制 issue close 前发 comment | §22 强制 commit + issue comment 都必须详细回答 4 Gate (≥3-5 行/Gate) + close 前必须发 comment | 每次 issue commit + close 必含 4 Gate 详细回答 |

---

## §23：commit hash 必须明示 (2026-07-31 新增, 硬规则, 配套 CLAUDE.md R21)

> **背景**: 用户 2026-07-31 反馈, Issue #78/#79/#80 issue comment 写了 "commit pending" 而非具体 commit hash. 实际 commit 8a761f6 已 push 完成, 但 reviewer 看 comment 时不知道 commit 在哪. R17/§19 模板有 commit hash 字段, 但 R11.5 自主决策时用 "commit pending" 占位 (commit 还在写时发 comment). §23 强制: 任何 commit 落地后, 必须补 comment 标注 commit hash + 修正"commit pending"占位. 不允许任何 comment / verdict / commit message 含 "commit pending" / "TBD" / "TODO" / "未确定" 等占位文本 (commit hash 维度).

### §23.1 实施细节

**comment 模板 (commit 落地 + push 之后, 含具体 hash)**:
```
**Issue #<N> R20+§23 强制 4 Gate 详细内容 + commit hash**

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): <PASS|FAIL|PARTIAL|STOP> per spec
- 关键数据: <util/collision/R@K/grad/ckpt path>
- 失败原因: <简洁根因>
- verdict 路径: <verdicts/task<N>_*_v2.md>
- commit: <hash>  ← §23 强制具体 hash, 不允许 pending

### Gate 2/3/4: ⏸ STOP per spec
- 原因: <前 Gate FAIL/PARTIAL 或 Issue spec 仅要求此 Gate>

### 关键产物
- commit hash: <hash>
- push: origin/main
- verdict: <verdicts/task<N>_*_v2.md>
- 整体决策: <GO|NO-GO|PARTIAL>
```

**verdict 模板修正**:
```
- verdict 路径: verdicts/task<N>_<...>_v2.md
- commit: <hash>  ← 落地后立即写入, 不允许 pending
- push: origin/main
```

**comment 顺序 (§23 + §22 联立强制, 顺序固定不可换)**:
1. commit + push (§15): `git add` + `git commit` + `git push`, 拿到 hash
2. 发 comment 含具体 hash (§23 + §22): `gh issue comment <N> --body-file` 含 4 Gate 详细 + commit hash
3. close issue (§18): `gh issue close <N> --reason completed`
4. ❌ **禁止** 任何位置写 "commit pending" / "TBD" / "TODO" / "未确定" 占位

### §23.2 与现有规则的关系

- **§23 加强 §22**: §22 强制 comment 含 4 Gate 详细内容, §23 加强必须含 commit hash (落地后立即)
- **§23 加强 §19**: §19 模板含 commit hash 字段, §23 强制 commit hash 必须具体 (不允许 pending)
- **§23 加强 §15**: §15 强制 push, §23 强制 push 后 comment 必须含 commit hash
- **§23 ⊂ §18**: §18 强制 issue 检查 + close, §23 强制 close 前必须先发 commit hash comment (step 2)
- **R11.5 (自主决策) > §23**: comment 模板是强制格式, 不需要等 owner 决策

### §23.3 关键 caveat

- ❌ **禁止** comment / verdict / commit message 任何位置 "commit pending" / "TBD" / "TODO" / "未确定" 占位文本 (§23 强制 owner 2026-07-31 反馈强化)
- ❌ **禁止** commit 落地前发 comment (comment 必须在 commit + push 之后发, 拿到 hash 才能发)
- ❌ **禁止** "comment 先发, commit 后补" 的两步走流程 (commit 必须先, comment 必须后)
- ❌ **禁止** 用 fallback "comment 已经在 push 之前发了, 不再补" (§23 强制补, R2 不允许 fallback)
- ✅ **允许** 仅当 commit 落地 + push 完成 + 拿到 hash 后才发 comment (§23 step 1: commit+push → step 2: comment 含 hash → step 3: close)
- ✅ **允许** verdict 文件落地后含具体 commit hash (无占位)
- ✅ **允许** R11.5 自主决策按顺序: commit → push → comment(含 hash) → close (符合 §23)

### §23.4 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-07-31 §23 v1 新增 | Issue #78/#79/#80 comment 写 "commit pending" 但 commit 8a761f6 已 push, reviewer 看 comment 不知 commit 在哪 | R11.5 自主决策允许占位, §19/§22 没强制落地后必须修正 | §23 强制 commit 落地后必须补 comment 标注 commit hash + 修正 pending 占位 | commit push 后立即 step 2 发 comment |
| 2026-07-31 §23 v2 (owner 反馈) | §23.3 caveat 允许"commit 落地前发 comment 含 pending"是漏洞, owner 不允许任何占位 | §23 v1 留下时间窗口漏洞 | §23 v2 禁止 comment / verdict / commit 任何位置 commit pending 占位, comment 必须在 commit + push 之后发 | comment 必须 commit+push 之后发, 拿到 hash 才能发 |

---

## §24：新 issue 立即开工, 不允许等待 (2026-07-31 新增, 硬规则, 配套 CLAUDE.md R22)

> **背景**: 用户 2026-07-31 反馈, "你就是 owner, 只要有新 issue 马上开工, 不允许等待". R10 v2 idle 允许 + §21 激进推进 协同的"idle 等待"语义被强化否定. 任何新 issue 一旦出现 (GitHub OPEN 或 owner 明确派工), 必须立即进入 §18 + §19 + §20 + §22 + §23 闭环流程, 不允许任何形式的等待/暂停/idle/汇报.

### §24.1 核心要求

- ✅ **新 issue 立即开工**: `gh issue list --state open` 发现新 issue → 立即按 §18 + §19 + §20 + §22 + §23 流程处理, 无任何"等下一轮"/"等 owner 拍板"/"等 GPU"/"等 issue 优先"
- ✅ **任何派工立即执行**: owner 在 prompt 中说"跑 X 实验"/"启动 Y"/"follow loop.md" → 立即开始, 不允许"是否启动?" 询问
- ✅ **GPU 抢占处理**: 4 卡全部占用 → 立即换 GPU / nohup & 后台排队 / 缩减任务规模 (R7 + R11.5 自主决策), 不允许"等已用卡释放"
- ✅ **覆盖 R10 v2 idle**: R10 v2 idle 仅在 "无 issue + §16 空 + 用户未派工" 时生效. 一旦有 issue 或 owner 派工, §24 立即覆盖 idle
- ✅ **覆盖 R11.4 (AskUserQuestion)**: R11.4 仅在 critical 决策用, §24 强制非 critical 决策立即自主推进
- ✅ **§24 > R10 v2 + R11.4**: 任何等待/汇报/保守路径都是 §24 违规
- ❌ **禁止** 任何"是否启动 GPU 训练?" / "是否 OK?" / "等 owner 拍板" 等询问/等待话术
- ❌ **禁止** "ROI 低" / "drift-cycle" 保守理由跳过新 issue
- ❌ **禁止** "no actionable work" 报告 (§18 + §20 + §21 + §24 联立必须找到 actionable work)
- ❌ **禁止** 串行处理多 issue (§21 + §24 联立强制跨 issue 并行)

### §24.2 决策速度

| 场景 | §24 行为 |
|------|----------|
| 任何 OPEN issue 出现 | 立即 §18 + §19 + §20 + §22 + §23 闭环, 不允许等待 |
| owner prompt 含实验/任务关键词 | 立即启动, 不询问"是否启动?" |
| 跨 issue 多实验 | 立即并行 (一张 GPU 一个, 4 卡并行) |
| GPU 全部占用 | 立即换 GPU / nohup & 后台排队 / 缩减规模, 不等待 |
| 实验 FAIL | 立即写 verdict + 启动下一个变体 (不 cartesian 串行) |

### §24.3 与现有规则的关系

- **§24 > R10 v2 idle**: §24 否定 idle 等待的合法性, 仅在严格条件下保留
- **§24 > R11.4 (AskUserQuestion)**: §24 全面作废等待授权, R11.4 仍生效只在"AI 内部决策"维度
- **§24 + §21**: §21 强调 owner 主动推进, §24 强调 issue/派工维度
- **§24 + §18**: §18 强制 issue 检查, §24 强制检查后立即开工
- **§24 + §19 + §20**: §19/§20 强制 commit + 实证, §24 强制立即进入这些流程
- **§24 + §22 + §23**: §22/§23 强制 comment + commit hash, §24 强制按 §23 顺序: commit+push → comment(含 hash) → close

### §24.4 关键 caveat

- ❌ **禁止** "loop tick 无 actionable work" 报告 (§18 + §20 + §21 + §24 联立必须找到 actionable work)
- ❌ **禁止** "等下一轮" / "下一 loop tick 处理" 等拖延话术
- ❌ **禁止** "已经启动 GPU 训练了, 等结果" 状态报告后不立即写 verdict
- ❌ **禁止** "R10 v2 idle 等待" 作为不立即开工的理由 (§24 强制立即开工)
- ✅ **允许** 实际无 issue + §16 空 + 用户未派工 + 4 卡空闲 的 idle 状态 (R10 v2 保留)
- ✅ **允许** 立即开工后, 实时报告状态 (commit hash + verdict 路径 + 进度), 但不允许"等待授权"

### §24.5 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|------|
| 2026-07-31 §24 新增 | owner 反馈 AI 处理新 issue 时偶有"等下一轮"/"等 owner 拍板"/"是否启动?"等待话术 | R10 v2 idle 允许 + R11.4 AskUserQuestion 留下等待空间 | §24 强制新 issue 立即开工, 不允许任何等待 | §24 + §21 + §18 联立强制 |

---

## §25：明显失败立即终止 (2026-08-01 新增, owner 反馈, 硬规则, 配套 CLAUDE.md R23)

> **背景**: owner 2026-08-01 反馈, 看到 Task #454 + #456 训练 val_R@10=0.0000 持续跨 epoch 5/10/15/20, 但 AI 没立即终止, 还在跑. owner 原话: "这两个已经明显失败了, 为什么还在跑" + "如果任务已经执行中间就看到明显失败了, 马上终止, 写 commit 关闭 issue, 不允许一直在训练等待".

### §25.1 核心要求

- ✅ **每个 loop tick 必须扫一眼所有活跃训练** (per §16 + TaskList in_progress 项)
- ✅ **明显失败信号** (满足任一即触发 §25):
  1. **val_R@10 = 0.0000 跨 ≥2 个连续 val checkpoint** (5/10/15/20 epoch 持续 0)
  2. **loss 不下降** (跨 ≥3 个 epoch loss 没改善)
  3. **val loss 反向** (跨 ≥3 个 val checkpoint 上升)
  4. **wrapper broken 已知 bug** (e.g. wrapper 内部 sigmoid/clamp 饱和导致 logits 偏离)
  5. **ckpt 不保存** (R12 强制落盘, 若违反 立即终止)
  6. **NaN/Inf 出现** (任何 loss/grad/logits NaN/Inf)
  7. **GPU 占用 100% 但训练 loss 不变** (可能死锁)
- ✅ **发现明显失败 → 立即终止**:
  1. **`kill <PID>` + `pgrep | xargs kill -9`** 强制 kill
  2. **写 NO-GO verdict** (verdicts/task<N>_*.md, R17+R20+R21 v2 4-Gate 详细)
  3. **commit + push** (R15 强制)
  4. **`gh issue close --reason completed`** (R16 强制)
  5. **更新 TaskList** (in_progress → completed)
  6. **更新 §16 当前活跃任务表** (删除已完成行, per R8)
- ❌ **禁止** "已经跑到 epoch X, 跑完再说" / "等 200 epoch 完成再判断" / "loss 在下降, 应该会好"
- ❌ **禁止** "等下一 loop tick 处理" 拖延话术 (§24 + §25 联立)
- ❌ **禁止** 用 fallback "looser proxy" 掩盖真实失败 (R2 强制)
- ❌ **禁止** 让 val_R@10=0 持续训练超过 1 个 val checkpoint 间隔 (§25 强制立即 kill)

### §25.2 实施细节

**每个 loop tick 必跑扫一眼 (§25.2 强制)**:
```bash
# 1. 列出所有活跃训练 PID + val_R@10 最近状态
ps aux | grep -E "task[0-9]+_issue" | grep -v grep | awk '{print $2, $11, $12}'
# 2. 看最新 log val_R@10 行
for log in $(ls logs/task*_v8.log 2>/dev/null | tail -5); do
    echo "=== $log ==="
    grep -E "val_R@10|val_loss|USAGE-KILL|NaN" "$log" | tail -5
done
# 3. 若 val_R@10 = 0.0000 跨 ≥2 个 checkpoint → 立即 §25 kill + 写 verdict + close issue
```

**kill 命令模板 (R7 + §25 联立)**:
```bash
PIDS=$(pgrep -f "task<NUM>_issue")
kill $PIDS 2>/dev/null
sleep 5
pgrep -f "task<NUM>_issue" | xargs -r kill -9
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
ps aux | grep -E "task<NUM>_issue" | grep -v grep
```

**verdict + commit + close 一气呵成 (§15+§18+§19+§22+§23+§25 联立)**:
1. `git status` 看 untracked files
2. 写 `verdicts/task<N>_*.md` (R17+R20+R21 4-Gate 详细)
3. `git add verdicts/task<N>_*.md` + `git commit -m "Issue #<N> ... (R17+R20+R21 4-Gate 详细)"`
4. `git push origin main`
5. `gh issue close <N> --reason completed`
6. 更新 TaskList (#N → completed)
7. 更新 §16 当前活跃任务表 (R8 强制)

### §25.3 与现有规则的关系

- **§25 > §24**: §24 强制新 issue 立即开工, §25 强制训练中明显失败立即终止. 两者协同 (开工激进 + 终止激进)
- **§25 > R7 (GPU 占用)**: R7 强制分配到空闲 GPU, §25 强制释放已占 GPU
- **§25 + R15 + R16 + R17 + R20 + R21**: §25 终止流程必须严格走完整闭环
- **§25 + R8 (loop.md §16 清理)**: 完成 verdict + close issue 后, 必须从 §16 删除行
- **§25 + R12 (ckpt 落盘)**: 训练未完成时, R12 强制每 N 步落盘 ckpt, §25 kill 时确保 best_val_r10 ckpt 已保存
- **§25 + R10 v2 idle**: 终止失败训练后, R10 v2 idle 允许生效

### §25.4 关键 caveat

- ❌ **禁止** "loss 还在下降, 跑完再说" — 训练失败的早期信号是 loss 健康但 val_R@10=0
- ❌ **禁止** "val_R@10=0.0000 但 protocol 是 strict 4-digit, 试试 loose" — R2 禁止 fallback
- ❌ **禁止** "再跑 50 epoch 看看能不能好" — §25 强制 ≥2 个 val checkpoint 0.0000 立即 kill
- ❌ **禁止** "kill 之前先写 Slack/Email 给 owner" — R11.5 自主决策 kill, 不需要等 owner
- ✅ **允许** 实时状态报告 (commit hash + verdict 路径 + PID + GPU), 但不允许"等待授权"
- ✅ **允许** 失败训练 kill 后, 立即在 §16 记录 (R8 强制)
- ✅ **允许** 同样失败模式的多个训练 (parallel GPU) 同时 kill

### §25.5 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-08-01 §25 新增 | Task #454 + #456 v8 训练 val_R@10=0.0000 跨 epoch 5/10/15/20, AI 没立即 kill, 还在跑 | §24 + §21 强调"立即开工", 但缺少"训练中失败立即终止"硬规则 | §25 新增: 明显失败 7 信号 + 立即 kill + verdict + close 强制流程 | 每个 loop tick 扫一眼活跃训练 val_R@10 状态 |
| 2026-08-01 #454 + #456 训练浪费 GPU | owner 反馈: "这两个已经明显失败了, 为什么还在跑" + "不允许一直在训练等待" | 训练 PID 3830057 + 3830462 跑了 epoch 5/10/15/20 仍未终止, 浪费 ~1.5 GPU hour | kill PID + 写 NO-GO verdict a2adffe + close issue #162/#163 | §25 强制 loop tick 扫一眼 val_R@10 |

---

## §26：禁止任务暂停等待 owner 拍板 (2026-08-01 新增, owner 反馈, 硬规则, 配套 CLAUDE.md R24)

> **背景**: owner 2026-08-01 反馈, "每次检查是否有任务在等待owner拍板, 如果有, 马上自行做决策, 绝对不允许任务暂停". 当前 #486 task458/task459 wrapper 实质性实现 in_progress 状态 = 实质暂停. R19 + R22 强调立即开工, 但缺少"等待 owner 拍板"专项禁止. §26 强制.

### §26.1 核心要求

- ✅ **每个 loop tick 检查是否有任务处于"等待 owner 拍板"状态** (per §16 + TaskList in_progress 项 + open issue 编号 = task<N>_issue<M>_* 模式)
- ✅ **处于"等待 owner 拍板"的任务 → 立即 R11.5 自主决策** (不抛回用户)
- ✅ **R11.5 决策后立即执行** (立即启动 / 立即 NO-GO / 立即修改 wrapper / 立即 kill)
- ✅ **决策理由必须明示** (R11.3 强制: 选了哪个 + 为什么 + 备选方案)
- ❌ **禁止** "等 owner 拍板" / "请 owner 决策" / "等下一轮" / "wrapper 复杂等 owner 拍板" 等任何等待话术
- ❌ **禁止** "ROI 低" / "drift-cycle" 保守理由跳过决策
- ❌ **禁止** "无法决定" / "需重新决策" / "用户授权后启动" 等阻塞表述

### §26.2 实施细节

**每个 loop tick 必跑 (§26.2 强制)**:
```bash
# 1. 扫一眼 TaskList in_progress 项
# 2. 检查任务是否真有 GPU 占用 / 脚本运行 / 文件写入 行为
ps aux | grep -E "task[0-9]+_issue" | grep -v grep | awk '{print $2, $11, $12, $13}'
# 3. 若 in_progress 但无活跃 PID + 无 file mtime 更新 → 判定为"等待 owner 拍板"
# 4. 立即 R11.5 自主决策 + 立即执行
```

**R11.5 自主决策 兜底顺序 (§26.2 强制)**:
1. **项目 CLAUDE.md / memory 已固化的偏好** (e.g. R5 数据集, R23 监控触发线)
2. **上游 framework 默认值** (e.g. paper 报告超参, official code default)
3. **论文原始方案** (e.g. arXiv:2405.13979 curvature-aware optimization)
4. **简单实用方案** (e.g. 找不到精确匹配时用近似, 记下偏差)

**§26.3 决策后立即执行 (§26 强制)**:
- 立即写代码 (R4 py_compile 验证)
- 立即后台启动 (R12 ckpt 落盘 + R19 激进)
- 立即写 verdict (R17 commit message 含 Gate + 失败原因)
- 立即 git push (R15)
- 立即 issue comment (R20 4 Gate 详细)
- 立即 issue close (R16)
- 立即更新 TaskList (in_progress → completed)
- 立即更新 §16 (R8 强制)

### §26.4 与现有规则的关系

- **§26 > §21**: §21 强调"立即开工", §26 加强"等待 owner 拍板"专项禁止
- **§26 > §24**: §24 强调"新 issue 立即开工", §26 加强"任何任务禁止暂停"
- **§26 > R10 v2 idle**: §26 否定"等待 owner 拍板"作为 idle 理由
- **§26 > R11.4**: R11.4 仅在"AI 内部决策"维度, §26 全面禁止等待
- **§26 + R11.5**: R11.5 提供决策顺序, §26 强制决策后立即执行
- **§26 + §25**: §25 强制"立即 kill", §26 强制"立即决策 + 立即执行"

### §26.5 关键 caveat

- ❌ **禁止** "决策需要时间" / "需要权衡" / "wrapper 复杂" 等拖延话术 — §26 强制 5 秒内决策
- ❌ **禁止** "建议 owner 拍板" / "请 owner 决策" — R24 强制立即开工
- ❌ **禁止** "无法决定 NO-GO 还是 GO" — R11.5 兜底顺序 + R19 激进必须选一个
- ❌ **禁止** "等下一 tick 再处理" — §26 立即 tick 内闭环
- ✅ **允许** R11.5 自主决策 + 立即执行 + verdict 记录决策理由
- ✅ **允许** 后置微调 (决策后 5 分钟内 polish), 但不允许重新决策
- ✅ **允许** 失败后立即调整 (R11.5 + R11.1 自主决策), 不需要等 owner

### §26.6 历史事故

| 事故 | 现象 | 根因 | 修复 | 防止措施 |
|------|------|------|------|----------|
| 2026-08-01 §26 新增 | task458/459 wrapper 实质性实现 in_progress 状态持续 2+ tick, 无 GPU 启动, 实质暂停 | R11.4 + R19 + R22 留有"无法决定"灰区, AI 借口"wrapper 复杂"等下一 tick | §26 新增: 禁止任务暂停等待 + R11.5 立即决策 + 立即执行 | 每个 tick 扫一眼 in_progress 任务是否有真实执行行为 |
| 2026-08-01 task458/459 卡住 | owner 反馈: "新建规则, 每次检查是否有任务在等待owner拍板, 如果有, 马上自行做决策, 绝对不允许任务暂停" | 之前 R11.4 + R19 + R22 留有灰区, AI 借口"wrapper 复杂" | §26 强制 R11.5 决策 + 立即执行, 不允许任何"等待"话术 | §26 强制每个 tick 扫一眼 in_progress 状态 |

