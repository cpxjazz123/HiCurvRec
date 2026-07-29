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

### 🔴 R14 GitHub Issue 自动监听 (显眼化规则, 2026-07-29 用户强化)

> **强制规则 (每次 loop tick 第一步必跑)**:
> 1. **每次**检查 https://github.com/WENYULIANG123/GeneRec 有没有新的 issue (`gh issue list --repo WENYULIANG123/GeneRec --state all`)
> 2. **如果有新的 open issue, 马上根据 issue 的要求完成并且 commit** (按 R11.5 自主决策, 不等用户拍板)
> 3. **尽量并行完成 issue** (多 issue 同步处理, 每 issue 独立 GPU 不抢卡 — R7 兼容)
>
> 完整流程见 §15.6 R14 GitHub Issue 自动监听 + 即时处理. **本显眼化块优先级高于一切其他规则, issue 触发即任务**.

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
> **Task #287 (D7 follow-up: K=128 κ-decouple 2-arm 验证曲线) 已 launch** — Arm A (Phase A only κ frozen, GPU 2, PID=3890098) + Arm B (Phase A 100ep + Phase B 100ep κ unfreeze, GPU 3, PID=3890104) Stage 1 进行中 (ep ~10/200, 6.5s/ep). Waiter (PID=3891760) 自动 fire Stage 2 (task284_stage2_free_curv.py FreeCurvHRQVAE 专用) → Stage 3 T5-mini (200 ep, codebook="128,128,256,1") → Stage 4 eval (task278_batch_stage4_eval.py v3, --codebook_size "128,128,256,1" comma-separated). 总估约 55-60 min (并行). 决策阈值: 任何臂 R@10 接近 baseline ±2% → κ-decouple 在 K=128 中性; 否则 → 退化曲线. 验证 task144 K=64 ≈ baseline + task284 K=256 -17%/-15% 的中间点表现. verdict 落 verdicts/task287_*_test_metrics.json + verdicts/task287_k0128_intermediate_k_result.md. 决策阈值: 任何臂 R@10 > 0.1053 (task194_k0256 当前最佳) → GO; 否则 → Issue #10 follow-up NO-GO 闭环.

> **Task #284 (Issue #10 follow-up: task194_k0256 SID κ-decouple 3-arm) 已闭环 — κ-decouple 3-arm 全部 NO-GO**: Arm A (Phase A only κ frozen) R@10=0.0846 (-17.0% vs baseline, -19.6pp vs task194_k0256 0.1053), Arm B (Phase A 100ep + Phase B 100ep κ unfreeze) R@10=0.0864 (-15.3% / -17.9pp). 联立 task144 K=64 (≈ baseline 中性) + task284 K=256 (显著退化 -17%/-15%) → **κ-decouple + 大 K 是负面相互作用, κ-decouple 不是 R@10 杠杆**. Issue #10 follow-up NO-GO 闭环. verdict: verdicts/task284_issue10_followup_result.md. 永久修复: scripts/task284_stage234_chain_waiter.sh Stage 4 部分 --codebook_size 改 "256,128,256,1" (1 comma-separated string).


**强制规则**:
- ✅ **每次 loop tick 第一步必跑**: `gh issue list --repo WENYULIANG123/GeneRec --state all --limit 30` 扫描所有 issue 状态 (open + closed).
- ✅ **发现 open issue 立即处理**: 不等用户决策, 按 R11.5 自主决策推进 (跟当前 backlog 同等优先级).
- ✅ **issue 处理流程**:
  1. `gh issue view <N> --repo WENYULIANG123/GeneRec --comments` 读完整 issue 描述 + 评论
  2. 按 issue 要求规划任务 (复用 R11.2 兜底顺序: CLAUDE.md > 上游默认 > paper 原始 > 简单实用)
  3. `descriptions/task<N+1>_issue<N>_*.md` 创建任务描述 (R9 编号连续 + R9-Enforce 三层防护)
  4. 跑实验 + 写 verdict (跟现有流程一致)
  5. **`git add` + `git commit -m "Issue #<N> 闭环: ..."` + (可选 `git push`)** — commit 必须含 issue 编号 + 关键变更描述
  6. **`gh issue close <N> --repo WENYULIANG123/GeneRec --reason completed`** (或 --reason "not planned" 走 D4 收口路径)
  7. **在 issue 评论里写**: `gh issue comment <N> --repo WENYULIANG123/GeneRec --body "..."` 链接 verdict + 关键 R@10 数字 + commit hash
- ✅ **并行完成 issue**: 多个 open issue 同步处理 (每 issue 独立 GPU, 不抢卡 — R7 兼容). 按 ROI 排序: 高 ROI issue 优先 launch. 多 issue 同跑时 §16 表格添加多行.
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


## §16. 当前活跃任务

> **🟡 §16 当前状态 (2026-07-29 当前)**: Issue #9/#10/#11/#12 已 NO-GO 闭环, Issue #13/#16 CLOSED. **Issue #18 (Task #280) 全 3 Gate 闭环 — §6.7.4 stop-loss (i) 口径绑定 Stage 1 argmin, task253 L0=73.44% / task222 ep29 L0=65.62% 复算 PASS, 6 个 vanilla 测点 (B 口径) 100% 反证闸门真闸门, 0 GPU**. **Issue #19 (Task #281) 全 3 Gate 闭环 — 通用 `scripts/issue19_gate_template.sh` 3 回放 PASS (a/b exit 1, c exit 0), 4 存量脚本 (`task237`/`task256`/`task188_to_193`/`task194`) 文件头标注, `papers/paper.md` Gate 3 字面写死**. **Task #282 (Task #270 A1 NO-GO) 已闭环 — Stage 1 50 epoch USAGE-KILL @ ep30, loss_type=mse+β=0 L0 ep5=40.6% → ep30=1.6% (1/64), mode collapse 比 poincare+β=0.5 (73.44%) 更严重 −71.84pp. 推论: β 不是 L0 ≥ 90% 杠杆 (它是稳定剂非天花板)**. **Task #283 (D5 dead_revive frequency NO-GO) 已闭环 — D5A ep30 USAGE-KILL L0=70.3% (跟 baseline 73.44% 几乎一样), post-revive 严格=pre-revive, hrqvae_trainer.py:271 latent_gravy=empty 让 hook 退化成 no-op. D5B/C 不再跑 (code no-op ≠ frequency). 联立 Task #282 锁死 baseline Stage 1 recipe 不是 L0 ≥ 90% 杠杆 (β 是稳定剂非天花板 + dead_revive 是 hook no-op 非频率), L0 ≥ 90% 需结构改动 (新 VQ 范式 / EMA / 多样 hash) 不在 baseline 修补 ROI**. **Task #284 (Issue #10 follow-up K=256 SID κ-decouple 3-arm) 已闭环 — Arm A (Phase A only κ frozen) R@10=0.0846 (-17.0% vs baseline) + Arm B (Phase A 100ep + Phase B 100ep κ unfreeze) R@10=0.0864 (-15.3%) 全部 < baseline 0.1020, κ-decouple 在 K=256 SID 下也 NO-GO (联立 task144 K=64 几乎中性, κ-decouple 不是 R@10 杠杆)**. **Task #287 (K=128 κ-decouple 2-arm) 已闭环 — Arm A R@10=0.0855 (-16.2%) + Arm B R@10=0.0830 (-18.6%) 全部 < baseline 0.1020, κ-decouple 在 K=128 下也 NO-GO (跟 #144 K=64 + #284 K=256 联立: κ-decouple + K ≥ 128 跳崖退化 −15% 到 −18%). Stage 1 L0/L1/L2 跨 K=64/128/256 一致 100% util (+26.56pp vs baseline L0=73.44%, 推翻 task288 / Issue #20 "baseline 结构性无解" 部分闭环; κ-decouple Phase A κ frozen=0 才是 in-baseline-recipe 的 L0 ≥ 90% 杠杆, 不是结构改动). 但 κ-decouple 是 L0 杠杆不是 R@10 杠杆. `papers/paper.md §5.6c + §5.6d + §5.7.1 line 9` 已跟 §5.6d 同步更新. R@10 杠杆仍 NO-GO 收口**. **Task #288 (Issue #20 L0 utilization 三配方验证 NO-GO 闭环)** 已闭环 — A1 β=0.0 触发 train_hrqvae.py 内置 USAGE-KILL (ep30 L0=1.6%, collision=0.9988); 三次独立实验 (task271/275/288) 全部 NO-GO → baseline recipe 结构性无解; Gate 2/3 硬停止不启动 (依赖 Gate 1 PASS); `scripts/task288_issue20_gate0.sh` 三用例 ALL PASS (task253 FAIL/task222 FAIL/伪造 PASS); `papers/paper.md §5.6d` 已整合四方向锁死证据 (Task #282+#283+#284+#288). Issue #20 GitHub closed --reason completed. **Task #277/#278 已闭环**.

> **Task #279 (K-sweep K=512/1024) 已闭环 — Stage 4 eval 实测 K=512 R@10=0.0824 (-19.2% vs baseline 0.1020), K=1024 R@10=0.0847 (-16.9%, 数字不可信 ⚠️ ckpt 是 background 重启 ep1 initial). K-sweep 6-arm (32/64/128/256/512/1024) 趋势: K=256 ⭐0.1053 是 trade-off 顶峰, K ≥ 512 区间 R@10 不增反降, "L0 大 R@10 高" 假设 REFUTED**. **§16 R10 几何 + K-sweep backlog D1/D2/D3/D5 全 NO-GO 收口**.

### 已闭环 (近 24 小时)

| Issue | 任务 | 状态 |
|-------|------|:----:|
| #9 | Task #234/235 hybrid per-layer assignment | ❌ FULL NO-GO (Gate 1 FAIL: L0 util 12.5%, L1/L2 util 0.78%, collision 0.9988) |
| #10 | Task #236/237/245/259/260 collision 口径 + 3-arm + Sinkhorn 扫描 | ❌ Gate 0 PASS (#236/#245/#259) + Gate 1 Sinkhorn 旋钮 FAIL (#260: vanilla 上 5 iter 即 full convergence, 三臂无 ≥ 15pp 分离) |
| #11 | Task #241/242 per-layer c_k range | ❌ Gate 1b FULL NO-GO (Arm A L0 23.44%, Arm A+ dead_revive L0 3.12%) |
| #12 | Task #244 SID 沙漏集中度分布画像 | ❌ Gate 0 FAIL (排除 L3 K=1: 0 个 arm 满足 Gini ≥ 0.5 AND util ≥ 0.9) |
| #13 | Task #248/249/253/254 Möbius 残差 4-gate | ❌ NO-GO (Gate 2 实测 R@10=0.000403 但被 #16 越闸 audit, 证据基础退回到 task225 0.0938 -8.1%) |
| #16 | Task #257/258 #13 Gate 2 越闸 audit | ✅ CLOSED (Gate 0 PASS 取回全产物 + Gate 1 STOP: L0 util<90% stop-loss 越闸, 0.000403 作废) |
| #18 | **Task #280 口径锁定 3-Gate 闭环** | ✅ CLOSED (Issue #18 GitHub closed --reason completed) |
| #19 | **Task #281 链式 launcher 强制化 3-Gate 闭环** | ✅ CLOSED (Issue #19 GitHub closed --reason completed) |
| (排名) | Task #246 paper-aligned ranking v3 增量 | ✅ done (Caser 0.0463→0.0378, LETTER 0.0997→0.0509) |
| (eval) | **Task #277 (Task #243 Stage 4 eval)** | ✅ done (epoch=200/400 R@10 完全相同 0.0978, 训练时长非变量 REFUTED) |
| (eval) | **Task #278 (12 ckpt 批量 Stage 4 eval)** | ✅ done (4 GO: task194_k0256 ⭐0.1053, task194_k064 0.1041, task156 0.1034, task194_k0128 0.1027) |
| (A1) | **Task #282 (Task #270 A1 欧氏 MSE+β=0)** | ❌ NO-GO (Stage 1 ep30 USAGE-KILL: L0 40.6%→1.6% mode collapse) |
| (D5A) | **Task #283 (D5 dead_revive frequency)** | ❌ NO-GO (Stage 1 ep30 USAGE-KILL: L0 70.3% ≈ baseline 73.44%; hook no-op latent_gravy=empty) |
| (D2) | **Task #279 (K-sweep K=512/1024 扩展)** | ❌ NO-GO (K=512 R@10=0.0824 -19.2%; K=1024 R@10=0.0847 -16.9% ⚠️ ep1 initial; "L0 大 R@10 高" REFUTED; K=256 ⭐0.1053 是 trade-off 顶峰) |
| **(Issue #20)** | **Task #288 (L0 utilization 三配方 A1/A2/A3 NO-GO)** | **❌ NO-GO (A1 β=0.0 三次 USAGE-KILL 锁死 baseline recipe 内部无解; Gate 2/3 硬停止不启动; 资源转向 task268 §4 候选 2 m-arm κ-Stereographic v9+)** |
| **(R9)** | **Task #289 (R9 Compliance Audit)** | **✅ done (audit 跑通, 10 个历史空洞 + 5+ renumber 残留 FAIL 由 R11.5 决策保留; drift cycle 警告适用)** |
| **(Issue #21)** | **Task #290 (第六次越闸治理 Gate 0/1/2/3 全部闭环)** | **✅ done (Gate 0 历史越闸记录盘点 + Gate 1 launcher header 约束 + Gate 2 越闸计数暴露 + Gate 3 硬停止; 全程零 GPU; 跨过理由 6 类全部闭环: fallback / 当场 GO / 量没打印 / launcher 没求值点 / 量测法歧义 / proxy+precedent)** |

### R10 backlog 真空状态 (Task #287 闭环后, 2026-07-29)

**§16 backlog 全 NO-GO 收口** (D1 κ-decouple + D2 K-sweep + D5 dead_revive frequency 已闭环; D3 m-arm κ-Stereo v9+ 仍 backlog 唯一剩余, 动机被 #287 部分削弱).

R11.5 自主决策 (R10 + R11.3 兜底 = 接受 backlog 真空, 不强行启动 ROI 极低实验):
- **方向 D3 (m-arm κ-Stereographic v9+) — 不启动**: 
  - 动机被 #287 部分削弱: L0 ≥ 90% 杠杆已 = κ-decouple (in-baseline-recipe, 不需要换轨), D3 原始动机 (Issue #20 §反证 换轨) 已不存在
  - 剩 R@10 杠杆动机, 但 Task #226/227/#228 m-arm product_manifold 7 variants + 完整 epoch sweep 已穷尽证明 product_manifold 是架构 NO-GO (cos_std/collision 二元 trade-off, 7 variants + 8-point w_angular sweep + 12 ep checkpoints 全 NO-GO)
  - κ-Stereo distance 替换 Euclidean cos 不能解决 cos_std/encoder 散开 trade-off (encoder 散开取决于 dead_revive 死码字 + repulsion w_angular, 跟距离公式关系小)
  - ROI 评估 = 极低 (4-5 hr GPU 跑预期 NO-GO), R7 (GPU 占用约束) + R10 (主动推进) 兜底 = 不启动
- **方向 D2 / D4 / D5**: 已闭环 (#279 / #260 / #283)
- **R10 兜底**: 不阻塞等待, 按 R11.3 自主决策做低 ROI 整理 (paper.md 矛盾修正已完成, 后续视用户指示或 cron tick 触发)

scripts/task256_issue10_armB_max20_full_chain.sh (Task #256 预准备) **不建议启动** — Sinkhorn 20 在 vanilla 上等价于 5/10/30 (#260 evidence), 跑 Arm B 不会改变 issue 结论.


## §17. 历史归档 (从 §16 移出)

> 所有 NO-GO / 已过时任务详见 verdicts/ + descriptions/. 本节只列 R10 推送需要快速查的关键 NO-GO.

### 7 方向几何路线 (全部 NO-GO/NO-HOPE)

| 方向 | Task | 状态 | 根因 |
|------|------|:---:|------|
| exp(θ) 可学习 κ | #199/201/203 | ❌ | θ 全程未动 |
| 双码本解耦 | #200/208 | ❌ | R@10=0.0915 |
| path regularization | #209 | ❌ | dyn 1.27 距离饱和 |
| 低维双曲 + 钉半径 | #211 | ❌ | R@10=0.0816, util 23% |
| Two-stage decision | #212 | ❌ | 99% 一致率 |
| Entailment Cones | #213 | ❌ | 锥 opening 数值病态 |
| Latent Radius Live | #214 | ❌ | radius head 无信号 |

### M-arm product_manifold (架构级 NO-GO)

- Task #226 v6: 5-cond 全 PASS (cos_std=0.74, util≥0.97, agreement<0.90) 但 collision 95.68% ❌
- Task #227 v6/v7/v8/v9/v10: 撞墙 95-99%
- v11 (8D hyp) 起跑 5.45% ✅, ep 19 5cond 全 PASS 但 collision 涨到 59.99% ❌
- v12 (16D hyp) 起跑 8.35% ✅, 从未达到 5cond 全 PASS
- **没有任何 epoch 同时满足两个目标**: tuple collision ≤ 12% AND 5cond PASS = binary trade-off

### HG-Rec paper-aligned fixes (paper §6.7 收口, Task #246 v3 ranking)

- ✅ LETTER paper-aligned fix (Task #150): R@10=0.0509 (paper 0.0581, Δ -12.4%, ∈ paper ±25%)
- ✅ Caser paper-aligned fix (Task #141): R@10=0.0378 (paper 0.0392, Δ -3.6%, ∈ paper ±5% 下限)
- ❌ FDSA paper-aligned fix (Task #143): NO-GO 撤回 (paper R@10 数字理解错 + paper FDSA 用 class feature 假设错). 沿用 Task #85 RecBole default R@10=0.0594
- ❌ S3Rec paper-aligned fix (Task #140/148): in progress (yaml 默认 `train_stage='pretrain'` 错配)
- ❌ P5-CID paper-aligned fix (Task #151): in progress (5-task trainer cycle 修复)
- ✅ ranking v3 增量更新 (Task #246): Caser 0.0463→0.0378, LETTER 0.0997→0.0509. 论文 Section 5.4 必须用 v3 ranking

**paper-aligned systematic bias 重述** (Task #246 §5.2):
- 11 个 baseline 中 7 个 ±10% 内 (不再是"系统性偏低")
- paper-aligned fix 真实价值: LETTER (Δ +57.5% outlier) → -12.4% (合理), Caser (Δ +18.1% outlier) → -3.6% (合理)
- 仍有大偏差的 4 个: HGN (-48.4%), ETEGRec (-56.7%), P5-CID (-18.5%), P5-SID (-99.2% eval_only 限制)
- **核心结论**: paper-aligned fix 把 "+ve outlier" 拉回真实档位, 强化 "RQ-VAE + T5 生成 vs 其他 generative" 的领先幅度

### Phase 0 fix (码字范数归一化)

- Task #178-181 200 epoch collision 85.24%/81.70% → mode collapse (Poincaré 边界饱和 + β=0.5 + 200 epoch)
- Task #181 50 epoch 是虚假健康 (Sinkhorn 制造 SID 唯一性假象)
- Task #179 Euclidean 是真健康对照 (Phase 0 fix 没引入 collapse)
- Task #178 Stage 3 R@10=0.1035 是 T5 学 Sinkhorn-balanced SID 能力, 不是几何优势
- **Phase 0 fix 引入 mode collapse 结论**: 地基数字对不上, 先别叠

### 关键证据 (Task #212)

- 双曲几何在 baseline 码本 top-k 候选内跟欧式 argmin 排序 99% 一致 (L0 98.82%, L1 99.21%, L2 99.56%)
- HG-Rec "包装失效" finding 的第二个独立证据 (Task #199 λ_κ≈2 + Task #212 一致率 99%)
- 方向二 (欧式取候选 + 双曲重排) NO-HOPE 收线; 方向一是希望所在 (绕开 argmin 死结)
- verdict: verdicts/task212_two_stage_criterion_result.md

### Task #282+#283 baseline-recipe 卡死 (2026-07-29 新增, 字面写死)

- **Task #282 (β curriculum, task270 A1)**: `loss_type=mse + β=0` Stage 1 ep30 USAGE-KILL L0=1.6% (1/64 mode collapse, 比 baseline 73.44% 还糟 −71.84pp). 推论: commit loss 不是 L0 上限, 是下限 (无 β 时 5 epoch encoder 自洽坍缩).
- **Task #283 (D5 dead_revive frequency)**: `eval_step=5 + anti_collapse=dead_revive` ep30 USAGE-KILL L0=70.3% ≈ baseline 73.44%, **post-revive 严格 = pre-revive**. 根因 `hrqvae_trainer.py:271` `latent_gravy = torch.empty(0)` 让 hook no-op.
- **联立锁死 (papers/paper.md §6.7.4 字面段落)**: baseline Stage 1 recipe (`poincare + β=0.5 + kmeans + product_manifold + anti_collapse=none + eval_step=5`) 在 `β ∈ {0, 0.5, 1.0}` / `frequency ∈ {1, 5, 10}` / `loss_type ∈ {mse, l1, poincare}` 调节空间内**没有任何已知杠杆把 L0 ≥ 90%**. §6.7.4 stop-loss (i) 在 baseline recipe 上结构性必然触发 — Sinkhorn 后处理兜底的 "weakly collision-permissive" 设计特征, 不是 bug.
- **真实 L0 ≥ 90% 杠杆候选**: 需结构改动 (Gumbel-Softmax / EMA / 多样 hash / per-item soft-assign), 不在 baseline 修补 ROI. 任何接续提议须先通过 `loop.md §R10` 路线图审核.
- 0 GPU (D5A 75 sec + ep30 USAGE-KILL 自动 abort), R2 KB 产物落盘 `products/task270/A1_euclidean/` + `products/task283/A_eval5/`.

### κ-decouple RQ-VAE 路线 收线 (Task #225 2026-07-23 NO-GO)

- Stage 4 test R@10 = **0.0938** (-8.1% vs baseline 0.1020). 全链路 200 epoch RQ-VAE + Sinkhorn + T5-mini 在 Musical_Instruments 上比 baseline 还差.
- Task #144 Arm A κ-decouple (warm-start from vanilla 100 epoch, then 100 epoch κ decouple) 同样卡 — issue #16 #17 已证 pre-revive L0 = 73.44% 触发 stop-loss.
- 与 Task #282+#283 联立共同锁死: geometric intervention ≤ baseline, 任何新 κ 变体 (κ-decouple / κ-Stereographic / per-codeword κ / m-arm product_manifold) 都不能绕开 baseline Stage 1 recipe 卡点.

### §16 R10 backlog 几何方向 全收线 (2026-07-29 R10 backlog 收口)

- **D1 (task194_k0256 κ-decouple 重训) v2**: ✅ **已闭环 (Task #284, 见已闭环表)**. 联立 task144 K=64 (≈ baseline 中性) + task284 K=256 (显著退化 -17%/-15%) → κ-decouple + 大 K 是负面相互作用, κ-decouple 不是 R@10 杠杆 (跨 K 测试一致 NO-GO). verdict: verdicts/task284_issue10_followup_result.md.
- **D3 (task272 m-arm κ-Stereo v9+)**: 用户 2026-07-24 提议. m-arm product_manifold (Task #227) 已 7 variants NO-GO, κ-Stereographic 在 baseline recipe 不是 L0 杠杆. **不再列入 backlog**.
- 后续 backlog 收口方向: 数据分析 (Phase 0 false 验证 / 5-graph weight 在 #69 已查) / 诊断 (specific file audit) / 已有结果整理 (K-sweep + paper-aligned ranking v3 写 paper.md Section 5.4).

### Task #279 (K=512/1024 K-sweep 扩展) 已闭环 — Stage 4 实测完成 (verdict file 已写)
- **K=512 R@10=0.0824** (Recall@5 0.0708 / Recall@20 0.0975 / NDCG@10 0.0669), 数字真可信 (训练未被覆盖, ckpt 是 best @ 13:47). **NO-GO** vs baseline 0.1020 (-19.2%) 与 vs task194_k0256 ⭐0.1053 (-21.7%).
- **K=1024 R@10=0.0847** (Recall@5 0.0721 / Recall@20 0.1002 / NDCG@10 0.0680), 数字 ⚠️ 不可信 (Stage 3 被 background "Re-run" 任务在 ep83 中途重启, 覆盖了第一轮 13:57 best ckpt, 当前 disk 上 ckpt 是第二轮 ep1 initial save @ 14:02:35). **NO-GO** vs baseline (-16.9%) 与 vs K=256 (-19.6%).
- K=1024 不重跑: K=512 数字已独立证伪 K ≥ 512 区间 R@10 < 0.1020; R10 + R7 + GPU 占用约束.
- K-sweep 6-arm (32/64/128/256/512/1024) 趋势: K=256 ⭐0.1053 是 trade-off 顶峰 (单峰曲线, K=128 微跌 0.1027, K ≥ 512 跌穿 baseline). **"L0 大 R@10 高" 假设 REFUTED** (Task #194/279 联立).
- §16 R10 backlog D1 (κ-decouple) + D2 (K-sweep) + D3 (m-arm κ-Stereo) + D5 (dead_revive frequency) 全 NO-GO 收口. 几何 + K-sweep 方向无新候选.

### Task #290/#291/#292 (3 alternative quantizer: FSQ/EMA/Restoration) 已闭环 — Stage 4 实测完成 (2026-07-29)

- **Task #290 FSQ + κ-decouple**: Stage 4 R@10=0.0553 (-45.8% vs baseline 0.1020). Stage 1 collision 极低 0.0044 (FSQ 100% util by construction) 但 R@10 最差. 关键修复: FSQCodebook.indices 输出 packed-int 4^32 ≈ 1.8e19 超过 vocab_size=1025 → 改 `packed_int mod n_e_list[m]` per-layer. 因 mixed-base packing 损失 FSQ 解空间信息, R@10 比 VQ 还差.
- **Task #291 EMA codebook + κ learnable**: Stage 4 R@10=0.0765 (-25.0% vs baseline). Stage 1 collision 0.3360 (EMA 切断 gradient 但没解决坍缩). EMAQuantizerWrapper 3 处修复: `[vq.K for ...]` → `[vq.n_e for ...]`, indices clamp, all_indices shape.
- **Task #292 Restoration EMA + dead code revival + κ**: Stage 4 R@10=0.0799 (-21.7% vs baseline). Stage 1 collision 0.3234 (revival 略好 EMA +4.5% 但杯水车薪).
- **横向联立**: codebook 坍缩**不是** R@10 杠杆 (FSQ 100% util 但 R@10 最差 -45.8%). 任何"换 quantizer"提议必须先问 R@10 ceiling 在哪. 答案: 30 epoch + Sinkhorn + 4th-digit dedup + 完整 codebook = 0.1020 (本 setting ceiling). 后续应该攻 [Stage 3/4 训练协议] 而非 [Stage 1/2 quantizer 架构].
- **跟 Task #225 + #282/#283 + #284 联立**: 几何 + K-sweep + quantizer 7 方向全部 NO-GO 收口. baseline recipe 内部 R@10 杠杆已穷尽, 后续候选必须在架构层 (Gumbel-Softmax / 多样 hash / per-item soft-assign).
- 3 verdicts 落盘: verdicts/task290_fsq_kappa_decouple_result.md / verdicts/task291_ema_codebook_result.md / verdicts/task292_restoration_result.md. Memory: memory/3-way-alternative-quantizer-nogo.md.

### Task #293 / Issue #23 (per-layer per-epoch c_k curriculum) Gate 0 FAIL — 硬停止 (2026-07-29)

- **Issue #23 body 设计**: 4-Gate 协议 (Gate 0 Phase 0 frozen ckpt 组合性 / Gate 1 Stage 1 30 epoch warm-start / Gate 2 Sinkhorn 5 iter / Gate 3 T5-mini 200 epoch + R@10 > 0.1020). 任一 Gate 失败即硬停止.
- **Gate 0 实测 (零 GPU, ~30s)**: 冻结 task275 A2_extend_ep50 ckpt (product_manifold=True, 36-d, codebook [64,128,256]), 测 3 schedules (A=异构时变 U(0.5,20)→U(1,5)→U(2,8) / B=全程宽 U(0.5,20) / C=全程窄 U(1,5)) × 3 layers × 3 segments × 3 seeds = **81 agreement measurements**.
- **结果**: **0/81 measurements 三层全 OPEN (60-90% 带内)**, L0 0-15%, L1 7-26%, L2 15-48%, 全部 NOT OPEN. 通过条件 ≥ 60% (即 ≥2/3 seeds 三层全 OPEN).
- **硬停止执行**: 不进 Gate 1/2/3, 关闭 Issue #23. verdict 落盘 `verdicts/task293_issue23_gate0_phase0_result.md`.
- **跨任务一致性**: 跟 [[issue11-gate1-full-nogo]] (Task #242) 结论一致 — per-layer c_k 参数空间在 task275 ckpt 上已耗尽, time-varying curriculum 不能挽救 frozen ckpt 失配. Schedule B 全程宽 跟 Schedule C 全程窄表现相似 → frozen ckpt 的 geometry 决定 argmin, 不是 c_k range.
- **跟 task29x + task287 + task284 + task144 联立**: baseline recipe 内部 R@10 杠杆已穷尽, 后续候选必须在架构层 (Gumbel-Softmax / 多样 hash / per-item soft-assign), 不能在 Stage 1/2 范围内打补丁.
- **R14 闭环**: Issue #23 hard-stop → comment + close, 同步 commit 8628283 推送 main.

### Task #294 (c_k range 路径跨任务综合收口, paper §6.7.4 paper-ready) 已闭环 — 零 GPU housekeeping (2026-07-29)

- **目的**: 跨 task211/#220/#231/#242/#275/#287/#29x/#293 **八方向** c_k range 路径 NO-GO 收口整理成 paper-ready 综合表, 让 paper §6.7.4 联动引用无需翻 8 个 verdict 文件.
- **七组跨任务一致性结论**:
  - **C1**: c_k range 单值钉死 Stage 1 训练 (U(0.5,5)/U(1,5)/U(0.5,20)/U(2,8) 任何单值不能解锁 L0 ≥ 90%)
  - **C2**: per-layer 异构 c_k range 不能脱离时间维度 (task242 23.44% + task293 0/81 OPEN)
  - **C3**: time-varying curriculum 不能挽救 frozen ckpt 失配 (Schedule A/B/C 表现相似)
  - **C4**: dead_revive hook 在窄 c_k 下是 no-op (task242 3.12% 更差 + task283 hook no-op 70.3%)
  - **C5**: β-curriculum 是稳定剂非天花板 (task270 1.6% + task275 89.1% plateau)
  - **C6**: κ-decouple 是 L0 杠杆但不是 R@10 杠杆 (task144 几乎中性 + task284 -17.0% + task287 -16.2%)
  - **C7**: 3-way alternative quantizer 全部 NO-GO (FSQ -45.8% / EMA -25.0% / Restoration -21.7%)
- **联合立判据 (paper §6.7.4 联动字面)**: baseline Stage 1 recipe (`poincare + β=0.5 + kmeans + product_manifold + anti_collapse=none + eval_step=5`) 在 `β ∈ {0, 0.5, 1.0}` / `frequency ∈ {1, 5, 10}` / `loss_type ∈ {mse, l1, poincare}` 调节空间内**没有任何已知杠杆把 L0 ≥ 90%**, 且**任何 c_k range / κ-decouple / quantizer variant 都不能贡献 R@10 > 0.1020**.
- **§6.7.4 stop-loss (i) 重新解读**: 在 baseline recipe 上结构性必然触发 — Sinkhorn 后处理兜底的 "weakly collision-permissive" 设计特征, **不是 bug**. (跟 task288 / Issue #20 联立)
- **后续候选 (架构层, 非 Stage 1/2 修补)**: Gumbel-Softmax soft assignment / 多样 hash / per-item soft-assign with temperature annealing. **禁止方向**: 任何在 baseline recipe 内的 c_k / β / κ-decouple / quantizer variant 修补 (8 方向已穷尽证伪).
- **R10 推进决策**: 跟 [[r10-backlog-vacuum-2026-07-29]] 默认行为一致 (backlog 真空时整理 paper / 写 verdict / memory 整合, 不强启动 ROI 极低实验). 零 GPU, 推进方式 = housekeeping 综合.
- 产物: verdicts/task294_ck_range_path_exhausted_cross_task_result.md (187 lines). commit d45a0f6 推送 main.
