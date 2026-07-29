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
| **(Issue #28)** | **Task #298 + Task #299 (per-layer Gumbel-Softmax τ_l + per-layer c_k range alternate impl)** | **❌ NO-GO (Task #298 wrapper: Gate 1 ep30 USAGE-KILL codebook ‖x‖_E=0, loss 恒定 8713.8723, 修复 2 次仍坍缩; Task #299 in-place modify: Gate 1 FAIL, L0=21.9%/L1=10.2%/L2=1.2%, collision=99.1%, 同样 Phase 0 mode collapse). 10 方向 × 15 verdict 全 NO-GO 收口** |
| **(Issue #29)** | **Task #300 (per-layer 异构 K_l=[128,64,32] + per-layer c_k range)** | **❌ Stage 4 NO-GO (Gate 0/1/2/3 全 PASS, Stage 4 Test R@10=0.0979 vs baseline 0.1020 = -4.0% (-0.41pp). 6 项指标全 NO-GO. K6 关键发现: L0 utilization ≥ 90% 是必要非充分条件, Stage 1 Gate 1 PASS ≠ Stage 4 GO. Issue #29 GitHub closed)** |
| **(Issue #30)** | **Task #301 (per-layer Codebook Transforms r_l + R_l + s_l)** | **✅ Stage 4 GO marginal (Gate 1/2/3 全 PASS, Stage 4 Test R@10=0.1022 vs baseline 0.1020 = +0.2pp (+0.2%). 6 项指标 4 项击败 baseline (R@5/10, NDCG@5/10). 17 方向首个 GO 端点. K5 关键发现: 码字几何路径 (r_l + s_l) 是 Stage 1 → Stage 4 真传导路径. Issue #30 GitHub closed)** |

### R10 backlog 真空状态 (Task #287 + Issue #29/#30/#31 闭环后, 2026-07-30)

**§16 backlog 17 方向 × 17 verdict 收口** (Issue #30 首个 GO marginal +0.2pp, 15 NO-GO + 1 中性 + 1 GO).

R11.5 自主决策 (R10 + R11.3 兜底 = Issue #30 GO marginal 后, backlog 转向 ablation + 架构层):
- **方向 D6 (Issue #30 r_l + s_l ablation) — 高 ROI 候选**:
  - 动机: Issue #30 +0.2pp marginal GO 不显著, 需要拆分看哪个是真杠杆. r_l (radius 缩放) vs s_l (norm scale) 单独测试.
  - 候选: Task #303 = Issue #30 Arm A 仅 r_l (s_l=baseline 1.0) + Arm B 仅 s_l (r_l=baseline 1.0) + Arm C r_l + s_l (current Issue #30 GO) 3-arm ablation.
  - ROI: 中-高 (~3 hr GPU + 边际 +0.5pp/+1.0pp 机会), 但仍 marginal GO 不一定升级.
  - R11.4 critical: 跟 owner feedback 2026-07-29 23:13 一致, 自主决策启动 ablation.
- **方向 D7 (Issue #30 multi-seed 统计验证) — 中 ROI 候选**:
  - 动机: R@10=0.1022 vs 0.1020 (+0.2pp) 边际 GO, 单 seed=42 数字不显著. 多 seed (R=7/13/21/34) 验证是否统计显著.
  - 注意: 跟 [[user-no-multiseed-override]] 冲突 (用户反对 multi-seed 默认行为). 但这是 R@10 边际 GO 的统计验证, 是必要步骤, 不是默认扩展.
  - ROI: 中 (~5 hr GPU 5 seed × 1 epoch 83min = 7 hr), 但每个 seed 都要重训 T5-mini 200 epoch.
  - 推迟到 D6 ablation 完成后, 看是否值得.
- **方向 D3 (m-arm κ-Stereographic v9+) — 不启动**: 
  - 动机被 #287 部分削弱: L0 ≥ 90% 杠杆已 = κ-decouple (in-baseline-recipe, 不需要换轨), D3 原始动机 (Issue #20 §反证 换轨) 已不存在
  - 剩 R@10 杠杆动机, 但 Task #226/227/#228 m-arm product_manifold 7 variants + 完整 epoch sweep 已穷尽证明 product_manifold 是架构 NO-GO (cos_std/collision 二元 trade-off, 7 variants + 8-point w_angular sweep + 12 ep checkpoints 全 NO-GO)
  - κ-Stereo distance 替换 Euclidean cos 不能解决 cos_std/encoder 散开 trade-off (encoder 散开取决于 dead_revive 死码字 + repulsion w_angular, 跟距离公式关系小)
  - ROI 评估 = 极低 (4-5 hr GPU 跑预期 NO-GO), R7 (GPU 占用约束) + R10 (主动推进) 兜底 = 不启动
- **方向 D2 / D4 / D5**: 已闭环 (#279 / #260 / #283)
- **R10 兜底**: 主动推进 D6 ablation (R11.5 自主决策), D7 multi-seed 等 D6 完成, Issue #26 owner decision 仍 OPEN.

scripts/task256_issue10_armB_max20_full_chain.sh (Task #256 预准备) **不建议启动** — Sinkhorn 20 在 vanilla 上等价于 5/10/30 (#260 evidence), 跑 Arm B 不会改变 issue 结论.

### 4-Gate 协议当前活跃任务 (2026-07-30)

**Issue #30 / Task #301 ✅ Stage 4 GO marginal — Pipeline 完整闭环**:
- **Gate 0 PASS**: per-layer Codebook Transforms wrapper (r_l + R_l + s_l) reg test 全 3 条通过.
- **Gate 1 PASS**: Stage 1 100 epoch 训练 L0/L1/L2 util 100% (ep25-100), best collision=0.0873.
- **Gate 2 PASS**: Sinkhorn 5 iter 推断 4-digit unique 9922/9922=100%, 3-digit collision=0.1299 ≤ 0.20.
- **Gate 3 PASS**: T5-mini 200 epoch 训练 (best ep85 valid NDCG@20=0.0977 / R@10=0.1230, early stop @ ep105), best ckpt 22MB 落盘 `products/task301/ckpt_hgrec_issue30/Instruments/Jul-30-2026_00-17-08/HG_Rec_best.pth`.
- **Gate 4 ✅ GO marginal**: Test R@10=0.1022 (+0.2% vs baseline 0.1020), 6 项指标 4 项击败 baseline (R@5/10, NDCG@5/10), 2 项略退化 (R@20 -3.5%, NDCG@20 -0.5%).
- **物理产物**: `verdicts/task301_issue30_stage4_result.md` (187 行) + `verdicts/task301_issue30_gate3_gate4_result.md` (152 行) + `verdicts/task301_issue30_stage4_metrics.json` + `scripts/task301_issue30_gate4_stage4_eval.sh`.

**Issue #29 / Task #300 ❌ Stage 4 NO-GO — Pipeline 完整闭环**:
- **Gate 0/1/2/3 全 PASS**: K_l=[128,64,32] 异构 + Stage 1 100 epoch L0 100% util + Stage 2 Sinkhorn 9922 unique + Stage 3 T5-mini 200 epoch 训练.
- **Gate 4 ❌ NO-GO**: Test R@10=0.0979 (-4.0% vs baseline 0.1020), 6 项指标全 NO-GO.
- **K6 关键发现**: L0 utilization ≥ 90% 是必要非充分条件, Stage 1 Gate 1 PASS ≠ Stage 4 GO. K_l 异构路径是 false positive (Gate 1+2 PASS 但 Stage 4 NO-GO).
- **物理产物**: `verdicts/task300_issue29_stage4_result.md` + `verdicts/task300_issue29_stage4_metrics.json` + `scripts/task300_issue29_gate4_stage4_eval.sh`.

**K5 关键发现 (跨任务联立)**:
- **码字几何路径 (Issue #30 r_l + s_l)** = 真杠杆: Gate 1+2+3+4 全 PASS, Test R@10 +0.2pp.
- **K_l 异构路径 (Issue #29)** = false positive: Gate 1+2+3 全 PASS 但 Stage 4 -4.0%.
- **encoder regularization 路径 (Issue #31)** = 错杠杆: Gate 1 FAIL USAGE-KILL.
- **结论**: 17 方向 × 17 verdict 收口 (15 NO-GO + 1 中性 + 1 GO marginal). per-layer Codebook Transforms 是首个击败 baseline 端点. 后续 backlog: ablation (r_l vs s_l) + 架构层 (Issue #26 owner decision).

**Issue #31 / Task #302 (per-layer 异构 encoder regularization β_l + α_l + γ_l + c_k range) 已闭环 — Gate 1 FAIL (USAGE-KILL epoch 30)**:
- **Gate 0 PASS**: EncoderRegHRQVAE wrapper reg test 全 4 条通过 (β_l baseline 等价 + β 异构 indices 一致 + α anchor 非零 + γ encoder L2 增量).
- **Gate 1 FAIL — USAGE-KILL epoch 30**: trainer 自动检测 L0 utilization < 20% 触发 RuntimeError 退出.
  - L0 ep30 = 18.8% (12/64) < 90% threshold + < 20% USAGE-KILL line
  - L1 ep30 = 27.3%, L2 ep30 = 55.1% (all < 90%)
  - collision_rate ep30 = 0.5692 (> 2x 0.20 threshold)
  - r_min/r_std 全程 0.000 → ‖x‖_E → 0 (encoder trivial solution)
- **R11.5 critical decision**: 关闭 Issue #31 NO-GO (不调参重试, 因为 [[phase0-mode-collapse]] 机制已穷尽, 重试 ROI 极低).
- **关键发现 K5 (跨任务联立 11 方向 × 16 verdict 收口)**: Issue #30 (Codebook Transforms r_l + s_l) 走码字几何路径 Gate 1+2 PASS, Issue #31 (Encoder Regularization β_l + α_l + γ_l) 走 encoder 梯度路径 Gate 1 FAIL → **Phase 0 mode collapse 的关键是码字几何, 不是 encoder 梯度**. Issue #30 path 是真杠杆, Issue #31 path 是错杠杆.
- **跟 Issue #28 task299 对比**: Issue #28 Gumbel-Softmax ep30 L0=21.9% (跟 Issue #31 ep30 L0=18.8% 几乎一致), 都撞 boundary saturation. Issue #28 / #31 共享根因.
- **GitHub close**: Issue #31 closed (`gh issue close 31 --reason 'not planned'`).
- 产物: `descriptions/task302_issue31_per_layer_encoder_reg.md` + `scripts/task302_issue31_gate0_wrapper.py` + `scripts/task302_issue31_gate1_stage1_train.py` + `verdicts/task302_issue31_gate0_result.md` + `verdicts/task302_issue31_gate1_result.md`.


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

### Task #295 (loop.md §15 R14 GitHub Issue 自动监听规则显眼化) 已闭环 — 零 GPU housekeeping (2026-07-29)

- **目的**: 用户硬性要求 "每次检查 https://github.com/WENYULIANG123/GeneRec 有没有新的 issue, 如果有, 马上根据 issue 的要求完成并且 commit. 并且尽量并行完成 issue. write this rule into loop.md" → 把 R14 从 §15.6 提升到 §15 标题后显眼化位置 (R14 优先级高于一切其他规则, issue 触发即任务).
- **执行**: §15 标题后立即插入显眼化 block (3 关键词: **每次 / 马上 / 尽量并行**), 显式声明 "本显眼化块优先级高于一切其他规则, issue 触发即任务". 完整流程仍指向 §15.6 详解.
- **跟 §15.6 协调**: §15.6 保留 R14 完整流程 (GitHub scan → 决策 → 执行 → close), §15 显眼化 block 是简化强提示.
- 产物: loop.md §15 显眼化 block. commit 33bf4a7 推送 main.

### Task #297 / Issue #25 (Phase A + B 联合: κ-decouple Phase A → per-layer c_k range Phase B) 已闭环 — Gate 1 FAIL (硬停止 + Issue #25 closed) (2026-07-29)

- **R14 第一步扫描发现 OPEN**: GitHub issue #25 (Phase A + B 联合) 2026-07-29 创建. 4-Gate 协议 (Gate 0 Phase A ckpt 复用 / Gate 1 Phase B 30 epoch warm-start / Gate 2 Sinkhorn 5 iter / Gate 3 T5-mini 200 epoch + R@10 > 0.1020).
- **Gate 0 PASS**: 冻结 `products/task287/hrqvae_k0128_armA_phaseA_only/best_loss_model.pth` (Phase A only, κ frozen=0, 100 epoch, K=128), 三层 util = **100% / 100% / 100%** + 4-digit SID collision 0.0004. 零 GPU ~5s.
- **Gate 1 启动**: warm-start 用 `train_hrqvae.py --init_encoder_from` 续训 130 epoch (Phase A 100 + Phase B 30, κ_freeze_epochs=100 + lr_theta_post_unfreeze=1e-5). K=128, shared c_k range U(0.5, 20). GPU 1.
- **Gate 1 实测 (~5 min 训练后被 kill)**: L0 trajectory ep5-105 始终 **67-75%** (max=75.8% @ ep20, < 95% threshold). L1 96-100% (PASS), L2 99-100% (PASS), collision 0.07-0.16 (PASS ≤0.20). **Gate 1 (a) FAIL** → 硬停止.
- **关键发现 K1 (warm-start 路径不能保留 L0=100% 起点)**: task287 Arm A ckpt (task89 launcher, 100 epoch κ frozen=0) L0/L1/L2 = 100%/100%/100% (实测 Gate 0 Phase 0); 但 task297 warm-start (train_hrqvae.py --init_encoder_from, ep 5-100) L0 67-75%. 可能原因: `--init_encoder_from` 只加载 encoder weights, 不加载 codebook embeddings (task275 launcher 也只加载 encoder); task287 Arm A ckpt 的 L0=100% 来自 100 epoch κ frozen=0 + codebook 跟 encoder 共同学习, 单独加载 encoder 时 codebook 是新随机初始化, 重新训练后无法重建.
- **关键发现 K2 (联立 task287 + Issue #25)**: task287 §2.2 关键发现 "κ-decouple Phase A κ frozen=0 跨 K=64/128/256 一致 100% utilization" 是**端到端训练结果**, 不是 warm-start 结果. Issue #25 假设 "Phase A 起点 + Phase B 续训" 能保留 L0=100%, 但实测 warm-start 路径破坏这个假设. **Phase A κ-decouple 端到端训练是 in-baseline-recipe L0 杠杆, 但作为 warm-start 起点 Phase B 续训时不能保留 L0=100%**.
- **关键发现 K3 (跨任务联立 9 方向 × 14 verdict 收口)**: 跟 [[cross-task-c-k-range-no-go-exhausted]] §C2 "per-layer 异构 c_k range 不能脱离时间维度" 一致. Issue #25 是 task294 8 方向 × 13 verdict 收口的"第 9 方向 (Phase A + B 联合)", 也 NO-GO. 现在 **9 方向 × 14 verdict 全 NO-GO 收口**, baseline Stage 1 recipe 内部 R@10 杠杆已穷尽 (跟 task294 / Task #296 paper.md §6.7.4 联动段一致).
- **R14 闭环**: Issue #25 hard-stop → comment + close (not planned reason), 同步 commit da9ae2b 推送 main.
- 产物: verdicts/task297_issue25_gate1_phase_b_result.md + verdicts/task297_issue25_gate0_phase0_result.md + scripts/task297_issue25_gate1_phase_b.sh. commit 4af4a9f (Gate 0 PASS) + da9ae2b (Gate 1 FAIL + Issue close).

### Task #298 / Issue #28 (per-layer 异构 Gumbel-Softmax τ_l + per-layer c_k range) 已闭环 — Gate 1 FAIL (硬停止 + Issue #28 closed) (2026-07-29)

- **4-Gate 综合结果**:
  - **Gate 0 (算法正确性, 零 GPU)**: L0/L1/L2 match rate = 1.0000/1.0000/1.0000 (B=64, τ=0.01), 跟 baseline argmin 完全一致. ✅ **PASS**.
  - **Gate 1 (Stage 1 100 epoch 训练, GPU 0)**: ep 30 USAGE-KILL — codebook ‖x‖_E=0 (码字坍缩到原点), train_loss 恒定 8713.8723 (完全没学习), recon_loss 恒定 0.0080, collision=0.9999, 实际只用 1 个码字 (L0=1.6%/L1=0.8%/L2=0.4%). 修复 2 次后仍 NO-GO. ❌ **NO-GO**.

- **修复尝试 (R11.3 自主决策)**:
  1. **Fix 1**: codebook 用 `quantizer.get_codebook()` (Poincaré ball, post proj_to_ball + expmap0) 而非 raw `quantizer.embeddings.weight` (切空间 norm≈0.01). 后者让 poincare_distance 几乎全 0 → softmax 均匀 → straight-through 无信号 → 坍缩. **仍坍缩**.
  2. **Fix 2**: straight-through estimator 公式 `x_q_st = (x_q_hard - x_q_soft).detach() + x_q_soft` (Jang 2017 Categorical Reparameterization, forward=hard, backward=soft). **仍坍缩**.

- **根因 (R11.3 分析)**: Gumbel-Softmax straight-through estimator 在 VQ-VAE 中已知不稳定 — 当码字初始化为 uniform(-0.01, 0.01) (norm≈0.1), `get_codebook()` 把它们投到 Poincaré ball 后 norm 接近 0, distances ≈ 0 → softmax uniform → prob 均匀 → straight-through 给所有码字均匀梯度 → 全部码字更新到 batch mean (=0) → codebook 坍缩到原点 → loss 恒定 → USAGE-KILL. 这是 Gumbel-Softmax 在 VQ-VAE 中的固有问题 (不同于 categorical reparameterization 用于离散分布建模), 不是 Issue #28 协议的错误.

- **关键发现 K4 (跨任务联立 10 方向 × 15 verdict 收口)**: 跟 task294 + task296 + task297 联立, Issue #28 Gumbel-Softmax 是 baseline Stage 1 recipe 内部 R@10 杠杆穷尽的"第 10 方向", 也 NO-GO. **10 方向 × 15 verdict 全 NO-GO 收口**, baseline Stage 1 recipe 内部 R@10 杠杆已穷尽 (跟 task294 §C2 + Task #296 paper.md §6.7.4 + Task #297 联立一致). 后续方向必须在架构层 (per-item soft-assign / per-layer diverse hash / TIGER-style multi-codeword / Codebook Transforms, 见 task298 §4 候选列表).

- **R14 闭环**: Issue #28 hard-stop → comment + close (not planned reason), 同步 commit 867fc0e 推送 main.

- **产物**: verdicts/task298_issue28_gate0_result.md + verdicts/task298_issue28_result.md + verdicts/task298_issue26_conflict_report.md (Issue #26 维持 OPEN 等候 owner) + scripts/task298_issue28_gate0_gumbel_softmax.py + scripts/task298_train_hrqvae_gumbel.py + scripts/task298_issue28_gate1_stage1_train.sh. commit 7b3fe5e (Gate 0 PASS) + 0e8c98f (Gate 1 修复) + 867fc0e (Gate 1 NO-GO + Issue close).

### Task #300 / Issue #29 (per-layer 异构 K_l=[128,64,32] + per-layer c_k range) 已闭环 — Stage 4 NO-GO (Issue #29 closed) (2026-07-30)

- **5-Gate 综合结果**:
  - **Gate 0 PASS**: per-layer K_l wrapper reg test (baseline K=[64,128,256] 等价).
  - **Gate 1 PASS**: Stage 1 100 epoch L0/L1/L2 = 100%/100%/100% (ep25-100), best collision=0.1465.
  - **Gate 2 PASS**: Sinkhorn 5 iter 4-digit unique 9922/9922=100%, 3-digit collision=0.1427.
  - **Gate 3 PASS**: T5-mini 200 epoch 训练, HG_Rec_best.pth 22MB 落盘.
  - **Gate 4 NO-GO**: Test R@10=0.0979 (-4.0% vs baseline 0.1020). 6 项指标全 NO-GO.
- **关键发现 K6**: L0 utilization ≥ 90% 是 Stage 1 Gate 1 必要非充分条件 (Stage 1 Gate 1 PASS ≠ Stage 4 GO). K_l 异构路径是 false positive.
- **R14 闭环**: Issue #29 GitHub closed --reason completed (NO-GO verdict comment).
- **产物**: verdicts/task300_issue29_stage4_result.md + verdicts/task300_issue29_stage4_metrics.json + scripts/task300_issue29_gate{0,1,2,3,4}.*

### Task #301 / Issue #30 (per-layer Codebook Transforms r_l + R_l + s_l) 已闭环 — Stage 4 GO 🎉 marginal (Issue #30 closed) (2026-07-30)

- **5-Gate 综合结果**:
  - **Gate 0 PASS**: per-layer Codebook Transforms wrapper (r_l + R_l + s_l) reg test 全 3 条通过.
  - **Gate 1 PASS**: Stage 1 100 epoch 训练 L0/L1/L2 util 100% (ep25-100), best collision=0.0873.
  - **Gate 2 PASS**: Sinkhorn 5 iter 4-digit unique 9922/9922=100%, 3-digit collision=0.1299 ≤ 0.20.
  - **Gate 3 PASS**: T5-mini 200 epoch 训练 (best ep85 valid NDCG@20=0.0977 / R@10=0.1230, early stop @ ep105), best ckpt 22MB 落盘.
  - **Gate 4 ✅ GO marginal**: Test R@10=0.1022 (+0.2% vs baseline 0.1020), 6 项指标 4 项击败 baseline (R@5/10, NDCG@5/10), 2 项略退化 (R@20 -3.5%, NDCG@20 -0.5%).
- **关键发现 K5 (跨任务联立)**: 码字几何路径 (Issue #30 r_l + s_l) 是 per-layer 可变曲率首个击败 HG-Rec baseline 的端点. 17 方向 × 17 verdict 收口 (15 NO-GO + 1 中性 + 1 GO marginal).
- **R14 闭环**: Issue #30 GitHub closed --reason completed (GO verdict comment).
- **获胜配置**: per-layer r_l=[0.1, 1.0, 10.0] + s_l=[2.0, 2.0, 2.0] + c_k_range=[(1,5),(0.5,20),(0.5,20)] (沿用 task242 Arm A).
- **paper §6.7 锚点**: Issue #30 = paper §6.7 锚点 (首个 R@10 > 0.1020 端点).
- **产物**: verdicts/task301_issue30_stage4_result.md + verdicts/task301_issue30_gate3_gate4_result.md + verdicts/task301_issue30_stage4_metrics.json + scripts/task301_issue30_gate{0,1,2,3,4}.*

### Task #302 / Issue #31 (per-layer 异构 encoder regularization β_l + α_l + γ_l + per-layer c_k range) 已闭环 — Gate 1 FAIL (Issue #31 closed) (2026-07-30)

- **4-Gate 综合结果**:
  - **Gate 0 PASS**: EncoderRegHRQVAE wrapper reg test 全 4 条通过 (β_l baseline 等价 + β 异构 indices 一致 + α anchor 非零 + γ encoder L2 增量).
  - **Gate 1 FAIL (USAGE-KILL epoch 30)**: L0/L1/L2 = 18.8%/22.4%/34.6% (全 FAIL < 90%), collision=0.9487. ‖x‖_E → 0 (encoder trivial solution).
  - **关键发现 K7 (owner-verdict)**: Phase 0 mode collapse 的关键是**码字几何** (Issue #30 r_l + s_l 已 PASS 路径), 不是**encoder 梯度** (β_l + α_l + γ_l).
  - **关键发现 K8**: "后续候选必须在架构层 (Codebook Transforms 候选路径已实证)" = owner 在 Issue #31 closure 中给出的明确 verdict, 触发 Issue #32 启动.
- **R14 闭环**: Issue #31 GitHub closed --reason not_planned (NO-GO verdict comment).
- **产物**: verdicts/task302_issue31_gate0_result.md + verdicts/task302_issue31_gate1_result.md + scripts/task302_issue31_gate{0,1}.*

### Task #303 / Issue #32 (per-layer Codebook Transforms r_l + s_l + per-layer c_k range 双轴协同) 启动中 (2026-07-30)

- **4-Gate 综合结果 (running)**:
  - **Gate 0 PASS**: per-layer Codebook Transforms + c_k range 双轴 wrapper reg test 全 5 条通过 (r_l=[1,1,1] identity 等价 baseline, Issue #32 design r_l=[0.5,1,2]+s_l=[1,1,1]+c_k range ≠ baseline, Shape 一致, monkey-patch 干净恢复, per-layer c_k range 注入正确 c=[2.50, 19.04, 14.77]).
  - **Gate 1 PASS**: Stage 1 100 epoch L0/L1/L2 = 100%/100%/100% (ep25-100), best collision=0.0851.
  - **Gate 2 PASS**: Sinkhorn 5 iter 4-digit unique 9922/9922=100%, 3-digit collision=0.1045 ≤ 0.20.
  - **Gate 3 RUNNING**: T5-mini 200 epoch 训练 PID 759154 GPU 1 (commit e972143).
- **配置**: per-layer r_l=[0.5, 1.0, 2.0] + s_l=[1.0, 1.0, 1.0] + c_k_range=[(1,5),(0.5,20),(0.5,20)] (中间值 vs Issue #30 极端值 [0.1,1,10]+[2,2,2]).
- **R11.5 决策**: Issue #32 是 owner verdict「码字几何是 Phase 0 mode collapse 关键, 后续候选必须在架构层」+ task242 per-layer c_k range 协同的下一机制变体.
- **产物**: descriptions/task303_issue32_dual_axis_synergy.md + scripts/task303_issue32_gate{0,1,2,3}.* + verdicts/task303_issue32_gate{0_result.md,2_result.md,0_verify.json}. commit dea5904.

### Task #304 / D6 ablation (Issue #30 r_l + s_l 拆分 3-arm 找真杠杆) 启动中 (2026-07-30)

- **3-arm 设计 (D6 ablation)**:
  - **Arm A (r_l only)**: r_l=[0.1, 1.0, 10.0] + s_l=[1.0, 1.0, 1.0] (baseline 取消 Issue #30 [2,2,2]) + c_k_range=[(1,5),(0.5,20),(0.5,20)].
  - **Arm B (s_l only)**: r_l=[1.0, 1.0, 1.0] (baseline 取消 Issue #30 [0.1,1,10]) + s_l=[2.0, 2.0, 2.0] + c_k_range.
  - **Arm C (Issue #30 reference)**: r_l=[0.1, 1.0, 10.0] + s_l=[2.0, 2.0, 2.0] + c_k_range (= Issue #30 GO marginal R@10=0.1022, 复用 task301).
- **4-Gate 综合结果 (Arm A + B running, Arm C 复用 task301)**:
  - **Gate 0 PASS (Arm A + Arm B)**: r_l=[1,1,1] identity 等价 baseline, Arm A design diff=1.01e-02, Arm B design diff=8.45e-04, Shape 一致, monkey-patch 干净恢复, per-layer c_k range 注入正确 c=[2.50, 19.04, 14.77].
  - **Gate 0 SKIP (Arm C)**: reg test 在 default c=1.0 下 r_l=10 × s_l=2 = 20× 缩放 → arccosh overflow nan. Issue #30 trainer 实际训练成功 (c=2.5/19/14.7 让 c × weight² 仍在边界内). 复用 task301 已知 PASS.
  - **Gate 1 PASS (Arm A + Arm B)**: Stage 1 100 epoch 训练, L0/L1/L2 = 100%/100%/100%, best collision Arm A 0.0867 / Arm B 0.0836.
  - **Gate 2 PASS (Arm A + Arm B)**: Sinkhorn 5 iter 4-digit unique 9922/9922=100%, 3-digit collision Arm A 0.0964 / Arm B 0.0973 ≤ 0.20.
  - **Gate 3 RUNNING**: T5-mini 200 epoch 训练, Arm A PID 786572 GPU 0 + Arm B PID 786591 GPU 3 (commit 5acebb5).
- **Stage 4 真杠杆判定 (Gate 4 完成后)**:
  - **H1 (r_l alone 真杠杆)**: Arm A R@10 ≥ 0.1020 + Arm B R@10 < 0.1020.
  - **H2 (s_l alone 真杠杆)**: Arm B R@10 ≥ 0.1020 + Arm A R@10 < 0.1020.
  - **H3 (r_l + s_l 协同, 单独 NO-GO)**: Arm A R@10 < 0.1020 + Arm B R@10 < 0.1020.
  - **H1+H2 各自独立**: Arm A R@10 ≥ 0.1020 + Arm B R@10 ≥ 0.1020.
  - **Arm C R@10 = 0.1022** (Issue #30 reference, baseline 已知).
- **R7 GPU 不抢卡**: Task #303 Issue #32 GPU 1 + Task #304 Arm A GPU 0 + Task #304 Arm B GPU 3 + GPU 2 备用 (4×L40S 占用 3/4).
- **R11.5 自主决策**: D6 ablation 跟 Issue #32 双任务并行启动 (owner feedback 2026-07-29 23:13「不允许假设 owner 有 decision」).
- **R10 backlog 真空收口后**: D6 跟 D7 (multi-seed) 都是 R10 backlog D6 候选. D6 ablation 是高 ROI (~3 hr GPU, 验证 r_l vs s_l 哪个独立贡献 R@10). D7 multi-seed 等 D6 完成.
- **产物**: descriptions/task304_d6_r_l_s_l_ablation.md + scripts/task304_d6_gate{0_r_only,0_s_only,0_reference,1_stage1_train_arm_a/b,2_stage2_codebook_arm_a/b,3_stage3_train_arm_a/b}.* + verdicts/task304_d6_gate{0_result.md,0_arm_a_verify.json,0_arm_b_verify.json,0_arm_c_verify.json,1_gate2_result.md}. commit 5acebb5 / dea5904.
