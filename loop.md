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

## §16. 当前活跃任务

> **🟡 §16 当前状态 (2026-07-29 当前)**: Issue #9/#10/#11/#12 已 NO-GO 闭环, Issue #13/#16 CLOSED. **Issue #18 (Task #280) 全 3 Gate 闭环 — §6.7.4 stop-loss (i) 口径绑定 Stage 1 argmin, task253 L0=73.44% / task222 ep29 L0=65.62% 复算 PASS, 6 个 vanilla 测点 (B 口径) 100% 反证闸门真闸门, 0 GPU**. **Issue #19 (Task #281) 全 3 Gate 闭环 — 通用 `scripts/issue19_gate_template.sh` 3 回放 PASS (a/b exit 1, c exit 0), 4 存量脚本 (`task237`/`task256`/`task188_to_193`/`task194`) 文件头标注, `papers/paper.md` Gate 3 字面写死**. **Task #277/#278 已闭环**.
> **Task #279 (K=512/1024 Stage 3 + Stage 4 eval) 仍在跑**: 12 stage3_train children 进程alive (父进程 spawn 后), K=512 + K=1024 HG_Rec_best.pth 已存. Waiter (`task279_stage4_eval.sh`) 每 2min check, 期望 Stage 4 完成后自动 fire eval.

### 已闭环 (近 24 小时)

| Issue | 任务 | 状态 |
|-------|------|:----:|
| #9 | Task #234/235 hybrid per-layer assignment | ❌ FULL NO-GO (Gate 1 FAIL: L0 util 12.5%, L1/L2 util 0.78%, collision 0.9988) |
| #10 | Task #236/237/245/259/260 collision 口径 + 3-arm + Sinkhorn 扫描 | ❌ Gate 0 PASS (#236/#245/#259) + Gate 1 Sinkhorn 旋钮 FAIL (#260: vanilla 上 5 iter 即 full convergence, 三臂无 ≥ 15pp 分离) |
| #11 | Task #241/242 per-layer c_k range | ❌ Gate 1b FULL NO-GO (Arm A L0 23.44%, Arm A+ dead_revive L0 3.12%) |
| #12 | Task #244 SID 沙漏集中度分布画像 | ❌ Gate 0 FAIL (排除 L3 K=1: 0 个 arm 满足 Gini ≥ 0.5 AND util ≥ 0.9) |
| #13 | Task #248/249/253/254 Möbius 残差 4-gate | ❌ NO-GO (Gate 2 实测 R@10=0.000403 但被 #16 越闸 audit, 证据基础退回到 task225 0.0938 -8.1%) |
| #16 | Task #257/258 #13 Gate 2 越闸 audit | ✅ CLOSED (Gate 0 PASS 取回全产物 + Gate 1 STOP: L0 util<90% stop-loss 越闸, 0.000403 作废) |
| (排名) | Task #246 paper-aligned ranking v3 增量 | ✅ done (Caser 0.0463→0.0378, LETTER 0.0997→0.0509) |
| (eval) | **Task #277 (Task #243 Stage 4 eval)** | ✅ done (epoch=200/400 R@10 完全相同 0.0978, 训练时长非变量 REFUTED) |
| (eval) | **Task #278 (12 ckpt 批量 Stage 4 eval)** | ✅ done (4 GO: task194_k0256 ⭐0.1053, task194_k064 0.1041, task156 0.1034, task194_k0128 0.1027) |

### 等用户决策 / R10 backlog 候选 (R11.5 自主决策推进)

候选方向 (按 Task #278 新发现):
- **方向 D1 (R10 推荐)**: 用 **task194_k0256 SID** (R@10=0.1053 当前最佳) 重跑 Issue #10 Gate 1 3-arm 曲线. 验证 κ-decouple Arm A (task144 0.1026) 在最优 K=256 SID 下是否突破 0.1053. 需 Stage 1 RQ-VAE 重训 + Stage 2 Sinkhorn + Stage 3 T5 + Stage 4 eval, 估约 4-6 小时 GPU.
- **方向 D2 (R10 备选)**: 探索 K=512/1024 (Task #194 K-sweep 提示 L0 越大越好). Stage 2 重训 + Stage 3/4, 估约 2-3 小时 GPU.
- **方向 D3 (backlog)**: task272 m-arm κ-Stereographic v9+ (用户 2026-07-24 提议 + R11.5 自主推进候选).
- **方向 D4 (低 ROI)**: Issue #10 接受方向 A2 NO-GO 闭环 (R11.5 默认决策).

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
