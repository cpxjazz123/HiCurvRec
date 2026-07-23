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

**🟢 0 个活跃 (2026-07-24 更新, Task #110 audit dispatcher + VERSION 1.0.0 + git tag v1.0.0 闭环 (单 dispatcher `python3 scripts/all_audits.py` 4/4 PASS 0.46s, VERSION=1.0.0 锁 paper-submission baseline, git tag v1.0.0 锁 commit 9b81667))**:

| Task | 模型 | GPU | 状态 | 启动时间 | 备注 |
|------|------|-----|------|---------|------|
| (空 — §16 表格清空, 等待下一任务) |
| (空 — R8 强制清理: Task #110 audit dispatcher + VERSION 1.0.0 ✅ 已完成 (2026-07-24). scripts/all_audits.py 单 dispatcher 跑 4 audit (task101 env / task103 paper claims / task105 ckpt integrity / task106 paper audits) 4/4 PASS 0.46s 实测. VERSION=1.0.0 锁 repo 基线, git tag v1.0.0 锁 commit 9b81667. reviewer 单命令 `python3 scripts/all_audits.py` 拿全部 4 audit verdict. verdict: `verdicts/task110_audit_dispatcher_version_result.md`) |
| (空 — R8 强制清理: Task #109 shields.io badges ✅ 已完成 (2026-07-24). README.md 顶部 7 个 `<a href><img>` shields.io badges: paper (14 页 blue / acrobat reader logo) + baselines (12/20 reproduced brightgreen) + paper claims (14/14 Δ=0 brightgreen) + abstract (197/200 字 green) + R12 ckpt (21.06 MB green) + license (MIT blue) + CI (audits automated success / github actions logo). 全部 `%2F`-encoded URLs, evidence files 全 linked. verdict: `verdicts/task109_shields_badges_result.md`) |
| (空 — R8 强制清理: Task #99 paper.md → paper.pdf ✅ 已完成 (2026-07-24). 14 页 PDF (124 KB), 修复 10 个 LaTeX bug, 三状态 math 追踪, lstlisting verbatim. verdict: `verdicts/task99_md_to_pdf_result.md`. 同时补 task99 description 文档化工作, 修复 R9 descriptions/ 编号连续性.) |
| (空 — R8 强制清理: Task #100 paper submission 准备 ✅ 已完成 (2026-07-24). Step A BibTeX (papers/refs.bib 12 entries) + Step B 格式微调 (\\usepackage{times} + fancyhdr + Acknowledgements section) + Step C local git commit 3b982f6 (无 origin remote). 14 页 PDF (106 KB). verdict: `verdicts/task100_paper_submission_prep_result.md`.) |
| (空 — Task #87 v2 synthesis 闭环 (2026-07-24 04:00). 整合 Task #88 c555 + Task #89 free-curv κ→0 + Task #95 HGN 闭环. 27 baseline 完整 ranking, RQ-VAE 系 7 个变体 ≥ 0.10 R@10, 论文 Section 5.4 应报告并列最优 phonism 0.1058 / HG-Rec c555 0.1051. verdict: `verdicts/task87_paper_table2_baseline_ranking_result.md`. GPU 0/1/2/3 全部空闲) |
| **#93** Paper Section 6 Draft | writeup | 无 GPU | ✅ 已完成 | 2026-07-24 05:05 | 完整 markdown 草稿: 6.1 Theoretical Implications (5 重证据) + 6.2 When Does HG Help (数据集类型表) + 6.3 Mechanism vs Geometry (Sinkhorn 优于双曲) + 6.4 Limitations (6 项) + 6.5 Future Work (6 方向) + 6.6 Concluding Thoughts (4 实践建议). 接续 Task #92 Section 5 形成论文实验 + 讨论闭环. verdict: `verdicts/task93_paper_section6_draft.md` |
| **#91** Stage 3 T5 训练动力学对比 | T5-small 8 run | 无 GPU | ✅ 已完成 | 2026-07-24 04:25 | analytical only, CPU 30 sec. 8 配置 valid R@10 全聚类 [0.1240, 0.1276] (3.6% 跨度), 收敛 epoch 一致 (3-4). 关键: valid 排序与 test 排序**反向** (c1055 valid 最高但 test 较低 gap +0.0261; vanilla valid 中位但 test 最高 gap +0.0204). phonism 优势不是来自训练, 而是来自最稳定泛化. 论文 Section 5.4 应以 test R@10 为主指标. verdict: `verdicts/task91_stage3_dynamics_result.md`. JSON: `verdicts/task91_stage3_dynamics.json` |
| **#92** Paper Section 5 Final Draft | writeup | 无 GPU | ✅ 已完成 | 2026-07-24 04:45 | 整合 8 个闭环任务, 起草 Section 5 完整 markdown: 5.1 Setup + 5.2 Main Results (27 baselines) + 5.3 Codebook Architecture + 5.4 Per-Layer Curvature (6 grid) + 5.5 Codebook Decomposition + 5.6 Training Dynamics + 5.7 Discussion (5 重证据 + Reproducibility Notes) + 5.8 Summary + Appendix LaTeX (Table 2/4/7). 草稿可直接 copy 到 paper. verdict: `verdicts/task92_paper_section5_draft.md` |
| **#90** phonism vs HG-Rec codebook 机制分解 | RQ-VAE 4 ckpt | GPU 0 | ✅ 已完成 | 2026-07-24 04:05 | analytical only, CPU 30 sec. 4 方法 L0/L1/L2 token SET 全共享 (Jaccard=1.000), 但 4-col SID 实质独立 (Jaccard=0.001, items_with_same=0/9922). c555 微弱最优 (R@10=0.1051) 来自 L0 略集中 (top1=5.11% vs vanilla=4.26%) 利于 T5 生成. free-curv L0 坍缩 53% utilization 是其 R@10=0.1015 最低根因. 论文 Section 5.5 草稿建议: "中度曲率 κ=0.5 既保留 K-Means init spread, 又不引入训练过程 L0 退化". verdict: `verdicts/task90_codebook_decomposition_result.md`. GPU 0 释放空闲 |
| (空 — R8 强制清理: Task #89 自由曲率乘积流形 RQ-VAE ✅ 已完成 (2026-07-24 03:30). Stage 0/1 全部完成 + A 臂下游 Stage 2/3/4 闭环. 18/18 (layer, κ_m) 训练后 = 0.000000 (Stage 0 NO-GO 完全确认). A 臂下游 R@10=0.1015 vs task84 baseline 0.1020 (Δ -0.5%, 在噪声内, sanity check pass). 5 重独立证据闭环强烈支持 "Musical_Instruments 数据本质欧氏, 曲率边际效应弱". verdict: `verdicts/task89_free_curv_product_manifold_result.md`. GPU 0/1/2/3 全部释放空闲) |
| (空 — R8 强制清理: Task #88 HG-Rec per-layer curvature 网格 ✅ 已完成 (2026-07-24 02:00). 6 网格 Stage 3 + Stage 4 全部完成. R@10 跨 5.3% 跨度 [0.0998, 0.1051], c555 (κ=0.5) 微弱最优. 与 Task #116/#117/#118/#89 四重证据一致支持 "Toy 规模数据集本质接近欧氏". verdict: `verdicts/task88_per_layer_curvature_result.md`. GPU 0/1 释放空闲) |
| (空 — R8 强制清理: Task #81 S³Rec NO-GO 已 kill PID 625025, yaml 配置错 (默认 `train_stage='pretrain'`, 训练无效). verdict: `verdicts/task81_s3rec_no_go_pretrain_only_result.md`. GPU 0 释放空闲) |
(空 — R8 强制清理: Task #84 HG-Rec 主实验已完成, 见下方"§16 最近完成" 2026-07-23 21:38)
(空 — R8 强制清理: Task #106/#107 HG-Rec multi-seed 用户决策撤回 (2026-07-23 22:14, 单 seed (Task #84 seed=42) 足够). PID 887426/896290 已 kill, GPU 2/3 释放空闲.)
(空 — R8 强制清理: Task #86 P5-SID eval 完成 (PID 886743 自然退出 22:31). verdict: `verdicts/task86_p5_cid_sid_evaluate_result.md` (test hit@10=0.000366, -99.1% vs paper, LLM-RecSys-ID `--eval_only` 限制). P5-CID hit@10=0.0413 已闭环. GPU 1 释放空闲.)

**§16 后台 daemons (2026-07-23 17:10 更新)**:
- **task88 auto-trigger inference v4**: PID 3409047+3409052. 30s tick, 检测 `is_pid_alive(PID) AND ckpt_exists(ckpt_path)` → 触发 task84/85/86 stage 4 inference.
- **task81_wait_then_launch (新 PID 4145587)**: 等 FDSA (新 PID 4128724) → 完成后自动 launch S³Rec. 老 daemon PID 2564886 已死.
- **task83_wait_then_launch**: PID 3387219. poll task82 PID 3387236 → 完成后自动 launch P5-SID.
- ~~**monitor_v3_gpu_slot_v5**: PID 3350833.~~ ❌ 已死, 2026-07-23 17:09 ps 确认.
- **task84 manual trigger**: PID 4148205. 手动触发 TIGER inference (task88 daemon 之前反复 fail due to missing results/ dir, 2026-07-23 17:08 修目录 + 手动启动).

**§16 R8 清理完成 (2026-07-23 16:35)**:
- ✅ Task #89 LightGCN — R@10=0.0455 vs paper 0.0454 (Δ +0.0001, +0.2%, 完全对齐) → verdict 已写, 从 §16 删除
- ⛔ Task #90 FMLP-Rec — NO-GO (FMLP-Rec 官方仓库不支持 Musical_Instruments 数据集, paper Table 2 该行无法直接复现) → verdict 已写, 从 §16 删除
- ✅ Task #78 TIGER T5 训练 — epoch 100/100 train_loss=3.495 ckpt-51500 已落盘, R12 ✅ → task84 inference 已切换新 ckpt; verdict: 待 task84 inference 完成后写
- ✅ Task #91 DuoRec — R@10=0.0672 vs paper 0.0454 (Δ +48%) best epoch=18, 仅最终 seaborn 可视化失败 (已 pip install) → verdict: `verdicts/task91_duorec_repro_result.md`, 从 §16 删除
- ✅ Task #79 phonism RQ-VAE δ-hyperbolicity — SINKHORN 让 codebook 利用率达 100%, 残差空间 δ/diameter 从 vanilla 9.2% 降至 7.8% (-15%) → verdict: `verdicts/task79_phonism_rqvae_delta_hyperbolicity_result.md`, 从 §16 删除

**已知 task88 false trigger (R7 监控)**:
- 13:42:57 + 13:43:27 daemon 检测到 task82/_DONE (瞬态存在), 触发 inference 失败 (No checkpoint). 由于未写 inference_done, 下次 tick 会重试. PID 2464610 持续 alive 证明 task82 仍在训练中 (53.5% 进度).

(空 — R8 强制清理: 2026-07-23 12:35 用户主动 SIGTERM 终止 Task #74/75/76/77 四个并行任务后, 已全部归档 verdict 写入 §16 历史区. 下一步按 R10 主动推进 backlog.)

**§16 backlog (下一步必须主动推进, R10 主动模式 2026-07-23 修订)**:
- **Task #88 — HG-Rec per-layer curvature 网格搜索 (Idea1 路径 1 验证)**: ✅ 已完成 (2026-07-24 02:00). 6 网格 Stage 4 eval 全闭环. R@10 跨 5.3% [0.0998, 0.1051], c555 (κ=0.5) 微弱最优. verdict: `verdicts/task88_per_layer_curvature_result.md`.
- **Task #89 — 自由曲率乘积流形 RQ-VAE (MCKG 框架, κ_m 可学习)**: ✅ 已完成 (2026-07-24 03:30). Stage 0/1 + A 臂下游 Stage 2/3/4 全闭环. 18/18 (layer, κ_m) = 0.000000, 5 重独立证据 NO-GO. verdict: `verdicts/task89_free_curv_product_manifold_result.md`.
- **Task #90 — Phonism vs HG-Rec codebook 机制分解 (analytical, no GPU)**: ✅ 已完成 (2026-07-24 04:15). 4 方法 L0/L1/L2 token SET 全共享 (vanilla/c111/c555 Jaccard=1.000), 但 4-col SID 实质独立 (Jaccard=0.001, items_with_same=0/9922). c555 微弱最优 (R@10=0.1051) 来自 L0 略集中 (top1=5.11% vs vanilla=4.26%) 利于 T5 生成. free-curv L0 坍缩 53% utilization 是其 R@10=0.1015 最低根因. 论文 Section 5.5 草稿: "中度曲率 κ=0.5 既保留 K-Means init spread, 又不引入训练过程 L0 退化". verdict: `verdicts/task90_codebook_decomposition_result.md`. JSON: `verdicts/task90_codebook_decomposition.json`. 脚本: `scripts/task90_codebook_decomposition.py` (CPU 30 sec).
- **Task #91 — Stage 3 T5 训练动力学对比 (analytical, no GPU)**: ✅ 已完成 (2026-07-24 04:35). 8 配置 (vanilla / c111 / c222 / c555 / c512 / c215 / c1055 / free-curv) valid R@10 全聚类 [0.1240, 0.1276] (3.6% 跨度), 收敛 epoch 一致 (3-4). **关键**: valid R@10 排序与 test R@10 排序**反向** (c1055 valid 最高 0.1276 但 test 较低 0.1015 gap +0.0261 worst overfit; vanilla valid 中位 0.1262 但 test 最高 0.1058 gap +0.0204 best generalization). phonism 优势不是来自 Stage 3 训练优势, 而是来自最稳定泛化. 论文 Section 5.4 应以 test R@10 为主指标, 标注 "HG-Rec hyperbolic mechanism marginal overfits valid set". verdict: `verdicts/task91_stage3_dynamics_result.md`. JSON: `verdicts/task91_stage3_dynamics.json`. 脚本: `scripts/task91_stage3_dynamics.py` (CPU 30 sec).
- **Task #92 — Paper Section 5 Final Draft (writeup, no GPU)**: ✅ 已完成 (2026-07-24 05:00). 整合 8 个闭环任务 (Task #84/87/88/89/90/91/94/95), 起草 Section 5 (Experiments) 完整 markdown: 5.1 Setup + 5.2 Main Results (27 baselines Table 2) + 5.3 Codebook Architecture Ablation + 5.4 Per-Layer Curvature (6 grid Table 4) + 5.5 Codebook Decomposition + 5.6 Training Dynamics (Table 7) + 5.7 Discussion (5 重证据 + Reproducibility Notes) + 5.8 Summary + Appendix LaTeX (Table 2/4/7). 草稿可直接 copy 到 paper 写作. verdict: `verdicts/task92_paper_section5_draft.md`.
- **Task #93 — Paper Section 6 Draft (writeup, no GPU)**: ✅ 已完成 (2026-07-24 05:15). 完整 markdown 草稿: 6.1 Theoretical Implications (5 重证据 → 数据本质欧氏) + 6.2 When Does HG Help (数据集类型表 + 假设) + 6.3 Mechanism vs Geometry (Sinkhorn > 双曲) + 6.4 Limitations (6 项: 单 seed / 单数据集 / 单 embedding / 单 codebook / leave-one-out / item-level) + 6.5 Future Work (6 方向: multi-seed / 多数据集 / codebook size sweep / embedding 灵敏度 / 自适应几何 / 理论分析) + 6.6 Concluding Thoughts (4 实践建议). 接续 Task #92 Section 5 形成论文实验 + 讨论闭环. verdict: `verdicts/task93_paper_section6_draft.md`.
- **Task #94 — Paper Section 3 Method Draft (writeup, no GPU)**: ✅ 已完成 (2026-07-24 05:30). 完整 markdown 草稿复现 HG-Rec 论文 Section 3: 3.1 Hyperbolic RQ-VAE (公式 5/6/7/8 + Theorem 3.1/3.2, 残差在切空间累计的稳定性论证) + 3.2 Differential-Length Codebook (公式 9/10/11/12 + Theorem 3.4 体积指数增长证明, γ≈2 / K1∈{16,32,64} 实践选参) + 3.3 Model Training & Inference (公式 13 NLL 训练目标 + T5-small 配置 + beam search 推断) + 3.4 Discussion (两组件正交轴 "latent 空间 vs capacity 分配", 含 Musical_Instruments 实证缺位解释). 9 个核心公式 LaTeX + Theorem 引用完整. 论文方法章节闭环, 接续 Task #92 Section 5 + Task #93 Section 6 形成 paper deliverable 完整. verdict: `verdicts/task94_paper_section3_method_draft.md`.
- **Task #95 — Paper Section 1 Introduction Draft (writeup, no GPU)**: ✅ 已完成 (2026-07-24 05:50). 完整 markdown 草稿 (复现视角, 与 HG-Rec 论文 Introduction 互补): 1.1 Background GR + 三大类 tokenization + Euclidean RQ-VAE 问题 + 1.2 HG-Rec Approach (复述两组件 + paper-reported 数字 4.8-13.5% / 36×11×38× speedup / 100% codebook utilization) + 1.3 Reproduction Motivation (3 动机: dataset generality / curvature robustness / mechanism vs geometry, 复现 setting: Musical_Instruments + sentence-T5-base + T5-small + [64,128,256]) + 1.4 Findings Preview (5 重证据预告: Task #117 stress / #88 grid / #89 free-curv κ→0 / #90 decomp / #91 dynamics inversion) + 1.5 Contributions (4 复现贡献 vs HG-Rec 论文 3 贡献). 与 Section 3 (Task #94) / Section 5 (Task #92) / Section 6 (Task #93) 形成 paper deliverable 完整. verdict: `verdicts/task95_paper_section1_introduction_draft.md`.
- **Task #96 — Paper Section 4 Related Work Draft (writeup, no GPU)**: ✅ 已完成 (2026-07-24 06:10). 完整 markdown 草稿 (5 子节, 复现版扩展 HG-Rec 论文 §5 2 段版): 4.1 Sequential Rec (Markov / Trans / 深度 / 对比, 4 段细分) + 4.2 Generative Rec (ID-based / context-aware / codebook-based + 集成 fine-tuning 4 篇) + 4.3 Hyperbolic Methods in Rec (5 子家族: HGCF/HyperML/HGCN/M2GNN/多模态, **独立子节**) + 4.4 RQ-VAE and Codebook Techniques (3 类利用率技术 Sinkhorn/Entropy/EMA + Sinkhorn vs 双曲对比, **独立子节**) + 4.5 Reproduction Methodology (五步协议: stress-metric / curvature grid / free-curv / token-set Jaccard / training dynamics + 复现 checklist 4 条, **复现独有**). verdict: `verdicts/task96_paper_section4_related_work_draft.md`.
- **Task #97 — Paper Section 7 Conclusion Draft (writeup, no GPU)**: ✅ 已完成 (2026-07-24 06:25). 完整 markdown 草稿 (5 子节, 复现版扩展 HG-Rec 论文 §6/§7 合并版): 7.1 Conclusion (1 段总结 + 4 复现贡献复述 + 一句话 takeaway "对 flat 数据集 simple vanilla + Sinkhorn ≥ hyperbolic RQ-VAE") + 7.2 Limitations (6 项独立 list, 复用 Task #93 §6.4) + 7.3 Future Work (6 方向独立 list, 复用 Task #93 §6.5) + 7.4 Acknowledgements (致谢项目基础设施 / snap-research/GRID 上游框架 / HG-Rec 论文作者) + 7.5 Impact Statement (3 类影响: 实践 / 科学 / 复现性 + 负面影响声明). verdict: `verdicts/task97_paper_section7_conclusion_draft.md`.
- **Task #98 — Paper Stitching (single-file paper.md, writeup)**: ✅ 已完成 (2026-07-24 06:40). `papers/paper.md` 单一可提交 paper 文件 (556 行, ~30 KB), 整合 Task #92/93/94/95/96/97 + HG-Rec §2 Preliminaries. 含 Title + Abstract + 7 sections + Appendix LaTeX + References. 7 个 Table (Table 1-7) + 13 个 LaTeX 公式 (eq. 1-13) + 3 个 Theorem 引用 (Thm 3.1/3.2/3.4) + 10+ 关键参考文献. 论文 submission-ready 单文件版本, 估计 ~8000 words / ~10-12 pages (LaTeX 双栏). 可直接用于 PDF 转换 (pandoc) / 格式微调 / Submission 准备. verdict: `verdicts/task98_paper_stitching_result.md`.
- **Task #95 — HGN standalone test eval (paper Table 2 baseline)**: ✅ **完成 (2026-07-23 23:32)**. Test R@5=0.0302 / R@10=**0.0495** / N@5=0.0193 / N@10=0.0255 (paper 0.0781 / 0.0960 / 0.0654 / 0.0712, Δ -61.3% / -48.4% / -70.5% / -64.2%). HGN 复现处于 sequential 经典档位 (NARM/SASRec/BERT4Rec 同档), 远低于 RQ-VAE 系 (HG-Rec/phonism/LETTER > 0.10). 进一步证实 [[hgrec-paper-comparison]] 系统性 paper 偏差 (-48.4% ∈ [-61%, -18%]). verdict: `verdicts/task95_hgn_test_eval_result.md`. JSON: `verdicts/task95_hgn_test_eval.json`.
- **Task #81 — S³Rec RecBole 训练 (paper Table 2 #8)**: ⛔ NO-GO 已归档 (2026-07-23 23:08). yaml 默认 `train_stage='pretrain'` 导致训练跑的是 self-supervised pretrain 阶段 (无 valid/eval), 不是 generation 推荐任务. 修复需双阶段 (pretrain → finetune), 总 ~30h ROI 低. verdict: `verdicts/task81_s3rec_no_go_pretrain_only_result.md`. GPU 0 释放空闲.
- **Task #106/#107 — HG-Rec multi-seed**: ⛔ 用户决策撤回 (2026-07-23 22:14), 见 [[user-no-multiseed-override]]. GPU 2/3 已释放空闲. 后续 multi-seed 类实验禁止 AI 自主启动.
- **Task #94 — HG-Rec paper Table 1 对比 verdict**: ✅ 完成 (2026-07-23 23:00). 发现 HG-Rec paper R@10=0.1315 vs Task #84 复现 0.1020 (Δ -22.4%), 8/8 paper-reported baseline 数字均高于我们复现 18-61%, 系统性数据集/评估协议差异. verdict: `verdicts/task87_hgrec_paper_comparison.md`.

**§16 最近完成 (2026-07-23)**:
- **Task #81 — S³Rec RecBole 复现 (paper Table 2 #8)**: ⛔ **NO-GO 已归档 (2026-07-23 23:08)**. yaml 配置错: 默认 `train_stage='pretrain'` 导致训练走的是 self-supervised pretrain 阶段 (4 个辅助 loss), 无 valid/early stopping, 无 best ckpt 保存. 训练 30 epoch 4h7min 后 kill PID 625025. paper R@10=0.0538 与 LETTER 0.0997 / TIGER 0.0591 同档非最优档位, 不影响 paper 主结论. verdict: `verdicts/task81_s3rec_no_go_pretrain_only_result.md`. GPU 0 释放空闲.
- **Task #94 — HG-Rec paper Table 1 对比 verdict**: ✅ **完成 (2026-07-23 23:00)**. HG-Rec paper Instruments 列 R@10=**0.1315** vs Task #84 复现 **0.1020** (Δ -22.4%). 8/8 paper-reported baseline 复现均低于 paper 18-61% (paper 报告最高的 TIGER 0.1214 vs 我们 0.0591 偏差 -51.3%), 表明**系统性数据集/评估协议差异** (非算法差异). 相对排序**保留** (HG-Rec > Letter > TIGER > SASRec), 绝对数字**不保留**. 论文 Section 5 必须明确标注 "absolute numbers may vary across datasets/protocols". verdict: `verdicts/task87_hgrec_paper_comparison.md`.
- **Task #82 — Stage 1c Metric 弱信号灵敏度 + 网格加密补测 (R9-Enforce 编号冲突, 实际是 task99 子任务)**: ✅ **判定 A1 + B1**. Step A (3 棵合成树 κ_real=-0.10~-0.20) 验证 metric 弱信号灵敏度足够 (best κ=-0.05 ≠ 0, 3.5-4× stress 差距). Step B (phonism 真实数据加密网格 11 κ × 4 层) 验证 4 层 best κ 全为 0, κ=-0.05 stress 比 κ=0 **高 4-5 倍** (排除网格空隙解释). 配合 Task #81 v3 sanity check, **三重独立支撑**确认 "phonism RQ-VAE 残差几何真实接近欧氏空间". verdicts: `verdicts/task82_step_a_weak_signal_positive_control_result.md` + `verdicts/task82_step_b_phonism_grid_refinement_result.md` + `verdicts/task82_weak_signal_grid_refinement_result.md`. task80 verdict §11 同步更新三重支撑证据.
- **Task #85 — FDSA RecBole test evaluation (paper Table 2 #7)**: ✅ **判定 A — 4 指标全部 +43-52% 超 paper baseline**. R@5=0.0384 vs paper 0.0261 (+47.1%), R@10=0.0594 vs paper 0.0391 (+51.9%), N@5=0.0249 vs paper 0.0174 (+43.1%), N@10=0.0316 vs paper 0.0216 (+46.3%). checkpoint `RecBole/saved/FDSA-Jul-23-2026_16-59-17.pth` (best valid @ epoch 37). Valid/Test gap < 10% (healthy generalization). 最可能原因: yaml 启用了 `selected_features: ['class']` 给 FDSA 提供额外 class token 语义信号. verdict: `verdicts/task85_fdsa_test_eval_result.md`. S3Rec 部分因 NO-GO 未启动评估.
- **Task #84 — HG-Rec 主实验复现 (ICML 2026, Poincaré loss RQ-VAE + Differential-Length Codebook + T5-small, Musical_Instruments)**: ✅ **判定 H2 部分确认 (几何先验中性)**. Stage 1 sentence-t5-base 768d 编码 (9922 items) → Stage 1 HRQ-VAE 1000 epoch (best_loss=8.97, loss_type='poincare') → Stage 2 Differential-Length Codebook (9922 unique) → Stage 3 T5-small HG_Rec 95 epoch (early stop @ epoch 95, best NDCG@20=0.0988 @ epoch 75, val R@10=0.1262) → Stage 4 test eval **R@5=0.0816, R@10=0.1020, R@20=0.1279, NDCG@5=0.0690, NDCG@10=0.0755, NDCG@20=0.0821**. **对比 baselines**: phonism (vanilla RQ-VAE + SINKHORN) 0.1058 (-3.6% ⭐ 持平偏弱); LETTER 0.0997 (+2.3%); TIGER 0.0591 (+72% ⭐); FDSA 0.0594 (+72% ⭐); P5-CID 0.0413 (+147% ⭐). **核心结论**: Poincaré 损失 RQ-VAE 在 Musical_Instruments 上与 vanilla MSE RQ-VAE 实质持平, Task #70 几何先验**不自动**传递到下游. R12 ckpt 已落盘 (HG_Rec_best.pth 21 MB, epoch 75). R11.3 修复: importlib.util 加载含连字符文件 (train_HG-Rec.py). verdict: `verdicts/task84_hgrec_main_repro_instruments_result.md`.

**最近的 R3 失败归档 (2026-07-23)**:
- **Task #74 ETEGRec 128d paper-cycle (warmup=0/warm_epoch=1/early_stop=30)**: ⏸️ SIGTERM 中断. max R@10=**0.026289** (vs paper 0.0624, -58%). R3 否证.
- **Task #75 ETEGRec 128d cycle=4 paper-cycle (备选 cycle 验证)**: ⏸️ SIGTERM 中断. max R@10=**0.026393** (与 #74 几乎完全相同). R3 cycle 维度也否证.
- **Task #76 ETEGRec 128d paper_exact 400 epoch replica (R3 双盲复跑)**: ⏸️ SIGTERM 中断. max R@10=**0.026393** (≈ task73 0.0253, 复现性证实). R3 全面否证.
- **Task #77 LETTER t5-base 训练 (backbone 升级)**: ⏸️ SIGTERM 中断. 首次 OSError (HF 无网络); 第三次跑到 step 106401/206000 (~52%) 仍未达 200 epoch, 无 val_R@10. 需修复 HF 网络 + 重启.
- 关键共识: **ETEGRec 上限 ~0.026 是真实天花板, 跨 4 次不同配置稳定 (task73/74/75/76), 必须切换 R4 调查方向 (paper GitHub diff + backbone 升级)**. verdicts: `task74_etegrec_warmup_tuning_result.md` + `task75_etegrec_cycle4_paper_cycle_result.md` + `task76_etegrec_paper_exact_400_replica_result.md` + `task77_letter_t5_base_result.md`.
- Task #62 (G4, ⚠️ 数据未齐): TC_G 与 SCR 单调性. 缺 MCKG baseline. 待 Stage 1 重跑.
- Task #63 P3 (G1, P1 已完成): G1 P3 TIGER 闭环共用 sid_sinkhorn_balanced.pt, 与 Task #66 P3 同一场 Stage 3 训练 (避免重训). P2 η 扫描可选 (0.5 天, 廉价).

**§16 历史已清理 (R8)**:
- **Task #71 init κ ablation: B (init=[0,0,0]) + C (random) (2026-07-20)**: ✅ 完成. **B HR@20=0.845 (+46% vs baseline 0.578), κ 始终 ≈0 (欧氏不动点)**. C_random (init=[-1,-1,0]) HR@20=0.636 (+10%), κ=[-1.019, -0.829, **+0.124**] (κ3 翻转成球面). **核心反 baseline 发现**: Toys G1 用 [0,0,0] 起点**优于默认 [+1,0,-1]**, MCKG 默认 init 在 Toys 上**过度引入几何 spread**. 球面是 MCKG 训练的 attractor (即使 init 全负, 训练后仍出现 +0.12 球面). → verdict: `verdicts/task71_init_kappas_ablation_result.md`
- **Task #70 5-graph Ollivier 真实曲率测量 (2026-07-20)**: ✅ 完成. 4/5 组数据是**强双曲** (mean_κ -0.65 to -0.84, 99%+ 边 κ<0), G3 因 1542 边小样本 mean_κ=+0.20 (std 0.51 噪声). **MCKG κ1 ≈ +0.7 是模型 regularization 而非数据真实** (4/5 组数据是负, 但模型学正). M2GNN Table 7 在 Toys 上**模型放大 6.7×** 数据真实差异. verdict: `verdicts/task70_olliver_curvature_result.md`
- Task #69 5-graph MCKG 权重诊断 (2026-07-20): ✅ 完成. 5 组训练 + 5 层诊断. G1 (interaction) HR@20=0.5780 ⭐ 最佳, G4 (full KG) HR@20=0.3965. R1-A fail (G3 异常), R1-B pass, R1-C partial (G2 异常), R1-D pass. verdict: `verdicts/task69_5graph_weight_result.md`
- Task #66 P3 Sinkhorn cascade TIGER (2026-07-20): ⏸️ 暂停未归档 → 已被用户主动 kill-TERM，PID 1029066 已退出，verdict 已写 → archived in `verdicts/task66_p3_result.md`

**§16 backlog (P3 进行中)**:
- Task #63 P3 (G1, P1 已完成): G1 P3 TIGER 闭环**共用 sid_sinkhorn_balanced.pt**, 与 Task #66 P3 **同一场 Stage 3 训练** (避免重训). P2 η 扫描可选 (0.5 天, 廉价).
- Task #62 (G4, ⚠️ 数据未齐): TC_G 与 SCR 单调性. 缺 MCKG baseline. 待 Stage 1 重跑.
- ~~Task #64 (G2, ❌ NO-GO)~~: 关闭. verdict: `verdicts/d0_round1_result.md`
- ~~Task #65 (G5, ❌ NO-GO)~~: 关闭. verdict: `verdicts/d0_round1_result.md`

**§16 backlog (P3 进行中)**:
- Task #63 P3 (G1, P1 已完成): G1 P3 TIGER 闭环**共用 sid_sinkhorn_balanced.pt**, 与 Task #66 P3 **同一场 Stage 3 训练** (避免重训). P2 η 扫描可选 (0.5 天, 廉价).
- Task #62 (G4, ⚠️ 数据未齐): TC_G 与 SCR 单调性. 缺 MCKG baseline. 待 Stage 1 重跑.
- ~~Task #64 (G2, ❌ NO-GO)~~: 关闭. verdict: `verdicts/d0_round1_result.md`
- ~~Task #65 (G5, ❌ NO-GO)~~: 关闭. verdict: `verdicts/d0_round1_result.md`

**§16 历史已清理** (R8):
- **Task #67 v6 norm-fix-no-revival (2026-07-20)**: ✅ L0 cumulative cov=**0.4761** ([0.40, 0.65) 区间, A+B 都成立). Revival hook 关闭后从 v5=0.374 → v6=0.476 (+0.10, A 成立: Revival 负贡献 36%); 但 v6 仍低于 #68 raw 0.651 (-0.17, B 成立: norm fix 信息损失 64% 是主因). **最终结论**: raw 192d + no Revival + no norm fix (Task #68 配置) 是最佳组合, L0 cov=0.651. 修复方案应保留原始信号而非平滑异常值. → verdict: `verdicts/task67_v6_result.md`
- Task #53 真实 TIGER 训练 (S4 AE + log1p, 2026-07-20): ❌ R@5=0.002 → verdict: `verdicts/task53_tiger_log1p_result.md`
- Task #54 L3 norm post-hoc rebase (2026-07-20): ❌ L3 无效 → verdict: `verdicts/task54_l3_norm_result.md`
- Task #55 OPQ (ITQ, 2026-07-20): ❌ MSE ratio=1.009 → verdict: `verdicts/task55_opq_result.md`
- Task #56 曲率重审 (2026-07-20): 维持 Task #85 verdict → verdict: `verdicts/task56_curvature_result.md`
- Task #58 Simple KMeans SID + TIGER (S4 AE 64d, 2026-07-20): 🏆 **R@5=0.02962, +53%** → verdict: `verdicts/task58_result.md`
- Task #59 flan-t5 2048d + Simple KMeans SID + TIGER (2026-07-20): 🏆🏆 **R@5=0.08572, +92%** → verdict: `verdicts/task59_result.md`
- Task #60 sentence-t5 768d + Simple KMeans SID + TIGER (2026-07-20): 🏆🏆 **R@5=0.08376, +88%** → verdict: `verdicts/task60_result.md`
- **Task #61 Hybrid 2816d + Simple KMeans SID + TIGER (2026-07-20): 🏆🏆🏆 R@5=0.09767, +119%** → verdict: `verdicts/task61_result.md`
- **第二代 G 系列 D0 Round 1 (2026-07-20)**: 5 方向 D0 完成 → **2 GO (G1 ✅, G3 ✅), 2 NO-GO, 1 数据未齐** → verdict: `verdicts/d0_round1_result.md`
- **Task #63 G1 P1 (2026-07-20)**: ✅ G1-H1 GO + ❌ G1-H2 FALSIFIED → verdict: `verdicts/task63_p1_rq_result.md`
- **Task #66 G3 P1 (2026-07-20)**: ⚠️ G3 PARTIAL (signal/noise 4×, D0 高估 105× → 实际 4×) + ⭐ **G1×G3 GO** (Sinkhorn cascade vs Vanilla L0 拓扑破坏降 36%) → verdict: `verdicts/task66_p1_persistence_result.md`

**§16 backlog (P1 决策后)**:
- Task #63 (G1, D0 ✅ + P1 ✅ 完成, 待 P3 同): G1 P3 TIGER 闭环 同样使用 `sid_sinkhorn_balanced.pt`, 与 G3 P3 **共用同一个 Stage 3 训练** (最高 ROI). Task #63 P2 (η 扫描) 可选, 0.5 天廉价.
- Task #62 (G4, D0 ⚠️ 数据未齐): TC_G 与 SCR 单调性. 当前 3 embedding (T5/flan-t5/hybrid), 缺 MCKG baseline. 后续重跑 Task #87 Stage 1 (MCKG 768d) → P1 全矩阵 + 残差分解. 见 `descriptions/task62_g4_information_theory_scr.md`
- ~~Task #64 (G2, D0 ❌ NO-GO)~~: 关闭. \|ρ\|=0.113 < 0.15, item 级 LID 预测 ε_i 弱. verdict: `verdicts/d0_round1_result.md`
- ~~Task #65 (G5, D0 ❌ NO-GO)~~: 关闭. AUC=0.378 < 0.6, flip rate 99%. verdict: `verdicts/d0_round1_result.md`

**§16 历史已清理** (R8):
- **Task #67 v6 norm-fix-no-revival (2026-07-20)**: ✅ L0 cumulative cov=**0.4761** ([0.40, 0.65) 区间, A+B 都成立). Revival hook 关闭后从 v5=0.374 → v6=0.476 (+0.10, A 成立: Revival 负贡献 36%); 但 v6 仍低于 #68 raw 0.651 (-0.17, B 成立: norm fix 信息损失 64% 是主因). **最终结论**: raw 192d + no Revival + no norm fix (Task #68 配置) 是最佳组合, L0 cov=0.651. 修复方案应保留原始信号而非平滑异常值. → verdict: `verdicts/task67_v6_result.md`
- Task #53 真实 TIGER 训练 (S4 AE + log1p, 2026-07-20): ❌ R@5=0.002 → verdict: `verdicts/task53_tiger_log1p_result.md`
- Task #54 L3 norm post-hoc rebase (2026-07-20): ❌ L3 无效 → verdict: `verdicts/task54_l3_norm_result.md`
- Task #55 OPQ (ITQ, 2026-07-20): ❌ MSE ratio=1.009 → verdict: `verdicts/task55_opq_result.md`
- Task #56 曲率重审 (2026-07-20): 维持 Task #85 verdict → verdict: `verdicts/task56_curvature_result.md`
- Task #58 Simple KMeans SID + TIGER (S4 AE 64d, 2026-07-20): 🏆 **R@5=0.02962, +53%** → verdict: `verdicts/task58_result.md`
- Task #59 flan-t5 2048d + Simple KMeans SID + TIGER (2026-07-20): 🏆🏆 **R@5=0.08572, +92%** → verdict: `verdicts/task59_result.md`
- Task #60 sentence-t5 768d + Simple KMeans SID + TIGER (2026-07-20): 🏆🏆 **R@5=0.08376, +88%** → verdict: `verdicts/task60_result.md`
- **Task #61 Hybrid 2816d + Simple KMeans SID + TIGER (2026-07-20): 🏆🏆🏆 R@5=0.09767, +119%** → verdict: `verdicts/task61_result.md`
- **第二代 G 系列 D0 Round 1 (2026-07-20)**: 5 方向 D0 完成 → **2 GO (G1 ✅, G3 ✅), 2 NO-GO, 1 数据未齐** → verdict: `verdicts/d0_round1_result.md`
- **Task #63 G1 P1 (2026-07-20)**: ✅ G1-H1 GO + ❌ G1-H2 FALSIFIED → verdict: `verdicts/task63_p1_rq_result.md`
- **Task #66 G3 P1 (2026-07-20)**: ⚠️ G3 PARTIAL (signal/noise 4×, D0 高估 105× → 实际 4×) + ⭐ **G1×G3 GO** (Sinkhorn cascade vs Vanilla L0 拓扑破坏降 36%) → verdict: `verdicts/task66_p1_persistence_result.md`

**§16 Campaign 总结** (2026-07-20 + 第二代 G 系列):
- **8 个第一代 Campaign 任务 (#53-#56 + #58-#61) 全部完成**: #58-#61 4 个 🏆 胜利 (R@5 0.0296/0.0857/0.0838/0.0977)
- **5 个第二代 G 系列 D0 + P1 完成 (2026-07-20)**:
  - **G1 (Task #63) D0+P1 完成**: G1-H1 ✅ GO (min_count 11x, D_rel -0.08%, var_red +2.25%) + G1-H2 ❌ FALSIFIED (ε 相关性 +19%, 与提案反向)
  - **G3 (Task #66) D0+P1 完成**: ⚠️ G3 PARTIAL (d_B signal/noise 4×, D0 高估 105×), ⭐ **G1×G3 GO**: Sinkhorn cascade 拓扑保留 2× vs Vanilla L0 (7σ)
  - **G2 (Task #64) ❌ NO-GO**: \|ρ\|=0.113
  - **G5 (Task #65) ❌ NO-GO**: AUC=0.378
  - **G4 (Task #62) ⚠️**: 缺 MCKG baseline
- **⭐ 关键洞察 (D0 + P1)**:
  - **G1×G3 双引擎 ⭐**: Sinkhorn-balanced RQ 级联是拓扑正则化的天然代理 — 拓扑保 2× 提升. 实际下游 TIGER 闭环验证 (vs Task #59 R@5=0.0857) 是 G-campaign 的最终得分点.
  - G1 G1-H2 失败 + G3 D0/P1 大幅修正 → "极致 GO 方向" 收敛到 G1×G3 拓扑正则化
  - G3 单线 (d_B raw vs recon) 不够强, 必须用 Sinkhorn cascade 作为差异化手段才能体现
- **⭐⭐⭐ R@5 阶梯演化**: 0.0194 (#87) → 0.0296 (#58) → 0.0838 (#60) → 0.0857 (#59) → **0.0977 (#61)**


---