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

**🟡 §16 当前活跃: 等用户最终决策 (Task #230/#231/#232 已完成 — 方向 H PCA 钉方向 + 方向 I c=10/100 + 方向 H+I 组合 也撞墙, 5-cond + collision trade-off 仍未突破; 跨 10 variants × 8-point w_angular sweep × κ-Stereographic × PCA 冻结 × c 放大 × H+I 组合穷尽). 决策方向: (a) 接受 v6 (5-cond PASS, collision 95.68%) 作为几何可解释性终点; (b) 攻命题前提 (方向 E κ-Stereographic / 方向 F per-codeword κ 67-76% OPEN / 方向 G Gromov 79.65% OPEN).**

> **🟢 Task #218 + #219 已完成 (verdicts 落盘 + memory 索引)**: 用户 2026-07-26 终极提议 (攻命题前提 b/d) Phase 0 判据检查两条都有层 OPEN (L0 用 Gromov 79.65% ✅, L1/L2 用 Per-Codeword κ 67-76% ✅). 已写完 4 份产物 (verdict + memory + paper §1.4-6.7.5 update). 0 卡消耗.
>
> **🟠 Task #226 + #227 + 后续 v9/v10/v11/v12 收口 (2026-07-27)**: M-arm product_manifold 架构在 tuple collision ≤ 12% AND 5-cond PASS 二元 trade-off 上**架构级 NO-GO**:
>   - Task #226 v6 5 条件全 PASS (cos_std=0.74, util≥0.97, agreement<0.90 ✓), 但 collision 95.68%
>   - Task #227 v6/v7/v8/v9/v10 撞墙 95-99%
>   - v11 (8D hyp) 起跑 5.45% collision **PASS** 但 cos_std=0.12 (FAIL), ep 19 5cond 全 PASS 但 collision 已涨到 59.99% (FAIL)
>   - v12 (16D hyp) 起跑 8.35% collision **PASS** 但 cos_std=0.049 (FAIL), 从未达到 5cond 全 PASS
>   - **没有任何 epoch 同时满足两个目标** (HRQVAE 直加载 calibration 验证, sweep 数值 = trainer ±0.03%)
>
> **GPU 状态**: 0-3 全部空闲 (4× L40S, 0-3% util, 0 MiB used, trainer 全部已 kill).
>
> **候选下一步 (R11.4 关键决策, 等用户决定)**:
>   - **方向 D (R11.3 自主推荐)** ⭐: 回退 vanilla 32D Poincaré (Task #84 baseline 路线), 已知 collision 9.07% 50 epoch, 1 天复现 Stage 1-4 验证 R@10=0.1020
>   - **方向 E**: κ-Stereographic (Berman-Metzler 2020) 替换 Poincaré 距离公式, 在 2D hyp 子空间增加角灵敏度, 3-5 天工程量
>   - **方向 F**: v11/v12 短训 (12 epoch 锁定, 不让 collision 爬升), 1 天 Stage 3 测 R@10 (期望低, 5cond FAIL)
>   - **方向 A**: 接受 5 条件 PASS 已达成 (Task #226) 作为几何可解释性研究终点, collision 降级为次要目标, 0 卡当天
>   - **方向 B**: 注册 M-arm Stage 2/3/4 队列 (即使 collision 高, 仍可走 RQ-VAE 路径测下游 R@10)
>
> **新增工程产物**:
>   - scripts/m_arm_step3_sweep_5cond.py (HRQVAE 直加载, collision 校准到 ±0.03%, 可复用)
>   - scripts/m_arm_step3_v11_angdim8.sh, scripts/m_arm_step3_v12_angdim16.sh (v11/v12 launchers)
>   - memory/m-arm-v8-collision-nogo.md 更新 v11/v12 epoch 全表 + Goldilocks 不存在结论

**📦 7 方向几何路线完整清单 (全部 NO-GO/NO-HOPE):**
> | 方向 | Task | 状态 | 根因 |
> |---|---|---|---|
> | exp(θ) 可学习 κ | #199/201/203 | ❌ | θ 全程未动 |
> | 双码本解耦 | #200/208 | ❌ | R@10=0.0915 |
> | path regularization | #209 | ❌ | dyn 1.27 距离饱和 |
> | 低维双曲 + 钉半径 | #211 | ❌ | R@10=0.0816, util 23% |
> | Two-stage decision | #212 | ❌ | 99% 一致率 |
> | Entailment Cones | #213 | ❌ | 锥 opening 数值病态 |
> | Latent Radius Live | #214 | ❌ | radius head 无信号 |

**📦 Task #215 + #216 产物:**
> - verdicts/task213_entailment_cones_phase0_result.md (158 行)
> - verdicts/task214_radius_live_phase0_result.md (106 行)
> - verdicts/task215_paper_section4_complete.md (103 行)
> - verdicts/index.md (117 行, 按主题分组 #193-#215)
> - papers/paper.md (590 → 615 行, +25 行 = §1.4×3 + §5.7.1×4 + §6.6×1)
> - 4 个 CPU 脚本 (/home/wlia0047/.claude/jobs/04ccf474/tmp/task{212,213_v2,214}_*.py)

**📦 paper.md 完整更新:**
> - §1.4: 7 → 11 evidence (加 finding 8/9/10/11)
> - §5.7.1: 7 → 11 evidence (加 evidence 8/9/10/11)
> - §6.1: "Seven" → "Eleven"
> - §6.6 (新): "Geometry Route Closure — 7 Directions All NO-GO/NO-HOPE"

**📦 关键证据 (Task #212):**
> - 双曲几何在 baseline 码本 top-k 候选内跟欧式 argmin 排序 **99% 一致** (L0 98.82%, L1 99.21%, L2 99.56%)
> - 任何 "欧式取候选 + 双曲重排" 都不改变最终选择 (单调变换, 不改变 argmin)
> - 这是 HG-Rec "包装失效" finding 的第二个独立证据 (Task #199 λ_κ≈2 + Task #212 一致率 99%)
> - **结论**: 方向二 NO-HOPE 收线; 方向一是希望所在 (绕开 argmin 死结, 用"包含"关系替代"最近"关系)
> - verdict: verdicts/task212_two_stage_criterion_result.md (147 行完整判据)
> - 产物: /home/wlia0047/.claude/jobs/04ccf474/tmp/task212_two_stage_decision_criterion.py (CPU 2min 重跑)

**📦 Task #211 已归档 (上一轮):**
> - 低维双曲 + 钉半径 4 stage 闭环 → ❌ NO-GO 收线. C1 R@10=**0.0816** (-20% vs baseline). Forward patch (HG-Rec/model/utils.py 5 处) + 单元测试 PASS.
> - 论文交付: 2×2 design space paper skeleton (verdicts/task30_2x2_design_space_paper_skeleton.md). 4 格失败模式 + 修复路径.
> - verdict: verdicts/task211_low_dim_pinned_radius_arch_infeasible.md

**已完成 / 归档:**
- **#208** 双码本几何解耦 (4 臂 B/C/D + E 欧式加权) | ⛔ **已过时** (用户 2026-07-26 新方案替代 → Task #209, 新增路径正则 + 5 臂 + 多种子 + 切片) | description: descriptions/task208_dual_codebook_geometry_decoupling.md. 继承: 双码本 + 显式 `r_target = ρ/2` 半径设定 + κ-Stereo 距离 (c=1.0) → Task #209.
- **#207** Euclidean vs Hyperbolic 全流水线对比 | ✅ COMPLETE | verdict: verdicts/task207_euclidean_vs_hyperbolic_result.md. 核心: Euc VQ 坍缩不可修复 (β/K0/Sinkhorn 全失败), hyp 稳定码本. Hyp test R@10=0.0914 (vs baseline 0.1020, △-10%), Euc SID 超 vocab 边界不可用. **论文基调结论**: 双曲几何是 VQ 稳定的必要条件.

**已完成 / 归档:**
- **#206n** Phase A: usage-target_r 半径语义项 (3 臂 w_rad 0/0.1/1.0, per-layer c=93/604/702, 500 epoch) | ❌ **全部 3 臂模式坍缩** (collision 93.6-99.9%) | verdict: verdicts/task206n_phaseA_mode_collapse_result.md. 根因: 强度对齐 c 过高 → λₖ 共形因子 L2 达 10⁵ → 梯度爆炸 → 全部码字退化到同一位置. w_rad=0 对照也坍缩, 说明不是 usage-target_r 问题而是曲率本身不可训练. **几何激活路线核心结论**: c ∈ [1, ~10] 安全, c ≥ 93 不可训练. Task #206 系列任务线终止.
- **#203** exp(θ) κ + scale normalization poincare commit/code | ❌ NO-GO | verdict: verdicts/task203_kappa_scale_norm_result.md.
- **#204** c=1 + weight×2.2 vs c=10 + weight=1.0 | 🟡 第三种结果 (Δ 0.86%) | verdict: verdicts/task204_quant_loss_weight_对照_result.md.
- **#201** exp(θ) κ with θ_init=log(10) | ❌ FAIL (θ 全程未动) | verdict: verdicts/task201_kappa_redo_result.md.
- **#200 dual_v5 Stage 3+4** | ❌ FAIL (R@10=0.0915) | verdicts/task200_dual_v5_stage3_4_result.md.
- **#199** exp(θ) κ 参数化 Stage 1 | ✅ 完成 | verdict: verdicts/task199_stage1_exp_theta_result.md.
- **#194** K0 容量扫描 | ✅ 完成 (用户决策: 不需要 Stage 3+4)

**队列 (待用户拍板):**
- **#196** Stage 1 软约束 γ 扫描 | ⛔ **已过时** (Task #206 线终止, 几何激活 c≥93 不可训练)
- **#197** Stage 2 双码本解耦 | ⛔ **已过时** (双码本 #200 已 FAIL)
- **#198 Stage 3 逐层可学习 κ** | ⛔ **已过时** (c≥10 退化, c=30+ 坍缩, #199+#201 链证伪)

**已完成 / 归档:**
- **#199** exp(θ) κ 参数化 Stage 1 (4 臂: B exp_global, C exp_per_layer, D c-扫描{1,10,30,100}) | ✅ 完成 (2026-07-26 02:56) | verdict: verdicts/task199_stage1_exp_theta_result.md. 关键: B/C 学到 c=0.7-1.0 (而非预测 30-550), D c=10 是 D 臂最佳 (8.26% < c=1 9.15%), D c=30 退化 (10.84%), D c=100 完全坍缩 (ep 14 29.67% → ep 164+ 99.99%). 用户"50× κ_max 修复 + c 进 [10,100] 健康窗口"假设**被实验数据明确证伪**: HG-Rec c=1.0 是次优但接近最优, c≤10 是健康区间.
- **#194** K0 容量扫描: Stage 1+2 已坐实 K0 controlling variable (collision 12.6→6.0% 单调↓). 用户 2026-07-26 00:14 决策: "不需要 #194 Stage 3+4". Stage 3 4 臂 (ep 90%+) + dispatcher 全部 kill. R12 best_ckpt 4 臂保留 (22 MB each).

**已完成 / 归档:**

| Task | 最终状态 | 关键数字 |
|------|---------|---------|
| **#181** Phase 0.6 官方对齐 | ✅ COMPLETE (full pipeline R@10=0.1057, +3.6% vs baseline) | verdicts/task181_phase0.6_result.md. Stage 3 early stop at epoch 91. best ckpt epoch 72 (val R@10=0.1262, N@20=0.1003). Test R@10=0.1057 vs HGRec baseline 0.1020 (+3.6%). 仍低于 paper 0.1315 (-19.6%). |
| **#182** 欧式 + loss×4 | ⛔ FAIL (Stage 1 complete, Best Collision 83.5%, final 99% mode collapse) | 欧式 MSE 无几何约束 → 码字坍缩. 不推进 Stage 2/3. |
| **#183 v1** 硬归一化 | ⛔ FAIL (epoch 54 coll 99.86%, fill=0.03) | 硬归一化移除了 argmin 径向信号 → 坍缩. pivot 到 v2 软正则化. |
| **#183 v2** 软正则化 | ⛔ FAIL (1000 epoch done, final collision=96.25%, fill=[0.70,0.85,0.93] ✓) | 软正则化只修复径向未修复角向坍缩 (encoder latent 方向未分化). → 登记 Task #184 product manifold. |
| **#184** Phase 1a product manifold | ⛔ FAIL (Stage 1 collision=95.48%, Stage 2 只有 50 unique 3-digit SIDs, Stage 3 CUDA assert 崩溃) | verdict: verdicts/task184_phase1a_result.md. Product manifold 假设"两轴独立可分" — 但 init 阶段两轴就耦合，复合 argmin 仍坍缩. 方向失败, 不推进 Stage 4. |
| **#185** disable early stop + resume 1000 epoch | ⏸️ STOPPED (epoch 8, val R@10=0.1062) | 用户叫停, 改用 fresh 0→1000 epoch 方案 (Task #186). description: descriptions/task185_disable_early_stop_resume.md. |
| **#186** fresh 1000 epoch (no resume, no early stop) | ⏸️ STOPPED (epoch 25, val NDCG@20=0.0887) | 用户 18:21 叫停, 不是失败. verdict: verdicts/task186_fresh_1000epoch_result.md. best_ckpt 已 R12 保存 (22 MB, epoch 24). GPU 0 全释放. |
| **#187** encoder 4 层 | ⏸️ STOPPED (epoch 6/200, best val NDCG@20=0.0630) | 用户 18:42 改方向到 Task #188 (paper Table 7 多 seed 复现). verdict 未写 (实验太短). description: descriptions/task187_encoder4_layers.md. |
| **#200 v3** 双码本 Phase 1 v3 | ⛔ FAIL (ep 14-176/1000 manual kill) | 用户 2026-07-26 5 点修正方案 (α_geo 0.1→1.0 + 三层 centering + emb_rec 自由). 隔离测试 v4 PASS (emb_geo grad ×10). 训练 ep 50+ (manual kill ep 176): train_loss 稳定 60-83 ✅ (vs v2 飞涨 15720), collision 95.85-99.56% (中位数 ~97%) ❌ (期望 30-50%). 修复路径走通 (no NaN), 但 collision 未改善, 等用户决策 A 接受/B 长训/C 调 β/D 启 Sinkhorn. verdict: verdicts/task200_phase1_v3_result.md. |
| **#200 dual_v5 Stage 3+4** | ❌ FAIL (Stage 3 silent death @ ep 93/200, Stage 4 test R@10=0.0915 < HG-Rec baseline 0.1020 -10.3%) | 用户 2026-07-26 选项 B: 验证 v5 collision 84% SID Stage 3+4 性能"持平". R12 best_ckpt 救场 22 MB. Stage 4 test R@10=0.0915, R@5=0.0756, R@20=0.1119, N@10=0.0697. 全面低于 baseline, 持平预测**失败**. 双码本 Phase 0+1 修复路径 (Phase 1 v5 Sinkhorn + 双码本解耦) 端到端验证**没救** baseline. verdict: verdicts/task200_dual_v5_stage3_4_result.md. |
| **#202** Sinkhorn-on Stage 3 K0=64 | ❌ NO-GO (val R@10=0.1065, vs #181 默认 SID 0.1057 持平) | 用户 2026-07-25 拍板"用 Sinkhorn 版 SID 重跑 Stage 3". Sinkhorn SID `_t5_rqvae_k064_sk0.003.npy`, T5-mini 9.18M, early_stop=20. early stop @ ep 117/200 (counter 20). best epoch (ep 113): val R@5=0.0909 R@10=0.1065 R@20=0.1223 N@5=0.0785 N@10=0.0836 N@20=**0.0876** (best). **用户预测 +1~3% 已 FAIL** (实际 +0.07% 持平). Sinkhorn 路径不进 paper recipe. verdict: verdicts/task202_sinkhorn_stage3_result.md. R12 best_ckpt 22 MB saved. |

