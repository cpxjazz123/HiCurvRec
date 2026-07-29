# Task #293 / Issue #23 — per-layer per-epoch c_k range curriculum (NCV)

**任务编号**: 293
**对应 Issue**: GitHub #23 (per-layer κ curriculum, 2026-07-29 owner 创建)
**目的**: 解锁 K=64 L0 hyperbolic codebook 90% utilization cap. 通过 per-layer per-epoch c_k range 时间函数 (而不是单值 c_k range) 让三层各自跨过 90% utilization, 并最终在 Stage 4 R@10 > 0.1020 击败 HG-Rec baseline.

---

## 1. 来源 (Issue #23 §依据)

承接 #242 (Issue #11 FULL NO-GO) / #235 (Issue #9 FULL NO-GO) / #275 #276 (A2 curriculum plateau 89.1% + Stage 4 R@10=0.0985 NO-GO).

之前所有 per-layer κ 失败 (task220 全层 U(0.5,5) L0=20.31% / task222 / task231 全层 U(1,5) L0=82.68% OPEN 但 Stage 1 坍缩 / task242 Arm A 逐层 U(1,5)+U(0.5,20) L0=23.44% / task242 Arm A+ L0=3.12%) 都是 **单一 c_k range 钉死在 Stage 1 全程**, 从未尝试让 c_k range 随 epoch 演化. task275 A2 curriculum 在 β 与 recon_loss 上做 curriculum, 但 c_k range 在训练全程保持 U(0.5,5) 不变. 这是 c_k curriculum 与 β-curriculum 的 **正交** 改进空间.

---

## 2. 假设 (Issue #23 §假设)

- **H1** (非平稳 c_k range 解锁 L0 cap): K=64 L0 在单值 c_k range 下 utilization 被钉在 82-89% 是 **range 单值** 的副作用. 让 c_k range 随 epoch 演化 (早期 U(0.5, 20) → 中期收缩 U(1, 5) → 后期 U(2, 8)), L0 utilization 能跨过 90%.
- **H2** (per-layer 解耦必要性): 三层 c_k curriculum 必须 **独立调度** (L0/L1/L2 各自的 epoch 演化曲线). 不能用同一时间函数.
- **H3** (plateau 不是 c_k 的终点): task275 A2 plateau 89.1% ×3 续训是 "β-curriculum 路径只能推到 89.1%", 本 issue 用 c_k curriculum (与 β curriculum 正交) 在 task275 ckpt 基础上继续推进.

---

## 3. 受控因素 (不改动)

- `num_emb_list=[64,128,256]`, `e_dim=32`, `seed=42`
- Musical_Instruments 5-core (9922 items / 24772 test examples)
- T5-mini 9.18M / 200 epoch / early_stop=20
- utilization 口径: Issue #18 Gate 1 落地的 Stage 1 直接 argmin
- collision 口径: Issue #10 Gate 0 锁定的 `collision_rate = 1 - unique_count / total_items`
- 继承 task275 A2 final ckpt 作为 Stage 1 warm-start (已 plateau 89.1% ×3 续训)

---

## 4. 阶段闸门 (每一道都是硬停止, 不是参考通过线)

### Gate 0 — Phase 0 c_k range 时间函数组合性 (零 GPU)

冻结 task275 final ckpt (products/task275/A2_extend_ep50/.../best_loss_model.pth). 在 Phase 0 forward-pass 上:

- 给 L0/L1/L2 各自采样 3 段 epoch schedule (每段 c_k range 不同), 共 3×3 = 9 种 schedule 组合 × 3 seeds × 3 layers = 81 个 agreement 测量
- 三段 schedule 模板:
  - Schedule A (早期宽 → 中期标准 → 后期密): U(0.5, 20) → U(1, 5) → U(2, 8)
  - Schedule B (全程宽): U(0.5, 20) 全程 (对照组)
  - Schedule C (全程窄): U(1, 5) 全程 (对照组, 跟 task275 一致)
- 测量每个 (schedule, seed, layer) 下的 agreement (per-codeword κ argmin vs Euclidean argmin)
- 检查三层 agreement 是否全在 60-90% OPEN 带内

**通过条件**: 81 个组合中有 ≥ 60% 三层全 OPEN (差 ≤ ±3pp 容忍度, 复用 task241 ±3pp 容忍).

**硬停止**: 若 < 60% 组合通过 → **STOP 并在 verdict 记录哪种 schedule 不通过**. **不得为补 Gate 0 通过而进入 Stage 1 训练**. Gate 0 失败意味着 c_k 时间函数在 forward-pass 都不兼容, Stage 1 训练必然失败.

### Gate 1 — Stage 1 三段 c_k curriculum 训练 (GPU, 仅在 Gate 0 通过后)

取 Gate 0 中表现最好的 3 段 schedule, 在 task275 ckpt 上做 warm-start Stage 1 训练:

- 三层独立 schedule: L0/L1/L2 各自的 epoch 边界不同
- 总 epoch 数: 30 (warm-start 跳过前 30 epoch, 避免 task275 已跑过的部分)
- 每 5 epoch 评估 per-layer utilization + collision
- dead_revive = off (task242 Arm A+ 已经证明 dead_revive 与窄 c_k range 互不相容)

**通过条件 (四条同时满足)**:
- (a) L0 utilization ≥ 90% at any evaluation step ≥ ep15
- (b) L1 utilization ≥ 90% at any evaluation step ≥ ep15
- (c) L2 utilization ≥ 90% at any evaluation step ≥ ep15
- (d) collision_rate ≤ 0.37 (task222 best known) at any evaluation step ≥ ep15

**硬停止**: (a)(b)(c)(d) 任一不满足 → **STOP, 不要进入 Gate 2**. **不得为补 Gate 1 通过而进入 Stage 2 推断 / Stage 3 T5 训练 / Stage 4 评估**. Gate 1 失败 = c_k curriculum 在 Stage 1 训练动力学下不能解锁 L0 cap = 本机制 NO-GO, 关闭本 issue.

### Gate 2 — Stage 2 Sinkhorn 推断 (cheap, 仅在 Gate 1 全条通过后)

用 Gate 1 最佳 ckpt 跑 Stage 2 Sinkhorn 推断, max_iters=5 (task260 已证 vanilla 5 iter 收敛), 记录 per-layer SID collision 与 3-digit / 4-digit SID 唯一性.

**通过条件**:
- 4-digit SID unique count ≥ 9500 (total=9922, 允许 4.2% 不唯一, 符合 task260 vanilla 4-digit dedup 0.0 collision 路径)
- per-layer utilization 与 Gate 1 终态偏差 ≤ 5pp (Sinkhorn 不破坏 stage 1 的 c_k curriculum 结果)

**硬停止**: unique count < 9500 → **STOP, 不要进入 Gate 3**. Stage 2 失败 = c_k curriculum SID 解码路径与 vanilla 不兼容 = NO-GO.

### Gate 3 — Stage 3 T5-mini 200 epoch 训练 + Stage 4 eval (仅在 Gate 0/1/2 全条通过后)

按 task278 batch Stage 4 eval recipe, 200 epoch 训练 + early_stop=20 + test eval R@10 / R@5 / R@20 / NDCG.

**通过条件 (一条)**:
- **Test R@10 > 0.1020** (HG-Rec baseline #84, 对照 task225/226/243/278 R@10 数字)

**硬停止**: R@10 ≤ 0.1020 → **STOP, 关闭本 issue, 写 verdict NO-GO**. **不得以 "再调一下 c_k range" 为由重跑 Gate 0-2**. Gate 3 失败 = c_k curriculum 路径不能贡献 R@10 增益. 本 issue 关闭, 承认 NO-GO.

**执行顺序**: Gate 0 → Gate 1 → Gate 2 → Gate 3. 任一 Gate 失败即在该 Gate 处写 verdict 结束, **不得跨 Gate 取数**.

---

## 5. 反证 / 压力测试 (Issue #23 §反证)

- **最强反证 (H1 不成立)**: 单一 c_k range 的 plateau 90% 是 K=64 hyperbolic codebook 的硬上限, 不是 range 单值副作用. 若此反证成立, 本 issue 在 Gate 1 必然失败. 本 issue 不假设此反证不成立.
- **H2 反证 (per-layer 解耦不需要)**: task275 A2 在 β-curriculum 上三层一起推进, plateau 89.1%/97%/98% — 表明 β-curriculum 已经让三层都 "动起来", 异构 c_k 时间函数未必比同构更优. 本 issue 不预设异构更优, Gate 0/Gate 1 都允许同构 schedule 也参与对比.
- **本自动跟进任务的强制力有限**: 该任务只读写 GitHub issue, **不运行也不监督实际训练作业**.
- **不申请跨过 Gate 0 的 Stage 1 训练**: Gate 1 训练需要 task275 ckpt warm-start.
- **不申请死码字复活**: task242 Arm A+ 已证 dead_revive 在窄 c_k range 下把 L0 砸到 3.12%.
- **不申请跨过 Stage 3 的 Stage 4 评估**: task276 Stage 4 端点 R@10=0.0985 已实证 "过 Stage 1 ≠ 过 Stage 4".
- **本 issue 不产出 R@10 增益的承诺**: 本 issue 只承诺 "如果 c_k curriculum 是正确的解锁机制, 它会被 Gate 0/1/2/3 实证". 若 Gate 3 不通过, 本 issue 关闭, 不允许 "继续 c_k curriculum 的下一变体".

---

## 6. 物理产物预期

- `descriptions/task293_issue23_per_layer_ck_curriculum.md` (本文件)
- `verdicts/task293_issue23_gate0_*_result.md` (Gate 0 通过后写)
- `verdicts/task293_issue23_result.md` (最终, 含 Gate 3 端点 R@10)
- `scripts/task293_issue23_gate0_phase0.py` (81 组合 agreement 测量)
- `scripts/task293_issue23_gate1_launcher.sh` (Stage 1 warm-start 训练)
- `products/task293/hrqvae_ck_curriculum/Jul-XX-XXXX/best_loss_model.pth` (Stage 1 ckpt)
- `HG-Rec/dataset/Instruments/Instruments_t5_ck_curriculum.npy` (Stage 2 SID)
- `products/task293/stage3_t5mini/Instruments/Jul-XX-XXXX/HG_Rec_best.pth` (Stage 3 ckpt)
- `products/task293/stage4_eval.json` (Stage 4 指标)

---

## 7. 依赖关系

- Issue #11 (FULL NO-GO) — 单一 c_k range 参数空间耗尽, 本 issue 的前提
- Issue #9 (FULL NO-GO) — hybrid per-layer assignment Stage 1 失败, 本 issue 的对照
- Issue #18 (completed) — utilization 口径绑定, Gate 1 (a)(b)(c) 的口径来源
- Issue #17 (completed) — Stage 1 per-layer utilization 打印, Gate 1 (a)(b)(c) 的量测基础
- Issue #19 (completed) — launcher 求值点, Gate 0/1 的 launcher 边界
- Issue #21 (closed completed 2026-07-29 by Task #290) — 防止 R11.4 自主决策跨过 §6.7.4
- Issue #20 (closed) — task270/271/275/276 chain, Gate 1 的 warm-start ckpt 来源
- `verdicts/task241_issue11_gate0_perlayer_ck_combo_pass.md` — Issue #11 Gate 0 PASS, Gate 0 的对照基线
- `verdicts/task242_issue11_gate1_perlayer_stage1_result.md` — Issue #11 Gate 1 FAIL, Gate 1 的反例
- `verdicts/task275_a2_a3_curriculum_parallel_result.md` — A2 plateau 89.1%, Gate 1 warm-start 起点
- `verdicts/task260_issue10_sinkhorn_strength_sweep_result.md` — Sinkhorn 5 iter 收敛, Gate 2 的 max_iters 选择
- `verdicts/task278_batch_stage4_eval_result.md` — task278 batch Stage 4 eval recipe, Gate 3 的对照
- `products/task275/A2_extend_ep50/Jul-29-2026_10-35-55_*/best_loss_model.pth` — Gate 0/1 冻结 ckpt

---

## 8. R11.3 自主决策点

- **c_k curriculum 启动 ROI**: Issue #23 body 已论证 H1/H2/H3 是 c_k curriculum 路径的剩余空间, 联立 task268 backlog 候选 3 + Issue #11/9 FULL NO-GO + task275 plateau 89.1%, 是当前 backlog 唯一未被验证的方向. ROI 中等 (Gate 0 零 GPU 验证, Gate 1~3 GPU 跑 3-5 hr 预期 NO-GO, 但可能 Stage 4 端点解锁).
- **同构 vs 异构 schedule**: 本 issue Gate 0 三组 schedule A/B/C 中, A 异构 + B/C 对照, 不会预判哪组最优.
- **不引入 EMA / FSQ / Restoration**: 本 issue 锁定 c_k curriculum 单一自变量, 不掺其他变体 (跟 task290/291/292 NO-GO 结论锁死一致).
- **Issue #21 governance**: 本 issue Gate 1 训练必须按 Issue #21 launcher header 约束 (`section_6_7_4_stop_loss_i=bound` + `auto_proceed_after_stage_2=false` + `precedent_override=forbidden` 默认), 不能用 task275 precedent 跨过 §6.7.4 stop-loss (i).
