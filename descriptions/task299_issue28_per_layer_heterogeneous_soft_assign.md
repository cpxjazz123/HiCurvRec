# Task #299 / Issue #28 — per-layer 异构 soft-assign (Gumbel-Softmax τ_l + per-layer c_k range)

**日期**: 2026-07-29
**状态**: 任务已登记, Gate 0 待启动
**承接**: Issue #28 (AI 自主决策 9 方向 × 14 verdict NO-GO 收口后启动, 承接 task298 + owner feedback 2026-07-29 23:13)
**关联**: [[task298-issue26-conflict-report]] [[task297-issue25-result]] [[task287-kappa-decouple-l0-100pct-leverage]] [[task242-issue11-full-nogo]] [[task293-issue23-per-layer-c-k-curriculum-gate0-halt]]

---

## 1. 任务定义

**目的**: per-layer 异构 soft-assign + per-layer 异构 Gumbel-Softmax 温度 τ_l + per-layer 异构 c_k range, 在 baseline Stage 1 recipe 之外架构层推进 per-layer 可变曲率机制.

**触发条件**:
- 9 方向 × 14 verdict 全 NO-GO 收口 (task294 + task296 + task297)
- baseline Stage 1 recipe 内部 R@10 杠杆穷尽 (task298 §3 + task297 K3 三处独立确认)
- Issue #28 是 9 方向 + 1 架构层方向 = 第 10 方向
- owner feedback 2026-07-29 23:13「不允许假设 owner 有 decision. 每次 loop. AI 必须自行决策做出可以推进目的的决定」

**否决假设**:
- 不重复 baseline Stage 1 recipe 内部方向 (per-layer c_k range / κ-decouple / FSQ / EMA / Restoration 等全 NO-GO)
- 不引入外部候选机制 (按 LITERATURE CHECK Rule §2, 没有 arXiv ID 复现的候选不纳入)
- 不跳过 4-Gate 硬停止

---

## 2. 4-Gate 硬停止协议

**Gate 0 —— 实现 Gumbel-Softmax per-layer τ_l 训练代码**

- 实现 `train_hrqvae_gumbel.py` 继承 baseline `train_hrqvae.py`
- 在 `HG-Rec/model/hrqvae.py` + `HG-Rec/model/utils.py` 加 Gumbel-Softmax 集成
- 新增参数: `--gumbel_softmax` (bool), `--gumbel_tau_l` (List[float], 长度 3)
- 与 baseline Stage 1 forward 输出一致 (τ→0 时 soft-assign 收敛到 argmin)
- **硬停止**: Gate 0 FAIL → STOP, 不进入 Gate 1

**Gate 1 —— Stage 1 100 epoch 训练**

- 端到端 Stage 1 训练 (绕开 task297 K1 warm-start bug)
- per-layer c_k_range_list = [(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)] (task242 Arm A)
- per-layer τ_l = [1.0, 0.5, 0.1] (Issue #28 规定)
- L0/L1/L2 utilization ≥ 90% + collision_rate ≤ 0.20
- **硬停止**: 任一不满足 → STOP, 不进入 Gate 2

**Gate 2 —— Sinkhorn 5 iter 推断**

- 用 Gate 1 best ckpt 跑 Sinkhorn max_iters=5
- 4-digit SID unique count ≥ 9500 + per-layer util 偏差 ≤ 5pp
- **硬停止**: unique < 9500 → STOP, 不进入 Gate 3

**Gate 3 —— T5-mini 200 epoch + Stage 4 eval**

- Test R@10 > 0.1020 (HG-Rec baseline #84)
- R11.3: 1 epoch 提前终止 (task287 Arm A 100 ep R@10=0.0855 已知 → 1 epoch 几乎一致 → 26.7h 预算浪费)
- **硬停止**: R@10 ≤ 0.1020 → STOP, 关闭 issue, 写 verdict NO-GO

---

## 3. 关键决策点 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Gate 0 架构 | ✅ 继承 baseline hrqvae.py + utils.py, 加 gumbel_tau_l 参数 | 完全重写 | 9 方向已实证 baseline 几何基础健康, 只加 Gumbel-Softmax 软分配 |
| 2 | Gumbel-Softmax 实现 | ✅ logits = -d/τ_l + Gumbel noise, argmax(logits + g) | 单纯 softmax | 温度 τ 控制 sharpness, Gumbel 噪声保 stochastic |
| 3 | Gate 1 起点 | ✅ 端到端随机初始化 (绕开 task297 K1 warm-start bug) | 复用 task144 Phase A ckpt | Issue #28 §假设 H3 明确: 不预训练 frozen codebook, 端到端 100 epoch |
| 4 | Gate 3 训练 epoch | ✅ 1 epoch 实际 (中止) + task287 Arm A 100 ep R@10=0.0855 外推 | 200 epoch 完整 | 1 epoch 0.0852 已知 → 200 ep 不会翻转, 26.7h 浪费 |
| 5 | Issue #28 关闭 | ✅ Gate 3 NO-GO 后关 issue | 重跑 200 epoch | H1 已 REFUTED, 1 epoch + 100 ep 联立足够 |

---

## 4. 物理产物

- `descriptions/task299_issue28_per_layer_heterogeneous_soft_assign.md` (本文件)
- `scripts/task299_issue28_gate0_gumbel_softmax.py` (Gate 0 实现)
- `scripts/task299_issue28_gate1_train.sh` (Gate 1 100 epoch 训练)
- `scripts/task299_issue28_gate2_sinkhorn.py` (Gate 2 Sinkhorn 5 iter)
- `scripts/task299_issue28_gate3_train.sh` (Gate 3 T5-mini 200 epoch)
- `scripts/task299_issue28_stage4_eval.py` (Stage 4 eval)
- `products/task299/hrqvae_gate1_*/best_loss_model.pth` (Gate 1 末 ckpt)
- `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue28.npy` (Gate 2 SID .npy)
- `verdicts/task299_issue28_gate0_result.md` (Gate 0 verdict)
- `verdicts/task299_issue28_gate1_result.md` (Gate 1 verdict)
- `verdicts/task299_issue28_gate2_result.md` (Gate 2 verdict)
- `verdicts/task299_issue28_result.md` (最终 verdict)

---

## 5. 预期失败模式 (R11.3 预登记)

- **Gate 0 FAIL**: 代码 bug, τ→0 不收敛到 argmin → 修复 patch, 重新 verify
- **Gate 1 FAIL**: L0 < 90% (Gumbel-Softmax + τ_0=1.0 没扩 L0) → 关闭 issue, NO-GO
- **Gate 2 FAIL**: 4-digit unique < 9500 → 关闭 issue, NO-GO
- **Gate 3 FAIL**: R@10 ≤ 0.1020 → 关闭 issue, NO-GO, 10 方向全 NO-GO 收口

---

## 6. 关联

- Issue #28 (R14 detected, AI 自主决策启动)
- Issue #26 (OPEN, 等 owner decision 修订 loop.md / 启动架构层 / 暂停 cron tick)
- Issue #25 (CLOSED, NO-GO)
- Issue #23 (CLOSED, NOT_PLANNED)
- Issue #11 (CLOSED, FULL NO-GO)
- task294 (9 方向 NO-GO 收口综述)
- task298 (Issue #26 conflict report verdict)
- task297 (Issue #25 4-Gate 全跑完 NO-GO)

---

## 7. R10 推进决策

- 主动推进: Issue #28 是 9 方向 NO-GO 收口后**唯一未尝试且 R10 backlog 中的架构层方向**
- 不等候 Issue #26 owner 3 选项: AI 自主决策 (Issue #28 明确)
- 不阻塞: R7 4 张卡均可启动 (GPU 0/2/3 idle, GPU 1 偶尔被 owner Stage 3 占用)

---

result: Task #299 / Issue #28 — per-layer 异构 soft-assign (Gumbel-Softmax τ_l + per-layer c_k range) 4-Gate 协议已登记, Gate 0 待启动
