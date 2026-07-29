# Task #300 / Issue #29 — per-layer 异构 K_l + per-layer c_k range

**日期**: 2026-07-29
**状态**: Gate 0 PASS, Gate 1 启动
**承接**: Issue #29 (AI 自主决策 — 9 方向 × 14 verdict NO-GO 收口后第 11 方向 per-layer 异构 K_l)
**关联**: [[issue29-task-body]] [[task298-issue28-result]] [[task297-issue25-result]] [[task287-kappa-decouple-l0-100pct-leverage]]

---

## 1. 任务定义

**目的**: per-layer 异构 K_l=[128, 64, 32] + per-layer c_k range, 在 baseline Stage 1 recipe 之外架构层推进 per-layer 可变"曲率容量"机制.

**触发条件**:
- 9 方向 × 14 verdict 全 NO-GO 收口 (task294 + task296 + task297)
- baseline Stage 1 recipe 内部 R@10 杠杆穷尽
- Issue #29 是 9 方向 + 2 架构层方向 (#28 + #29) = 第 11 方向
- owner feedback 2026-07-29 23:13「不允许假设 owner 有 decision. 每次 loop. AI 必须自行决策做出可以推进目的的决定」

**否决假设**:
- 不重复 K-sweep 全层同 K (K=128/256 task287/task284 已 NO-GO)
- 不引入外部候选机制 (按 LITERATURE CHECK Rule §2, 没有 arXiv ID 复现的候选不纳入)
- 不跳过 4-Gate 硬停止

---

## 2. 4-Gate 硬停止协议

**Gate 0 —— 实现 per-layer 异构 K_l 训练代码**

- 实现 `train_hrqvae_perlayer_k.py` 继承 baseline `train_hrqvae.py` + 新 CLI `--num_emb_list` (覆盖 [64,128,256] 为 [128,64,32])
- 复用 baseline `--c_k_range_list` (per-layer c_k range 沿用 task242 Arm A)
- 回归测试: K_l=[64,128,256] 输入下 forward 与 baseline 完全一致 (Gate 0 验证 PASS)
- **硬停止**: Gate 0 FAIL → STOP, 不进入 Gate 1

**Gate 1 —— Stage 1 100 epoch 训练**

- 端到端 Stage 1 训练 (绕开 task297 K1 warm-start bug)
- per-layer K_l = [128, 64, 32] (L0 K 翻倍, L1/L2 K 减半)
- per-layer c_k_range_list = [(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)] (task242 Arm A)
- L0/L1/L2 utilization ≥ 90% + collision_rate ≤ 0.20
- **硬停止**: 任一不满足 → STOP, 不进入 Gate 2

**Gate 2 —— Sinkhorn 5 iter 推断**

- 用 Gate 1 best ckpt 跑 Sinkhorn max_iters=5
- 4-digit SID unique count ≥ 9500 + per-layer util 偏差 ≤ 5pp
- **硬停止**: unique < 9500 → STOP, 不进入 Gate 3

**Gate 3 —— T5-mini 200 epoch 训练 + Stage 4 评估**

- Stage 4 eval: Test R@10 > 0.1020
- **硬停止**: R@10 ≤ 0.1020 → STOP, 不允许"继续 K_l 的下一变体"

---

## 3. 关键决策点 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 不修改 HG-Rec/model/ | ✅ 复用 baseline CLI --num_emb_list | wrapper class monkey-patch | R11.4 critical decision, 不动上游 |
| 2 | per-layer K_l 默认 | ✅ [128, 64, 32] (Issue #29 默认) | [256, 128, 64] / [64, 32, 16] | Issue #29 body 明确 |
| 3 | per-layer c_k range | ✅ task242 Arm A [(1,5),(0.5,20),(0.5,20)] | 共享 c_k range | Issue #29 跟 task242 兼容 |
| 4 | 回归测试口径 | ✅ baseline vs K_l=[64,128,256] 完全一致 | forward 输出数值 diff < 1e-5 | Issue #29 body §Gate 0 |

---

## 4. 物理产物

- `scripts/task300_issue29_gate0_per_layer_k.py` (Gate 0 验证 + 回归测试 PASS)
- `scripts/task300_issue29_gate1_stage1_train.sh` (Gate 1 100 epoch 训练 launcher, GPU 0)
- `verdicts/task300_issue29_gate0_verify.json` (Gate 0 verify 机器可读结果)

---

## 5. Gate 0 验证结果 (2026-07-29)

| 验证项 | 实测 | 决策 |
|-------|------|------|
| 回归测试 baseline vs K_l=[64,128,256] | max \|diff\| = 0.00e+00 | ✅ PASS |
| Issue #29 K_l=[128,64,32] vs baseline | mean \|diff\| = 1.10e-03 | ✅ PASS (新设计, 非零) |
| Shape 一致性 | baseline [4,768] / Issue#29 [4,768] | ✅ PASS |

**Gate 0 通过决策**: ✅ 进入 Gate 1 (Stage 1 100 epoch GPU 0)

---

result: Task #300 / Issue #29 Gate 0 PASS. per-layer 异构 K_l=[128,64,32] + per-layer c_k range 上游 HRQVAE 已支持 (无需 patch). 回归测试 baseline vs K_l=[64,128,256] max diff = 0. Gate 1 Stage 1 100 epoch 训练 launcher 已就位 (GPU 0). 进入 Gate 1.
