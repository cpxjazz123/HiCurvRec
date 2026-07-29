# Task #282 / Task #270 A1 — Stage 1 欧氏 MSE + β=0 NO-GO 闭环 verdict

> **完成日期**: 2026-07-29
> **状态**: ❌ **A1 (loss_type=mse + β=0.0 纯欧氏 VQ-VAE) NO-GO 闭环**
> **触发层**: Stage 1 训练 ep30 触发 USAGE-KILL (utilization < 20%)
> **GPU 占用**: GPU 2 一次性 ~75 sec

---

## 1. 实验目的

R11.5 自主推进 task270 description (高 ROI backlog 候选 3, Issue #18 闭环后解锁):
- 验证假设: **β 是 L0 utilization ≥ 90% 的杠杆**——把 commit loss 关掉, 码字能自由散开
- 配方 A1: `--loss_type mse --beta 0.0` (= 纯欧氏 VQ-VAE without commit loss)

---

## 2. 实测 step2 monitor 序列 (Stage 1 argmin 口径 = Issue #18 锁定量)

| Epoch | L0 usage | L1 usage | L2 usage | collision | 备注 |
|------:|---------:|---------:|---------:|----------:|------|
| ep5   | **40.6% (26/64)** | 68.8% (88/128) | 72.3% (185/256) | 0.7852 | kmeans_init 初始扩散 (基础) |
| ep10  | 6.2% (4/64)       | 32.0% (41/128) | 41.4% (106/256)| 0.9763 | **ep5 → ep10 暴跌 −34.4pp** |
| ep15  | 1.6% (1/64)       | 5.5% (7/128)   | 12.5% (32/256) | 0.9956 | **完全坍缩到 1 个码字** |
| ep20  | 1.6% (1/64)       | 2.3% (3/128)   | 6.6% (17/256)  | 0.9980 | — |
| ep25  | 1.6% (1/64)       | 1.6% (2/128)   | 4.7% (12/256)  | 0.9988 | — |
| ep30  | 1.6% (1/64)       | 1.6% (2/128)   | 4.7% (12/256)  | 0.9988 | **USAGE-KILL 触发** |

---

## 3. 反证 task270 假设

| 假设 | 期望 | 实测 | 结论 |
|------|------|------|------|
| A1: 去掉 commit loss → 码字自由散开 → L0 ≥ 90% | L0 ≥ 90% at ep30 | L0=1.6% (1/64) at ep30 | **H1 FAIL** |
| A1 至少 L0 ≥ poincare+β=0.5 baseline 73.44% | L0 ≥ 73.44% | L0=1.6% | **H2 FAIL** |
| A1 ≥ task270 description 期望 | L0 涨到 ≥ 90% | L0 卡 1.6% 比 baseline 更糟 | ❌ |

**关键发现**:
1. kmeans_init (ep5) 给出 L0=40.6% baseline 扩散
2. **无 commit loss 时**, 5 epoch 内 encoder 已"自洽地把所有 embedding 压到最近的 1 个码字"
3. 重建 loss 完美卡 0.0021, **码字利用率**: 1 个码字 = 重建 9922 items → mode collapse
4. 这个 L0=1.6% 比 poincare+β=0.5 的 task253 73.44% (Issue #17 闭环) 还要差 −71.84pp

---

## 4. 与历史 baseline 对比

| 来源 | loss_type | β | L0 usage | L1 | L2 |
|------|-----------|--:|---------:|---:|---:|
| **Task #282 A1 (本任务)** | mse | 0.0 | **1.6%** | 1.6% | 4.7% |
| Task #253 (Issue #17) | poincare | 0.5 | 73.44% | 100% | 100% |
| Task #84 baseline (vanilla) | poincare | 0.5 | ~85-100% | high | high |
| Task #265 smoke run ep3 | poincare | 0.5 | 1.6% | 1.6% | 0.4% |
| Task #270 A1 v1 (旧 30 epoch) | (poincare 假) | 0.0 | 26.6% (ep30) | 23.4% | 5.9% |

**注意**: Task #270 A1 v1 (Jul-29-2026_10-11-33) hrqvae.log 显示 L0 ep30=26.6%, 但 launcher 写 `--loss_type poincare --beta 0.0` — 该 run 实际是 poincare+β=0 (hyperbolic distance 但 commit=0). 我之前以为是欧氏 MSE + 0 commit. 看到真实 hrqvae.py compute_loss 后才知道 loss_type 实际是 poincare, 跟 Issue #17 同族.

**双 NO-GO**:
- 真正"loss_type=mse + β=0" (本 task A1 v2): L0=1.6% → worse
- "loss_type=poincare + β=0" (task270 v1 Jul-29-2026_10-11-33): L0=26.6% → 也 fail ≥ 90%

结论: **无论 loss_type=mse 还是 poincare, β=0 都不能让 L0 ≥ 90%**. commit loss 不是 L0 瓶颈.

---

## 5. 推论 — L0 ≥ 90% 不是单 β 问题

任务270 description 的核心假设 (β 控制 L0 utilization) 是错的. 这一发现给出两条推论:

### 推论 1: Commit loss 是 L0 "稳定剂", 不是 L0 "天花板"
- β=0 时 encoder 自由 → 码字 5 epoch 内坍缩到 1 个
- β=0.5 时码字能在双曲约束下维持 73.44% 利用
- **commit loss 控制的是码字"被推开"vs"被拉拢"的力**. 没有 commit loss, encoder 永远收敛到 mode collapse.

### 推论 2: L0 ≥ 90% 需要更激进的机制
候选机制 (按 §6.7.4 单一变量约束, 需逐项验证):
- **dead_revive 频率** (Issue #17 Gate 1 verifier post-revive 实测 100%, pre-revive 73.44%) — 需在 pre-revive 阶段改 frequency
- **Sinkhorn 端点频次 / ε 调度** — 已知 vanilla 上 0pp 分离 (#260), 但 pre-Stage-3 量化时生效
- **kmeans_init 重频** — 现状 1000 iter 一次性 ep0, 也许周期化重 init dead cluster
- **curriculum β ramp** — A2 设计: A1 预训 → β=0.5 fine-tune. 既然 A1 端点 1.6% 不能稳定, A2 设计前提崩塌, **NO-GO**

---

## 6. R12 + USAGE-KILL 配对验证 (0 GPU 风险)

| 维度 | 状态 |
|------|------|
| R12 ckpt 保存 | ✅ best_collision_model.pth + best_loss_model.pth 已落盘 (ep24 + ep29) |
| USAGE-KILL 触发 (ep30 < 20%) | ✅ RuntimeError: epoch 30 utilization < 20%, killed (Issue #17 R12 fix 工作) |
| 训练未污染 GPU 0/1 task279 | ✅ GPU 2 单独跑, < 90 sec 内退出 |
| 0 重跑 / 0 抢卡 | ✅ |

---

## 7. 不应做的反推

- ❌ **不得反推 "β 应该重新打开"** — Task #84 baseline 就是 β=0.5 已 73.44% < 90%, 当前配方 β=0 是 NO-GO 闭环
- ❌ **不得反推 "Stage 2 Sinkhorn 解决 L0"** — Issue #18 已证: Stage 2 口径 L0 = 100% 但语义不同, Stage 1 口径不变
- ❌ **不得反推 "A2/A3 仍有价值"** — A2 依赖 A1 端点 ≥ 50%, A1 端点 1.6% → A2 NO-GO; A3 是 task193 encoder freeze 已知路径, ROI 不高
- ❌ **不得要求重跑 A1** — 50 epoch 在 USAGE-KILL ep30 已经 abort, 跑完不改变结论

---

## 8. 后续 (R11.5 自主决策)

| 候选 | ROI | 备注 |
|------|-----|------|
| Task #279 K=512/1024 Stage 4 eval waiter 自动 fire | 高 | 在跑 |
| Task #194 K-sweep ranking 整合 | 中 | Task #278 已跑过, verdict 待整合 |
| Issue #10 Gate 1 retry w/ task194_k0256 SID | 高 | D1 R10 backlog: 重跑 κ-decouple 在最优 K=256 SID, 需 4-6h GPU |

---

## 9. 物理产物 (commit)

- `scripts/task270_utilization_curriculum.sh` (loss_type="none" → loss_type="mse" bug 修复, 0 GPU)
- `verdicts/task282_task270_a1_no_go.md` (本文件)
- `products/task270/A1_euclidean/Jul-29-2026_13-39-43_*/hrqvae.log` (ep5 → ep30 step2 monitor 序列, R2 KB)
- `products/task270/A1_euclidean/Jul-29-2026_13-39-43_*/best_collision_model.pth` (ep24 ckpt, R2 KB)
- `descriptions/task270_utilization_curriculum_design.md` 不动 (高 ROI backlog 候选 3, 历史记录)

---

## 10. result

result: Task #282 / Task #270 A1 — 纯欧氏 MSE + β=0 NO-GO 闭环. Stage 1 50 epoch 在 ep30 触发 USAGE-KILL (Issue #17 R12 fix 起作用): kmeans_init ep5 L0=40.6% → ep10 L0=6.2% → ep15-30 L0 卡 1.6% (1/64 unique), 完全 mode collapse. **推论**: commit loss (β) 不是 L0 ≥ 90% 的杠杆——它是"码字稳定剂"不是"码字天花板". 无 commit loss 时 encoder 5 epoch 自洽坍缩到 1 个码字; 有 commit loss (β=0.5) 时至少维持 73.44%. L0 ≥ 90% 需要更激进机制 (dead_revive frequency / Sinkhorn curriculum / kmeans 重 init). A2 (curriculum β) 前提崩塌 NO-GO. 0 GPU 风险 (Stage 1 75 sec + ep30 提前 kill), ckpt 已落盘, task279 GPU 0/1 未受影响.
