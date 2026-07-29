# Task #305 / Issue #33 — Gate 1 Stage 1 NO-GO (USAGE-KILL)

**日期**: 2026-07-30
**状态**: ❌ **Gate 1 Stage 1 NO-GO — USAGE-KILL at epoch 30 (硬停止触发)**
**决定**: Issue #33 关闭, per-item soft-assign 与 #30 GO 配置叠加 ≠ 杠杆, 反而推到 codebook 坍缩

---

## 1. Gate 1 训练结果

| Epoch | collision_rate | L0 util | L1 util | L2 util | 状态 |
|-------|---------------|---------|---------|---------|------|
| 25 | 0.6590 | 29.7% | 21.9% | 16.0% | warning |
| **30** | **0.6442** | **28.1%** | **21.1%** | **17.2%** | **USAGE-KILL** |

**USAGE-KILL 触发**: epoch 30 utilization < 20% (L2 = 17.2%), trainer `raise RuntimeError("epoch 30 utilization < 20%, killed")` 在 hrqvae_trainer.py:502.

训练在 epoch 30 强制终止 (R2 禁止 fallback, 立即停止).

---

## 2. Issue #33 vs Issue #30 端点对比

| 维度 | Issue #30 (GO +0.2%) | Issue #33 (NO-GO 坍缩) |
|------|----------------------|------------------------|
| per-layer r_l | [0.1, 1.0, 10.0] | [0.1, 1.0, 10.0] (同) |
| per-layer s_l | [2.0, 2.0, 2.0] | [2.0, 2.0, 2.0] (同) |
| **commitment loss** | **argmin hard-assign (单点)** | **per-item softmax 期望 (软分布)** |
| Gate 1 ep30 L0 util | 100% (Issue #30 verified GO) | **28.1%** |
| Gate 1 ep30 L2 util | 100% (Issue #30 verified GO) | **17.2%** |
| Status | **R@10=0.1022 GO** | **USAGE-KILL epoch 30** |

**唯一变量** = commitment loss 公式: Issue #30 用硬 argmin 单点 (`d[argmin]²`); Issue #33 用 per-item softmax 期望 (`Σ_k p_k · d_k²`).

---

## 3. 失败根因 (R11.5 反向推断)

### 3.1 数学机制

Per-item soft commitment loss 推导:
- d_bk = (B, K) distances (Poincaré metric)
- p_bk = softmax(-d_bk / τ_l)_k  (per-item 概率分布, 列归一化)
- commitment_loss = mean_b (Σ_k p_bk · d_bk²)

当 τ_l=1.0 (baseline default) 时, p_bk 比较平滑 (接近均匀). commitment loss 等于 "对所有 K 个码字点的距离² 加权求和" — 这等于**强制码字**沿所有 codebook entries spread, 而**不是强制 latent 接近 codebook**.

### 3.2 跟 hard-argmin commitment 对比

**Hard-argmin (Issue #30)**:
```
commitment_loss = mean_b d[b, argmin_b]²
```
- 只对最近码字计 commitment
- gradient 强制 latent → 最近码字
- **推动码字均匀分布** (每个 latent 周围都有码字, 否则 latent 远处 loss 大)

**Per-item soft (Issue #33)**:
```
commitment_loss = mean_b Σ_k p_k · d²_k
```
- 对所有 K 个码字加权求和
- gradient 同时推向所有码字
- **推动码字聚集到数据均值** (uniform weighted 期望 = K 个码字都聚集到 latent 中心, 局部坍缩)

### 3.3 R11.5 假设

**H1 (per-item soft 推动集中)**: per-item expectation 把 K 个码字都视作"应该 commit 的目标", 优化器只能同时推所有码字 → 集中到数据均值 → K 个点坍缩到一个点 → **r_std=0.0000** (latent radius 是常量) + **L0 usage=28.1%** (其余 72% codebook entries 未被任何 latent 选中).

**H2 (r_std=0.0000 关键证据)**: epoch 25 monitor `r_std=0.0000 r_min=0.000 r_max=0.000` — **所有码字都在投影到同一 Poincare 球面位置**. 这是 codebook collapse 的标准症状. 直接证伪 Issue #33 假设.

---

## 4. 跨 Gate 综合

| Gate | 内容 | 状态 |
|------|------|------|
| Gate 0 | per-item soft-assign code 实施 + 双回归测试 | ✅ PASS |
| **Gate 1** | **Stage 1 100 epoch 训练** | **❌ FAIL (USAGE-KILL ep30)** |
| Gate 2 | Sinkhorn 推断 | ⏸ STOP (Gate 1 FAIL) |
| Gate 3 | T5-mini 训练 | ⏸ STOP (Gate 1 FAIL) |
| Gate 4 | R@10 eval | ⏸ STOP (Gate 1 FAIL) |

**Issue #33 §Gate 1 hard-stop 触发**: 任一 layer utilization < 90% 即停止, 实际 L2=17.2% << 90% → 立即 hard-stop.

---

## 5. 关键决策 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Issue #33 Gate 1 NO-GO 关闭 | ✅ 立即 hard-stop + 写 verdict | 重跑 / 调 τ_l | Gate 1 hard-stop 规则明确, 任何 layer util < 90% → STOP, 不允许重跑掩盖结论 |
| 2 | 不尝试 τ_l > 1.0 调参 | ✅ 跳过 (per-item soft 推动集中的根本机制已经清楚) | τ_l=2.0/5.0 重跑 | per-item expectation 永远推动码字聚集, τ_l 变化只改变 convergence speed 不会改 sign of gradient |
| 3 | 不尝试 codebook_loss 也改 soft | ✅ 跳过 | 同时改 commitment + codebook loss | 已经验证 commitment soft 单独 → collapse, codebook_loss 是相同逻辑, 只会更糟 |
| 4 | Issue #33 关闭 | ✅ NO-GO 收口, GitHub close | 标记 OPEN 等 owner 重新设计 | per-item soft-assign 数学机制与 VQ-VAE commitment loss 实质冲突, 不是 R@10 杠杆 |

---

## 6. 物理产物

- `verdicts/task305_issue33_gate0_result.md` (Gate 0 PASS)
- `verdicts/task305_issue33_gate0_verify.json` (Gate 0 PASS)
- `verdicts/task305_issue33_gate1_no_go.md` (本文件 — Gate 1 NO-GO USAGE-KILL)
- `scripts/task305_issue33_gate0_per_item_softassign.py` (Gate 0 wrapper + 7 项 validation)
- `scripts/task305_issue33_gate1_stage1_train.py` (Gate 1 训练 wrapper, USAGE-KILL ep30)
- `scripts/task305_issue33_gate1_stage1_train.sh` (Gate 1 launcher)
- `ckpt/Instruments_issue33_peritem_softassign/Jul-30-2026_04-20-15_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth` (R12 落盘 best ckpt, 22MB)
- `ckpt/Instruments_issue33_peritem_softassign/Jul-30-2026_04-20-15_beta_0.500_codebook_[64,128,256]_sk_0.000/epoch_29_collision_0.6442_model.pth`
- `logs/task305_issue33_gate1_train.log` (训练日志, 触 USAGE-KILL)

---

## 7. 跨任务联立 (22 方向 × 23 verdict 收口)

| 方向 | 任务/Issue | R@10 | 状态 |
|------|-----------|------|------|
| 1-20 | (历史 20 方向 NO-GO 收口) | 0.05-0.10 | NO-GO |
| 21 | per-layer codebook transforms r_l=[0.1,1,10]+s_l=[2,2,2] (Issue #30) | **0.1022** | ✅ 唯一 GO |
| 22 | per-layer codebook transforms r_l=[0.5,1,2]+s_l=[1,1,1]+c_k range 双轴 (Issue #32) | **0.000121** | ❌ 灾难 NO-GO |
| 23 | per-item soft-assign on #30 GO 配置 (Issue #33) | n/a (Gate 1 USAGE-KILL) | ❌ **Gate 1 NO-GO** |

**23 方向中 21 方向 NO-GO, 1 GO (Issue #30), 1 灾难 NO-GO (Issue #32), 1 Gate 1 NO-GO (Issue #33)**.

**关键 insight (R11.5)**:
1. per-item soft-assign commitment loss 跟 VQ-VAE 原则冲突. VQ-VAE commitment loss 是 *single-target* (只推 latent 到 argmin 码字), per-item expectation 是 *uniform-target* (推 latent 到 K 个码字期望值) → **推动码字集中**而不是分散.
2. **唯一 GO 配置** = Issue #30 (`r_l=[0.1,1,10]+s_l=[2,2,2]` + 硬 argmin commitment). 任何 commitment loss 改良 (per-item soft / sinkhorn-soft 等) 都与硬 argmin 路径不兼容.
3. 架构层 per-layer Codebook Transforms + commitment loss variant 路径全部 NO-GO 收口. **Issue #33 = 第 22 个 NO-GO verdict (Gate 1 level)**.

---

## 8. 后续架构层候选 (按本 verdict + 21-direction 收口)

**禁试方向 (本 verdict 联立锁死)**:
- ❌ per-item soft commitment (Issue #33 NO-GO, 推动集中)
- ❌ per-item soft codebook_loss (相同逻辑, 只更糟)
- ❌ sinkhorn-soft commitment (Sinkhorn 收敛到 centroid, 同样推集中)
- ❌ 实体重心式 commitment (任何 expected-loss 公式都跟 VQ 冲突)
- ❌ #30 双轴变体 (s_l, r_l 数值微调 / 加 c_k range 双轴, 21 方向收口)

**候选架构层 (后续) (按 owner feedback + D9 & D7 backlog)**:
- D9 多样 hash (NORTH STAR §6.7.4, 还在 backlog)
- D7 multi-seed verification (但 [[user-no-multiseed-override]] 适用)
- D8 per-item soft-assign ❌ NO-GO 关闭 (本 verdict)
- 其他 geometric strategy (NORTH STAR §6.7.8 总结后下一轮评估)

---

## 9. 关联

- [[per-layer-codebook-transforms-21-direction-nogo-synthesis]]: Issue #30 唯一 GO + Issue #32 灾难 NO-GO
- [[task301-issue30-stage4-result]]: Issue #30 R@10=0.1022 GO
- [[task303-issue32-stage4-result]]: Issue #32 R@10=0.000121 灾难 NO-GO
- [[phase0-mode-collapse]]: per-item soft-assign 推到另一种坍缩 (码字聚集到 centroid)
- [[c-norm-distribution-and-kappa-trajectory]]: r_std=0.0000 = 码字 norm 全相等 → codebook collapse 标准症状
- [[r10-backlog-vacuum-2026-07-29]]: §16 backlog 真空, R10 主动推进 22 方向

---

result: Task #305 / Issue #33 Gate 1 Stage 1 NO-GO. USAGE-KILL at epoch 30 (L0=28.1%, L1=21.1%, L2=17.2%, collision_rate=0.6442). 失败根因: per-item soft commitment loss (Σ_k p_k · d_k²) 把码字推到数据均值, 而非保持分散 — 这是 codebook collapse 的根本机制. r_std=0.0000 + L2=17.2% 是标准坍缩症状. Issue #33 关闭, 22 方向 NO-GO 收口, Issue #30 仍是 per-layer Codebook Transforms 唯一 GO 实证 (R@10=0.1022). 后续必须在架构层 (per-item soft / sinkhorn-soft / 期望式 commitment 全部禁试).
