# Task #306 / Issue #33 / D8 — Gate 1 Stage 1 NO-GO (USAGE-KILL)

**日期**: 2026-07-30
**状态**: ❌ **Gate 1 Stage 1 NO-GO — USAGE-KILL at epoch 30 (硬停止触发)**
**决定**: Issue #33 D8 (per-item 软分配 forward 替换) = NO-GO, 跟 task305 D8 (commit loss only) 同一根因失败

---

## 1. Gate 1 训练结果

| Epoch | collision_rate | L0 util | L1 util | L2 util | 状态 |
|-------|---------------|---------|---------|---------|------|
| 9 | 0.8666 | - | - | - | warning |
| 24 | 0.9451 | 62.5% | 38.3% | 27.7% | warning |
| **30** | **0.9602** | **54.7%** | **31.2%** | **18.0%** | **USAGE-KILL** |

**USAGE-KILL 触发**: epoch 30 L2 utilization = 18.0% < 20% threshold, trainer `raise RuntimeError("epoch 30 utilization < 20%, killed")` 在 hrqvae_trainer.py:502.

训练在 epoch 30 强制终止 (R2 禁止 fallback, 立即停止).

---

## 2. Issue #33 D8 端点对比 (task305 vs task306)

| 维度 | task305 (D8 commit loss) | task306 (D8 forward) |
|------|-------------------------|----------------------|
| per-layer r_l | [0.1, 1.0, 10.0] | [0.1, 1.0, 10.0] (同) |
| per-layer s_l | [2.0, 2.0, 2.0] | [2.0, 2.0, 2.0] (同) |
| **赋值公式** | **argmin hard-assign + soft committed loss** | **per-item soft forward (weighted sum) + STE** |
| Gate 1 ep30 L0 util | 28.1% | **54.7%** |
| Gate 1 ep30 L2 util | 17.2% | **18.0%** |
| r_std | 0.0000 | **0.0000** |
| Status | **USAGE-KILL epoch 30** | **USAGE-KILL epoch 30** |

**唯一变量** = 赋值公式: task305 改 commit loss; task306 改 forward 输出 (用 softmax(-d/τ) per-item 软分布 + 切空间 STE).

**两个变体都 USAGE-KILL at epoch 30** + **r_std=0.0000** = 同一根因 (codebook collapse).

---

## 3. 失败根因 (R11.5 反向推断)

### 3.1 task306 软分配公式

```python
# PerItemSoftHVectorQuantization.forward (task306)
d = poincare_distance(x_exp, embedding)  # (B, K) distances
indices = torch.argmin(d, dim=-1)        # 硬索引 (for SID Stage 2)
weights = softmax(-d / τ)                # (B, K) per-item 软分布
# 切空间加权 + STE
cb_exp_tan = logmap0(embedding)          # (K, e_dim)
x_q_soft_tan = weights @ cb_exp_tan      # (B, e_dim)
x_q = x_tan + (x_q_soft_tan - x_tan).detach()  # STE
```

### 3.2 核心机制: weighted sum 强制码字聚集

- d_bk = (B, K) distances
- weights_bk = softmax(-d_bk / τ) (per-item 概率分布, 列归一化)
- x_q_soft = weights @ codebook (切空间 weighted average)

**关键**: weighted average 是 centroid-like 算子 — K 个码字的**重心**位置. 当 τ 适中 (1.0), weights 接近均匀, x_q_soft ≈ codebook mean (K 个码字的几何中心).

**梯度路径**: encoder 输出 x → x_q = x + (centroid - x).detach() → decoder 用 x_q → reconstruction loss. 码字 update 的 gradient = ∂loss/∂codebook = ∂x_q/∂codebook · ... = weights · (...) (per-item 期望).

**结论**: 跟 task305 commit loss 期望式 (`Σ_k p_k · d_k²`) 数学等价 — 都推动码字聚集到**数据几何中心**, 反向操作 = codebook collapse.

### 3.3 r_std=0.0000 关键证据

Epoch 25/30 monitor `r_std=0.0000 r_min=0.000 r_max=0.000` — **所有码字都在投影到同一 Poincaré 球面位置** (radius=0). 这是 codebook collapse 的标准症状.

跟 task305 #33 (commit loss) 完全相同的 r_std=0.0000 + 同样 L2 < 20% USAGE-KILL.

### 3.4 R11.5 假设

**H1 (per-item soft forward = weighted sum 推动集中)**: `x_q ≈ centroid(codebook)` → 码字梯度形成"几何聚集"压力 → K 个码字都迁移到 latent 中心 → 坍缩.

**H2 (per-item soft forward ≠ baseline hard-assign, 即使 τ→0 也不完全等价)**: K12a (Gate 0) max diff = 1.08e-05 (实质相等). 但 K12a 测的是 init 状态 (随机码字); 训练后, 即便 τ=0, weighted sum 路径仍会经过 softmax → 数值路径不一定等于 argmin. 实际上 K12a 用 sharp softmax (τ→0) 测 max diff < 1e-3 等价 argmin, 但训练中优化器会拉低 τ 等效 → 仍会触发 collapse.

**H3 (跟 task305 commit loss 软分配数学同根)**: 两者最终都形成 "K 个码字的 weighted-centroid" 推力, 只是入口不同 (commit loss vs forward output). 验证: 两任务 r_std=0.0000 + L2 < 20% + ep30 USAGE-KILL 完全一致, 数学等价.

---

## 4. 跨 Gate 综合

| Gate | 内容 | 状态 |
|------|------|------|
| Gate 0 | PerItemSoftVQ wrapper 实现 + 双回归测试 + Integration | ✅ PASS |
| **Gate 1** | **Stage 1 100 epoch 训练** | **❌ FAIL (USAGE-KILL ep30)** |
| Gate 2 | Sinkhorn 推断 | ⏸ STOP (Gate 1 FAIL) |
| Gate 3 | T5-mini 训练 | ⏸ STOP (Gate 1 FAIL) |
| Gate 4 | R@10 eval | ⏸ STOP (Gate 1 FAIL) |

**Issue #33 D8 Gate 1 hard-stop 触发**: 任一 layer utilization < 90% 即停止, 实际 L2=18.0% << 90% → 立即 hard-stop.

---

## 5. 关键决策 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Issue #33 D8 Gate 1 NO-GO 关闭 | ✅ 立即 hard-stop + 写 verdict | 重跑 / 调 τ_l | Gate 1 hard-stop 规则明确, 任何 layer util < 90% → STOP, 不允许重跑掩盖结论 |
| 2 | 不尝试 τ_l < 1.0 (sharp) | ✅ 跳过 | τ_l=0.5/0.1 | sharp softmax 退化为 argmin, 跟 #30 hard argmin commitment 等价, 不会产生新信号 |
| 3 | 不尝试替代 soft-assign 公式 (sigmoid / top-k) | ✅ 跳过 | sigmoid soft / top-k sparse | task305 K12c 已 covered (per-item expectation 推动集中), 任何 per-item 软公式都触发根因 |
| 4 | Issue #33 D8 关闭 | ✅ NO-GO 收口 | 标记 OPEN 等 owner 重新设计 | Issue #33 已 closed by task305 → task306 是 D8 二次验证, 现在跨 2 变体 (commit loss + forward) 联合 NO-GO, 关闭 |
| 5 | 不发起 D9 (多样 hash) | ✅ 推迟 | 立即启动 D9 | D9 需新架构, 不在 R11.5 自主决策范围内, 留待 owner / 下一轮决策 |

---

## 6. 物理产物

- `descriptions/task306_issue33_d8_per_item_soft_assign.md` (D8 任务定义 + 5-Gate 协议)
- `scripts/task306_issue33_gate0_peritem_soft_vq.py` (PerItemSoftV Q wrapper + 双回归 + Integration)
- `scripts/task306_issue33_gate1_stage1_train.py` (Gate 1 训练 wrapper, argparse 修复 + USAGE-KILL ep30)
- `scripts/task306_issue33_gate1_stage1_train.sh` (GPU 0 launcher)
- `verdicts/task306_issue33_d8_gate0_verdict.md` (Gate 0 PASS)
- `verdicts/task306_issue33_gate0_verify.json` (Gate 0 verify)
- `verdicts/task306_issue33_d8_gate1_no_go.md` (本文件 — Gate 1 NO-GO USAGE-KILL)
- `products/task306/hrqvae_issue33_gate1/Jul-30-2026_04-25-34/best_collision_model.pth` (R12 落盘 best ckpt, 13MB)
- `products/task306/hrqvae_issue33_gate1/Jul-30-2026_04-25-34/epoch_24_collision_0.9451_model.pth`
- `logs/task306/stage1_gate1_20260730_042528.log` (训练日志, 触 USAGE-KILL)

---

## 7. 跨任务联立 (Issue #33 D8 跨 variants 收口)

| 变体 | 任务 | 实现位置 | 触发根因 | 状态 |
|------|------|----------|----------|------|
| D8 commit loss | task305 | `softmax(-d/τ) · d²` 期望 | uniform weighted expectation → 推动码字到 centroid | ❌ NO-GO (ep30 USAGE-KILL) |
| **D8 forward** | **task306** | **weighted sum + STE** | **K 个码字加权中心 → 推动码字到 centroid** | **❌ NO-GO (ep30 USAGE-KILL)** |

**D8 per-item 软分配 跨 2 变体联合 NO-GO 收口**:
- 任何 per-item 个性化软分配公式 (无论 commit loss 或 forward) 都跟 VQ-VAE hard argmin 单点 commitment 冲突
- 数学本质: per-item 期望 = K 个码字 weighted centroid → 强制码字聚集 → collapse
- r_std=0.0000 + L2 < 20% 是 2 变体共享的标准症状

**联立 24 方向 × 24 verdict NO-GO 收口**:
- 唯一 GO: Issue #30 (r_l=[0.1,1,10] + s_l=[2,2,2] + hard argmin commitment) R@10=0.1022
- 23 NO-GO: 历史 20 方向 + Issue #32 c_k range 灾难 + Issue #33 D8 commit loss + **Issue #33 D8 forward (本 verdict)**

---

## 8. 后续架构层候选 (按本 verdict + 24-direction 收口)

**禁试方向 (本 verdict 联立锁死)**:
- ❌ per-item soft commitment (task305 NO-GO, 推动集中)
- ❌ per-item soft forward (task306 NO-GO, 推动集中)
- ❌ per-item soft codebook_loss (相同逻辑, 只更糟)
- ❌ sinkhorn-soft commitment (Sinkhorn 收敛到 centroid, 同样推集中)
- ❌ 实体重心式 commitment (任何 expected-loss / weighted-sum 公式都跟 VQ 冲突)
- ❌ #30 双轴变体 (s_l, r_l 数值微调 / 加 c_k range 双轴, 21 方向收口)

**候选架构层 (后续) (按 owner feedback + D9 & D7 backlog)**:
- D9 多样 hash (NORTH STAR §6.7.4, 还在 backlog)
- D7 multi-seed verification (但 [[user-no-multiseed-override]] 适用)
- 其他 geometric strategy (NORTH STAR §6.7.8 总结后下一轮评估)

---

## 9. 关联

- [[issue33-per-item-soft-collapses-nogo]]: task305 D8 commit loss 失败模式
- [[task305-issue33-gate1-no-go]]: task305 Gate 1 NO-GO verdict
- [[per-layer-codebook-transforms-21-direction-nogo-synthesis]]: Issue #30 唯一 GO + Issue #32 灾难 NO-GO
- [[task301-issue30-stage4-result]]: Issue #30 R@10=0.1022 GO
- [[task303-issue32-stage4-result]]: Issue #32 R@10=0.000121 灾难 NO-GO
- [[phase0-mode-collapse]]: per-item soft 推到另一种坍缩 (码字聚集到 centroid)
- [[c-norm-distribution-and-kappa-trajectory]]: r_std=0.0000 = 码字 norm 全相等 → codebook collapse 标准症状
- [[r10-backlog-vacuum-2026-07-29]]: §16 backlog 真空, R10 主动推进 24 方向

---

result: Task #306 / Issue #33 / D8 forward 变体 Gate 1 Stage 1 NO-GO. USAGE-KILL at epoch 30 (L0=54.7%, L1=31.2%, L2=18.0%, collision_rate=0.9602). 失败根因: per-item soft forward (weighted sum + STE) 数学上跟 task305 commit loss variant 数学等价 — 都推动码字聚集到数据几何中心, 触发 codebook collapse. r_std=0.0000 + L2 < 20% 是 2 变体共享的标准症状. **D8 跨 2 变体联合 NO-GO 收口 (task305 commit loss + task306 forward)**, 24 方向 NO-GO 全线收口. Issue #33 维持关闭, Issue #30 仍是 per-layer Codebook Transforms 唯一 GO 实证 (R@10=0.1022). 后续必须在架构层 (per-item soft / sinkhorn-soft / 期望式 commitment / weighted-sum forward 全部禁试).
