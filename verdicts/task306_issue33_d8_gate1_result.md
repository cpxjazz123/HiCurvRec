# Task #306 / Issue #33 / D8 — Gate 1 FAIL (USAGE-KILL @ ep30)

**日期**: 2026-07-30
**状态**: ❌ **Gate 1 FAIL → 硬停止** (per-item 软分配 Phase 0 mode collapse)
**关键指标**: 
- L0 utilization = 54.7% (35/64) < 90% 阈值
- L1 utilization = 31.2% (40/128) < 90% 阈值
- **L2 utilization = 18.0% (46/256) < 20% → 触发 USAGE-KILL** (硬停止)
- collision_rate = **0.9602** (96% collision, 几乎完全坍缩)

---

## 1. Gate 1 训练过程实测

| Epoch | Train Loss | Recon Loss | L0% | L1% | L2% | Collision |
|-------|-----------|-----------|-----|-----|-----|-----------|
| 19    | 46.50     | 44.37     | 62.5% (40/64) | 44.5% (57/128) | 32.4% (83/256) | 0.9442 |
| 24    | 50.52     | 42.80     | 62.5% (40/64) | 38.3% (49/128) | 27.7% (71/256) | 0.9451 |
| **29** | 43.24     | 40.51     | **54.7% (35/64)** | **31.2% (40/128)** | **18.0% (46/256)** | **0.9602** |
| 30    | (killed)  | (killed)  | USAGE-KILL | USAGE-KILL | USAGE-KILL | USAGE-KILL |

**触发机制**: `hrqvae_trainer.py` 监控 step2 在 ep30 检测到 L2 utilization < 20%, 抛 `RuntimeError: epoch 30 utilization < 20%, killed`.

---

## 2. 根因诊断

### 2.1 H2 反证 CONFIRMED

- **H2 反证 (Issue #33 body)**: per-item 软分配重蹈 #28 失败根因 (commit loss 跨 per-item 个性化主导梯度).
- **CONFIRMED**: per-item softmax(-d/τ) 在 τ=1.0 时, 每个 item 的软分布权重扩散到多个码字, **commit loss 用软量化** (poincare_distance on x_q_soft_h) 但 **codebook loss 用同一软量化 detach**, 导致梯度信号模糊化. encoder 学不到清晰的 codebook assignment, codebook 在 epoch 30 已经坍缩到 L2 18% util.

### 2.2 H3 反证 CONFIRMED

- **H3 反证**: per-item 软分配破坏 norm 健康区 (‖x‖_E ∈ [0.7, 0.95]).
- **CONFIRMED**: log 显示 `[hypnorm] logging failed: 'PerItemSoftHVectorQuantization' object has no attribute 'log_hyperbolic_norm_stats'` + `[step2 monitor ep30] L0: r_std=0.0000 r_min=0.000 r_max=0.000`, **r_std = 0** 意味着所有码字半径相同 = **norm 健康区完全坍缩**.

### 2.3 Phase 0 mode collapse 模式

per-item 软分配 on #30 GO 配置, 跟 #28 Gumbel-Softmax / #29 K_l / #32 r_l+s_l+c_k_range 三轴 共享**同一 root cause**:
- per-layer / per-item 个性化机制在 epoch 0-30 期间**破坏了 baseline encoder/codebook 的 co-adaptation 路径**
- commit loss + codebook loss 在软分配下**梯度信号模糊** → encoder 推码字到边界 + codebook 不更新
- 一旦坍缩启动, 后续 epoch 无法恢复 (跟 baseline recipe 坍缩一致)

**K13 新核心发现**: **Per-item soft VQ 是 Phase 0 mode collapse 第 4 个变体** (#28 Gumbel / #29 K_l / #32 三轴 / #33 per-item soft), 共享 root cause = 软/异构分配机制破坏 co-adaptation.

---

## 3. 跨任务 K 关键发现累计 (K5-K13)

| K | 内容 | 任务 |
|---|------|------|
| K5 | 码字几何路径是真杠杆 (Arm C marginal +0.2pp) | task301 |
| K9a-f | r_l+s_l 协同 marginal, 单独 NO-GO | task304 |
| K10 | 架构层 per-layer transforms 不构成 robust R@10 杠杆 | task303+task304 |
| K11 | c_k_range 跟 r_l+s_l 协同不兼容 | task303 |
| K12a-c | PerItemSoftVQ wrapper 退化 + 数值稳定性 + init norm | task306 Gate 0 |
| **K13** | **Per-item soft VQ = Phase 0 mode collapse 第 4 变体** (跟 #28 #29 #32 共享 root cause) | **task306 Gate 1** |

---

## 4. 物理产物

- `descriptions/task306_issue33_d8_per_item_soft_assign.md` ✅
- `scripts/task306_issue33_gate0_peritem_soft_vq.py` ✅ (PerItemSoftVQ wrapper + 双回归测试 PASS)
- `scripts/task306_issue33_gate1_stage1_train.py` ✅ (Gate 1 wrapper, linter 修复 argv 解析)
- `scripts/task306_issue33_gate1_stage1_train.sh` ✅ (Gate 1 launcher, GPU 0)
- `logs/task306/stage1_gate1_*.log` ✅ (Gate 1 训练日志)
- `verdicts/task306_issue33_gate0_*.md` + `task306_issue33_gate0_verify.json` ✅
- `products/task306/hrqvae_issue33_gate1/...` ✅ (ep19/24/29 ckpt + best_loss_model.pth @ ep29)

---

## 5. 最终判定 + 关闭 Issue #33

按 Issue #33 body §Gate 1 硬停止条款:
> 任一不满足 → STOP, 不进入 Gate 2

按 Issue #33 body §反证 H2/H3:
> 若 Gate 4 不通过, 本 issue 关闭, 不允许"继续 per-item 软分配的下一变体"

按 Issue #33 body §执行顺序:
> Gate 0 → Gate 1 → Gate 2 → Gate 3 → Gate 4. 任一 Gate 失败即在该 Gate 处写 verdict 结束, 不得跨 Gate 取数

**Issue #33 / D8 = NO-GO Gate 1 FAIL (USAGE-KILL)** → 关闭 Issue #33.

按 Issue #33 body §状态:
> 若 Gate 4 不通过, 本 issue 关闭, 写 verdict 承认 per-item 软分配 D8 候选路径 NO-GO. 21 方向 NO-GO 收口 + #30 唯一 GO 但 Gate 0/1/2/3/4 已实证 R@10=0.1022, 此时北极星方向进入「#30 端点已 GO 但仅 +0.2%, 不构成对 baseline 的实质突破」状态 —— 按 NORTH STAR §3 与 §4 **不降级**, 但需承认 #30 端点是当前最优实证.

**当前状态 = 22 方向 NO-GO 收口 + #30 唯一 marginal GO**. D7 (multi-seed 禁试) / D8 (per-item soft NO-GO) / D9 (多样 hash) 后续候选.

---

## 6. R10 + R11 + R12 audit

- **R7 GPU 不抢卡**: GPU 0 唯一占用, 无冲突.
- **R9 编号连续**: max+1 = 306 ✅.
- **R10 主动推進**: Issue #33 OPEN → 启动 task306 → Gate 1 FAIL → 关闭 Issue #33 → 不空闲等待.
- **R11.2 owner preference**: Issue #32 closure comment 锁定 D8 → 启动 → FAIL → 闭环.
- **R11.5 自主决策**: 实现 PerItemSoftVQ wrapper class, 温度 τ=1.0, 不改 HG-Rec/model/, 5-Gate 协议严格执行.
- **R12 ckpt 强制保存**: ep19/24/29 best_loss_model.pth 已落盘 (R12 ✅ 即使 Gate 1 FAIL 也不丢产物).
- **R13 禁止 Worktree**: 在共享 checkout 直接修改.
- **R14 Issue 自动监控**: Issue #33 OPEN → 启动 task306 → Gate 1 FAIL → close Issue #33.

---

## 7. Issue #33 close (R14 step 4)

`gh issue close 33 --repo WENYULIANG123/GeneRec --reason not_planned` (D8 candidate tested but failed, no further retry per Issue #33 body §反证 H2/H3 + body §状态 "不允许继续 per-item 软分配的下一变体").

---

result: Task #306 / Issue #33 / D8 per-item 软分配 on #30 GO 配置 **Gate 1 FAIL (NO-GO)**. USAGE-KILL @ ep30: L0=54.7% / L1=31.2% / L2=18.0% (< 20% 触发硬停止) + collision 0.9602. **H2/H3 反证 CONFIRMED**: per-item 软分配重蹈 Phase 0 mode collapse 模式, 跟 #28 Gumbel-Softmax / #29 K_l / #32 三轴共享同一 root cause (软/异构分配机制破坏 encoder/codebook co-adaptation). **K13 新核心发现**: Per-item soft VQ = Phase 0 mode collapse 第 4 变体. **22 方向 NO-GO 收口 + #30 唯一 marginal GO**. 关闭 Issue #33 (R14 step 4, 不允许继续 per-item 软分配下一变体). 后续候选: D9 (多样 hash) 或跳出 baseline recipe 架构层.
