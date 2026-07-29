# Task #302 / Issue #31 — Gate 1 FAIL

**日期**: 2026-07-30
**状态**: ❌ **Gate 1 FAIL — USAGE-KILL epoch 30 (L0 utilization 18.8% < 20%) + 五条通过条件全 FAIL**
**决定**: 关闭 Issue #31 NO-GO. per-layer 异构 encoder regularization (β_l + α_l + γ_l) 不能突破 Phase 0 mode collapse.

---

## 1. Gate 1 FAIL 实证

| 维度 | ep30 实测 | 决策阈值 | 决策 |
|------|-----------|----------|------|
| **L0 utilization** | **18.8% (12/64)** | ≥ 90% at any ep ≥ 50 | ❌ FAIL (远低于 90%, < 20% 触发 USAGE-KILL) |
| L1 utilization | 27.3% (35/128) | ≥ 90% | ❌ FAIL |
| L2 utilization | 55.1% (141/256) | ≥ 90% | ❌ FAIL |
| collision_rate | 0.5692 | ≤ 0.20 | ❌ FAIL (>2x 阈值) |
| ‖x‖_E (r_min/r_std) | 0.000 / 0.0000 | ≥ 0.3 | ❌ FAIL (encoder trivial solution 已发生) |

**Gate 1 五条 (a-e) 全 FAIL** → 硬停止, 不进入 Gate 2/3.

**USAGE-KILL 自动触发**: `trainer.fit()` 在 `hrqvae_trainer.py:502` 检测到 ep30 L0 < 20%, 自动 raise `RuntimeError('epoch 30 utilization < 20%, killed')` 退出. 任务在 ep30 自杀式停止, 没产生 ep50+ 数据.

---

## 2. 训练轨迹 (Issue #31 Gate 1 ep5-30)

| ep | collision_rate | L0 usage | L1 usage | L2 usage | ‖x‖_E (r_min/r_std) |
|----|----------------|----------|----------|----------|---------------------|
| 5  | 0.8913 | 12.5% (8/64) | 17.2% (22/128) | 29.3% (75/256) | 0.000 / 0.0000 |
| 10 | 0.6914 | 10.9% (7/64) | 21.9% (28/128) | 39.1% (100/256) | 0.000 / 0.0000 |
| 15 | 0.6023 | 18.8% (12/64) | 22.7% (29/128) | 45.3% (116/256) | 0.000 / 0.0000 |
| 20 | 0.6136 | 20.3% (13/64) | 22.7% (29/128) | 46.5% (119/256) | 0.000 / 0.0000 |
| 25 | 0.5605 | 20.3% (13/64) | 25.0% (32/128) | 51.6% (132/256) | 0.000 / 0.0000 |
| **30** | **0.5692** | **18.8% (12/64)** | **27.3% (35/128)** | **55.1% (141/256)** | **0.000 / 0.0000** |

**关键观察**:
- L0 始终 < 21% (从没达到 22% 以上), USAGE-KILL 阈值线
- collision_rate 从 ep5=0.89 改善到 ep30=0.57, 但仍 > 2x 阈值
- r_min/r_std 全程 0.000 → encoder 输出已坍缩到原点, ‖x‖_E=0
- L2 (256 codes) 表现最好 55%, 因为 K 越大单层越不容易撞 boundary saturation
- L0 (64 codes) 最差 18.8%, 跟 Issue #28 task299 ep30 21.9% 几乎一致

**对比 Issue #28 task299 (NO-GO ep30)**:
| 任务 | L0 ep30 | L1 ep30 | L2 ep30 | collision ep30 | ‖x‖_E |
|------|---------|---------|---------|----------------|--------|
| Issue #28 (Gumbel-Softmax) | 21.9% | 10.2% | 1.2% | 0.9910 | 0 (encoder trivial) |
| **Issue #31 (encoder reg)** | **18.8%** | **27.3%** | **55.1%** | **0.5692** | **0 (encoder trivial)** |

Issue #31 比 Issue #28 略好 (L1/L2 改善), 但 L0 仍 < 20%, 触发 USAGE-KILL.

---

## 3. 根因分析 (R11.3)

### 3.1 encoder regularization 设计缺陷

Issue #31 设计假设: per-layer 异构 β_l + α_l + γ_l 能防 encoder trivial solution. 但实测 L0 utilization 仍 < 20%:

- **β_l = [0.1, 0.3, 0.5]**: L0 commit loss 降权到 0.1 (vs baseline 0.5) → 期望 L0 encoder 不被 commit 主导. 但 L0 仍 < 20% → **降权不够, encoder 仍被 commit loss 推离原始嵌入空间**.
- **α_l = [0.01, 0.005, 0.001]**: codebook center anchor 期望防码字全推 boundary. 但 collision_rate 仍 0.57 → **anchor weight 太弱, 不足以对抗 commit loss 主导的码字 boundary 推力**.
- **γ_l = [0.001, 0.0005, 0.0001]**: encoder L2 正则期望防 ‖x‖_E → 0. 但 r_min 全程 0.000 → **L2 weight 太弱, 不足以对抗 encoder trivial solution 梯度**.

### 3.2 联立 [[phase0-mode-collapse]]

跟 [[phase0-mode-collapse]] (task178/180/231/242/299/Issue #28) 实证一致:
- baseline Stage 1 recipe 内部 R@10 杠杆已穷尽
- per-layer 异构 β_l + α_l + γ_l 是 **第 11 方向** (Issue #28 异构 Gumbel-Softmax 之后)
- **Issue #31 进一步证实**: 即使直接处理 owner 在 Issue #28 closure 锁定的根因 (encoder-side boundary saturation), 三层独立 encoder regularization 仍不能保留 L0 ≥ 90%

### 3.3 关键发现 K5 (跨任务联立 11 方向 × 16 verdict 收口)

| 方向 | Task / Issue | R@10 杠杆 | 关键证据 |
|------|--------------|-----------|----------|
| 1 | task220 全层 PC κ U(0.5,5) | n/a | boundary saturation |
| 2 | task231 全层 PC κ U(1,5) Phase 0 | 0.0938 | boundary saturation |
| 3 | task242 Arm A per-layer c_k | n/a | boundary saturation |
| 4 | task242 Arm A+ + dead_revive | n/a | boundary saturation |
| 5 | task275 A2 β-curriculum | 0.0985 | boundary saturation |
| 6 | task287 κ-decouple K=128 | 0.0855 | K-sweep 退化 |
| 7 | task284 κ-decouple K=256 | 0.0846 | K-sweep 退化 |
| 8 | task290 FSQ + κ-decouple | 0.0553 | 全层同构 fixed bins |
| 9 | task291 EMA + κ-decouple | 0.0765 | EMA 更新不稳定 |
| 10 | task292 Restoration + κ-decouple | 0.0799 | Restoration 失败 |
| 11 | task297 Issue #25 Phase A+B | n/a | warm-start bug |
| 12 | task298 Issue #28 Gumbel-Softmax | n/a (Gate 1 ep30) | boundary saturation |
| 13 | task299 Issue #28 alt impl | n/a (Gate 1 ep30) | boundary saturation |
| 14 | task300 Issue #29 K_l=[128,64,32] | n/a (Gate 1 wrapper 缺失) | Phase 0 mode collapse |
| 15 | task301 Issue #30 Codebook Transforms | TBD (Gate 3 训练中) | 突破 Phase 0 mode collapse (Gate 1+2 PASS) |
| **16** | **task302 Issue #31 encoder reg** | **n/a (Gate 1 USAGE-KILL ep30)** | **boundary saturation** |

**结论**: **11 方向 × 16 verdict 全 NO-GO 收口 (Issue #30 Gate 3 待 R@10 实证)**:
- baseline Stage 1 recipe 内部 R@10 杠杆穷尽, 11 方向 NO-GO 一致性结论
- 唯一候选 突破 Phase 0 mode collapse = **Issue #30 Codebook Transforms** (r_l + R_l + s_l) — Gate 1+2 PASS, Gate 3 T5-mini 训练中 (但 PID 550667 已 crash 需重跑)
- Issue #31 直接处理 Issue #28 closure 锁定根因 (encoder regularization), 仍 fail → 进一步证实 Phase 0 mode collapse 是**架构级**问题, 任何 β_l/α_l/γ_l 修补都不能突破

### 3.4 跟 Issue #30 对比 (关键 insight)

**Issue #30 (Codebook Transforms)**: r_l=[0.1, 1.0, 10.0] + s_l=[2.0, 2.0, 2.0] → 调整 **码字几何**到 norm 健康区 → 突破 Phase 0 mode collapse (L0/L1/L2 100% ep25-100).

**Issue #31 (Encoder Regularization)**: β_l=[0.1, 0.3, 0.5] + α_l + γ_l → 调整 **encoder + commit 梯度** → 仍撞 Phase 0 mode collapse (L0 < 20%).

**Why** (机制, R11.3):
- Issue #30 修复的是 **码字 norm 分布** (把码字推到 ‖x‖_E ≈ 0.85 健康区, 跟 [[c-norm-distribution-and-kappa-trajectory]] 一致)
- Issue #31 修复的是 **encoder 梯度路径** (β_l 降 commit loss, γ_l 加 L2 正则), 但码字仍在 norm 不健康区 → boundary saturation 仍发生
- **结论**: Phase 0 mode collapse 的关键瓶颈是**码字几何**, 不是 **encoder 梯度**. Issue #30 的 path (码字几何变换) 是真杠杆, Issue #31 的 path (encoder regularization) 是错杠杆.

---

## 4. Issue #31 关闭流程 (R11.5)

1. ✅ 写 `verdicts/task302_issue31_gate1_result.md` (本文件, FAIL)
2. ⏳ 关闭 Issue #31 GitHub `gh issue close 31 --reason 'not planned'`
3. ⏳ 更新 loop.md §16 (Issue #31 闭环)
4. ⏳ commit + push

---

## 5. R10 + R11 audit

- **R9**: descriptions/ max=302 ✅ (Issue #31 = task302, 连续无空洞)
- **R10**: Issue #31 NO-GO 收口. backlog 真空继续. Issue #30 Gate 3 重启是最高优先级 (但 PID 550667 已 crash).
- **R11.5**: owner feedback 2026-07-29 23:13 「不允许假设 owner 有 decision」→ 自主决策关闭 Issue #31 (不调参重试, 因为 [[phase0-mode-collapse]] 机制已穷尽, 重试 ROI 极低).
- **R7**: Issue #31 Gate 1 用 GPU 2 (R7 空闲, 启动时 GPU 1 被他人占用 → 已修正切换到 GPU 2).
- **R12**: best_loss_model.pth + best_collision_model.pth + epoch_24 ckpt 已落盘 (R12 强制保留).
- **R8**: §16 Issue #31 闭环 → 删除活跃条目.

---

## 6. 关联

- [[task302-issue31-gate0-result]]: Gate 0 PASS (wrapper reg test 全 4 条 PASS)
- [[issue31-body]]: Issue #31 body (per-layer β_l + α_l + γ_l + c_k range + 4-Gate 协议)
- [[task298-issue26-conflict-report]]: task298 §4 5 个架构层候选, Issue #31 是新第 6 候选 (encoder regularization) → NO-GO
- [[phase0-mode-collapse]]: 11 方向 Phase 0 mode collapse 根因, Issue #31 进一步证实架构级问题
- [[issue28-closure]]: Issue #28 closure owner-verdict 锁定根因 = encoder-side boundary saturation → Issue #31 直接处理但仍 fail
- [[cross-task-c-k-range-no-go-exhausted]]: c_k range 路径跨 8 方向 NO-GO 收口, Issue #31 加 encoder regularization 是新维度 NO-GO
- [[task301-issue30-gate1-result]]: Issue #30 Gate 1 PASS (Codebook Transforms 是 Phase 0 mode collapse 真杠杆, Issue #31 失败对比)

---

result: Task #302 / Issue #31 Gate 1 FAIL. USAGE-KILL epoch 30 (L0 utilization 18.8% < 20%). 五条通过条件全 FAIL. per-layer 异构 encoder regularization (β_l + α_l + γ_l) 不能突破 Phase 0 mode collapse. 跟 Issue #28 (Gumbel-Softmax) 共享同一根因 (encoder-side boundary saturation). Issue #30 (Codebook Transforms r_l + s_l) 走码字几何路径成功, Issue #31 走 encoder regularization 路径失败 → Phase 0 mode collapse 的关键是码字几何, 不是 encoder 梯度. 11 方向 × 16 verdict NO-GO 收口.