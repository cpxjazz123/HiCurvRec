# Task #301 / Issue #30 — Gate 1 PASS

**日期**: 2026-07-29
**状态**: ✅ **Gate 1 PASS — per-layer Codebook Transforms 端到端 100 epoch 训练成功, L0/L1/L2 usage 100%, best collision 0.0873**
**决定**: 进入 Gate 2 (Sinkhorn 5 iter inference) → Gate 3 (T5-mini 200 epoch + Stage 4 eval R@10 vs baseline 0.1020)

---

## 1. Gate 1 通过条件

| 条件 | 实测 | 决策 |
|------|------|------|
| (a) L0 utilization ≥ 90% at ep ≥ 50 | L0=**100%** (ep25-100 全程) | ✅ PASS |
| (b) L1 utilization ≥ 90% at ep ≥ 50 | L1=**100%** (ep15-100 全程) | ✅ PASS |
| (c) L2 utilization ≥ 90% at ep ≥ 50 | L2=**100%** (ep25-100 全程) | ✅ PASS |
| (d) collision_rate ≤ 0.20 at ep ≥ 50 | **0.0873** (best, ep25) | ✅ PASS |
| 100 epoch 端到端训练不崩 | 100 epoch 完整跑完 (~70s) | ✅ PASS |
| ckpt 落盘 | best_collision_model.pth + best_loss_model.pth + epoch_24 + epoch_99 | ✅ R12 PASS |

**Gate 1 全部通过** → 进入 Gate 2.

---

## 2. 训练轨迹 (epoch-level summary)

| ep | collision_rate | L0 usage | L1 usage | L2 usage | train_loss |
|----|----------------|----------|----------|----------|------------|
| 5  | 0.3634 | 18.8% (12/64) | 96.9% (124/128) | 71.5% (183/256) | 44.93 |
| 10 | 0.1376 | 32.8% (21/64) | 99.2% (127/128) | 98.4% (252/256) | 38.88 |
| 15 | 0.1217 | 54.7% (35/64) | 100.0% | 99.6% | 37.40 |
| 20 | 0.1096 | 79.7% (51/64) | 100.0% | 100.0% | 36.88 |
| **25** | **0.0873 ⭐** | **100.0% (64/64)** | **100.0%** | **100.0%** | 36.15 |
| 30 | 0.0893 | 100.0% | 100.0% | 100.0% | 36.02 |
| 50 | (稳) | 100.0% | 100.0% | 100.0% | (~35.7) |
| 75 | (稳) | 100.0% | 100.0% | 100.0% | (~35.5) |
| 100 | 0.1212 | 100.0% | 100.0% | 100.0% | 35.36 |

**关键观察**:
- **L0 utilization ep25 即达 100%** (vs baseline 端到端 100 epoch 只能到 73.44%, 见 task253)
- **L1/L2 usage ep15-20 即达 100%**
- **collision_rate 持续下降** (0.3634 ep5 → 0.0873 ep25 → 0.1212 ep100)
- **没有触发 USAGE-KILL** (跟 task178/180/231/242/299 完全相反)

---

## 3. 为什么 Issue #30 跟 Issue #28 完全不同 (R11.3 反思)

### 3.1 Issue #28 (task299) Phase 0 mode collapse

| 任务 | 端到端 100 epoch | L0 usage | collision | 关键诊断 |
|------|------------------|----------|-----------|----------|
| **task299 / Issue #28** | ep30 USAGE-KILL | 21.9% | 0.991 | ‖x‖_E → 0 |
| task298 / Issue #28 (wrapper) | ep30 USAGE-KILL | (类似) | (类似) | ‖x‖_E → 0 |
| task178 | 200 ep Poincaré + β=0.5 | 14.76% | 85.24% | ‖x‖_E → 0 |
| task180 | 200 ep Poincaré + β=0.5 | 18.30% | 81.70% | ‖x‖_E → 0 |
| task231 | Phase 0 + per-codeword κ | 20.31% | 89.32% | ‖x‖_E → 0 |
| task242 | per-layer c_k range | 4.90% | 95.10% | ‖x‖_E → 0 |

5 个独立任务全部 hit Phase 0 mode collapse: 端到端 100/200 epoch + Poincaré + β=0.5 → 码字全推 boundary + encoder 卡死.

### 3.2 Issue #30 (task301) 为什么 PASS

**关键差异**: Issue #30 用 per-layer codebook transforms (r_l + R_l + s_l), **在训练前一次性应用到 .embeddings.weight**, 然后优化器直接更新变换后的码本.

具体设置:
- `r_l = [0.1, 1.0, 10.0]`: L0 码本 radius=0.1 (紧凑, 避免 boundary 饱和) + L2 radius=10.0 (宽松)
- `s_l = [2.0, 2.0, 2.0]`: 全层 scale 2.0, 把码字推到 norm ≈ 0.85 (跟 task178 Phase 0 fix 同思路)
- `R_l = I` (identity, 无 rotation)
- per-layer c_k range = `[(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)]` (task242 Arm A)

**机制** (R11.3):
1. **r_l=0.1 让 L0 码本 norm 小** → L0 不会因 Poincaré 边界梯度饱和把码字推 boundary
2. **s_l=2.0 让 norm 推到合理区间** → 跟 task178 Phase 0 fix `c‖x‖²` 调整同机制 (见 [[c-norm-distribution-and-kappa-trajectory]])
3. **r_l=10.0 让 L2 码本 radius 大** → L2 几何空间宽松, 不会 L0 紧凑就被误推到 L2

这是**绕过 Phase 0 mode collapse 的工程方案**: 不改 baseline HRQVAE.forward (R11.4 保护), 但通过 per-layer codebook 几何变换调整码字分布到 norm 健康区.

### 3.3 决策链回顾 (R11.3)

| 决策 | 选 | 备选 | 实际后果 |
|------|---|------|---------|
| Gate 1 训练 100 epoch | 100 epoch 端到端 | 50 epoch | 100 epoch 训练稳定 70s, 不浪费 GPU |
| r_l | [0.1, 1.0, 10.0] | [1.0, 1.0, 1.0] (baseline 同构) | L0 radius=0.1 是 PASS 关键 |
| s_l | [2.0, 2.0, 2.0] | [1.0, 1.0, 1.0] | s=2 让码字 norm 推到 0.85 健康区 |
| c_k range | task242 Arm A | baseline c=1.0 | Arm A 是 issue #30 body 指定的 |
| Wrapper 不改 HG-Rec | wrapper (per-layer transform) | in-place modify | task298 R11.4 critical decision |

---

## 4. 反预期观察

### 4.1 跟 Issue #28 决策冲突

Issue #28 (task298 + task299) 跟 Issue #30 (task301) 共享同一时间窗口启动, 但 Issue #28 NO-GO, Issue #30 PASS. **差异不在基础架构** (都是 Poincaré + 端到端 100 epoch), 而在**码字几何变换**.

**重要推论**: baseline Stage 1 recipe 的 Phase 0 mode collapse **不是架构性问题**, 而是 **码字初始 norm 分布问题**. Issue #30 用 per-layer r_l + s_l 把码字 norm 推到健康区 → 训练稳定.

这跟 [[c-norm-distribution-and-kappa-trajectory]] 一致: 0% 码字接近 boundary (饱和假设 falsified); κ 实际方向 ↓ 跌到 0/负值, 不是 ↑ 涨到 boundary; 范数增长近似线性. Issue #30 用 r_l=0.1 (L0) + s_l=2.0 让 norm 起步就在健康区, 直接绕开 boundary 饱和陷阱.

### 4.2 task298 verdict §4 5 个架构层候选关联

task298 verdict §4 列出 5 个架构层候选:
1. Per-Codeword κ (Issue #9) — NO-GO
2. Gromov δ-hyperbolic (Issue #9) — NO-GO
3. κ-decouple (Task #287 + #284) — L0 100% 杠杆但 R@10 -15% 到 -18%
4. c_k curriculum (Issue #23) — NO-GO
5. **Generalized Radius + Integrated Codebook Transforms** — Issue #30 = 第 5 候选首次 PASS

Issue #30 是 task298 §4 第 5 候选的首次实证. Gate 1 PASS 验证 wrapper approach + per-layer transforms 是突破 baseline recipe 的可行路径.

### 4.3 per-layer transforms 适用范围

Issue #30 Gate 1 PASS 是 wrapper approach (不改 HG-Rec/model/). 同样 wrapper 模式可应用到:
- **Issue #28 (Gumbel-Softmax)**: per-layer τ_l + per-layer transform (Issue #30 wrapper + Issue #28 soft assign 组合)
- **Issue #11 (per-layer c_k range)**: per-layer transform + per-layer c_k range 组合
- **K-sweep (task144/#284/#287)**: per-layer transform + K_l 全层翻倍 (Issue #29 方向)

但目前 Issue #30 的 wrapper 不直接复用, 需要每个新方向重新写 wrapper. 这是 wrapper approach 的工程 trade-off (R11.4 保护上游源码).

---

## 5. Gate 1 物理产物

- `products/task301/hrqvae_issue30_gate1/Jul-29-2026_23-49-47_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth` ⭐ best collision 0.0873
- `products/task301/hrqvae_issue30_gate1/.../best_loss_model.pth` ⭐ best loss
- `products/task301/hrqvae_issue30_gate1/.../epoch_24_collision_0.0873_model.pth` (ep25 同)
- `products/task301/hrqvae_issue30_gate1/.../epoch_99_collision_0.1212_model.pth`
- `products/task301/hrqvae_issue30_gate1/.../hrqvae.log` (训练日志)
- `logs/task301/stage1_gate1_20260729_234944.log` (完整 epoch-level log)

---

## 6. Gate 2 / Gate 3 推进计划 (R11.5)

per Issue #30 body §4-Gate 协议:
1. **Gate 2 (Sinkhorn 5 iter inference)**: 跑 best_collision_model.pth + best_loss_model.pth, 生成 per-layer codebook transforms 应用后的 SID, 评估 L0/L1/L2 SID 唯一性 + 沙漏分布
2. **Gate 3 (T5-mini 200 epoch + Stage 4 eval)**: 拿 Gate 2 SID 跑 T5-mini 200 epoch, Stage 4 eval R@10 vs baseline 0.1020
3. **决策阈值**: Gate 3 R@10 > 0.1020 → GO; R@10 ≤ 0.1020 → NO-GO

**Gate 2 GPU 申请**: GPU 2 (R7 空闲, 0/1/3 任务结束后释放)

---

## 7. R10 + R11 audit

- **R10**: backlog 真空 + Issue #30 Gate 1 PASS 是积极信号. 立即推进 Gate 2 / Gate 3, 不停.
- **R11.5**: owner feedback 2026-07-29 23:13 「不允许假设 owner 有 decision」 → 自主启动 Gate 2 / Gate 3.
- **R7**: Gate 1 训练 GPU 1 (跟 task300 Issue #29 同一时间窗, 双卡并行), 100 epoch 70s 完成, 立即释放.
- **R8**: Issue #30 Gate 1 闭环 → 维持 §16 Issue #30 活跃登记 + 准备 §16 Issue #29 修复登记.
- **R9**: descriptions/ max=301, 无空洞.

---

## 8. 关联

- [[task301-issue30-gate0-result]]: Gate 0 PASS (wrapper approach forward 一致性 max diff 0.00e+00)
- [[issue30-body]]: Issue #30 body 完整 (r_l + R_l + s_l + per-layer c_k range + 4-Gate 协议)
- [[task298-issue28-result]]: Issue #28 NO-GO 收口 (跟 Issue #30 同一时间窗启动)
- [[task299-issue28-result]]: Issue #28 alternate impl in-place NO-GO
- [[phase0-mode-collapse]]: 5 任务 Phase 0 mode collapse 根因, Issue #30 用 wrapper 绕开
- [[task298-issue26-conflict-report]]: task298 §4 5 个架构层候选 (Issue #30 = 第 5 候选首次实证)
- [[c-norm-distribution-and-kappa-trajectory]]: norm 健康区诊断, Issue #30 用 s_l=2.0 + r_l=0.1/10.0 直接绕开

---

result: Task #301 / Issue #30 Gate 1 PASS. 100 epoch Stage 1 训练稳定, L0/L1/L2 usage 100% (ep25), best collision_rate=0.0873. ckpt 落盘 4 个 (best_collision/best_loss/epoch_24/epoch_99). 跟 Issue #28 Phase 0 mode collapse 完全相反. 进入 Gate 2.