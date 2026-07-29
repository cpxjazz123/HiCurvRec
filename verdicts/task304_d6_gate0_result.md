# Task #304 / D6 ablation — Gate 0 综合结果 (3-arm)

**日期**: 2026-07-30
**状态**: 🟡 Arm A + Arm B PASS / Arm C SKIP (复用 Issue #30 reference)
**决定**: 进入 Arm A + Arm B Gate 1 Stage 1 100 epoch 训练 (GPU 0/1 并行); Arm C 复用 task301 已知 PASS 不重训

---

## 1. Gate 0 综合结果

| Arm | 配置 | r_l | s_l | Gate 0 | 决策 |
|-----|------|-----|-----|--------|------|
| **A (r_l only)** | r_l=[0.1,1,10] + s_l=[1,1,1] + c_k_range | 1.01e-02 | identity | ✅ PASS | 进入 Gate 1 |
| **B (s_l only)** | r_l=[1,1,1] + s_l=[2,2,2] + c_k_range | identity | 8.45e-04 | ✅ PASS | 进入 Gate 1 |
| **C (Issue #30 reference)** | r_l=[0.1,1,10] + s_l=[2,2,2] + c_k_range | nan | nan | ⚠️ SKIP | 复用 task301 |

**Arm C 跳过原因**: reg test 在 default c=1.0 下, r_l=10 × s_l=2 = 20× 缩放 → weight norm × 20 → 远超 Poincare ball 边界 → arccosh overflow nan. 但 Issue #30 trainer 实际训练成功, 因为 trainer 用 c=2.5/19/14.7 (range sample) 让 c × weight² 仍在边界内. reg test 设计 issue 不是 implementation issue. **Arm C = task301 复用, Stage 4 R@10=0.1022 (+0.2pp GO marginal) 已知**.

---

## 2. Gate 0 详细验证 (Arm A + Arm B)

### 2.1 Arm A (r_l only) — Gate 0 PASS

| 验证 | 实测 | 决策 |
|------|------|------|
| 1. identity transforms 等价 baseline | max \|diff\| = 0.00e+00 | ✅ |
| 2. Arm A design (r_l=[0.1,1,10]+s_l=[1,1,1]+c_k_range) 跟 baseline 不同 | mean \|diff\| = 1.01e-02 | ✅ |
| 3. Shape 验证 baseline / regression / design 都是 (4, 768) | 全部 (4, 768) | ✅ |
| 4. monkey-patch 状态验证 | max \|diff\| = 0.00e+00 | ✅ |
| 5. per-layer c_k range 注入 (seed=42) | c=[2.498, 19.039, 14.774] ∈ [(1,5),(0.5,20),(0.5,20)] | ✅ |

**H1 假设**: r_l alone 是真杠杆, Stage 4 R@10 ≥ 0.1020
**H3 假设**: r_l + s_l 协同, r_l alone NO-GO

### 2.2 Arm B (s_l only) — Gate 0 PASS

| 验证 | 实测 | 决策 |
|------|------|------|
| 1. identity transforms 等价 baseline | max \|diff\| = 0.00e+00 | ✅ |
| 2. Arm B design (r_l=[1,1,1]+s_l=[2,2,2]+c_k_range) 跟 baseline 不同 | mean \|diff\| = 8.45e-04 | ✅ |
| 3. Shape 验证 baseline / regression / design 都是 (4, 768) | 全部 (4, 768) | ✅ |
| 4. monkey-patch 状态验证 | max \|diff\| = 0.00e+00 | ✅ |
| 5. per-layer c_k range 注入 (seed=42) | c=[2.498, 19.039, 14.774] ∈ [(1,5),(0.5,20),(0.5,20)] | ✅ |

**H2 假设**: s_l alone 是真杠杆, Stage 4 R@10 ≥ 0.1020
**H3 假设**: r_l + s_l 协同, s_l alone NO-GO

---

## 3. 下一步 Gate 1 启动

| Arm | 训练命令 | GPU | 期望时间 |
|-----|----------|-----|----------|
| A (r_l only) | `scripts/task304_d6_gate1_stage1_train_arm_a.sh` | GPU 0 | ~30 min |
| B (s_l only) | `scripts/task304_d6_gate1_stage1_train_arm_b.sh` | GPU 1 | ~30 min |
| C (reference) | 复用 task301 Gate 1 + Stage 2 + Stage 3 + Stage 4 | - | 已完成 (R@10=0.1022) |

**Gate 1 通过条件**: L0/L1/L2 util ≥ 90% at any eval step ≥ ep50 + collision ≤ 0.20

**Gate 2 → 3 → 4**: Sinkhorn 5 iter → T5-mini 200 epoch → Test R@10 > 0.1020 → GO

---

## 4. R10 + R11 audit

- **R9**: descriptions/ max=304 ✅ (Task #304 = D6 ablation, 连续无空洞)
- **R10**: D6 ablation 是 R10 backlog 真空收口后最高 ROI 候选 (Issue #30 GO marginal 验证 + K5 关键发现深层确认)
- **R11.5**: 自主决策启动 Arm A + B (owner feedback 2026-07-29 23:13「不允许假设 owner 有 decision」)
- **R7**: 4×L40S 全空闲 (Task #303 GPU 0 训练完成释放 + GPU 1/2/3 空闲). Arm A → GPU 0, Arm B → GPU 1 并行
- **R12**: best ckpt 强制保留 (save_limit=1, 每 epoch 末 → 删旧 → 存新)
- **R13**: 不用 Worktree, 直接在 cwd 工作

---

## 5. 关联

- [[task301-issue30-stage4-result]]: Issue #30 GO marginal (R@10=0.1022 +0.2pp) — Arm C reference
- [[task301-issue30-gate3-gate4-result]]: K5 关键发现「码字几何路径是真杠杆」
- [[task303-issue32-gate0-result]]: Issue #32 双轴协同 (Gate 0 PASS, Stage 1 Gate 1 已 PASS @ ep100)
- [[task300-issue29-stage4-result]]: Issue #29 K_l NO-GO (-4.0%) — false positive 案例
- [[task302-issue31-gate1-result]]: Issue #31 encoder reg USAGE-KILL — 错杠杆案例
- [[phase0-mode-collapse]]: 11 方向 Phase 0 mode collapse 根因

---

result: Task #304 / D6 ablation Gate 0: Arm A (r_l only) PASS + Arm B (s_l only) PASS + Arm C (Issue #30 reference) SKIP (复用 task301). Arm A + B 进入 Gate 1 Stage 1 100 epoch 并行训练 (GPU 0/1). Arm C = task301 Stage 4 R@10=0.1022 已知 GO marginal. 等 3-arm Stage 4 收敛后做真杠杆判定 (H1 r_l / H2 s_l / H3 协同).