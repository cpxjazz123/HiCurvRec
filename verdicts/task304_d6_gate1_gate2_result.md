# Task #304 / D6 ablation — Gate 1+2 PASS (Arm A + Arm B, Arm C SKIP)

**日期**: 2026-07-30
**状态**: ✅ **Arm A + Arm B Gate 1+2 PASS** / Arm C SKIP (复用 Issue #30 reference)
**决定**: 进入 Gate 3 — Stage 3 T5-mini 200 epoch 训练 (Arm A PID 786572 GPU 0, Arm B PID 786591 GPU 3)

---

## 1. 综合结果 (Arm A + Arm B)

| Arm | 配置 | Gate 1 (L0/L1/L2 util + collision) | Gate 2 (Sinkhorn 5 iter + 4-digit unique) | Gate 3 (Stage 3 T5) |
|-----|------|--------------------------------------|--------------------------------------------|---------------------|
| **A (r_l only)** | r_l=[0.1,1,10] + s_l=[1,1,1] + c_k_range | L0/L1/L2 100% / 100% / 100% / collision 0.134 (ep100) ✅ PASS | 3-digit 0.0964 / 4-digit 9922/9922 ✅ PASS | PID 786572 GPU 0 训练中 |
| **B (s_l only)** | r_l=[1,1,1] + s_l=[2,2,2] + c_k_range | L0/L1/L2 100% / 100% / 100% / collision 0.138 (ep100) ✅ PASS | 3-digit 0.0973 / 4-digit 9922/9922 ✅ PASS | PID 786591 GPU 3 训练中 |
| **C (Issue #30 reference)** | r_l=[0.1,1,10] + s_l=[2,2,2] + c_k_range | (复用 task301) | (复用 task301) | (复用 task301 Stage 4 R@10=0.1022) |

---

## 2. Gate 1 详细 (Arm A + Arm B)

### 2.1 Arm A (r_l only) Stage 1 训练轨迹

| Epoch | collision_rate | L0/L1/L2 util | 决策 |
|-------|----------------|---------------|------|
| ep40 | 0.1069 | 100% / 100% / 100% | ✅ |
| ep45 | 0.1163 | 100% / 100% / 100% | ✅ |
| ep50 | 0.1216 | 100% / 100% / 100% | ✅ Gate 1 PASS |
| ep60 | 0.1277 | 100% / 100% / 100% | ✅ |
| ep90 | 0.1308 | 100% / 100% / 100% | ✅ |
| ep100 (final) | 0.1340 | 100% / 100% / 100% | ✅ best_collision 0.0867 |

### 2.2 Arm B (s_l only) Stage 1 训练轨迹

| Epoch | collision_rate | L0/L1/L2 util | 决策 |
|-------|----------------|---------------|------|
| ep10 | 0.1313 | 35.9% / 99.2% / 98.8% | 启动中 |
| ep15 | 0.1063 | 64.1% / 100% / 100% | 上升中 |
| ep20 | 0.0836 | 100% / 100% / 100% | ✅ |
| ep90 | 0.1396 | 100% / 100% / 100% | ✅ |
| ep100 (final) | 0.1377 | 100% / 100% / 100% | ✅ best_collision 0.0836 |

---

## 3. Gate 2 详细 (Arm A + Arm B Sinkhorn 5 iter)

### 3.1 Arm A (r_l only) Sinkhorn

| Stage | Collision | Unique |
|-------|-----------|--------|
| Initial pass | 0.0889 | 9040 / 9922 |
| SK iter 0 | 625 groups | - |
| SK iter 4 | 955 groups | - |
| Final 3-digit | **0.0964** | - |
| 4-digit dedup | - | **9922 / 9922 (100%)** ✅ PASS |

### 3.2 Arm B (s_l only) Sinkhorn

| Stage | Collision | Unique |
|-------|-----------|--------|
| Initial pass | 0.0889 | 9040 / 9922 |
| SK iter 0 | 618 groups | - |
| SK iter 4 | 940 groups | - |
| Final 3-digit | **0.0973** | - |
| 4-digit dedup | - | **9922 / 9922 (100%)** ✅ PASS |

---

## 4. K5 + K6 关键发现跨任务联立

- **K5** (task301 verdict): 码字几何路径是真杠杆, L0 utilization ≥ 90% 是必要非充分条件
- **K6** (task300 verdict): L0 utilization ≥ 90% 不保证 Stage 4 R@10 > baseline
- **D6 ablation 目的**: Issue #30 r_l + s_l 拆分 3-arm 验证 r_l vs s_l 哪个独立贡献 R@10
  - **H1 (r_l alone 真杠杆)**: Arm A Stage 4 R@10 待 Gate 4 验证
  - **H2 (s_l alone 真杠杆)**: Arm B Stage 4 R@10 待 Gate 4 验证
  - **H3 (r_l + s_l 协同, 单独 NO-GO)**: 跟 H1/H2 对照

---

## 5. Stage 4 真杠杆判定 (Gate 4 完成后)

| Stage 4 R@10 | 判定 | 结论 |
|---------------|------|------|
| Arm A R@10 ≥ 0.1020 + Arm B R@10 < 0.1020 | **H1 成立** | r_l alone 是真杠杆 |
| Arm B R@10 ≥ 0.1020 + Arm A R@10 < 0.1020 | **H2 成立** | s_l alone 是真杠杆 |
| Arm A R@10 ≥ 0.1020 + Arm B R@10 ≥ 0.1020 | **H1 + H2 成立** | r_l + s_l 各自是真杠杆 (独立贡献) |
| Arm A R@10 < 0.1020 + Arm B R@10 < 0.1020 | **H3 成立** | r_l + s_l 协同 (单独 NO-GO) |
| Arm C R@10 = 0.1022 (Issue #30 reference) | baseline 已知 | r_l + s_l 组合 GO marginal |

---

## 6. R10 + R11 audit

- **R9**: descriptions/ max=304 ✅
- **R10**: D6 ablation 跟 Issue #32 双任务并行 (R10 backlog 真空收口后最高 ROI)
- **R11.5**: 自主决策启动 (owner feedback 2026-07-29 23:13)
- **R7**: Arm A GPU 0 + Arm B GPU 3 并行 (4×L40S 占用 3/4)
- **R12**: Stage 1 ckpt (best_collision_model.pth) + Stage 2 SID 都已落盘
- **R14**: Task #304 D6 验证 Issue #30 真杠杆

---

## 7. 关联

- [[task301-issue30-stage4-result]]: Issue #30 GO marginal (R@10=0.1022 +0.2pp) — Arm C reference
- [[task303-issue32-gate0-result]]: Issue #32 双轴协同 Gate 0 PASS
- [[task303-issue32-gate2-result]]: Issue #32 Gate 2 Sinkhorn PASS (跟 Arm A/B 同样路径)
- [[task304-d6-gate0-result]]: Task #304 Arm A/B Gate 0 wrapper PASS
- [[task304-d6-r-l-s-l-ablation]]: Task #304 D6 ablation description
- [[phase0-mode-collapse]]: 11 方向 Phase 0 mode collapse 根因

---

result: Task #304 / D6 ablation Gate 1+2 PASS. Arm A (r_l only) + Arm B (s_l only) Stage 1 L0/L1/L2 100% util + Sinkhorn 5 iter 3-digit collision 0.0964/0.0973 + 4-digit 9922/9922 unique. 进入 Gate 3 Stage 3 T5-mini 200 epoch 训练 (Arm A GPU 0 + Arm B GPU 3 并行). 等 Gate 4 Test R@10 完成后做真杠杆判定 (H1 r_l / H2 s_l / H3 协同).