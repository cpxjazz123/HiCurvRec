# Task #304 / D6 ablation Arm A (r_l only) — Gate 4 Stage 4 NO-GO

**日期**: 2026-07-30
**状态**: ❌ **H1 NO-GO** — r_l alone 不是真杠杆
**关键指标**: Recall@10 = **0.0990** (vs HG-Rec baseline 0.1020, **-2.9pp**)

---

## 1. Stage 4 Test Metrics (Arm A r_l only)

| 指标 | Arm A (r_l only) | HG-Rec baseline | Δ |
|------|------------------|-----------------|---|
| Recall@5 | 0.0825 | 0.0816 | +1.1pp |
| **Recall@10** | **0.0990** | **0.1020** | **-2.9pp** ❌ |
| Recall@20 | 0.1223 | 0.1279 | -4.4pp |
| NDCG@5 | 0.0708 | 0.0690 | +2.6pp |
| NDCG@10 | 0.0761 | 0.0755 | +0.8pp |
| NDCG@20 | 0.0820 | 0.0821 | -0.1pp |

---

## 2. D6 ablation 真杠杆判定 — H3 协同 CONFIRMED

| 配置 | Recall@10 | 判定 |
|------|-----------|------|
| HG-Rec baseline | 0.1020 | reference |
| Arm C (Issue #30 r_l+s_l) | 0.1022 | +0.2pp GO marginal |
| **Arm A (r_l only)** | **0.0990** | **-2.9pp NO-GO** |
| Arm B (s_l only) | 0.0943 | -7.5pp NO-GO |

- **H1 (r_l alone 真杠杆)**: ❌ **NO-GO** (R@10=0.0990 < 0.1020)
- **H2 (s_l alone 真杠杆)**: ❌ **NO-GO** (R@10=0.0943 < 0.1020)
- **H3 (r_l + s_l 协同, 单独 NO-GO)**: ✅ **CONFIRMED** — 必须 r_l + s_l 协同才达 0.1022 marginal GO

**关键发现**: Issue #30 的 R@10=0.1022 边际 GO 不是单一杠杆驱动, 而是 **r_l + s_l 协同效应**. 拆开后 r_l 单独 -2.9pp / s_l 单独 -7.5pp, 协同 +0.2pp. **协同效应是真杠杆的机制**.

---

## 3. 物理产物

- **best_ckpt**: `products/task304/ckpt_hgrec_arm_a/Instruments/Jul-30-2026_02-29-04/HG_Rec_best.pth` (22MB)
- **Stage 3 训练**: PID 786572 GPU 0, early stop @ epoch ~75-80 (val NDCG 平台期)
- **Stage 4 评估**: 03:50:45-03:51:31 (~46 sec)
- **结果 JSON**: `verdicts/task304_d6_stage4_arm_a_metrics.json`

---

## 4. K9 关键发现 (D6 ablation)

| 发现 | 内容 | 启示 |
|------|------|------|
| **K9a** | r_l alone R@10=0.0990 (-2.9pp) | r_l 不单独工作 |
| **K9b** | s_l alone R@10=0.0943 (-7.5pp) | s_l 不单独工作 |
| **K9c** | r_l + s_l R@10=0.1022 (+0.2pp) | 协同是真杠杆 |
| **K9d** | r_l vs s_l 不对称 (-2.9 vs -7.5) | s_l 单独退化更严重, r_l 部分承担 |

**联立 K5/K6/K7/K8/K9**:
- K5 (task301): 码字几何路径是真杠杆
- K6 (task300): L0 ≥ 90% 不充分
- K7 (task302): Phase 0 mode collapse 关键是码字几何
- K8 (task302): 后续候选必须在架构层
- **K9 (D6 ablation)**: 即使架构层, r_l vs s_l 必须协同才能边际改善 (单一不工作)

---

## 5. D6 ablation 最终 verdict 联动

等 Task #303 / Issue #32 (r_l + s_l + c_k_range 三轴) Stage 4 完成后合并写最终 verdict:
- Issue #32 三轴 R@10 vs Arm C 双轴 R@10=0.1022 → 验证 c_k_range 增量贡献
- 如 Issue #32 R@10 ≥ 0.1022 → 边际 GO 同等水平, c_k_range 不增杠杆
- 如 Issue #32 R@10 > 0.1022 → 三轴协同优于双轴, c_k_range 是 marginal 真杠杆

---

## 6. R10 + R11 + R12 audit

- **R7 GPU 不抢卡**: Stage 4 eval Arm A 启动时 GPU 2 空闲 (Arm A 训练停止释放 GPU 0). 立即利用.
- **R10 主动推进**: Arm A Stage 3 early stop 后**立即** launch Stage 4 eval (不等 5 min cron tick).
- **R11.5 自主决策**: launch decision 在 R10 框架下自主执行.
- **R12 ckpt 复用**: best_ckpt 从 Stage 3 直接加载.

---

result: Task #304 D6 ablation Arm A (r_l only) Gate 4 Stage 4 **NO-GO**. Recall@10 = 0.0990, -2.9pp vs HG-Rec baseline 0.1020. **H1 (r_l alone) 不成立**. D6 ablation **H3 协同 CONFIRMED**: r_l + s_l 必须协同 (单独都不行). Issue #30 的边际 GO (R@10=0.1022) 是 r_l + s_l 协同效应, 不是单杠杆. 等 Task #303 / Issue #32 Stage 4 完成后合并写 D6 + Issue #32 最终 verdict.