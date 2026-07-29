# Task #304 / D6 ablation Arm B (s_l only) — Gate 4 Stage 4 NO-GO

**日期**: 2026-07-30
**状态**: ❌ **H2 NO-GO** — s_l alone 不是真杠杆
**关键指标**: Recall@10 = **0.0943** (vs HG-Rec baseline 0.1020, **-7.5pp**)

---

## 1. Stage 4 Test Metrics (Arm B s_l only)

| 指标 | Arm B (s_l only) | HG-Rec baseline | Δ |
|------|------------------|-----------------|---|
| Recall@5 | 0.0776 | 0.0816 | -4.9pp |
| **Recall@10** | **0.0943** | **0.1020** | **-7.5pp** ❌ |
| Recall@20 | 0.1134 | 0.1279 | -11.3pp |
| NDCG@5 | 0.0673 | 0.0690 | -2.5pp |
| NDCG@10 | 0.0726 | 0.0755 | -3.8pp |
| NDCG@20 | 0.0775 | 0.0821 | -5.6pp |

---

## 2. 真杠杆判定 (H1/H2/H3 框架)

| 配置 | Recall@10 | 判定 |
|------|-----------|------|
| HG-Rec baseline | 0.1020 | reference |
| Arm C (Issue #30 r_l+s_l) | 0.1022 | +0.2pp GO marginal |
| **Arm B (s_l only)** | **0.0943** | **-7.5pp NO-GO** |
| Arm A (r_l only) | TBD | 待 Gate 4 评估 |

- **H2 (s_l alone 真杠杆)**: ❌ **NO-GO**
  - R@10=0.0943 < 0.1020, s_l 单独使用导致 -7.5% 退化
  - 排除 s_l 作为独立杠杆

---

## 3. 物理产物

- **best_ckpt**: `products/task304/ckpt_hgrec_arm_b/Instruments/Jul-30-2026_02-29-04/HG_Rec_best.pth` (22MB)
- **Stage 3 训练**: PID 786591 GPU 3, early stop @ epoch ~75-80 (val NDCG 平台期)
- **Stage 4 评估**: PID 867718 GPU 2, 03:38:46-03:39:32 (~46 sec)
- **结果 JSON**: `verdicts/task304_d6_stage4_arm_b_metrics.json`

---

## 4. K5 + K6 + K8 联立更新

- **K5** (task301 verdict): 码字几何路径是真杠杆, L0 utilization ≥ 90% 是必要非充分条件
- **K6** (task300 verdict): L0 utilization ≥ 90% 不保证 Stage 4 R@10 > baseline
- **K7** (task302 owner-verdict): Phase 0 mode collapse 的关键是**码字几何**, 不是 encoder 梯度
- **K8** (task302 owner-verdict): 后续候选必须在架构层 (Codebook Transforms 候选路径已实证)
- **新发现 K9 (D6 Arm B)**: s_l 单独使用 NO-GO (-7.5pp). r_l vs s_l 是非对称: r_l 单独可能成立 (待 Arm A 验证), s_l 单独失败.

---

## 5. 后续任务

- **Task #304 Arm A** (r_l only): Stage 3 训练中, PID 786572 GPU 0. Stage 4 eval launcher 已就绪 (GPU 2)
- **Task #303 / Issue #32** (r_l + s_l + c_k_range): Stage 3 训练中, PID 759154 GPU 1. Stage 4 eval launcher 已就绪 (GPU 2)
- **D6 ablation 整体结论**: 等 Arm A + Issue #32 Stage 4 完成后做最终 H1/H2/H3 真杠杆判定

---

## 6. R10 + R11 + R12 audit

- **R7 GPU 不抢卡**: Stage 4 eval Arm B 启动时 GPU 2 空闲 (Task #304 Arm A + B 训练中, GPU 0 + 3; Task #303 GPU 1). 立即利用空闲 GPU 2.
- **R10 主动推进**: Arm B Stage 3 early stop 后**立即** launch Stage 4 eval (不等 5 min cron tick).
- **R11.5 自主决策**: launch decision 在 R10 框架下自主执行, 不等用户拍板.
- **R12 ckpt 复用**: best_ckpt 从 Stage 3 直接加载, 无重训.
- **R14 Issue #32 跟踪**: 仍 OPEN (等 Task #303 Stage 4 完成).

---

result: Task #304 D6 ablation Arm B (s_l only) Gate 4 Stage 4 **NO-GO**. Recall@10 = 0.0943, -7.5pp vs HG-Rec baseline 0.1020. **H2 (s_l alone) 不成立**. Stage 3 best_ckpt + Stage 4 results 已落盘. 待 Arm A + Issue #32 Stage 4 完成后做 D6 ablation 最终判定.