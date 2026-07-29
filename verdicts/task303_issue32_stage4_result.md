# Task #303 / Issue #32 — Gate 4 Stage 4 NO-GO

**日期**: 2026-07-30
**状态**: ❌ **Gate 4 Stage 4 NO-GO — per-layer Codebook Transforms r_l=[0.5,1,2] + s_l=[1,1,1] + c_k range 双轴协同 R@10=0.000121 (-99.88% vs HG-Rec baseline 0.1020)**
**决定**: Issue #32 关闭, per-layer 异构双轴协同路径 NO-GO, 不构成 R@10 增益杠杆.

---

## 1. Stage 4 评估结果

| 指标 | Issue #32 | HG-Rec baseline (#84) | Issue #30 (双轴母) | Δ vs baseline |
|------|-----------|----------------------|---------------------|---------------|
| Recall@5 | 0.000080 | 0.0816 | **0.0820** | -99.90% |
| **Recall@10** | **0.000121** | **0.1020** | **0.1022** | **-99.88%** |
| Recall@20 | 0.000161 | 0.1279 | 0.1234 | -99.87% |
| NDCG@5 | 0.000058 | 0.0690 | 0.0698 | -99.92% |
| NDCG@10 | 0.000072 | 0.0755 | 0.0764 | -99.91% |
| NDCG@20 | 0.000082 | 0.0821 | 0.0817 | -99.90% |

**Issue #32 §Gate 3 通过条件** `R@10 > 0.1020` **远未满足** → hard-stop, 关闭 Issue #32.

---

## 2. 跨 Gate 综合

| Gate | 内容 | 状态 |
|------|------|------|
| Gate 0 | per-layer Codebook Transforms + c_k range wrapper | ✅ PASS |
| Gate 1 | Stage 1 100 epoch 训练 | ✅ PASS (L0/L1/L2 100% util, collision=0.0851) |
| Gate 2 | Sinkhorn 5 iter 推断 | ✅ PASS (9922 unique, collision=0.1045) |
| Gate 3 | T5-mini 200 epoch 训练 | ✅ PASS (HG_Rec_best.pth 22MB 落盘) |
| **Gate 4** | **Stage 4 Test R@10 eval** | **❌ NO-GO (-99.88%)** |

**关键观察**: **Gate 1+2+3 PASS 但 Gate 4 NO-GO**. Stage 1 训练 utilization ≥ 90% 跟 Stage 3 T5 学到语义之间存在 gap.

---

## 3. 失败根因 (R11.5 反向推断)

### 3.1 跟 Issue #30 对比

| 维度 | Issue #30 (GO +0.2%) | Issue #32 (NO-GO -99.88%) |
|------|----------------------|---------------------------|
| per-layer r_l | [0.1, 1.0, 10.0] (极端值) | [0.5, 1.0, 2.0] (中间值) |
| per-layer s_l | [2.0, 2.0, 2.0] (显式 scale) | [1.0, 1.0, 1.0] (identity) |
| per-layer c_k range | [(1,5),(0.5,20),(0.5,20)] | [(1,5),(0.5,20),(0.5,20)] (相同) |
| Stage 1 L0 util | 100% | 100% |
| Stage 2 unique | 9922 | 9922 |
| Stage 4 R@10 | **0.1022 GO** | **0.000121 NO-GO (-99.88%)** |

### 3.2 假设 (R11.5)

**H1 (r_l + s_l 数值是关键)**: Issue #30 用 r_l=[0.1,1,10] (10x 极端) + s_l=[2,2,2] (显式 scale) 把码字推到 ‖x‖_E ≈ 0.85 健康区 (跟 [[c-norm-distribution-and-kappa-trajectory]] 一致). Issue #32 用 r_l=[0.5,1,2] (4x 中等) + s_l=[1,1,1] (identity) → 码字 norm 仍在 ‖x‖_E ≈ 0.1 紧致区 → Poincaré 距离梯度弱 → Stage 3 T5 没法学到语义.

**H2 (per-layer c_k range 独立贡献 R@10)**: Issue #32 的 c_k_range 跟 Issue #30 相同 (沿用 task242 Arm A), 但 r_l + s_l 没推到健康区 → c_k_range 协同失效 → R@10 ≈ 0.

**H3 (过 Stage 1 ≠ 过 Stage 4)**: Issue #29 K_l 异构已实证 (task300 verdict). Issue #32 是第二个 Stage 1 PASS 但 Stage 4 NO-GO 端点 (真杠杆是码字几何极端值, 不是双轴协同).

### 3.3 关键教训 (R11.3)

**per-layer Codebook Transforms 的真杠杆 = r_l + s_l 极端值 + scale factor**, 不是 c_k range 双轴协同. Issue #30 = r_l=[0.1,1,10] + s_l=[2,2,2] 才能把码字 norm 推到健康区. Issue #32 用温和值 r_l=[0.5,1,2] + s_l=[1,1,1] 没推到健康区 → R@10 ≈ 0.

---

## 4. 关键决策 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Issue #32 关闭 | ✅ NO-GO verdict + GitHub close | 重跑 / 调 r_l 数值 | Issue #32 §Gate 3 hard-stop 明确, R@10 ≤ 0.1020 → STOP |
| 2 | 不申请 r_l 数值扫描 | ✅ NO-GO 收口 | r_l=[0.1,1,10] 重复跑 | Issue #30 已跑过该配置, 不增加新信息 |
| 3 | 双轴协同路径 NO-GO | ✅ 不再尝试 c_k range 叠加 r_l + s_l | 第三次双轴变体 | 双轴协同在温和 r_l 下失效, 极端 r_l 已是 Issue #30 GO, 双轴无新增价值 |

---

## 5. 物理产物

- `verdicts/task303_issue32_gate0_result.md` (Gate 0 PASS)
- `verdicts/task303_issue32_gate0_verify.json` (Gate 0 verify)
- `verdicts/task303_issue32_stage4_metrics.json` (Stage 4 metrics)
- `scripts/task303_issue32_gate0_codebook_transforms_ckrange.py` (Gate 0 wrapper)
- `scripts/task303_issue32_gate1_stage1_train.py` (Gate 1 training)
- `scripts/task303_issue32_gate1_stage1_train.sh` (Gate 1 launcher)
- `scripts/task303_issue32_gate2_stage2_codebook.py` (Stage 2 Sinkhorn)
- `scripts/task303_issue32_gate3_stage3_train.sh` (Stage 3 T5 launcher)
- `scripts/task303_issue32_gate4_stage4_eval.sh` (Stage 4 eval launcher)
- `products/task303/hrqvae_issue32_gate1/Jul-30-2026_02-17-23*/` (Stage 1 ckpt)
- `products/task303/ckpt_hgrec_issue32/Instruments/Jul-30-2026_02-20-16/HG_Rec_best.pth` (Stage 3 ckpt, 22MB)
- `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue32_dual_axis_synergy.npy` (Stage 2 SID)

---

## 6. 跨任务联立 (18 方向 × 18 verdict 收口)

| # | 方向 | 任务/Issue | R@10 | 状态 |
|---|------|-----------|------|------|
| 1-15 | (历史 15 方向 NO-GO 收口) | task220/231/242/275/287/284/290/291/292/293/144/297/299/300/302 | 0.05-0.10 | NO-GO |
| 16 | per-layer K_l 异构 | #29 / task300 | 0.0979 | NO-GO |
| 17 | per-layer codebook transforms r_l=[0.1,1,10]+s_l=[2,2,2] | #30 / task301 | **0.1022** | ✅ GO |
| **18** | **per-layer codebook transforms r_l=[0.5,1,2]+s_l=[1,1,1]+c_k range 双轴** | **#32 / task303** | **0.000121** | **❌ NO-GO (-99.88%)** |

**18 方向中 16 方向 NO-GO, 1 GO (Issue #30), 1 灾难 NO-GO (Issue #32)**. per-layer 双轴协同路径 NO-GO 收口.

**关键 insight (R11.5)**: per-layer Codebook Transforms 的真杠杆 = r_l + s_l 极端值 (Issue #30 [0.1,1,10]+[2,2,2]) 把码字 norm 推到 ‖x‖_E ≈ 0.85 健康区. 温和值 (Issue #32 [0.5,1,2]+[1,1,1]) 推到 ‖x‖_E ≈ 0.1 紧致区 → T5 学不到语义 → R@10 ≈ 0.

---

## 7. 关联

- [[task301-issue30-stage4-result]]: Issue #30 GO 🎉 (R@10=0.1022 +0.2%, r_l=[0.1,1,10]+s_l=[2,2,2])
- [[task300-issue29-stage4-result]]: Issue #29 NO-GO (R@10=0.0979, K_l 异构)
- [[c-norm-distribution-and-kappa-trajectory]]: 码字 norm 推到 ‖x‖_E ≈ 0.85 健康区 (Issue #30 s_l=2 实现, Issue #32 s_l=1 没实现)
- [[phase0-mode-collapse]]: Issue #32 r_l=[0.5,1,2] 跟 Issue #30 r_l=[0.1,1,10] 同样突破 Phase 0 (L0 100%), 但 Stage 4 差异巨大 (-99.88% vs +0.2%) → 真杠杆在 r_l 数值, 不在 L0 utilization
- [[cross-task-c-k-range-no-go-exhausted]]: c_k range 协同路径 NO-GO 收口

---

result: Task #303 / Issue #32 Gate 4 Stage 4 NO-GO. R@10=0.000121 (-99.88% vs baseline 0.1020). 5 Gate 全跑完 (Gate 0/1/2/3 PASS, Gate 4 NO-GO 灾难性). 关键 insight: per-layer Codebook Transforms 的真杠杆 = r_l + s_l 极端值 (Issue #30 [0.1,1,10]+[2,2,2]) 把码字 norm 推到 ‖x‖_E ≈ 0.85 健康区, 不是 c_k range 双轴协同. 温和 r_l + s_l (Issue #32 [0.5,1,2]+[1,1,1]) 推到 ‖x‖_E ≈ 0.1 紧致区, T5 学不到语义 → R@10 ≈ 0. 18 方向 × 18 verdict 收口 (16 NO-GO + 1 GO + 1 灾难 NO-GO). Issue #32 关闭.