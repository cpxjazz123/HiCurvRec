# Task #305 / Issue #33 — per-item soft-assign on #30 GO 配置 (D8 候选)

**日期**: 2026-07-30
**Issue**: #33 (D8 候选, owner #32 closure verdict)
**任务**: 在 #30 GO 配置 (r_l=[0.1,1,10] + s_l=[2,2,2]) 基础上新增 per-item soft-assign 维度, 验证 R@10 > 0.1022 (严格高于 #30 端点).

---

## 来源

承接 Issue #32 closure comment (2026-07-29 18:01 owner-verdict 5-Gate 闭环 NO-GO, task303 完成) + task304 D6 ablation (r_l alone / s_l alone NO-GO) + owner 关键 R11.5 insight 「per-layer Codebook Transforms 真杠杆 = r_l + s_l 极端值, 不是 c_k range 双轴协同」+ owner #32 closure D8 verdict 「per-item soft-assign 替代 argmin hard-assign」.

**Issue #33 申请 GPU** (5-Gate 全跑):
- Gate 0: 代码实现 + 双回归测试 (退化到 baseline + 复现 #30 端点) (0 GPU)
- Gate 1: Stage 1 100 epoch 训练 (GPU 1)
- Gate 2: Sinkhorn 5 iter 推断 (cheap)
- Gate 3: Stage 3 T5-mini 200 epoch 训练 (GPU 2)
- Gate 4: Stage 4 Test R@10 > 0.1022 (GPU 3)

**硬停止 (每道 Gate 都是 hard-stop, 不是参考通过线)**:
- Gate 0 FAIL → STOP
- Gate 1 (a)(b)(c)(d)(e) 任一不满足 → STOP
- Gate 2 unique < 9500 → STOP
- Gate 3 NaN / 早期 early_stop → STOP
- Gate 4 R@10 ≤ 0.1022 → STOP, 关闭本 issue

---

## 5-Gate 综合协议

**Gate 0 代码实现 + 双回归测试**:
- 继承 task301 wrapper (PerLayerCodebookTransformHRQVAE)
- 新增 per-item soft-assign 模块 (方案 A 默认: per-item softmax over negative distances, 温度 τ_l = baseline default)
- 双回归测试:
  1. r_l=[1,1,1]/s_l=[1,1,1]/per-item-soft-assign OFF → matches baseline
  2. r_l=[0.1,1,10]/s_l=[2,2,2]/per-item-soft-assign OFF → matches task301 Issue #30 端点 (R@10=0.1022 GO)

**Gate 1 Stage 1 100 epoch**:
- 沿用 #30 GO 端点 r_l=[0.1,1,10] + s_l=[2,2,2] + per-item soft-assign 打开
- L0/L1/L2 ≥ 90% util + collision ≤ 0.20 + ‖x‖_E ∈ [0.7, 0.95] norm 健康区

**Gate 2 Sinkhorn 5 iter**: 4-digit SID unique ≥ 9500

**Gate 3 Stage 3 T5-mini 200 epoch**: 稳定 (no NaN, no early_stop before ep100)

**Gate 4 Stage 4 Test R@10**: **严格 > 0.1022** (高于 #30 端点 = per-item soft-assign 贡献 R@10 增益)

---

## 与 #28 / #29 / #30 / #31 / #32 关系

| Issue | 路径 | 状态 |
|-------|------|------|
| #28 | per-layer 异构 Gumbel-Softmax τ_l (全局扰动) | NO-GO |
| #29 | per-layer 异构 K_l | NO-GO |
| #30 | per-layer 异构 r_l + s_l 极端值 (唯一 GO) | R@10=0.1022 GO |
| #31 | per-layer 异构 encoder regularization | NO-GO |
| #32 | r_l + s_l + c_k range 双轴 (温和毁坏 #30 杠杆) | 灾难 NO-GO |
| **#33** | **#30 GO 配置 + per-item soft-assign (D8 候选, per-item 个性化 ≠ #28 全局扰动)** | **待实证** |

**关键区别**:
- #28: per-layer Gumbel-Softmax (全局同分布 over codebook, committing the codebook globally) → NO-GO
- #33: per-item 软分配 (per-item 个性化 over codebook, committing per item not globally) → 待实证

---

## 禁止方向

- ❌ 不重蹈 #28 全局扰动路径 (per-layer Gumbel-Softmax 同分布)
- ❌ 不重蹈 #29 K_l 路径
- ❌ 不重蹈 #31 encoder regularization 路径
- ❌ 不重蹈 #32 温和 r_l + s_l 路径 (必须保留 #30 GO 极端值配置)
- ❌ 不引入 multi-seed verification (D7 禁试 per [[user-no-multiseed-override]])
- ❌ 不引入 dead_revive (task242 Arm A+ 已证伪 + task283 hook no-op)

---

## 产物

- descriptions: `descriptions/task305_issue33_per_item_softassign.md` (本文件)
- scripts: `scripts/task305_issue33_gate{0_per_item_softassign,1_stage1_train,2_stage2_codebook,3_stage3_train,4_stage4_eval}.{py,sh}`
- verdicts: `verdicts/task305_issue33_gate{0,1,2,3,4}_*.md`

---

## R9 编号 +1 check

descriptions/ max = 304, 本任务 = 305. ✅