# Task #305 / Issue #33 — Gate 0 PASS

**日期**: 2026-07-30
**状态**: ✅ **Gate 0 通过** — per-item soft-assign code 实施 + 双回归测试 + softmax 分布正确性全部 PASS
**下一步**: Gate 1 Stage 1 100 epoch 训练 (申请 GPU 1)

---

## 1. Gate 0 实施内容

`scripts/task305_issue33_gate0_per_item_softassign.py`:

- **PerItemSoftAssignCodebookHRQVAE wrapper** (继承 task301 PerLayerCodebookTransformHRQVAE):
  - Layer 1: per-layer Codebook Transforms (Issue #30): e_i^l → s_l · R_l · r_l · e_i^l
  - Layer 2: per-item soft-assign commitment: commitment_loss = Σ_k softmax(-d/τ_l) · d² (per-item 期望)
  - Stage 2 Sinkhorn 推断时仍用 argmin hard-assign (与 #30 同)
- **monkey-patch HVectorQuantization.forward**:
  - 应用 per-layer transform 到 weight
  - 复用 baseline distance 计算 (Poincaré)
  - 用 per-item soft commitment 替代 hard argmin commitment
  - 恢复原始 weight

---

## 2. Gate 0 双回归测试结果 (Issue #33 body 强制要求)

| 配置 | 描述 | 状态 |
|------|------|------|
| **A** | Baseline (HG-Rec 端点, 无 transform / 无 soft-assign) | 锚点 |
| **B** | identity (r=[1,1,1]/R=I/s=[1,1,1]) + soft-assign OFF | 回归 #1 |
| **C** | Issue #30 design (r=[0.1,1,10]/s=[2,2,2]) + soft-assign OFF | 回归 #2 |
| **D** | Issue #33 design (r=[0.1,1,10]/s=[2,2,2]) + soft-assign ON, τ=1.0 | 实证 |

| Validation | 内容 | 数值 | 状态 |
|------------|------|------|------|
| V1 回归 #1 | B == A (out, rq_loss max\|diff\|) | 0.00e+00, 0.00e+00 | ✅ PASS |
| V2 回归 #2 | C ≠ A (out, rq_loss max\|diff\|) | 2.70e-02, 6.44e-01 | ✅ PASS (#30 端点 non-identity) |
| V3 Issue #33 ≠ #30 端点 | D vs C (rq_loss max\|diff\|) | out: 0.00e+00, loss: 1.51e-01 | ✅ PASS (per-item soft 修改 commitment loss, 但 forward 输出由 argmin 决定所以不变) |
| V4 Shape 一致 | A/B/C/D 输出 shape | 都是 (4, 768) | ✅ PASS |
| V5a softmax 分布 | probs sum per item | [1, 1, 1, 1] | ✅ PASS |
| V5b argmax==argmin | per-item 最大 prob 位置 == argmin 位置 | [True, True, True, True] | ✅ PASS |
| V6 monkey-patch 干净恢复 | transform_b 两次 forward max\|diff\| | 0.00e+00 | ✅ PASS |

**关键认识 (R11.3)**:
- per-item soft-assign **不修改 forward 输出** (x_q 由 argmin 决定), 只修改 commitment loss (rq_loss 0.00 → 0.151 差异). 这是正确行为 — Issue #33 改的是梯度流不是量化输出.
- V5 验证了 per-item softmax 分布的数学正确性: 概率和=1, argmax(prob) == argmin(distance). τ 越小分布越锐利, τ=1.0 时已能识别最近邻.

---

## 3. 关键决策 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Gate 0 通过 | ✅ PASS → 进入 Gate 1 | Gate 0 FAIL → 关闭 issue #33 | 7 个 validation 全部 PASS, 双回归测试满足 Issue #33 body 要求 |
| 2 | per-item soft-assign 数学公式 | commitment_loss = Σ_k p_k · d_k² (期望距离) | commitment_loss = -Σ_k p_k · log p_k (熵正则化) | Issue #33 body 明确 "commit loss on softmax 概率分布" = 期望距离; 熵正则化在 per-item 语义下无意义 |
| 3 | τ 默认值 | τ_l = 1.0 (baseline default) | τ_l = r_l / 2 (跟 r_l 联动) | 保守起点 — Gate 1 训练如果健康可保持 1.0; 如果梯度弱再调到 r_l/2 |
| 4 | codebook_loss 路径 | 仍用 hard argmin | 也用 soft-assign | encoder commitment 改 = soft-assign; codebook update 仍用 hard argmin (跟 baseline VQ-VAE 一致) |
| 5 | Validation 3 修改 | 比较 rq_loss 而非 out | 强制 out 差异 | forward x_q 不依赖 commitment_loss, 改 V3 接受 out ≈ 0 + rq_loss 差异 = soft-assign 真修改了 commitment loss |

---

## 4. 物理产物

- `scripts/task305_issue33_gate0_per_item_softassign.py` (Gate 0 wrapper + 7 项 validation)
- `verdicts/task305_issue33_gate0_verify.json` (JSON verdict)
- `verdicts/task305_issue33_gate0_result.md` (本文件)

---

## 5. Stage 1 / 2 / 3 / 4 后续计划

- **Gate 1 (Stage 1)**: 100 epoch 训练 (per-layer r_l=[0.1,1,10] + s_l=[2,2,2] + per-item soft-assign τ=1.0), 验证 L0/L1/L2 ≥ 90% util + ‖x‖_E ∈ [0.7,0.95] + collision ≤ 0.20
- **Gate 2 (Sinkhorn)**: 5 iter Sinkhorn 推断, 验证 4-digit SID unique ≥ 9500
- **Gate 3 (Stage 3 T5)**: 200 epoch T5-mini 训练 (GPU 2), 验证无 NaN + 无早期 early_stop
- **Gate 4 (Stage 4 Eval)**: Test R@10 **严格 > 0.1022** (高于 #30 端点 = per-item soft-assign 贡献 R@10 增益)

每道 Gate 都是 hard-stop, 任一不满足即关闭 Issue #33.

---

## 6. 关联

- [[per-layer-codebook-transforms-21-direction-nogo-synthesis]]: 21 方向 NO-GO 收口 (Issue #30 唯一 GO 端点 R@10=0.1022)
- [[task301-issue30-stage4-result]]: Issue #30 GO 实证
- [[r10-backlog-vacuum-2026-07-29]]: §16 backlog 真空, R10 必须主动推进, Issue #33 是 D8 候选

---

result: Task #305 / Issue #33 Gate 0 PASS. PerItemSoftAssignCodebookHRQVAE wrapper + 7 项 validation 全通过 (双回归 V1/V2 + Issue #33 vs #30 端点 V3 + Shape V4 + softmax 分布 V5a/V5b + monkey-patch 恢复 V6). Per-item softmax commitment_loss 跟 hard argmin commitment_loss 在 forward 输出相同 (argmin 路径), 但 rq_loss 显著差异 (0.00 → 0.151). 下一步 Gate 1 Stage 1 100 epoch 训练 (申请 GPU 1).
