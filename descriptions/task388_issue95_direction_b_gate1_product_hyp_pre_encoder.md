# Task #388 / Issue #95 [方向B Gate1] product 分量 HypPreEncoder 距离公式验证

**日期**: 2026-07-31
**触发**: Issue #95 [方向B Gate1] product 分量 HypPreEncoder 距离公式验证 — R16 + R22 强制立即开工
**前置**: Issue #92 Gate 1 FAIL (commit 0ef9eb9, step1 agree3=100%); Issue #43 HypPreEncoder 机制 PASS (task334)
**任务**: 1. R18 4 维度对比 vs Issue #92; 2. Product3CompHRQVAEWithHypPre + 每 component 接入 HypPreEncoder; 3. 50 epoch 验证 step1 agree3 < 100%; 4. Gate 1 PASS
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #95 vs Issue #92)

| 维度 | Issue #92 (kmeans_init + β=0 product FAIL) | Issue #95 (product 分量 HypPreEncoder 验证) | 一致? |
|------|------|------|------|
| **D1 spec** | kmeans_init=True + β=0 first epoch product 修复 | per-component HypPreEncoder + κ-Stereographic 距离公式 (learned-κ + fixed-hyp + Euclidean) | ❌ |
| **D2 实施** | Product3CompHRQVAE + kmeans_init + β=0 | Product3CompHRQVAEWithHypPre (HypPreEncoder BEFORE encoder, 每个 component 都用) | ❌ |
| **D3 失败机制** | 假设 commitment loss 根因 | 假设输入端几何失真 → expmap0 让三个 component 距离尺度可比 | ❌ (新方向) |
| **D4 引用文献** | arXiv:2307.04514 + DOI 10.1109/ICDM51629.2021.00021 | 同 (weighted product manifolds) | ✅ |

→ R18 3/4 维度不一致, 必须做架构层实证.

## 2. Issue #95 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **per-component HypPreEncoder** | learned-κ + fixed-hyp + Euclidean 三个 component 都用 HypPreEncoder | Issue #95 spec Gate1 |
| **保留 product 结构** | 不删除 learned + fixed + Euclidean 三分量 + mixing logits | Issue #95 spec Gate1 |
| **报告** | step1/step50/final L0/L1/L2 per-component max_load, util, component argmin agreement, pairwise dist, codebook norm, mixing weight, θ/κ, NaN/Inf | Issue #95 spec Gate1 |
| **PASS** | step1 不再 agree3=100%; final L0/L1/L2 util ≥ 90%; mixing weights 非单点塌缩; SID collision ≤ 0.20 | Issue #95 spec Gate1 PASS |
| **FAIL** | step1 agree3 仍 100%, 或 component argmin 一致, 或 final util < 90% | Issue #95 spec Gate1 FAIL |

## 3. Issue #95 Gate 决策

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1** | ⏳ 进行中 |  |
| Gate 2/3/4 | ⏸ STOP per spec |  |

## 4. 实施策略

- 复刻 task334 wrapper 思路, 但 base=Product3CompHRQVAE
- HypPreEncoder BEFORE encoder (整个输入端一次性映射, 三个 component 共享映射后表示)
- 50 epoch + per-step per-component trace
- GPU 1 立即启动 (R7 + R19)

---

result: Issue #95 [方向B Gate1 product HypPreEncoder 验证] R22 + R19 立即开工 (GPU 1). 3 步: 1) Product3CompHRQVAEWithHypPre; 2) 50 epoch 训练 + per-component trace; 3) Gate 1 验证 + commit + push + comment(含 hash) → close. ⏳ 进行中.