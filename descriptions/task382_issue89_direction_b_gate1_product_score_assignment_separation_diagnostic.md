# Task #382 / Issue #89 [方向B Gate1] product score → assignment 坍缩分离诊断

**日期**: 2026-07-31
**触发**: Issue #89 [方向B Gate1] product score → assignment 坍缩分离诊断 — R16 + R22 强制立即开工
**前置**: Issue #85 CLOSED (commit 09f78a8, 三分量 product Stage1 FAIL, mixing 健康但 codebook 坍缩)
**任务**: 1. R18 4 维度对比 vs Issue #85; 2. 10 epoch 训练 + per-component trace; 3. 三选一判定 (component/mixing/codebook-loss collapse); 4. Gate 1 验证
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #89 vs Issue #85)

| 维度 | Issue #85 (三分量 product 训练 FAIL) | Issue #89 (三分量 product 分离诊断) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | 50 epoch 训练 + mixing 学习 + 最终 util | 10 epoch 训练 + per-component trace (component score quantiles / argmin agreement / mixed score top-k margin / mixing weights / θ / κ / pairwise dist / util / collision / grad) | ❌ 不同 (#89 是诊断) |
| **D2 实施核心** | Product3ComponentHRQVAE 50 epoch 完整训练 | Product3ComponentHRQVAE 10 epoch + per-step per-component trace | ❌ 不同 |
| **D3 失败机制假设** | mixing collapse (后被证伪, mixing 健康) | component-level / mixing-level / codebook-loss collapse 三选一 | ❌ 不同 |
| **D4 引用文献** | arXiv:2307.04514 + DOI 10.1109/ICDM51629.2021.00021 | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做三分量 product 分离诊断实证 (R18 强制)**.

## 2. Issue #89 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **10 epoch 训练** | 不需 50 epoch 完整训练, 10 epoch 足够定位 | Issue #89 spec Gate1 |
| **per-component trace** | per-layer × per-component (learned-hyp / fixed-hyp / Euclidean) score quantiles, argmin agreement rate, mixed score top-k margin, mixing weights | Issue #89 spec |
| **3 选 1 判定** | component-level collapse / mixing-level collapse / codebook-loss collapse | Issue #89 spec |
| **PASS** | 明确给出 3 选 1 判定, L0/L1/L2 trace 完整, 无 NaN/Inf, verdict/commit 可追踪 | Issue #89 spec Gate1 |
| **禁止** | 只看 final util / 无法区分 3 种机制 / mixing detach / theta detach / 未落仓 | Issue #89 spec 框架合规 |

## 3. Issue #89 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 三分量 trace)** | ⏳ 进行中 | 10 epoch 训练 + per-component trace + 3 选 1 判定 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP per Issue #89 spec | Gate 1 PASS 后才允许 |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP per Issue #89 spec | Gate 1+2 PASS 之后 |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per Issue #89 spec | Gate 1+2+3 PASS 之后 |

---

## 4. 实施策略 (R11.5 自主决策)

- 复用 task378 Product3ComponentHRQVAE wrapper, 改成 10 epoch + per-step per-component trace
- 三分量分离 trace: per-component argmin, agreement rate, mixed top-k margin
- 输出: products/task382_issue89_product_separation/trace_per_step.jsonl
- 判定: 哪个先坍缩 (component → mixing → codebook-loss)
- GPU 1 立即启动 (R7 + R19 激进, 4 卡空闲)

---

result: Issue #89 [方向B Gate1 product score → assignment 坍缩分离诊断] R22 + R19 立即开工 (GPU 1). 3 步: 1) 10 epoch 训练 + per-component trace (score quantiles/argmin agreement/mixed top-k margin/mixing/θ/κ/codebook dist/util/collision/grad); 2) 3 选 1 判定 (component-level/mixing-level/codebook-loss collapse); 3) Gate 1 验证 + commit + push + comment(含 hash) → close. ⏳ 进行中.