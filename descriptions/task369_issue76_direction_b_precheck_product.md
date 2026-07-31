# Task #369 / Issue #76 [方向B 预检] product 三分量零训练梯度证据

**日期**: 2026-07-31
**触发**: Issue #76 [方向B 预检] product 三分量零训练梯度证据 — R16 强制 GitHub OPEN 处理
**前置**: Issue #73 闭环完成 (task366 旧 verdict, R18 实证, 3/4 autograd PASS, 但 θ_m gradient 缺失)
**任务**: R18 4 维度 vs #73 对比 + 实施 product 三分量 (learned-κ + fixed-hyp + Euclidean + softmax logits) + 零训练 autograd 证据 (扰动 theta/logits 改变 loss, gradient 有限非零, 无 detach)
**结果**: ⏳ 进行中 (task382 跟踪)

---

## 1. R18 4 维度对比 (Issue #76 vs Issue #73)

| 维度 | Issue #73 (三分量 product 合同验证) | Issue #76 (零训练梯度证据) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | precheck 5/5 PASS + 实施 + autograd 3/4 PASS (theta_m gradient 缺失) | 每层 learned-κ + fixed-hyp + Euclidean + w_l = softmax(logits_l) + 扰动 theta/logits 改变 loss + gradient 有限非零 + 无 detach | ❌ 不同 (Issue #76 明确"无 detach"硬约束 + gradient 完整) |
| **D2 实施核心** | FreeCurvMixtureVectorQuantization 子类 | 同上 + explicit no-detach check + gradient 范数报告 | ❌ 不同 |
| **D3 Gate 1 失败机制** | 三分量 product 合同未落到可微路径 | 同 + 任何 detach 都是 implementation failure | ❌ 不同 |
| **D4 引用文献** | arXiv:2307.04514 + ACE-HGNN | arXiv:2307.04514 + ACE-HGNN DOI 10.1109/ICDM51629.2021.00021 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做实验 (R18 强制). Issue #76 强调"无 detach"硬约束 + gradient 有限非零 (vs #73 theta_m gradient 缺失)**.

## 2. Issue #76 关键字段 (per spec)

- 每层必须有 learned 主分量 θ_l → κ_l() + fixed-hyperbolic + Euclidean
- 每层必须有 w_l = softmax(logits_l) 或等价 mixing 权重, logits 进入 optimizer
- 三分量 score 必须在 assignment 前组合, 并进入 commitment/codebook loss 的可微路径
- **零训练证据**: 扰动任一 θ_l 或 logits_l,m 改变 score/loss; **梯度有限非零; 无 detach**
- 禁止: 只在日志 / argmin 后处理 / SID postprocess / fixed-only branch 中体现 mixing
- Gate 1 后续: mixing weights 不长期单分量 >0.98, theta/mixing 非边界静止

## 3. Issue #76 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⏸ STOP (Issue #76 是预检任务) | Issue #76 spec 是预检 + 零训练证据, 不要求 Stage 1 训练 |
| **Gate 1.5 = 预检 + autograd** | ⏳ 进行中 | 实施 + 零训练 autograd (no detach + 完整 gradient) |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

---

result: Issue #76 [方向B 预检 product 三分量] R18 3/4 维度 vs #73 不一致必须做实验. 强调"无 detach"硬约束 + gradient 有限非零 (vs #73 theta_m gradient 缺失). ⏳ 进行中 (task382 跟踪).