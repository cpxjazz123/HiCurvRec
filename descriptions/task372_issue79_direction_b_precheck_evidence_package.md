# Task #372 / Issue #79 [方向B 预检] product 三分量梯度证据包

**日期**: 2026-07-31
**触发**: Issue #79 [方向B 预检] product 三分量零训练梯度证据包 — R16 强制 GitHub OPEN 处理
**前置**: Issue #76 闭环 (task369 R18 实证 10/10, Product3ComponentVectorQuantization + 10 markers)
**任务**: R18 4 维度 vs #76 对比 + 完整证据包 (代码路径 + 参数表 + 扰动实验 + 梯度证据)
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #79 vs Issue #76)

| 维度 | Issue #76 (零训练梯度证据) | Issue #79 (零训练梯度证据包) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | no detach + gradient 有限非零 | 证据包 = 代码路径 + 参数表 + 扰动实验 + 梯度证据 | ❌ 不同 (Issue #79 要求完整证据集合) |
| **D2 实施核心** | Product3ComponentVectorQuantization | 同 + structured 参数表输出 + 完整扰动实验报告 | ❌ 不同 |
| **D3 Gate 1 失败机制** | 三分量 product 合同未落到可微路径 | 同 + 证据包要可审计可复现 | ❌ 不同 |
| **D4 引用文献** | arXiv:2307.04514 + ACE-HGNN DOI | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做实验 (R18 强制). Issue #79 要求"证据包" = 完整证据集合 (代码路径 + 参数表 + 扰动实验 + 梯度证据)**.

## 2. Issue #79 证据包要求

| 证据类型 | 内容 | 来源 |
|------|------|------|
| **代码路径** | Product3ComponentVectorQuantization (继承 FreeCurvVectorQuantization) + per-component kappa_lm + logits_l + softmax + 三分量 score 组合 + 软量化无 detach | scripts/task372 Issue #79 实施 |
| **参数表** | n_e=64/128/256, e_dim=32, M=3, kappa_max=2.0, fixed_hyp_kappa=1.0, 三分量 = learned-κ + fixed-hyp + Euclidean, w_l = softmax(logits_l) | 实施参数 |
| **扰动实验** | (1) logits_l[0,0] += 0.1 → diff=4.77e-7 > 1e-8; (2) theta_m[0] += 0.1 → diff=8.96e-5 > 1e-8; (3) re-perturb logits → diff=4.77e-7 | autograd_evidence 函数 |
| **梯度证据** | logits_l.grad.max()=6.34e-6, theta_m.grad.max()=1.0e-2, embeddings.grad.max()=1.58e-3, 全部 finite | backward() 后 .grad.abs().max() |
| **verdict 路径** | verdicts/task372_issue79_evidence_package_v2.md | 本次 |

## 3. Issue #79 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⏸ STOP (Issue #79 是预检任务) | Issue #79 spec 是预检, 不要求 Stage 1 训练 |
| **Gate 1.5 = 预检 + autograd** | ⏳ 进行中 | 完整证据包实施 + 10 markers + 结构化输出 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

---

result: Issue #79 [方向B 预检 product 三分量梯度证据包] R18 3/4 维度 vs #76 不一致必须做实验. 要求完整证据集合 (代码路径 + 参数表 + 扰动实验 + 梯度证据). ⏳ 进行中.