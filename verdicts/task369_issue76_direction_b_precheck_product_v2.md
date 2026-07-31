# Task #369 / Issue #76 Gate 1.5 — product 三分量零训练梯度证据 PASS (R18 强制实证)

**日期**: 2026-07-31
**触发**: Issue #76 [方向B 预检] product 三分量零训练梯度证据 — GitHub OPEN
**前置**: Issue #73 闭环 (task366 R18 实证, 3/4 autograd PASS, θ_m gradient 缺失)
**任务**: R18 4 维度 vs #73 对比 + 实施 product 三分量 + 零训练 autograd 证据 (no detach + gradient 有限非零)
**结果**: ✅ **10/10 PASS** (Issue #76 spec 全部硬约束满足)

---

## 1. R17/§19 Gate 决策

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⏸ STOP (Issue #76 是预检任务) | Issue #76 spec 是预检 + 零训练证据, 不要求 Stage 1 训练 |
| **Gate 1.5 = 预检 + autograd** | ✅ **PASS (10/10)** | 全部硬约束满足 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

## 2. R18 4 维度对比 (Issue #76 vs Issue #73)

| 维度 | Issue #73 (三分量 product 合同验证) | Issue #76 (零训练梯度证据) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | precheck 5/5 PASS + autograd 3/4 PASS (θ_m grad 缺失) | no detach + gradient 有限非零 + 完整 | ❌ 不同 |
| **D2 实施核心** | FreeCurvMixtureVectorQuantization | Product3ComponentVectorQuantization (learned 始终走 geodesic) | ❌ 不同 |
| **D3 Gate 1 失败机制** | 三分量 product 合同未落到可微路径 | 同 + detach 是 implementation failure | ❌ 不同 |
| **D4 引用文献** | arXiv:2307.04514 + ACE-HGNN | 同 + DOI 10.1109/ICDM51629.2021.00021 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做实验 (R18 强制)**.

## 3. 实施 + 10 markers (R19 强制)

### 3.1 实施
- Product3ComponentVectorQuantization 子类 (继承 FreeCurvVectorQuantization)
- 三分量: learned-κ (走 κ-Stereographic 始终) + fixed-hyperbolic + Euclidean
- w_l = softmax(logits_l), logits 是 nn.Parameter 进入 optimizer
- 三分量 score = sum_m w_lm * distance_m, 在 assignment 前组合
- 软量化 + soft argmin via softmax(-dist), 全程无 detach
- 关键修复: learned 分量始终走 κ-Stereographic (避免 kappa=0 走 Euclidean 不依赖 kappa)

### 3.2 10 markers 真实数据 (R18 强制)

| Marker | 内容 | 数据 | 结果 |
|------|------|------|------|
| T1 logits gradient 非零 (no detach) | loss_ref.backward() 后 logits_l.grad | 6.34e-6 | ✅ PASS |
| T2 theta gradient 非零 (no detach) | loss_ref.backward() 后 theta_m.grad | 1.0e-2 量级 | ✅ PASS |
| T3 embeddings gradient 非零 (no detach) | loss_ref.backward() 后 embeddings.grad | 1.58e-3 | ✅ PASS |
| T4 第一次 logits perturb 改变 loss | logits_l[0,0] += 0.1 → diff | 4.77e-7 > 1e-8 | ✅ PASS |
| T5 theta perturb 改变 loss | theta_m[0] += 0.1 → diff | 8.96e-5 > 1e-8 | ✅ PASS |
| T6 第二次 logits perturb 改变 loss | re-perturb logits → diff | 4.77e-7 > 1e-8 | ✅ PASS |
| T7 第二次 backward logits grad 非零 | RE-backward logits.grad | 6.23e-6 > 1e-8 | ✅ PASS |
| T8 第二次 backward theta grad 非零 | RE-backward theta.grad | 1.0e-2 量级 | ✅ PASS |
| T9 mixing weights 不塌缩到单一分量 | max(w_l[0]) < 0.98 | [0.356, 0.322, 0.322] | ✅ PASS |
| T10 gradient 全部 finite (无 NaN/Inf) | torch.isfinite(grad).all() | True | ✅ PASS |
| **TOTAL** | | | **10/10 PASS** |

### 3.3 Issue #76 spec 硬约束全部满足

- ✅ 每层 learned 主分量 θ_l → κ_l() + fixed-hyperbolic + Euclidean
- ✅ 每层 w_l = softmax(logits_l), logits_l 是 nn.Parameter 进入 optimizer
- ✅ 三分量 score 在 assignment 前组合, 进入 commitment/codebook loss 的可微路径
- ✅ 扰动任一 θ_l 或 logits_l,m 改变 score/loss
- ✅ Gradient 有限非零
- ✅ 无 detach (全程 .detach() 检查)
- ✅ 不在 argmin 后处理 / SID postprocess / fixed-only branch

## 4. 产物清单

| 路径 | 内容 |
|------|------|
| descriptions/task369_issue76_direction_b_precheck_product.md | 本 description |
| verdicts/task369_issue76_direction_b_precheck_product_v2.md | 本 verdict (R18 实证) |
| scripts/task369_issue76_product_3component_evidence.py | Product3ComponentVectorQuantization 实施 + 10 markers |
| logs/task369_issue76_product_evidence/training.log | autograd 实证日志 |

---

result: Issue #76 [方向B 预检 product 三分量零训练梯度证据] PASS (10/10). R18 3/4 维度 vs #73 不一致已做实证. 实施 Product3ComponentVectorQuantization (learned 始终 geodesic) + softmax(logits_l) + 软量化无 detach. 真实数据: logits/theta/embeddings gradient 非零, perturb 改变 loss, mixing weights 不塌缩, gradient 全 finite. Gate 1.5 ✅ PASS, Gate 1/2/3/4 ⏸ STOP per spec.