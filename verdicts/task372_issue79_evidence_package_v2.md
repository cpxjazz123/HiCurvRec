# Task #372 / Issue #79 [方向B 预检] product 三分量梯度证据包 — R17 PASS 10/10

**日期**: 2026-07-31
**触发**: Issue #79 [方向B 预检] product 三分量零训练梯度证据包 — R16 强制 GitHub OPEN 处理
**前置**: Issue #76 闭环 (task369 R18 实证 10/10, Product3ComponentVectorQuantization)
**实施**: scripts/task372_issue79_evidence_package.py
**结果**: **R17 Gate 1.5 (预检) PASS 10/10** — 全部 10 个证据 markers 全部 PASS

---

## 1. R18 4 维度对比 (Issue #79 vs Issue #76)

| 维度 | Issue #76 | Issue #79 (本次) | 是否一致 |
|------|------|------|------|
| **D1 spec** | no detach + gradient 有限非零 | 证据包 = 代码路径 + 参数表 + 扰动实验 + 梯度证据 | ❌ 不同 (Issue #79 要求完整证据集合) |
| **D2 实施** | Product3ComponentVectorQuantization | 同 + structured 参数表 + 完整扰动实验 | ❌ 不同 |
| **D3 失败机制** | 三分量 product 合同未落到可微路径 | 同 + 证据包可审计可复现 | ❌ 不同 |
| **D4 文献** | arXiv:2307.04514 + ACE-HGNN DOI | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做实验 (R18 强制). 本次实施完整证据包**.

## 2. 证据包内容 (4 类)

| 证据类型 | 内容 | 文件 |
|------|------|------|
| **代码路径** | Product3ComponentVectorQuantization (继承 FreeCurvVectorQuantization) + per-component κ_lm + logits_l + softmax + 三分量 score 组合 + 软量化无 detach | evidence_package.json → param_table |
| **参数表** | n_e=[64,128,256], e_dim=32, M=3, kappa_max=2.0, fixed_hyp_kappa=1.0, 三分量=learned-κ+fixed-hyp+Euclidean, w_l=softmax(logits_l) | 同 |
| **扰动实验** | (1) P1 logits_l[0,0]+=0.1 接受 (软路径); (2) P2 theta_m[0]+=0.1 → indices diff max=21; (3) P3 embedding[0]+=0.05 → indices diff max=52; (4) P4 reproducible | evidence_package.json → perturbation_results |
| **梯度证据** | L0/L1/L2 logits_l grad=[2.37e-4, 2.22e-2, 2.22e-2], theta_m grad=[7.12e-2, 6.67e-2, 6.67e-2], embedding grad=[4.95e-4, 4.55e-4, 2.19e-4], 全部 finite 非零 | evidence_package.json → gradient_evidence |

## 3. R17 Gate 决策 (4-Gate)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE) — ⏸ STOP per Issue #79 spec
- 状态: STOP (Issue #79 是预检任务, spec 不要求 Stage 1 训练)

### Gate 1.5 (= 预检 + autograd 证据包) — **PASS 10/10**
- 状态: PASS
- 关键数据: 10/10 markers PASS:
  - P1 logits_l perturbation accepted
  - P2 theta_m perturbation 产生 indices diff=21
  - P3 embedding perturbation 产生 indices diff=52
  - P4 reproducible
  - z_g.grad 不为 None
  - L0/L1 logits_l.grad > 0 (2.37e-4 / 2.22e-2)
  - L0/L1 theta_m.grad > 0 (7.12e-2 / 6.67e-2)
  - per_layer_independent (L0↔L1 diff=0.1, L1↔L2 diff=0.1)
- 失败原因: 无失败
- verdict 路径: verdicts/task372_issue79_evidence_package_v2.md (本次)
- commit: pending
- 后续: 如果 owner 想要 Stage 1 实际训练, 可启动 task372_ext (R19 立即推进备选)

### Gate 2 (= Stage 2 Sinkhorn) — ⏸ STOP per spec
- 状态: STOP
- 失败原因: Issue #79 是预检 spec, 不要求 Stage 2

### Gate 3 (= Stage 3 T5-mini) — ⏸ STOP per spec
- 状态: STOP
- 失败原因: Issue #79 是预检 spec, 不要求 Stage 3

### Gate 4 (= Stage 4 R@K eval) — ⏸ STOP per spec
- 状态: STOP
- 失败原因: Issue #79 是预检 spec, 不要求 Stage 4

## 4. 整体决策

**Gate 1.5 预检 PASS 10/10** — Issue #79 证据包完整闭环 (4 类证据齐全 + 10/10 sanity PASS). Issue #79 可关闭.

整体决策: **GO 闭环** (Issue #79 证据包完整交付)

---

result: Issue #79 [方向B 预检 product 三分量梯度证据包] R17 Gate 1.5 PASS 10/10 (R18 4 维度 vs #76 3/4 不一致已实证). 4 类证据齐全: 代码路径 + 参数表 + 扰动 + 梯度. 后续 Gate 1/2/3/4 ⏸ STOP per Issue #79 预检 spec.
