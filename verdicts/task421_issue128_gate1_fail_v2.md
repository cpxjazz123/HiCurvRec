# Task #421 / Issue #128 [方向B Gate1] Product-space 三分量 Poincaré 域合同 — verdict

**日期**: 2026-08-01
**任务**: 修复 #125 NaN 失败, 加每 hyperbolic 分量独立 ball projection + d_mix 前域断言 + per-codeword α
**结果**: ❌ Gate 1 NO-GO (5-step audit 通过 ✅ 但训练 USAGE-KILL @ ep5, util 0.009)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ⚠️ PARTIAL PASS (5-step audit PASS, training FAIL)

**[Pre-check 数据]**: ✅ SHA256 OK
- dataset X.shape=(9922, 768), X.norm mean=1.000

**[5-step product audit — Main config (projection+assertion)]**: ✅ PASS
- domain_ok: ✅ True (per-layer 2 hyp 分量 sqrt(c)·‖x‖ < 1-eps)
- posterior_changed: ✅ True (per-component offset 扰动 softmax-invariant fix 生效)
- loss_changed: ✅ True
- grad_finite_nz: ✅ True (kappa grad 5.56e-5/2.73e-5/7.13e-7, alpha grad 1.34e-4/6.64e-5/7.58e-6 — 全 finite 非零)
- hard_assign_recorded: ✅ True
- ⭐ **d_mix 前域断言 (assert_in_ball) 集成成功** — forward pass 任何 NaN/Inf 都 raise

**[30 epoch 训练 Main config]**: ❌ USAGE-KILL @ ep5
- ep1: loss=0.0040, util=0.012
- ep5: loss=0.0007, util=0.009 ❌
- final util L0/L1/L2 = 0.0156/0.0078/0.0039
- max_load = 1.0/1.0/1.0 (单码字吞所有 token)

**[30 epoch 训练 Control config (no projection+assertion, #125 unstable)]**: ❌ 同样 USAGE-KILL
- final util L0/L1/L2 = 0.0156/0.0078/0.0039
- max_load = 1.0/1.0/1.0

**[失败原因 — 关键发现]**: **3 分量数学等价灾难**:
- task305 + task306 联立已证 per-item soft / per-item weighted sum 等价 → 推动码字聚集到数据几何中心 → codebook collapse
- 本 task (per-codeword α_l,k softmax(g_l,k) + 3 分量) 也是同一机制: 即使 α 3 分量保持差异化 (0.135/0.072/0.793 等), d_mix 仍是 K 个码字的加权距离 → argmin 仍推向数据几何中心
- d_mix 前域断言成功防止 NaN 传播 ✅, 但**不能防止码字几何聚集**

**[实施]**: ProductManifoldHRQVAE (3 分量 per-codeword α_blending + 每 hyp 分量独立 ball projection + d_mix 前 assert_in_ball), scripts/task421_issue128_product_manifold_contract.py (~440 lines)

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 PARTIAL (audit PASS, training FAIL)

---

## 跨方向联立 (R18 v2 4 维度)

| 维度 | Issue #125 (task418, NO-GO) | Issue #128 (本 task, NO-GO) |
|------|------------------------------|------------------------------|
| **D1 spec 摘录** | hard argmin + per-codeword α + 3 分量 | 同样 + 每分量 ball projection + d_mix 前 assertion |
| **D2 实施核心** | HG-Rec `poincare_pairwise` 3 分量叠加放大 NaN | 每 hyp 分量独立 projector + 域断言 + α finiteness check |
| **D3 Gate 1 失败机制** | grad=NaN (数值不稳) | ✅ audit 通过, 训练坍缩 (per-codeword α 等价于码字加权) |
| **D4 引用文献** | α-blending decoder (Gao 2020) | 同 + Nickel-Kiela 2017 ball projection |

**R18 v2 判定**: 数值层面成功 (audit PASS, 域断言集成), 但 per-codeword α 等价于"per-item soft 数学坍缩" (跟 task305/306 联立). 后续方向需走架构层 (Sinkhorn during train / EMA codebook / kmeans_init in geodesic space).

**联立 #125 → #128**: 数值稳定 (域断言集成), 架构坍缩根因 (per-item 等价) 未触及. 跟 #127 一样, projection 是必要非充分.

**联立 task305/task306/task418 → task421**: 3 方向全 NO-GO, 共同结论 = **per-item soft posterior 数学等价灾难** (任何 per-item 加权都坍缩到码字加权 → 几何中心 → collapse).

---

## Gate 1 整体决策

| 检查 | 状态 | 数据 |
|------|------|------|
| 5-step product audit (含域断言) | ✅ PASS | 5/5 检查通过 |
| 30 epoch main util ≥ 0.9 | ❌ FAIL | util=0.009 (≪ 0.9) |
| 30 epoch control util ≥ 0.9 | ❌ FAIL | util=0.009 (≪ 0.9) |
| Gate 1 PARTIAL → STOP | ✅ STOP per spec |

### 关键产物
- commit hash: pending (this commit)
- push: origin/main (after push)
- verdict: verdicts/task421_issue128_gate1_fail_v2.md (本文件)
- 实施: scripts/task421_issue128_product_manifold_contract.py
- verdict.json: products/task421_issue128_product_manifold_contract/verdict.json
- 整体决策: ❌ Gate 1 NO-GO 收口 (域断言集成成功, per-item 数学等价坍缩根因仍存在)