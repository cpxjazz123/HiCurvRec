# Task #418 / Issue #125 [方向B Gate1] hard SID + 软后向 mixed-distance posterior 恢复 gate gradient — verdict

**日期**: 2026-08-01
**任务**: 实施 hard argmin forward + per-codeword alpha_l,k = softmax(g_l,k) 进入 soft posterior `softmax(-d_mix/τ)` backward, 验证 alpha gradient 是否能通过 soft bridge 流通
**结果**: ❌ Gate 1 NO-GO (5-step audit grad=NaN, 数值不稳定)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 HRQ-VAE): ❌ FAIL (5-step audit grad=NaN)

**[Pre-check]**: ✅ 数据 OK
- dataset X.shape=(9922, 768), X.norm mean=1.000 ✅

**[5-step functional audit]**:
- d_mix_changed: ❌ False (alpha perturb 没传到 d_mix — 反向路径有问题)
- z_q_soft_changed: ❌ False
- **grad_finite_nz: ❌ False — grad_max_per_layer = [NaN, NaN, NaN]** ← 核心 FAIL
- hard_assign_recorded: ✅ True
- roundtrip_ok: ✅ True (gate_logits save/load 一致)
- two_components_ge_01: 1.000 (init randn*0.3 → 各分量都 > 0.1)

**失败原因 (核心发现 — 数值不稳定)**:
- 跟 #124 同根因: poincare_pairwise 数值不稳定
- per-codeword alpha 三分量 (learn κ hyp / fixed κ=1 hyp / Euclidean) 在混合 d_mix 时, 当 κ→0.793 + norm→1 → NaN
- d_mix_changed=False 是反向特征: alpha 扰动没影响 d_mix → 进一步证明数值崩了
- 修复方向: 同 #124 — c.clamp_min(0.5) + 数值稳定化

**[实施]**:
- HardSoftPosteriorHRQVAE: hard argmin + per-codeword alpha (3 分量) 进入 soft posterior
- d_mix(x, c_k) = Σ_j alpha_l,k,j · d_l,j(x, c_k), 3 components
- poincare_pairwise(z_e, codebook, c) → (B, K) via HG-Rec utils
- 5-step audit: alpha 扰动 + grad check
- 代码: scripts/task418_issue125_hard_soft_posterior.py (~410 lines)

**[后续 30 epoch 训练]**:
- 5-step audit FAIL → per spec "FAIL, 不训练", 直接 exit

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (5-step audit grad=NaN)

---

## 跨方向联立 (R18 v2 4 维度)

| 维度 | Issue #122 (task415, closed) | Issue #125 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | per-codeword alpha + STE detach | **hard SID + soft posterior (softmax(-d_mix/τ)) backward** ✅ |
| **D2 实施核心** | STE detach 让 gate grad=0 | **soft posterior 让 alpha grad 通过 soft qz 流回** ✅ |
| **D3 Gate 失败机制** | argmin + STE 切断 alpha gradient | **poincare_pairwise 数值不稳定 → grad=NaN** ❌ (不同失败根因) |
| **D4 引用文献** | arXiv:2307.04514 | arXiv:2307.04514 ✅ |

**R18 v2 强制结论**: Issue #125 跟 #122 路径**有差异** (soft posterior vs STE detach), 但两者都 Gate 1 FAIL, 根因不同:
- #122: 架构 (argmin + STE 切断 grad)
- #125: 数值 (per-codeword alpha + poincare 数值崩)

→ per-codeword alpha soft posterior **架构上**应该能恢复 grad, 但**数值上**需要稳定的 poincare distance.

### 关键产物
- commit hash: pending (this commit)
- push: origin/main (after push)
- verdict: verdicts/task418_issue125_gate1_fail_v2.md (本文件)
- 实施: scripts/task418_issue125_hard_soft_posterior.py
- 整体决策: ❌ Gate 1 NO-GO (grad=NaN, 数值不稳定)
