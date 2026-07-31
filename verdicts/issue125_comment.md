## Issue #125 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### Gate 1 (= Stage 1 per-codeword alpha soft posterior): ❌ FAIL (5-step audit grad=NaN)

**[Pre-check]**: ✅ 数据 OK
- dataset X.shape=(9922, 768), X.norm mean=1.000

**[5-step functional audit]**:
- d_mix_changed: ❌ False (alpha perturb 没传到 d_mix)
- z_q_soft_changed: ❌ False
- **grad_finite_nz: ❌ False — grad_max_per_layer = [NaN, NaN, NaN]** ← 核心 FAIL
- hard_assign_recorded: ✅ True
- roundtrip_ok: ✅ True (gate_logits save/load 一致)
- two_components_ge_01: 1.000 (init randn*0.3 → 各分量都 > 0.1)

**失败原因 (核心 — 数值不稳定)**:
- 跟 #124 同根因: poincare_pairwise 数值不稳定
- per-codeword alpha 三分量 (learn κ hyp / fixed κ=1 hyp / Euclidean) 在混合 d_mix 时, κ→0.793 + norm→1 → NaN

**[实施]**: HardSoftPosteriorHRQVAE (hard argmin + per-codeword alpha 3 分量 d_mix + soft posterior), scripts/task418_issue125_hard_soft_posterior.py (~410 lines)

**[后续 30 epoch 训练]**: ⏸ SKIPPED per spec (5-step audit FAIL)

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (5-step audit grad=NaN)

### 跨方向联立 (R18 v2 4 维度)
- vs Issue #122: 路径有差异 (soft posterior vs STE detach), 失败根因不同 (#122 架构 / #125 数值)

### 关键产物
- commit hash: 18f5645
- push: origin/main
- verdict: verdicts/task418_issue125_gate1_fail_v2.md
- 实施: scripts/task418_issue125_hard_soft_posterior.py
- 整体决策: ❌ Gate 1 NO-GO 收口 (grad=NaN, 数值不稳定)
