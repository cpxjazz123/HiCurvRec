## Issue #124 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### Gate 1 (= Stage 1 HRQ-VAE soft posterior κ bridge): ❌ FAIL (5-step audit grad=NaN)

**[Pre-check]**: ✅ SHA256 verified
- dataset X.shape=(9922, 768), X.norm mean=1.000

**[5-step functional audit]**:
- kappa_changed: ✅ True (-0.7931 → -1.0741 after +0.5 perturbation)
- z_q_soft_changed: ❌ False
- **grad_finite_nz: ❌ False — grad_max_per_layer = [NaN, NaN, NaN]** ← 核心 FAIL
- roundtrip_ok: ✅ True

**失败原因 (核心 — 数值不稳定)**:
- poincare_pairwise 使用 HG-Rec `mobius_add(-x, y, c)` + `artanh(sqrt_c * ||diff||)`
- 当 κ→0.793 (init) + norm→1 时, artanh argument 越界 → NaN 传播到 grad
- 修复方向: c.clamp_min(0.5) 或 arctanh-stable form, 不再直接用 HG-Rec 默认

**[实施]**: HardSoftBridgeHRQVAE (hard argmin forward + soft posterior backward), scripts/task417_issue124_hard_soft_bridge.py (~410 lines)

**[后续 30 epoch 训练]**: ⏸ SKIPPED per spec (5-step audit FAIL)

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (5-step audit grad=NaN)

### 跨方向联立 (R18 v2 4 维度)
- vs Issue #121: 路径有差异 (soft posterior vs STE detach), 失败根因不同 (#121 架构 / #124 数值)
- 联立建议: soft posterior bridge 架构正确, 但需数值稳定的 poincare distance

### 关键产物
- commit hash: 18f5645
- push: origin/main
- verdict: verdicts/task417_issue124_gate1_fail_v2.md
- 实施: scripts/task417_issue124_hard_soft_bridge.py
- 整体决策: ❌ Gate 1 NO-GO 收口 (grad=NaN, 数值不稳定)
