# Task #417 / Issue #124 [方向A Gate1] hard 前向 + 软后向 κ gradient bridge — verdict

**日期**: 2026-08-01
**任务**: 实施 hard argmin forward + soft posterior `p_l(k|x)=softmax(-d_hyp(κ_l,scale_l)/τ)` backward, 验证 κ gradient 是否能通过 soft bridge 流通
**结果**: ❌ Gate 1 NO-GO (5-step audit grad=NaN, 数值不稳定)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 HRQ-VAE): ❌ FAIL (5-step audit grad=NaN)

**[Pre-check]**: ✅ SHA256 verified
- task84 ckpt: 56d046dbabdb1930691f1361419b030b6912409853fe84e645c67072ffecb86e ✅
- SID: 9773e96a57fad9323ed8a37b99d3d3eb5cff5e7ac40895ccd90fcc5d828537b8 ✅
- dataset X.shape=(9922, 768), X.norm mean=1.000 ✅

**[5-step functional audit]**:
- kappa_changed: ✅ True (after +0.5 perturbation: -0.7931 → -1.0741)
- z_q_soft_changed: ❌ False (perturbation + radial rescale net effect on z_q_soft = 0)
- **grad_finite_nz: ❌ False — grad_max_per_layer = [NaN, NaN, NaN]** ← 核心 FAIL
- roundtrip_ok: ✅ True (codebook rescale < 1.0)

**失败原因 (核心发现 — 数值不稳定)**:
- poincare_pairwise 使用 HG-Rec `mobius_add(-x, y, c)` + `artanh(sqrt_c * ||diff||)`
- 当 κ→0.793 (init) + 1/sqrt_c norm overflow → NaN 传播到 grad
- 即使采用 soft backward, 数值计算本身崩了, 无法验证机制
- 修复方向: 需要 c.clamp_min(0.5) 避免 norm * sqrt_c → 1, 或者用 arctanh-stable form

**[实施]**:
- HardSoftBridgeHRQVAE: hard argmin forward + soft posterior backward
- poincare_pairwise(z_e, codebook, c) → (B, K) proper pairwise distance via HG-Rec utils
- 径向 codebook 同步 + 5-step DIRECT perturbation audit
- 代码: scripts/task417_issue124_hard_soft_bridge.py (~410 lines)

**[后续 30 epoch 训练]**:
- 5-step audit FAIL → per spec "FAIL, 不训练", 直接 exit

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (5-step audit grad=NaN)

---

## 跨方向联立 (R18 v2 4 维度)

| 维度 | Issue #121 (task414, closed) | Issue #124 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | 径向重标定 + STE detach | **hard 前向 + soft posterior bridge** ✅ |
| **D2 实施核心** | STE detach 让 κ grad=0 | **soft posterior 让 κ grad 通过 qz 流回** ✅ |
| **D3 Gate 失败机制** | argmin + STE 切断 κ gradient | **poincare_pairwise 数值不稳定 → grad=NaN** ❌ (不同失败根因) |
| **D4 引用文献** | arXiv:2405.13979 | arXiv:2405.13979 ✅ |

**R18 v2 强制结论**: Issue #124 跟 #121 路径**有差异** (soft posterior vs STE detach), 但两者都 Gate 1 FAIL, 根因不同:
- #121: 架构 (argmin + STE 切断 grad)
- #124: 数值 (poincare distance 在 κ→0.79 + norm→1 时 NaN)

→ soft posterior bridge **架构上**应该能恢复 grad, 但**数值上**需要稳定的 poincare distance (clamp c ≥ 1.0, arctanh 稳定形式).

### 关键产物
- commit hash: pending (this commit)
- push: origin/main (after push)
- verdict: verdicts/task417_issue124_gate1_fail_v2.md (本文件)
- 实施: scripts/task417_issue124_hard_soft_bridge.py
- 整体决策: ❌ Gate 1 NO-GO (grad=NaN, 数值不稳定)
