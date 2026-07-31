## Issue #121 R18+R20 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 强制)

### Gate 1 (= Stage 1 κ 更新 + 径向传输 + 训练): ❌ FAIL (5-step audit grad signal = 0)

**[5-step functional audit]**:
- κ_changed: ✅ True (manual perturbation 改变 kappa_l)
- assignment_consistent: ✅ True (传输前后 assignment 不变 — 径向重标定数学正确)
- ranking_consistent: ✅ True (ranking 不变)
- **grad_finite_nz: ❌ False — grad_max_per_layer = [0.0, 0.0, 0.0]**
- roundtrip_ok: ✅ True

**失败原因 (核心发现)**:
- 5-step audit 中 grad_finite_nz 失败, grad 通过 kappa_l_raw 实际为 **0.0**
- **架构根本问题**: 当前 HRQ-VAE forward 用 `argmin(d_hyp)` + STE (`z_q_st = z_e + (z_q - z_e).detach()`)
- 梯度路径: Loss → decoder(z_q) → z_q = codebook[assign] → argmin (non-diff) → STE detach
- κ 只通过 hyperbolic distance 进入 d_hyp, 但 d_hyp 只用于 argmin, **没有进入 loss 路径**
- 因此 κ 没有任何 gradient signal, optimizer.step() 不会改变 κ
- 这是 #121 提案机制本身的根本问题, 跟 #115/#116/#118 NO-GO 模式同源

**[后续 30 epoch 训练]**:
- 5-step audit FAIL 后, per spec "FAIL, 不训练", 直接 exit

**关键产物**:
- 实施脚本: `scripts/task414_issue121_radial_codebook_sync.py` (~430 lines)
- 径向重标定公式: r = sqrt(κ_old/κ_new), Voronoi-preserving
- per-layer optimizer step hook 实现完成
- 5-step audit (DIRECT perturbation 验证机制)

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (5-step audit grad signal = 0)

### 跨方向联立 (R18 v2 4 维度)
- #115 + #116 + #118 + #121 全部 NO-GO 收口, 共同根因 = **κ 没有 gradient 路径** (argmin + STE detach)
- 任何 κ 参数化变体 (R137 fix tanh / softplus / codebook sync) 都无法绕开

### 关键产物
- commit hash: 4399c5c
- push: origin/main
- verdict: verdicts/task414_issue121_gate1_fail_v2.md
- 整体决策: ❌ NO-GO 收口 (grad=0 架构问题)