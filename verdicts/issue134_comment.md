## Issue #134 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ❌ FAIL (5-step audit FAIL)
- 关键数据:
  - `hard_argmin_ok=True` (forward SID 由 stable hyp cost + hard argmin 产生)
  - `domain_ok=True` (baseline margin=0.610, < 1-eps)
  - `loss_changed=True` (0.0078 → 0.0145, 损失对 perturb 变化敏感)
  - `grad_finite_nz=False` (κ grad=0.0, encoder grad=6.45e-5 / 7.79e-5 / 2.59e-5 — κ 锁死导致 κ 无梯度, 不符合 5-step audit κ grad 有限非零)
  - `reg_affects_loss=True` (-0.789 → -0.124, count regularizer 作用有效)
  - `ema_drift_finite=True` (ema_drift=0.134, log→tangent centroid→exp map 闭环有效)
  - `grad_max_per_layer`: L0 `{kappa:0.0, encoder:6.45e-5}`, L1 `{kappa:0.0, encoder:7.79e-5}`, L2 `{kappa:0.0, encoder:2.59e-5}`
- **失败原因**: **R137 κ lock 设计特性** (`kappa_l_raw=0` + `softplus(0)=ln2≈0.693`, `kappa_l=-0.1-0.693≈-0.793` 是常数, 不依赖 `kappa_l_raw` 的 grad). 5-step audit 第 3 项要求 κ grad 有限非零, 而 R137 fix 故意让 κ 不动. 这是 spec 兼容性问题, 不是 script bug.
- 实施: scripts/task426_issue134_hard_ema.py (~330 lines, HardEMAHRQVAE + geodesic_ema_update + count_regularizer)

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Gate 1 FAIL — 无 SID 产出可推断 Sinkhorn
- Issue spec 强制: Gate 2 目标 = Sinkhorn 5 iter + 4-digit unique ≥9500/9922, 前置 = Gate 1 端到端 PASS

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 STOP

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 STOP

### 关键产物
- commit hash: 600c1a3
- push: origin/main
- verdict: verdicts/task426_issue134_gate1_fail_v3.md
- 实施: scripts/task426_issue134_hard_ema.py
- 整体决策: ❌ Gate 1 NO-GO 收口 (5-step audit FAIL: grad_finite_nz=False, R137 κ lock 与 spec κ grad 兼容性冲突)
- 联立 #131/#132/#133 → #134: **任何软/硬正则化路径都被 R137 κ lock 与 5-step audit κ grad 要求冲突**. 下一步必须明确"是否需要 κ grad" 二选一.