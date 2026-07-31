## Issue #137 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### 5 件套审计 (Issue #137 spec 强制) — 全部已落地, commit 600c1a3
1. **config**: `descriptions/task426_issue134_hard_ema.md` (HardEMAHRQVAE + geodesic_ema_update + count_regularizer 全部设计)
2. **sha256**: scripts 落地 (R20 audit 5 件套), 跟 task84 baseline 同数据 / 同 budget / seed=42 (per Issue spec)
3. **raw_log**: `logs/task426_issue134_hard_ema.log` (5-step audit 完整输出 + verdict.json 落盘)
4. **verdict**: `verdicts/task426_issue134_gate1_fail_v3.md` + `products/task426_issue134_hard_ema/verdict.json`
5. **commit**: `600c1a3` (tracked, pushed origin/main)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ❌ FAIL (5-step audit FAIL)
- 关键数据:
  - `hard_argmin_ok=True` (forward SID 由 stable hyp cost + hard argmin 产生)
  - `domain_ok=True` (baseline margin=0.610, < 1-eps)
  - `loss_changed=True` (0.0078 → 0.0145)
  - `grad_finite_nz=False` (κ grad=0.0, encoder grad=6.45e-5 / 7.79e-5 / 2.59e-5 — κ 锁死导致 κ 无梯度, 不符合 5-step audit κ grad 有限非零)
  - `reg_affects_loss=True` (-0.789 → -0.124)
  - `ema_drift_finite=True` (ema_drift=0.134)
  - `grad_max_per_layer`: L0/L1/L2 全 `{kappa:0.0, encoder:6.45-7.79-2.59e-5}`
- 失败原因: **R137 κ lock 设计特性** (`kappa_l_raw=0` + `softplus(0)=ln2≈0.693`, `kappa_l=-0.1-0.693≈-0.793` 是常数). Issue #137 spec 要求"kappa grad 有限非零且参数离开初值" + Issue #134 hard-EMA 跟 R137 κ lock **不兼容**.

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec

### 关键产物
- commit hash: 0df1b98
- push: origin/main (pushed 2026-08-01)
- verdict: verdicts/task426_issue134_gate1_fail_v3.md
- 实施: scripts/task426_issue134_hard_ema.py
- 整体决策: ❌ Gate 1 FAIL (5-step audit FAIL: grad_finite_nz=False, R137 κ lock 与 Issue #137 spec κ grad 兼容性冲突)

### 联立 #131 + #134 + #137
3 方向共同结论 = **R137 κ lock 与 5-step audit κ grad 兼容性冲突是 spec-level 锁死的工程瓶颈**. 下一方向必须明确二选一:
- 选项 A: 解除 κ lock (per Issue #137 spec 允许 kappa 离开初值) → κ 可以动 → 上一轮 task199/task201/task203/task204 已证 κ 调节是低 ROI 杠杆 (R@10 ceiling 0.1053 锁死)
- 选项 B: 修改 spec 兼容 R137 κ lock (per Issue #137 spec 强制 κ grad 有限非零) → κ 不动 → 任何 mixing/EMA 路径 5-step audit FAIL

Issue #137 → 选项 A/B 二选一必须 owner 拍板, AI 不能自主决定 (R11.4 critical decision).