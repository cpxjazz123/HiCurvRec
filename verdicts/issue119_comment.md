## Issue #119 R18+R20 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 强制)

### Gate 1 (= Stage 1 d_mix gate 因果断言 + 训练): ⚠️ PARTIAL PASS (Phase 0) + ❌ FAIL (Phase 1/2 训练)

**Phase 0 (Functional causality assertion)**: ✅ PASS per Issue spec §Gate1 1
- L0_alpha_before: [0.665, 0.245, 0.090] (softmax([1.0, 0.0, -1.0]))
- L0_alpha_after_perturb: [0.000333, 0.00669, 0.993] (asymmetric perturb [-3.0, 0.0, +5.0]) ✅ α 改变
- L0_d_mix_before: 4.513, L0_d_mix_after: 3.323 ✅ d_mix 改变 (-26.4%)
- L0_loss_before: 0.1515, L0_loss_after: 0.1517 ✅ loss 改变
- L0_assignment_changed: True ✅ argmin 改变
- L0_grad_norm: 3.23e-4 (finite non-zero) ✅ gradient 流通
- L0_assignment_changed_by_perm: True ✅ permuted alpha (swap [1,0,2]) 改变 argmin

**Phase 1 (Main config d_mix gate 30 epoch)**: ❌ FAIL
- best_epoch: 1, best_avg_util: 0.353 (USAGE-KILL @ ep 5)
- final_metrics (ep 5): L0 util=100% ✓, L1 util=3.1% < 90% ❌, L2 util=2.3% < 90% ❌
- L0 max_load=5.4%, L1 max_load=45.2%, L2 max_load=71.8% ❌
- Gate weights stuck at init: c0=0.665, c1=0.245, c2=0.090 (跟 init 一致)
- Gate grad norm: 4.49e-4 (finite non-zero, 但优化器在 mode collapse 区域推动有限)
- κ_l (L0) 卡在 ≈ -0.793 (init, softplus(0) = ln(2))

**Phase 2 (Control uniform gate 30 epoch)**: ❌ FAIL
- best_epoch: 1, best_avg_util: 0.353 (USAGE-KILL @ ep 5)
- 跟 Phase 1 一样 mode collapse (L1/L2 坍缩)
- Gate weights: c0=0.333, c1=0.333, c2=0.333 (uniform, requires_grad=False)

**Round-trip**: ✅ PASS (max_diff=0.9938 < 1.0 tolerance)

**失败原因**:
1. Phase 0 因果断言 PASS — d_mix gate 真的进入 forward path, gradient 链 α→d_mix→loss 通畅
2. Phase 1/2 训练 mode collapse — 跟 #115/#116/#118/#408/#409/#411 联立同模式失败
3. 30 epoch 受限短训 + L0 K64 容量 + L1/L2 训练不足 → mode collapse 是确定性事件, 跟 d_mix 公式无关

**关键产物**:
- ckpt_path: `products/task412_issue119_dmix_gate/ckpt/task412_best_epoch_01.pth`
- main_final_metrics: L0=100%, L1=3.1%, L2=2.3% (USAGE-KILL @ ep 5)
- 实施脚本: `scripts/task412_issue119_dmix_gate_train.py` (~600 lines)

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (L1/L2 util < 90%)

### 关键产物
- commit hash: b333e31
- push: origin/main
- verdict: verdicts/task412_issue119_gate1_fail_v2.md
- 整体决策: ⚠️ PARTIAL PASS (Phase 0) + ❌ NO-GO 收口 (Phase 1/2)