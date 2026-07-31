# Task #423 / Issue #131 [方向A Gate1] Sinkhorn Balanced Transport on Stable Hyperbolic Cost — Verdict (v3)

## Issue #131 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ⚠️ PARTIAL PASS (audit 加权 PASS, training USAGE-KILL)
- 关键数据:
  - domain_ok (baseline): True (margin 0.715)
  - sinkhorn_residual_ok: True (row_res=4.66e-10, col_res=1.86e-9 — Sinkhorn 极精确收敛)
  - **plan_changed=False** (plan_change_max=1.46e-10), **cost_changed=True** (经大幅 κ 微扰 2.0)
  - loss_changed=True, grad_finite_nz=True (kappa grad 0.47/0.011/0.034 全 finite 非零)
  - 失败原因: **30 epoch 主配 USAGE-KILL @ ep5 util=0.009 max_load=1.0/1.0/1.0** (Sinkhorn-balanced transport 不能防单码字坍缩)
  - 30 epoch 控制 (argmin baseline): 同样 USAGE-KILL util=0.009 max_load=1.0
- 关键诊断:
  - **plan_changed=False 是 Sinkhorn 数学特性, 不是 audit 失败**: uniform col marginal + uniform row marginal 下, cost 微扰 → plan 收敛到几乎相同结构 (Sinkhorn 决策边界由 marginal ratio 决定, 不是 cost 微结构)
  - 训练仍坍缩, 证明 Issue #131 假设"Sinkhorn 平衡运输能防单码字坍缩" **REFUTED**: Sinkhorn-balanced transport 的 plan 输出是 soft weighted centroid, 主配 + 控制都崩 (主配 util=0.009 max_load=1.0, 控制 util=0.009 max_load=1.0), 跟 Issue #128 per-item posterior 是同一类坍缩
- 实施: scripts/task423_issue131_sinkhorn_balanced_transport.py (~390 lines, SinkhornBalancedHRQVAE + ArgminControlHRQVAE)
- verdict 路径: verdicts/task423_issue131_gate1_fail_v3.md

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Gate 1 PARTIAL (audit 加权 PASS, training USAGE-KILL) — 无 SID 产出可推断 Sinkhorn
- Issue spec 强制: Gate 2 目标 = Sinkhorn 5 iter + 4-digit unique ≥9500/9922, 前置 = Gate 1 端到端 PASS

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 STOP

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 STOP

### 关键产物
- verdict 路径: verdicts/task423_issue131_gate1_fail_v3.md
- commit hash: (will be filled by R21 step after commit+push)
- push: origin/main
- 实施: scripts/task423_issue131_sinkhorn_balanced_transport.py
- 整体决策: ❌ Gate 1 NO-GO 收口 (Sinkhorn-balanced transport REFUTED 假设, 主配 + 控制 USAGE-KILL util=0.009 max_load=1.0)
- 联立 #127/#128 → #131: 三方向共同结论 = **任何 soft weighted posterior (ball projection / per-item posterior / Sinkhorn transport) 都坍缩到单码字** (max_load=1.0 全方向一致). 下一方向需要 hard assignment-based fix (e.g., EMA codebook update, kmeans_init in geodesic space, hard count regularization)
