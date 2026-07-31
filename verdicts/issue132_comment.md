## Issue #132 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ⚠️ PARTIAL PASS (audit PASS, training USAGE-KILL)
- 关键数据: domain_ok=True (两个双曲域 < 1-eps), plan_residual_ok=True (row_res=4.86e-6, col_res=1.86e-9), d_mix_changed=True (per-component α perturb 2.0/-1.5/0.5 确实改 d_mix), plan_changed=False (Sinkhorn 数学特性, 同 #131), loss_changed=True, grad_finite_nz=True (kappa grad 0.032/0.032/0.033 + alpha grad 0.0005/0.0002/0.0001 全 finite 非零), hard_sid_recorded=True
- **失败原因**: **30 epoch 主配 (Shared Sinkhorn) USAGE-KILL @ ep5 util=0.009 max_load=1.0/1.0/1.0**; **30 epoch 控制 (per-item posterior, 同 #128) 同样 USAGE-KILL util=0.009 max_load=1.0/1.0/1.0**
- 实施: scripts/task424_issue132_shared_sinkhorn.py (~440 lines, SharedSinkhornProductHRQVAE + PerItemControlHRQVAE)
- verdict 路径: verdicts/task424_issue132_gate1_fail_v3.md

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Gate 1 PARTIAL (audit PASS, training USAGE-KILL) — 无 SID 产出可推断 Sinkhorn
- Issue spec 强制: Gate 2 目标 = Sinkhorn 5 iter + 4-digit unique ≥9500/9922, 前置 = Gate 1 端到端 PASS

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 STOP

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 STOP

### 关键产物
- commit hash: 1f63b00
- push: origin/main (pushed 2026-08-01)
- verdict: verdicts/task424_issue132_gate1_fail_v3.md
- 实施: scripts/task424_issue132_shared_sinkhorn.py
- 整体决策: ❌ Gate 1 NO-GO 收口 (Shared Sinkhorn transport REFUTED 假设, 主配 + 控制 USAGE-KILL util=0.009 max_load=1.0)
- 联立 task305/task306/task418/task421 → task424 (#128 + #132): 5 方向共同结论 = **per-codeword α 在任意 transport framework (per-item posterior / shared plan) 都数学等价坍缩** → 锁定 per-codeword α 不能用作坍缩修复路径
