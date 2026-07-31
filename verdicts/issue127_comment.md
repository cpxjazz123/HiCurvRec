## Issue #127 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ⚠️ PARTIAL PASS (5-step audit PASS, training FAIL)
- 关键数据: domain_ok=True (margin 0.709/0.715/0.687), posterior_changed=True, loss_changed=True, grad_finite_nz=True (kappa grad 5.13e-8/9.22e-8/1.07e-9)
- 失败原因: 30 epoch main USAGE-KILL @ ep5 util=0.012 max_load=1.0 (单码字吞所有 token); 30 epoch control 同样 USAGE-KILL util=0.014
- 实施: scripts/task420_issue127_ball_projection.py (BallProjHRQVAE, ~390 lines)
- verdict 路径: verdicts/task420_issue127_gate1_fail_v2.md
- commit: 3bb3bd6

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Gate 1 PARTIAL (audit PASS, training FAIL) — 无 SID 产出可推断 Sinkhorn
- Issue spec 强制: Gate 2 目标 = Sinkhorn 5 iter + 4-digit unique ≥9500/9922, 前置 = Gate 1 端到端 PASS

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 STOP
- Issue spec 强制: Gate 3 训练 + 前置 = Gate 1+2 PASS

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 STOP
- Issue spec 强制: R@10 > 0.1020 阈值 + 标记 [TARGET REACHED] 条件

### 关键产物
- commit hash: 3bb3bd6
- push: origin/main (pushed 2026-08-01)
- verdict: verdicts/task420_issue127_gate1_fail_v2.md
- 实施: scripts/task420_issue127_ball_projection.py
- 整体决策: ❌ Gate 1 NO-GO 收口 (数值修复成功 — 域投影+acosh 稳定 ✅, 架构坍缩根因仍存在 — β=0.25+VQ 结构 + max_load=1.0)
- 联立 #124 → #127: ball projection 是**必要非充分**修复. 后续方向需要换码字初始化 / Sinkhorn during train / EMA 等架构层修复, 不能再叠投影