## Issue #128 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ⚠️ PARTIAL PASS (5-step product audit PASS, training FAIL)
- 关键数据: domain_ok=True, posterior_changed=True, loss_changed=True, grad_finite_nz=True (kappa grad 5.56e-5/2.73e-5/7.13e-7, alpha grad 1.34e-4/6.64e-5/7.58e-6 — 全 finite 非零)
- **d_mix 前域断言 (assert_in_ball) 集成成功** — forward pass 任何 NaN/Inf 都 raise, 物理隔离 NaN 传播
- 失败原因: 30 epoch main USAGE-KILL @ ep5 util=0.009 max_load=1.0/1.0/1.0; 30 epoch control 同样 util=0.009
- 实施: scripts/task421_issue128_product_manifold_contract.py (ProductManifoldHRQVAE, ~440 lines)
- verdict 路径: verdicts/task421_issue128_gate1_fail_v2.md
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
- verdict: verdicts/task421_issue128_gate1_fail_v2.md
- 实施: scripts/task421_issue128_product_manifold_contract.py
- 整体决策: ❌ Gate 1 NO-GO 收口 (域断言集成成功 — assert_in_ball 防 NaN 传播 ✅, per-item 数学等价坍缩根因仍存在 — per-codeword α 等价 task305/306 → 码字加权 → 几何中心 → collapse max_load=1.0)
- 联立 task305/task306/task418 → task421: 3 方向共同结论 = **per-item soft posterior 数学等价灾难** (任何 per-item 加权都坍缩到码字加权 → 几何中心 → codebook collapse)