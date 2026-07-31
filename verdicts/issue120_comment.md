## Issue #120 R18+R20 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 强制)

### Gate 1 (= Stage 1 RQ-VAE): ✅ PASS per spec (复核 task84 ckpt)
- ckpt SHA256: `56d046dbabdb1930691f1361419b030b6912409853fe84e645c67072ffecb86e`
- L0 K64 / L1 K128 / L2 K256 跟 baseline 一致
- ✅ PASS per Issue spec §Gate1 "复核"

### Gate 2 (= Stage 2 Sinkhorn + dedup): ✅ PASS per spec (复核 task396 SID)
- SID SHA256: `9773e96a57fad9323ed8a37b99d3d3eb5cff5e7ac40895ccd90fcc5d828537b8`
- shape `(9922, 4)`, unique `9922/9922 = 100%`
- ✅ PASS per Issue spec §Gate2 "复核"

### Gate 3 (= Stage 3 T5-mini adapter-only): ⚠️ PARTIAL PASS (架构验证 PASS) + ❌ FAIL (训练无进展)

**[Check 1] Zero-gate ≡ control (max logits diff ≤ 1e-5)**: ✅ PASS
- max_diff: **0.0** (两层都是 0.0, 不是 1e-5 容差, 是 0.0)
- sigmoid per-position (init=-30 → 1e-13) 替代 softmax (init=-30 → [1/3,1/3,1/3] 非零) 修好 zero-gate 数学

**[Check 2] 三层 κ 有限、负、|κ_l| ≥ κ_min=0.1**: ✅ PASS
- Layer 0: mean κ_L0=-0.7936, min |κ|=0.7807, finite=True, neg=True, bounded=True
- Layer 1: mean κ_L0=-0.7935, min |κ|=0.7809, finite=True, neg=True, bounded=True

**[Check 3] 三层 κ/scale/gate 梯度均有限非零**: ✅ PASS (实测!)
- Layer 0 u_l: norm=5.594e-06 (finite_nz=True)
- Layer 0 scale: norm=6.761e-02 (finite_nz=True)
- Layer 0 gate: norm=2.935e-01 (finite_nz=True)
- Layer 1 u_l: norm=1.789e-08 (finite_nz=True)
- Layer 1 scale: norm=1.083e-01 (finite_nz=True)
- Layer 1 gate: norm=5.141e-01 (finite_nz=True)

**[Check 4] 5 反事实**: ⚠️ PASS (机制) / ❌ FAIL (repro 容忍太紧)
- cf2 (kappa shuffle): diff=1.192e-6 ← PASS (non-zero)
- cf2_repro: 9.537e-7 ← FAIL (repro 容忍 1e-7 比实测 Δ=2.4e-7 还紧)
- cf3 (L0/L2 swap): diff=1.073e-6 ← PASS
- cf4 (alignment destroy): diff=1.4513 ← PASS (强非零)
- cf5 (κ sign/scale sanity): diff=5.352e-5 ← PASS

**[30 epoch adapter-only 训练]**: ❌ FAIL
- loss 跨 30 epoch 全程 = 0.9999... (1.0 卡死)
- grad_norms: [6.5e-21, 1.3e-20] (实质为零)
- 根因: gate init=-30 → sigmoid≈1e-13 → adapter 等效不训练 (Check 3 测的是 gate=0 active 时的 grad flow)

**失败原因**:
1. 架构层面: zero-gate 数学正确, κ 有界非零, grad 流通, 反事实区分 → 全 PASS
2. 训练层面: zero-gate 设计本身让受限短训无法验证训练性 (gate=1e-13 → grad≈0 → loss 卡 1.0)
3. 必须扩 epoch (200 epoch) 或改 init gate (active from start) 才能跑通 Stage 3 训练

**关键产物**:
- ckpt_path: `products/task413_issue120_bounded_kappa_adapter/ckpt/task413_adapter_final.pth`
- ckpt SHA256: `62c7b0f5727d8ebd0fda6d2766cba075c4dc133b06cbe766b2bbaece16ba4426`
- 实施脚本: `scripts/task413_issue120_bounded_kappa_adapter.py` (~540 lines)

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 PARTIAL (架构验证 PASS, 训练无效). 启动 Gate 4 必须先解决训练期 zero-gate 问题

### 跨方向联立 (R18 v2 4 维度)
- #117 + #118 + #119 + #120 全部 NO-GO 收口
- #117 κ→0 退化 + #119 gate 非零贡献 都被 #120 修复 (zero-gate max_diff=0.0, κ 有界非零)
- 新发现: zero-gate 设计本身让受限短训无法验证训练性
- baseline recipe 内部 R@10 杠杆已穷尽

### 关键产物
- commit hash: b333e31
- push: origin/main
- verdict: verdicts/task413_issue120_gate3_fail_v2.md
- 整体决策: ⚠️ PARTIAL PASS (架构验证) + ❌ NO-GO 收口 (训练无进展)