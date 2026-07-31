## Issue #152 R20+R21 强制 4 Gate 详细内容 + commit hash

**Issue**: #152 [方向B Gate1] 样本条件 product 权重与容量硬分配审计
**Task**: #443
**commit**: `79132cc` (push: origin/main `b238f48..79132cc`)
**verdict**: `verdicts/task443_issue152_gate1_fail_v1.md`

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ❌ FAIL per spec (util FAIL + weight entropy 退化)
- **关键数据**:
  - Precheck PASS: aux_loss → κ grad = [3.97e-5, 1.31e-6, 1.73e-5] + weight_mlp grad = [4.36e-3, 2.52e-3, 2.02e-3]; hard SID → κ grad = [0, 0, 0] (isolated=True)
  - Control 1000-step (no-aux): grad_κ=[0, 0, 0] (R137 κ lock 复现)
  - Calibration 1000-step (with sample-conditioned aux + Hungarian capacity):
    - step 0: loss=1.98, util=[0.91, 0.69, 0.52], max_load=[0.020, 0.012, 0.008] ✓, comp=L0:[0.33, 0.33, 0.34] (3 分量均衡)
    - step 500: loss=1.89, util=[0.81, 0.70, 0.61], max_load PASS, **comp=L0:[0.020, 0.020, 0.960]** ⚠️ weight 退化成 one-hot
    - step 900: loss=1.05, util=[0.95, 0.68, 0.53], max_load PASS, **comp=L0:[~0, ~0, 1.0]** + L2:[0.68, 0.14, 0.18]
  - **最终 κ**: [-0.790, -0.809, -0.736]
  - **最终 util**: L0=0.83 / L1=0.67 / L2=0.56 (期望 >= 90%, **L0/L1/L2 全部 FAIL**)
  - **最终 max_load**: L0=1.95% / L1=1.17% / L2=0.78% (**全部 < 5% PASS** ✅, 跟 #149 100% 反例)
  - **最终 sample_weights entropy**: **L0=1.33e-12 ≈ 0 (退化)** / L1=1.086 / L2=1.003 (期望 >= 0.999 ❌, L0 FAIL)
  - **component contribution ≥2 per layer**: L0=False / L1=True / L2=True (期望 3/3 PASS, **1/3 FAIL**)
  - all batch feasible: True ✓
- **失败原因**:
  1. **核心根因**: per-sample weight MLP 退化成 one-hot (L0 entropy 从 step 0 的 1.0989 衰到 step 900 的 1.33e-12, entropy reg α=0.1 力度不够 vs contrastive_loss 推 weight 到单分量)
  2. Hungarian 容量约束成功 (复用 #151), max_load 全程 < 5%
  3. util 仍 fail (encoder 仍聚到中心, 跟 #145/#148/#149/#151 共享 collapse family)
- **R18 4 维度路径对比 vs #149**:
  - D1 spec: per-sample weight MLP + entropy reg + Hungarian capacity vs continuous soft-anchor + softplus + random neg — 不同
  - D2 实施: weight MLP(z_e) → softmax (per-sample α_l ∈ Δ) vs ProductDistanceModel 静态 mixing scalars — 不同
  - D3 Gate 1 失败机制: weight entropy 退化 (L0 ~0) + util 不达标 vs loss=0 + util 不达标 — 不同 mechanism, 同 collapse family
  - D4 引用文献: arXiv:2307.04514 同文献 (同 #149) 不同实施
- **verdict 路径**: `verdicts/task443_issue152_gate1_fail_v1.md`
- **commit**: `79132cc`
- **后续**: ⏸ STOP per spec: Gate 1 FAIL, util + weight entropy 不达标, 不进入 Gate 2/3/4

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (util L0/L1/L2 全部不达标 + weight entropy 退化), 无 SID 产出可推断 Sinkhorn / 训练 T5 / R@K eval

### 关键产物
- **commit hash**: `79132cc`
- **push**: origin/main `b238f48..79132cc`
- **verdict**: `verdicts/task443_issue152_gate1_fail_v1.md`
- **实施**: `scripts/task443_issue152_sample_conditioned_product.py`
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS (Hungarian 修复 max_load) + 配方级 FAIL (util + weight entropy 退化)
