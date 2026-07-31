## Issue #151 R20+R21 强制 4 Gate 详细内容 + commit hash

**Issue**: #151 [方向A Gate1] 精确容量约束硬双曲分配与κ校准联合审计
**Task**: #442
**commit**: `79132cc` (push: origin/main `b238f48..79132cc`)
**verdict**: `verdicts/task442_issue151_gate1_fail_v1.md`

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ⚠️ PARTIAL FAIL per spec (max_load PASS + util 部分 FAIL)
- **关键数据**:
  - Precheck PASS: aux_loss → κ grad = [3.70, 4.08, 4.08] (provides_grad=True); hard SID → κ grad = [0, 0, 0] (isolated=True)
  - Control 1000-step (no-aux): grad_κ=[0, 0, 0] (R137 κ lock 复现)
  - Calibration 1000-step (with pairwise aux + Hungarian capacity):
    - step 0: loss=38.32, grad_κ=[3.70, 4.08, 4.08], util=[0.83, 0.80, 0.65], **max_load=[0.020, 0.012, 0.008]** ✓
    - step 500: loss=28.36, grad_κ=[1.74, 3.25, 3.20], util=[0.86, 0.68, 0.54], max_load=[0.020, 0.012, 0.008] ✓
    - step 900: loss=25.74, grad_κ=[1.44, 2.74, 2.68], util=[0.92, 0.74, 0.55], max_load=[0.020, 0.012, 0.008] ✓
  - **最终 κ**: [-1.327, -1.346, -1.338]
  - **最终 util**: L0=0.9375 ✓ / L1=0.6719 ❌ / L2=0.5820 ❌ (期望 >= 90%, L0 PASS, L1/L2 FAIL)
  - **最终 max_load**: L0=1.95% / L1=1.17% / L2=0.78% (**全部 < 5% PASS** ✅, 跟 #148/#149 100% 完全反例)
  - all batch feasible: True ✓ (Hungarian 100% 可行)
- **失败原因**:
  1. Hungarian 容量约束成功消除 max_load (跟 #148/#149 反例), 但 L1/L2 util 仍不达标
  2. cap = ceil(B/K)+1 太松 (L2 cap=2, 256 码字只能用 512 slot 但 batch=256 实际只需 256 slot)
  3. encoder z_e 仍聚到中心 (跟 #145/#148/#149 共享 collapse family)
- **R18 4 维度路径对比 vs #148**:
  - D1 spec: capacity-hard Hungarian + per-codeword cap vs #148 triplet+diversity EXPAND — 不同
  - D2 实施: linear_sum_assignment (Hungarian) + cap slot expansion vs hinge+64-sample diversity — 不同
  - D3 Gate 1 失败机制: max_load PASS (0.020/0.012/0.008) vs #148 100% — 反例对照成功, 但 L1/L2 util 共享 family
  - D4 引用文献: arXiv:2405.13979 + CrossRef VQ optimization (新检索)
- **verdict 路径**: `verdicts/task442_issue151_gate1_fail_v1.md`
- **commit**: `79132cc`
- **后续**: ⏸ STOP per spec: Gate 1 PARTIAL, util 不达标, 不进入 Gate 2/3/4

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 PARTIAL FAIL (util L1/L2 不达标), 无 SID 产出可推断 Sinkhorn / 训练 T5 / R@K eval

### 关键产物
- **commit hash**: `79132cc`
- **push**: origin/main `b238f48..79132cc`
- **verdict**: `verdicts/task442_issue151_gate1_fail_v1.md`
- **实施**: `scripts/task442_issue151_capacity_hard_hyperbolic.py`
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS (max_load 修复) + 配方级 FAIL (util 部分不达标)
