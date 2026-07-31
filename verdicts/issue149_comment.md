## Issue #149 R20+R21 强制 4 Gate 详细内容 + commit hash

### Gate 1 (= Stage 1 RQ-VAE / HRQVAE): ❌ FAIL (架构 PASS + 配方 FAIL)
- **关键数据**:
  - **Precheck PASS** ✅: aux_loss → κ grad = [2.45e-5, 5.42e-6, 5.94e-5], mixing grad = [6.15e-4, 4.66e-4, 1.13e-4]; hard SID → κ/mixing grad = [0, 0, 0] (isolated=True)
  - **架构修复实证成功**: 跟 Issue #146 (Task #436) 反例对照成功 (continuous soft-anchor 修了 loss=0 反例)
  - **Control 1000-step (no-aux, 跟 #144 同根因)**:
    - step 200/400/600/800/1000: grad_κ=[0, 0, 0], grad_mixing=[0, 0, 0] (R137 κ lock 复现)
  - **Calibration 1000-step (with continuous-anchor + softplus)**:
    - step 200: loss=1.9774, grad_κ=[4.70e-5, 6.71e-6, 7.49e-6], grad_mixing=[0.007, 0.001, 0.002]
    - step 400: loss=1.9300, grad_κ=[1.62e-5, 4.87e-5, 9.98e-6], grad_mixing=[0.069, 0.011, 0.006]
    - step 600: loss=0.9751, grad_κ=[4.16e-6, 1.45e-5, 2.79e-2], grad_mixing=[0.177, 0.049, 0.075]
    - step 800: loss=0.5598, grad_κ=[2.26e-7, 2.07e-5, 6.72e-3], grad_mixing=[0.090, 0.101, 0.022]
    - step 1000: loss=0.3067, grad_κ=[6.70e-6, 4.65e-6, 6.10e-3], grad_mixing=[0.046, 0.048, 0.017]
    - **loss 显著下降 1.97 → 0.31 (-85%)** ✓ (跟 #146 不同, #146 反例)
  - **最终 κ**: [-0.785, -0.794, -0.693]
  - **最终 grad_κ**: [6.7e-6, 4.7e-6, 6.1e-3] (finite nonzero ✓)
  - **最终 grad_mixing**: [0.046, 0.048, 0.017] (finite nonzero ✓)
  - **两分量贡献 > 0.1**: 1/3 layers (期望 >= 2) ❌
  - **最终 util**: L0=3.13%, L1=1.56%, L2=0.39% (期望 >= 90% ❌)
  - **最终 max_load**: L0=99.90%, L1=99.95%, L2=100% (期望 < 5% ❌)
  - **hard SID round-trip**: True ✓
  - **all loss finite**: True ✓
- **失败原因**:
  1. **核心根因**: soft-anchor continuous 路径 + softplus margin 仍 fail codebook spread. 即便 loss 显著下降 (1.97 → 0.31), codebook 仍 collapse 到 max_load=100%. 跟 #148 同 collapse family
  2. **两分量贡献不达标**: L1/L2 两分量 contribution < 0.1 (期望 >= 2 layers), 说明 mixing 跟 data 不匹配
  3. **soft-anchor 修复 loss=0 反例成功但 usage 仍 fail**: Issue #149 spec 明确要求"全 10 个记录点 κ/mixing grad finite nonzero" ✓ (跟 #146 反例区分), 但 usage/max_load 阈值 (跟 #148 同 pattern)
  4. **跟 #144/#146/#148 共享 collapse family**: 任何"aux loss 让 κ/mixing 拿 grad 但不强制 codebook spread"的 recipe 都失败, encoder z_e 仍聚到中心
- **verdict 路径**: `verdicts/task439_issue149_gate1_fail_v1.md`
- **commit**: `36056e4` (R21 fix: `1a0854f`)

### Gate 2 (= Stage 2 Sinkhorn + dedup): ⏸ STOP per spec
- **原因**: Gate 1 FAIL per spec, 无 SID 产出可推断 Sinkhorn
- **Issue spec 强制**: Gate 1 PASS 前禁止 Gate 2/3/4

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- **原因**: Gate 2 STOP
- **Issue spec 强制**: Gate 3 训练 + 前置条件

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- **原因**: Gate 3 STOP
- **Issue spec 强制**: R@10 阈值 (vs HG-Rec baseline 0.1020) + 标记 [TARGET REACHED] 条件

### 关键产物
- **commit hash**: `36056e4`
- **R21 fix**: `1a0854f` (R21 强制填入 commit hash)
- **push**: origin/main (R15 强制)
- **verdict**: `verdicts/task439_issue149_gate1_fail_v1.md`
- **实施**: `scripts/task439_issue149_continuous_anchor_product.py`
- **8 件套审计**: config.json + SHA256(item_emb=`1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`) + d_mix_graph_proof.json + raw_log + verdict.json + commit
- **整体决策**: ⚠️ PARTIAL — 架构级 PASS (soft-anchor+softplus 修好 loss=0 反例, grad 全程 finite nonzero, 跟 #146 反证) + 配方级 FAIL (两分量贡献不达标 + usage/max_load 不达标). Issue #149 关闭

### R18 4 维度对比 vs #146
- D1 spec: continuous soft-anchor + softplus vs argmin + ReLU — **完全不同**
- D2 实施: continuous soft-anchor (NO argmin 切断) + softplus (NO 0 区间) vs argmin + relu — **完全不同**
- D3 Gate 1 失败机制: 两分量贡献不足 + usage 不达标 vs argmin 切断 + relu 0 grad — **不同 mechanism, 同 collapse family**
- D4 引用文献: arXiv:2307.04514 同文献不同实施

### 累计 NO-GO 收口
- 13 方向 NO-GO 收口 + Issue #145/#146/#147/#148/#149 架构级 PARTIAL = **baseline recipe 路径 + 修复尝试 联合耗尽**
- **新方向候选** (R11.5 决策, 等 owner 拍板): enforce codebook spread (orthogonality / K-means init / dead revival / EMA) + Stage 3 protocol split + 架构 pivot

当前任务已完成，请做下一个任务的指示。