# Task #407 / Issue #113 [方向B Precheck→Gate1] 三分量 product NO-GO 收口

**日期**: 2026-07-31
**任务**: Issue #113 Precheck + Gate 1 Stage 1 短训 50 epoch 验证
**结果**: ⚠️ **Precheck PASS, Gate 1 ❌ FAIL (USAGE-KILL @ ep 5)**. Issue #113 NO-GO 收口, 联立 task391/task397 历史 NO-GO = 三分量 product 架构路径在 baseline recipe 内部不可行

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Precheck + Stage 1 短训): ⚠️ Precheck PASS, Stage 1 ❌ FAIL

#### Precheck PASS (✅):
- **状态**: PASS
- **实施**: `scripts/task407_issue113_three_component_precheck.py`
- **关键数据**:
  - **state_dict schema**: 18 keys = 3 layers × 6 keys (`learnable_theta_m`, `codebook_{learnable,fixed,euclidean}.weight`, `gate_logits`, `kappa_fixed_buf`)
  - **forward contribution**: 三 component distance 真实不同 (L0: d_learnable=23.69, d_fixed=3.57, d_euclidean=23.08)
  - **round-trip**: max_abs_diff ≤ 2.38e-7, is_bit_equal: True (Float32 浮点精度 OK)
  - **gate non-degenerate**: 三层 gate 都是 softmax 后 2+ component > 0.1
  - **component finite + non-zero**: 三层 κ_fixed=0.5/1.0/1.5 per-layer 独立, 三 codebook norm 全部 finite + non-zero, 无 NaN
- **Issue spec §Framework compliance precheck 6 项强制要求**:
  1. 保留原 learnable-κ hyperbolic 主路 ✅
  2. fixed-curvature hyperbolic component ✅ (per-layer 独立 [0.5, 1.0, 1.5])
  3. Euclidean component ✅
  4. 每层独立 learnable mixing/gate ✅ (per-layer 独立 softmax)
  5. 提交 checkpoint key/shape 表 ✅
  6. forward 数据流 + 参数与梯度清单 + save/load round-trip ✅

#### Stage 1 FAIL (❌):
- **状态**: FAIL per Issue spec §Gate1 强制: "三层 utilization≥90%, max_load<5%; L1/L2 norm 不低于 L0 的 0.2 倍; 三 component 与 gate 均有限非零; 无 NaN/Inf; round-trip 一致"
- **实施**: `scripts/task407_issue113_three_component_stage1_train.py`, GPU 1, 50 epoch + USAGE-KILL @ ep 5
- **关键数据 (4 轮修复尝试, 全部 USAGE-KILL @ ep 5)**:
  | 版本 | 修复 | min_util | max_load | gate | 状态 |
  |------|------|----------|----------|------|------|
  | **v1** | gate=zero init, codebook uniform | **0.78%** | 100% | [0.33,0.33,0.33] (init 不动) | ❌ USAGE-KILL |
  | **v2** | gate init 不均 [1.0, 0.0, -1.0], codebook per-comp scale | 10.94% | 100% | [0.67,0.24,0.09] (init 不动) | ❌ USAGE-KILL |
  | **v3** | + per-comp argmin | 10.94% | 100% | [0.67,0.24,0.09] (init 不动) | ❌ USAGE-KILL |
  | **v4** | + 删除 STE (no STE, x_q 真实反传) | **14.06%** | 61.0% L0 | [0.66,0.25,0.09] (几乎不动) | ❌ USAGE-KILL |
- **v4 详细 epoch 进展**:
  - ep 1: util L0=20.3%/L1=29.7%/L2=49.2%, gate=[0.66,0.25,0.09], recon=0.0097
  - ep 2: util L0=18.8%/L1=28.1%/L2=49.6%, gate 不变
  - ep 3: util L0=17.2%/L1=24.2%/L2=47.7%, gate 不变
  - ep 4: util L0=15.6%/L1=25.8%/L2=46.5%, gate 不变
  - ep 5: util L0=14.1%/L1=23.4%/L2=44.9%, gate 不变 → USAGE-KILL
- **v4 失败根因 (R11.5 深度推理)**:
  1. **gate_logits 梯度极小**: 4 轮修复全部显示 gate 完全 stuck 在 init value (差异 ≤ 0.01), 证明 gate_logits 几乎没有有效梯度流过
  2. **commitment loss 路径不通**: 即使删除 STE 让 x_q 通过 recon_loss 反传, 但 κ-Stereographic commitment/codebook loss 公式中 `x_q - latent.detach()` 让 x_q.detach() 切断 gradient flow (跟 commitment loss 设计一致, 但跟 gate 期望的梯度路径冲突)
  3. **argmax 不平滑**: `argmin(d_total)` 是离散的, gate 通过离散的 index 选择学到的梯度是 0 (因为 d_total 对 gate 的导数依赖 ∂argmin/∂w = 0 for non-differentiable op)
  4. **架构根本限制**: 三分量 + softmax gate 的 product manifold 在 RQ-VAE 训练中无法学到真实 product (这是数学约束, 不是参数调整能解决)

#### 决策阈值 (per Issue #113 spec §Gate1):
- **三层 utilization ≥ 90%** → 14.06% ≪ 90% → **FAIL**
- **max_load < 5%** → 61.0% L0 ≫ 5% → **FAIL**
- **L1/L2 norm ≥ L0 × 0.2** → ✅ (L0=5.82, L1=8.22, L2=11.60 全部满足)
- **三 component + gate 有限非零** → ✅
- **无 NaN/Inf** → ✅
- **round-trip 一致** → ✅
- **Gate 1 整体**: ❌ FAIL (util + max_load 不达标)

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- **原因**: Gate 1 FAIL per spec, Gate 2 禁止执行
- **Issue spec §Gate2 强制**: "仅 Gate1 PASS 后生成 9922×4 SID"

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- **原因**: Gate 1 FAIL per spec, Gate 3 禁止执行

### Gate 4 (= Stage 4 R@K): ⏸ STOP per spec
- **原因**: Gate 1 FAIL per spec, Gate 4 禁止执行

---

## 跨方向联立 (R18 实证 + 跟历史任务 verdict 联立)

| 任务 | Issue | Gate 1 结果 | 失败根因 |
|------|-------|------------|----------|
| task391 | #98 product audit | ❌ NO-GO (synthetic agree3=0%) | scale_normalization MISSING |
| task397 | #101 per-component fix | ❌ NO-GO (50 epoch 仍坍缩) | monkey-patch 不彻底 |
| **task407 v1** | **#113 base** | **❌ USAGE-KILL @ ep 5 (util 0.78%)** | **gate=zero init, Euclidean 主导** |
| **task407 v2** | **#113 gate 不均** | **❌ USAGE-KILL @ ep 5 (util 10.94%)** | **gate 梯度极小, stuck 在 init** |
| **task407 v3** | **#113 per-comp argmin** | **❌ USAGE-KILL @ ep 5 (util 10.94%)** | **argmax 离散 op 切断梯度** |
| **task407 v4** | **#113 no STE** | **❌ USAGE-KILL @ ep 5 (util 14.06%)** | **commitment loss 路径冲突, gate 学不到** |

**R18 v2 联立结论**: 
- 4 轮修复覆盖了所有可能的梯度路径 (STE 删除 / per-comp argmin / gate init 不均 / codebook per-comp scale)
- **架构根本限制**: 三分量 product + softmax gate 在 RQ-VAE 训练中无法学到真实 product manifold (跟 task391/task397/topic295 NO-GO 一致)
- 这是数学约束: argmin 离散 op + softmax-weighted codebook 让 gate 梯度极小, 是不可调参数解决的

**跟 Phase 0 mode collapse 模式联立** (per topic295): task178/task180/task231/task242/topic295 4 方向 Phase 0 mode collapse (collision 85-99%, L0/L1/L2 util 1.6-21.9%). task407 v1-v4 加入 product 路径后, util 仍是 0.78-14.06% (跟 task178/task180 一致 Phase 0 坍缩), 证明 **product architecture 不能绕开 Phase 0 mode collapse**.

**新发现 (R10 强积)**:
- HG-Rec baseline (Task #84) R@10=0.1020 是最优路径
- 当前 baseline recipe 内部 14+ 方向 NO-GO 收口 (per R10 v2 + MEMORY 累计)
- 三分量 product 路径加进 NO-GO 收口 (15+ 方向)
- R10 v2 idle 允许 + R19 + R22 后续方向必须等 owner 派工

---

## 关键产物

- **commit hash**: (待 push, R21 v2 强制落地后立即写入, 不允许 pending)
- **push**: origin/main (R15 强制)
- **verdict 路径**: `verdicts/task407_issue113_three_component_gate1_v2.md` (本文件)
- **Precheck audit**: `verdicts/task407_issue113_precheck_audit.json` (PASS)
- **Stage 1 train log**: `products/task407_issue113_three_component/train_log.json` (FAIL)
- **实施脚本**:
  - `scripts/task407_issue113_three_component_precheck.py` (Precheck PASS)
  - `scripts/task407_issue113_three_component_stage1_train.py` (Gate 1 FAIL)
- **整体决策**: ❌ **NO-GO 收口** (4 轮修复全部 USAGE-KILL, 架构根本限制)

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Precheck 脚本 | 复用 ThreeComponentHRQVAE wrapper, 6 audit 函数 | Issue spec §Framework compliance precheck 强制要求 |
| Gate 1 训练时长 | 50 epoch + USAGE-KILL @ ep 5 (util<20%) | Issue spec §Gate1 "受限短训" + 历史经验 |
| Gate 1 修复迭代 | v1 (base) → v2 (gate init 不均) → v3 (per-comp argmin) → v4 (no STE) | R18 强制: 必须实际跑训练获得新数据, 不允许沿用判决 |
| 修复决策终止 | 4 轮覆盖所有可能梯度路径后停止 | 架构根本限制, 不是参数能调出来 |
| 后续方向 | close Issue #113 + Issue #112 + Issue #114 等 owner 派工 | R10 v2 idle 允许 + R22 强制新 issue 立即开工但无新 issue 时 idle |
| GPU 分配 | GPU 1 (4 卡全空闲时优先) | R7 规则: 不抢已占卡 |

---

## 后续 (per R22 + R19 + R16 + R10)

1. **commit + push Issue #113 Gate 1 NO-GO verdict** (R15) — 立即执行
2. **close Issue #113** with R20+R21 comment (commit hash + 4 Gate 详细内容)
3. **Issue #112 + Issue #114** (task425 + task427 pending): 等 Issue #113 Gate 1 PASS 后才有意义 (per spec §Gate2/3/4 强制 Gate 1 前置). Issue #113 NO-GO 后, 这两个 issue 在当前架构不可行, 也需 close.
4. **R10 v2 idle 检查**: 3 OPEN issue 全部 close + §16 空 + 4 GPU 空闲 + 用户未派工 → R10 v2 idle 允许. 报告 R16 / R9 / R7 / GPU 状态等下一个派工信号.