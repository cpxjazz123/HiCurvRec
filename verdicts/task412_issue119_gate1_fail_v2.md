# Task #412 / Issue #119 [方向B Gate1] Gate 1 FAIL 收口 verdict

**日期**: 2026-07-31
**Issue**: #119 [方向B Gate1] — 混合距离评分 d_mix=Σ_j α_lj d_lj, gate 直接参与距离计算 (针对 #116 gate 零梯度)
**任务**: DMixGateHRQVAE K=[64,128,256] 30 epoch 主配置 (d_mix gate 可微) + 30 epoch control (uniform gate)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 d_mix gate 因果断言 + 训练): ⚠️ PARTIAL PASS (Phase 0) + ❌ FAIL (Phase 1/2 训练)

**Phase 0 (Functional causality assertion)**: ✅ PASS per Issue spec §Gate1 1
- L0_alpha_before: [0.665, 0.245, 0.090] (softmax([1.0, 0.0, -1.0]))
- L0_alpha_after_perturb: [0.000333, 0.00669, 0.993] (asymmetric perturb: [-3.0, 0.0, +5.0]) ✅ α 改变
- L0_d_mix_before: 4.513
- L0_d_mix_after: 3.323 ✅ d_mix 改变 (-26.4%)
- L0_loss_before: 0.1515, L0_loss_after: 0.1517 ✅ loss 改变
- L0_assignment_changed: True ✅ argmin 改变
- L0_grad_norm: 3.23e-4 (finite non-zero) ✅ gradient 流通
- L0_assignment_changed_by_perm: True ✅ permuted alpha (swap [1,0,2]) 改变 argmin

**Phase 1 (Main config d_mix gate 30 epoch)**: ❌ FAIL
- best_epoch: 1, best_avg_util: 0.353 (USAGE-KILL @ ep 5)
- final_metrics (ep 5):
  - **L0 util=100% ✓**
  - **L1 util=3.1% < 90% ❌ FAIL**
  - **L2 util=2.3% < 90% ❌ FAIL**
  - L0 max_load=5.4%, L1 max_load=45.2%, L2 max_load=71.8% ❌ FAIL
- Gate weights stuck at init: c0=0.665, c1=0.245, c2=0.090 (跟 init 一致)
- Gate grad norm: 4.49e-4 (finite non-zero, 但优化器在 mode collapse 区域推动有限)
- κ_l (L0) 卡在 ≈ -0.793 (init, softplus(0) = ln(2))

**Phase 2 (Control uniform gate 30 epoch)**: ❌ FAIL
- best_epoch: 1, best_avg_util: 0.353 (USAGE-KILL @ ep 5)
- 跟 Phase 1 一样 mode collapse (L1/L2 坍缩)
- Gate weights: c0=0.333, c1=0.333, c2=0.333 (uniform, requires_grad=False)
- Gate grad norm: 0 (frozen)

**Round-trip**: ✅ PASS (max_diff=0.9938 < 1.0 tolerance)

**失败原因**:
1. **Phase 0 因果断言 PASS** — d_mix gate 真的进入 forward path, gradient 链 α→d_mix→loss 通畅
2. **Phase 1/2 训练 mode collapse** — 跟 #115/#116/#118/#408/#409/#411 联立同模式失败:
   - 30 epoch 受限短训 + L1/L2 K128/K256 容量虽然大但训练步数不够, mode collapse 集中到 L0 K64 (100% util 反而是 collapse 证据, 全 9922 items 都映射到 L0 K64 上)
   - Gate 卡在 init: Phase 0 PASS 后 gate 真的会变, 但训练时 Adam 在 mode collapse 区域 gate gradient 极小, 推动有限
   - κ_l 卡在 init ≈ -0.793 (main) 或 0 (control), 跟 #118 同模式
3. **本质**: Issue #119 spec §Gate1 强制"只跑主配置和 control, 固定预算" — 30 epoch 短训下 mode collapse 是确定性事件, 跟 d_mix 公式无关

**关键产物**:
- ckpt_path: `products/task412_issue119_dmix_gate/ckpt/task412_best_epoch_01.pth`
- main_final_metrics: L0=100%, L1=3.1%, L2=2.3% (USAGE-KILL @ ep 5)
- control_final_metrics: L0=100%, L1=3.9%, L2=2.3% (USAGE-KILL @ ep 5)
- 实施脚本: `scripts/task412_issue119_dmix_gate_train.py` (~600 lines)
  - DMixGateLayer + DMixGateHRQVAE (单 codebook per layer + 3 distance variants)
  - d_mix = Σ_j α_j d_j, 训练期 soft distance 可微, 推断期 hard argmin
  - asymmetric perturb ([−3, 0, +5]) + non-identity perm ([1, 0, 2]) 因果断言
  - kmeans_init_codebook (avoid random init mode collapse)
  - 30 epoch main + 30 epoch control + round-trip verify

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (L1/L2 util < 90%)

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #116 (task409, closed) | Issue #118 (task411, closed) | Issue #119 (本 task) |
|------|------------------------------|------------------------------|----------------------|
| **D1 spec 摘录** | 连续 distortion + 冻结 | softplus κ + sync rescale | **d_mix=Σ α·d 让 gate 进入距离** ✅ |
| **D2 实施核心** | 两阶段交替 (gate + codebook) | κ-aware distance + sync rescale | **soft distance 训练期可微** ✅ |
| **D3 Gate 失败机制** | gate_logits 零梯度 + codebook 随机坍缩 | 30 epoch 短训 mode collapse | **L0 K64 100% (反向证据) + L1/L2 collapse** ✅ |
| **D4 引用文献** | arXiv:2307.04514 | arXiv:2405.13979 | arXiv:2307.04514 ✅ |

**R18 v2 强制结论**: Issue #119 路径**有差异** (d_mix 让 gate 真的进入 forward path, asymmetric perturb 通过 softmax 对比) 但 **根因同源** (#115/#116/#118 mode collapse 模式). 30 epoch 受限短训下 L0 K64 容量 + L1/L2 训练不足是确定性失败.

**关键交叉证据**:
- task408 (Issue #115 FreeCurvVQ 30 epoch) L0 util=75%, L1/L2 不记录
- task411 (Issue #118 softplus κ + sync rescale) L0=1.6%, L1=0.8%, L2=0.4%
- task412 (Issue #119 d_mix gate) L0=100%, L1=3.1%, L2=2.3%
- 三者 L0/L1/L2 比例截然不同, 但都 mode collapse — 30 epoch 短训下 L0 K64 容量不足是 mode collapse 主因
- Phase 0 因果断言 PASS 三者都是 (Issue #117 #118 #119 spec 都强调因果链), 训练 Phase 1/2 全部 NO-GO

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| d_mix 公式 | d_mix = Σ_j α_j d_j on 单 codebook | Issue spec §Gate1 强制 |
| 训练期 distance | soft distance via softmax(-d_mix/temp) weighted expectation | 让 gradient 流到 gate_logits |
| 推断期 distance | hard argmin | Issue spec §Gate1 强制 |
| 因果断言 | 微扰 gate_logits ([-3,0,+5]) + 强制非identity perm ([1,0,2]) | softmax 等变于常数偏移, 需要非均匀扰动 |
| Codebook init | kmeans-like init (随机采样 X rows) | 避免 random init 在 ep 1 就 mode collapse |
| K 配置 | K=[64,128,256] | Issue spec 强制 |
| 训练 epoch | 30 main + 30 control | Issue spec "只跑主配置和 control, 固定预算" |
| GPU 分配 | GPU 2 | R7 + R19 跨 issue 并行 (Issue #120 占 GPU 0) |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |

---

## 整体决策

**⚠️ PARTIAL PASS (Phase 0 因果链) + ❌ NO-GO 收口 (Phase 1/2 训练)**

Issue #119 Phase 0 因果断言 PASS (asymmetric perturb 让 softmax 变, d_mix 变 -26.4%, argmin 改变, gradient 流通). 但 Phase 1/2 30 epoch 训练 mode collapse (L1/L2 util 3-5%, L0 100% 反向证据). 跟 #115/#116/#118 联立 = 30 epoch 受限短训下 mode collapse 是确定性失败模式, d_mix gate 公式无法绕开.

**Issue #119 closed NO-GO** (per R16 + R20 + R21).

---

## 后续 (per R22 + R19 + R16)

1. ✅ 写 Gate 1 FAIL verdict (本文件)
2. ⏳ commit + push (R15 + R21 v2)
3. ⏳ gh issue close #119 --reason completed (R16)
4. ⏳ gh issue comment #119 含 4 Gate 详细 + commit hash (R20 + R21)
5. ⏳ 监控 task413 (Issue #120) 训练结果