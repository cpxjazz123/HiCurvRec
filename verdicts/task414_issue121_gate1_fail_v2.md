# Task #414 / Issue #121 [方向A Gate1] Gate 1 FAIL 收口 verdict

**日期**: 2026-08-01
**Issue**: #121 [方向A Gate1] — κ 更新后 assignment-preserving 径向 codebook 同步传输
**任务**: RadialSyncHRQVAE K=[64,128,256] 30 epoch main (有径向传输) + control (无径向传输)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 κ 更新 + 径向传输 + 训练): ❌ FAIL (5-step audit grad signal = 0)

**[5-step functional audit]**:
- κ_changed: ✅ True (manual perturbation 改变 kappa_l)
- assignment_consistent: ✅ True (传输前后 assignment 不变 — 径向重标定数学正确)
- ranking_consistent: ✅ True (ranking 不变)
- grad_finite_nz: ❌ **False — grad_max_per_layer = [0.0, 0.0, 0.0]**
- roundtrip_ok: ✅ True

**失败原因 (核心发现)**:
- 5-step audit 中 grad_finite_nz 失败, grad 通过 kappa_l_raw 实际为 **0.0**
- **架构根本问题**: 当前 HRQ-VAE forward 用 `argmin(d_hyp)` + STE (`z_q_st = z_e + (z_q - z_e).detach()`)
- 梯度路径: Loss → decoder(z_q) → z_q = codebook[assign] → argmin (non-diff) → STE detach
- κ 只通过 hyperbolic distance 进入 d_hyp, 但 d_hyp 只用于 argmin, **没有进入 loss 路径**
- 因此 κ 没有任何 gradient signal, optimizer.step() 不会改变 κ
- 这是 #121 提案机制本身的根本问题, 跟 #115/#116/#118 NO-GO 模式同源 (Phase 0 修复把 geometric signal 推到 boundary 但 κ 失去梯度)

**[后续 30 epoch 训练]**:
- 5-step audit FAIL 后, per spec "FAIL, 不训练", 直接 exit

**关键产物**:
- 实施脚本: `scripts/task414_issue121_radial_codebook_sync.py` (~430 lines)
  - RadialSyncHRQVAE: per-layer κ_l + radial codebook rescale after optimizer step
  - 5-step functional audit (DIRECT perturbation 替代 optimizer.step 验证机制)
  - Main + Control 30 epoch 配置 (未启动, 因 audit FAIL)
- 路径差异 vs #118: 实施路径**有差异** (径向 codebook 传输 vs scale rescale), 但根因同源 (κ 梯度为零)

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (5-step audit grad signal = 0)

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #118 (task411, closed) | Issue #121 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | softplus κ + sync rescale | **径向重标定 + per-layer optimizer step hook** ✅ |
| **D2 实施核心** | scale 跟 κ 同步 | **codebook 径向重标定 + optimizer hook** ✅ |
| **D3 Gate 失败机制** | 30 epoch 短训 mode collapse | **grad through κ = 0 (架构问题, 跟训练时长无关)** ✅ |
| **D4 引用文献** | arXiv:2405.13979 | arXiv:2405.13979 ✅ |

**R18 v2 强制结论**: Issue #121 路径**有差异** (径向 codebook 传输 vs scale rescale). 实施后发现根因更深层: **HRQ-VAE 的 argmin + STE 设计让 κ 没有 gradient 流入 loss 路径**. 即使径向重标定数学正确 (assignment_consistent=True), κ 也不会因为 loss 优化而改变.

**联立 #115 + #116 + #118 + #121**: 4 个 issue 全部 NO-GO 收口, 共同根因 = **κ 没有 gradient 路径** (argmin + STE detach). 任何 κ 参数化变体 (R137 fix tanh / softplus / codebook sync) 都无法绕开这个架构问题.

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 径向重标定公式 | r = sqrt(κ_old/κ_new) | 数学保证 Voronoi-preserving (均匀缩放) |
| optimizer step hook | per-layer per-step | Issue spec §Gate1 1 强制 |
| Audit 方法 | DIRECT perturbation (替代 optimizer.step) | 避免 Adam lr=1e-4 1 step 不改变 κ 的伪 FAIL |
| grad threshold | 1e-20 | grad_max=0.0 即使阈值=1e-20 仍 FAIL |
| 训练 epoch | 30 main + 30 control | Issue spec "相同有限预算" |
| GPU 分配 | GPU 0 | R7 + R19 跨 issue 并行 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |
| Gate 1 决策 | ❌ FAIL (grad=0) | 机制验证诚实记录, 不允许 fudge |

---

## 整体决策

**❌ NO-GO 收口 (5-step audit FAIL: grad through κ = 0)**

Issue #121 径向重标定数学正确 (assignment + ranking 一致), 但 HRQ-VAE 架构让 κ 失去 gradient path. 任何 κ 参数化变体都无法绕开 argmin + STE detach 的根本问题.

**Issue #121 closed NO-GO** (per R16 + R20 + R21).

---

## 后续 (per R22 + R19 + R16)

1. ✅ 写 Gate 1 FAIL verdict (本文件)
2. ⏳ commit + push (R15 + R21 v2)
3. ⏳ gh issue close #121 --reason completed (R16)
4. ⏳ gh issue comment #121 含 4 Gate 详细 + commit hash (R20 + R21)
5. ⏳ task415 (Issue #122) 同样 grad=0, FAIL 收口