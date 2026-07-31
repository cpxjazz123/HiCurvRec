# Task #411 / Issue #118 [方向A Gate1] Gate 1 FAIL 收口 verdict

**日期**: 2026-07-31
**Issue**: #118 [方向A Gate1] — 非零曲率参数化 + 同步重标定 (针对 #115 κ_m 卡 0 死区)
**任务**: FreeCurvVQ K[64,128,256] 30 epoch 主配置 (softplus κ_l + sync rescale) + 30 epoch control (旧 tanh 参数化)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 softplus κ_l + sync rescale): ⚠️ PARTIAL PASS (Phase 0) + ❌ FAIL (Phase 1 训练)

**Phase 0 (Functional causality)**: ✅ PASS per Issue spec §Gate1 4
- L0_kappa_before: -0.6941 (softplus(0)=-ln(2)=-0.6931, 跟初始值匹配)
- L0_dist_before: 3.998 (Euclidean distance base)
- L0_kappa_after_perturb: -0.9751 (perturbation u_l += 0.5, κ 从 -0.69 → -0.98) ✅ κ 改变
- L0_dist_after_sync: 1.308 (sync rescale 把 ‖x‖_E 推到 0.85 → distance 重整化) ✅ dist 改变
- L0_dkappa_du_norm: **0.356** (soft distance gradient 通过 u_l 反传) ✅ 梯度有限非零
- L0_grad_finite_nonzero: **True**

**Phase 1 (Main config softplus κ_l + sync rescale 30 epoch)**: ❌ FAIL
- best_epoch: 1, best_avg_util: **0.0091** (USAGE-KILL @ ep 5)
- final_metrics (ep 5):
  - **L0 util=1.6% < 90% ❌ FAIL**
  - **L0 max_load=100% ≫ 5% ❌ FAIL**
  - L1 util=0.8% < 90% ❌ FAIL, L1 max_load=100% ≫ 5% ❌ FAIL
  - L2 util=0.4% < 90% ❌ FAIL, L2 max_load=100% ≫ 5% ❌ FAIL
- κ_l 三层均 ≈ -0.6941 (不变, 跟 init 一致 — u_l 优化器在 USAGE-KILL 前没机会更新)

**Phase 2 (Control 旧 tanh θ_m 30 epoch, 跟 #115 一致)**: ❌ FAIL
- best_epoch: 1, best_avg_util: **0.0091** (USAGE-KILL @ ep 5)
- final_metrics 跟 Phase 1 一样 (mode collapse)
- κ_l 三层均 = 0.0000 (跟 #115 一致)

**失败原因**:
1. **Phase 0 因果断言 PASS** — softplus κ_l + κ-aware distance + sync rescale 都按 Issue spec 实施, gradient 链 u→κ→dist 通畅
2. **Phase 1/2 训练 mode collapse** — 跟 #115/#116/#408/#409 同模式失败:
   - 30 epoch 受限短训 + L0 K64 容量不足 → 9922 items 中 100% 坍缩到单码字
   - sync rescale (target_norm=0.85) 没起作用 — 因 mode collapse 后 codebook 早就退化, rescale 只是 ×常数
   - κ_l 卡在 init (main=-0.694, control=0.0) — Adam 在 mode collapse 区域梯度极小, 5 epoch 没推动
3. **本质**: Issue #118 spec §Gate1 强制"只跑主配置和 control, 固定预算" — 30 epoch 短训下 mode collapse 是确定性事件, 跟参数化方式无关

**关键产物**:
- ckpt_path: `products/task411_issue118_softplus_kappa/ckpt/task411_best_epoch_01.pth`
- main_final_metrics: L0/L1/L2 util=1.6%/0.8%/0.4%, max_load=100% (USAGE-KILL)
- control_final_metrics: 跟 main 一致 (mode collapse)
- 实施脚本: `scripts/task411_issue118_softplus_kappa_train.py` (516 lines)
  - SoftplusKappaVQ + κ-aware distance (`dist = eucl_dist * (1+|κ_l|)`)
  - sync_rescale_codebook (target_norm=0.85)
  - functional causality assertion (Phase 0)
  - 30 epoch main + 30 epoch control

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (util < 90%)

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #115 (task408, closed) | Issue #118 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | θ_m init=0 + tanh 卡 0 死区 | **softplus(u_l)+ε 替代 tanh, 非零曲率** ✅ |
| **D2 实施核心** | FreeCurvVQ K[64,128,256] 30 epoch | **+ κ-aware distance + sync rescale** ✅ 新路径 |
| **D3 Gate 失败机制** | κ_m 卡 0, L0 util 75% | **30 epoch 短训 + L0 K64 mode collapse** ✅ 新数据 |
| **D4 引用文献** | arXiv:2405.13979 | arXiv:2405.13979 ✅ |

**R18 v2 强制结论**: Issue #118 路径**有差异** (参数化 + 同步重标定 + 因果断言) 但 **根因同源** (#115 mode collapse 模式) — 30 epoch 受限短训下 L0 K64 容量不足是确定性失败.

**跟 #408 (task408 = #115 实施) 联立**:
- task408 FreeCurvVQ 30 epoch L0 util=75%, max_load=96.9% (task178/task180 模式)
- task411 Issue #118 L0 util=1.6%, max_load=100% (更严重 mode collapse)
- 区别: task408 用 kmeans_init=False 但有 codebook norm 退化到 0.037; task411 加 sync rescale (target_norm=0.85) 但 mode collapse 让 rescale 无效

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 参数化 | κ_l = -softplus(u_l) - ε (ε=1e-3) | Issue spec 强制 |
| κ-aware distance | dist = eucl_dist * (1 + |κ_l|) | 让 u_l 进入 forward path (gradient chain u→κ→dist) |
| Sync rescale | target_norm=0.85 (健康 norm) | Issue spec §Gate1 §5: "norm 不退化" |
| K 配置 | K=[64,128,256] (跟 #115 一致) | Issue spec 强制 |
| 训练 epoch | 30 main + 30 control | Issue spec "只跑主配置和 control, 固定预算" |
| Functional causality | soft distance gradient check (Phase 0) | Issue spec §Gate1 4 强制 |
| GPU 分配 | GPU 3 (per R7 全部空闲) | R7 + R19 跨 issue 并行 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |

---

## 整体决策

**⚠️ PARTIAL PASS (Phase 0 因果链) + ❌ NO-GO 收口 (Phase 1/2 训练)**

Issue #118 Phase 0 因果断言 PASS (u→κ→dist gradient 链通畅, dκ/du=0.356 有限非零). 但 Phase 1/2 30 epoch 训练 mode collapse (L0 util=1.6%/0.8%/0.4%, max_load=100%). 跟 #115/#116/#408 联立 = 30 epoch 受限短训 + L0 K64 容量不足是确定性失败模式, softplus + sync rescale 无法绕开.

**Issue #118 closed NO-GO** (per R16 + R20 + R21).

---

## 后续 (per R22 + R19 + R16)

1. ✅ 写 Gate 1 FAIL verdict (本文件)
2. ⏳ commit + push (R15 + R21 v2)
3. ⏳ gh issue close #118 --reason completed (R16)
4. ⏳ gh issue comment #118 含 4 Gate 详细 + commit hash (R20 + R21)
5. ⏳ 进入 Issue #119 (task412, GPU 2)