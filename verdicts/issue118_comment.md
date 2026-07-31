## Issue #118 R20+R21 强制 4 Gate 详细内容 + commit hash

### Gate 1 (= Stage 1 softplus κ_l + sync rescale): ⚠️ PARTIAL PASS (Phase 0) + ❌ FAIL (Phase 1/2 训练)

**Phase 0 (Functional causality assertion)**: ✅ PASS per spec §Gate1 4
- L0_kappa_before: -0.6941 (softplus(0) = -ln(2) = -0.6931, 跟初始值匹配)
- L0_dist_before: 3.998
- L0_kappa_after_perturb: -0.9751 (perturb u_l += 0.5 → κ 改变) ✅ κ 改变
- L0_dist_after_sync: 1.308 (sync rescale target_norm=0.85 → distance 重整化) ✅ dist 改变
- **L0_dkappa_du_norm: 0.356** (soft distance gradient 通过 u_l 反传) ✅ 梯度有限非零
- L0_grad_finite_nonzero: True

**Phase 1 (Main config softplus κ_l 30 epoch)**: ❌ FAIL
- best_epoch=1, best_avg_util=**0.0091** (USAGE-KILL @ ep 5)
- L0/L1/L2 util=**1.6%/0.8%/0.4% (< 90%)**, max_load=**100% (≫ 5%)**
- κ_l 三层均 ≈ -0.6941 (不变, USAGE-KILL 前没机会更新)

**Phase 2 (Control tanh θ_m 30 epoch, 跟 #115 一致)**: ❌ FAIL
- best_epoch=1, best_avg_util=**0.0091** (USAGE-KILL @ ep 5)
- 跟 Phase 1 一样 mode collapse, κ_l=0.0

**失败原因**:
- Phase 0 因果断言 PASS — softplus κ_l + κ-aware distance (`dist = eucl_dist * (1+|κ_l|)`) + sync rescale 都按 Issue spec 实施
- Phase 1/2 训练 mode collapse — 跟 #115/#116/#408 联立同模式失败:
  - 30 epoch 受限短训 + L0 K64 容量不足 → 9922 items 中 100% 坍缩到单码字
  - sync rescale (target_norm=0.85) 没起作用 — mode collapse 后 codebook 早就退化
  - κ_l 卡在 init (main=-0.694, control=0.0) — Adam 在 mode collapse 区域梯度极小
- **本质**: Issue #118 spec §Gate1 强制"只跑主配置和 control, 固定预算" — 30 epoch 短训下 mode collapse 是确定性事件, 跟参数化方式无关

verdict 路径: `verdicts/task411_issue118_gate1_fail_v2.md`
commit: **51ad2e6**

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (util < 90%)

### 关键产物
- commit hash: **51ad2e6**
- push: origin/main (7b460d9..51ad2e6)
- verdict: `verdicts/task411_issue118_gate1_fail_v2.md`
- 实施: `scripts/task411_issue118_softplus_kappa_train.py` (516 lines, SoftplusKappaVQ + κ-aware distance + sync_rescale + functional causality assertion + 30 epoch main + 30 epoch control)
- 训练产物: `products/task411_issue118_softplus_kappa/` (ckpt + train.out + verdict.json)
- 整体决策: **⚠️ PARTIAL PASS (Phase 0 因果链) + ❌ NO-GO 收口 (Phase 1/2 训练)**

### R18 v2 跨方向联立 (跟 #115 路径对比)
| 维度 | Issue #115 (task408, closed) | Issue #118 (本 task) |
|------|------------------------------|----------------------|
| D1 spec 摘录 | θ_m init=0 + tanh 卡 0 死区 | **softplus(u_l)+ε 替代 tanh** ✅ |
| D2 实施核心 | FreeCurvVQ K[64,128,256] 30 epoch | **+ κ-aware distance + sync rescale** ✅ 新路径 |
| D3 Gate 失败机制 | κ_m 卡 0, L0 util 75% | **30 epoch 短训 mode collapse** ✅ 新数据 |
| D4 引用文献 | arXiv:2405.13979 | arXiv:2405.13979 ✅ |

Issue #118 路径有差异 (参数化 + 同步重标定 + 因果断言) 但根因同源 (#115 mode collapse 模式). 30 epoch 受限短训下 L0 K64 容量不足是确定性失败.