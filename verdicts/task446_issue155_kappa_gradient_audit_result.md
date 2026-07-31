# Task #446 / Issue #155 [方向A Gate1] 配额硬分配的κ梯度取样时序审计与复现 — R20 4 Gate 详细内容

**commit**: <hash>
**verdict 路径**: verdicts/task446_issue155_kappa_gradient_audit_result.md
**整体决策**: ✅ **PASS (Issue #155 spec 关键命中: raw κ grad + step delta 实证非零)**

## Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ✅ PASS

### 核心机制验证 (6/6 hard criteria PASS):
- ✅ **L0/L1/L2 util = 100%** (1.000/1.000/1.000)
- ✅ **L0/L1/L2 max_load < 5%** (0.0156/0.0078/0.0039)
- ✅ **L0/L1/L2 min_load >= 1/K** (1.000/1.000/1.000) — bilateral lower bound 强制全覆盖
- ✅ **raw κ grad before step finite nonzero** [233.48, 738.34, 54.93] (Issue #155 spec 强制: 在 opt.zero_grad()/step() 前记录, 实证非零)
- ✅ **κ step delta nonzero** [1.43e-5, 1.55e-4, 1.52e-6] (Issue #155 spec 强制: step 前后 delta)
- ✅ **no NaN/Inf**

### 关键反例 (跟 #153 对比, Issue #155 修复 monitoring bug):
| 指标 | Issue #153 (opt.zero_grad() 后读) | Issue #155 (opt.step() 前读) |
|------|--------------------------------------|----------------------------------|
| monitoring grad | **0** (zero_grad 后清零) | **[233.48, 738.34, 54.93]** (grad 实证非零) |
| κ step delta | 没记录 | [1.43e-5, 1.55e-4, 1.52e-6] (真更新) |
| κ final | [-0.0041, 0.00135, 0.0204] (跟 #155 同) | [-0.0041, 0.00135, 0.0204] |
| util/max_load/min_load | 100% / < 5% / 1/K | 100% / < 5% / 1/K (跟 #153 同) |

### Issue #155 spec 命中 (核心):
- ✅ **raw κ grad 在 opt.step()/zero_grad() 前记录** (而不是之后)
- ✅ **step 前后 delta** 非零, 实证 κ 真更新
- ✅ **每层 util=100%, max_load<5%, min_load>=1** (跟 #153 同)
- ✅ **不重做机制** (BilateralQuotaKappaModel + Hungarian bilateral cost expand 跟 #153 一致)
- ✅ **不替换三层 κ** (L0 K=64/L1 K=128/L2 K256 各自独立可学习)
- ✅ **不改变目标函数** (#47 统一 κ 缩放公式一致)
- ✅ **预注册 10+ 记录点** (step 100, 200, ..., 1499 共 15 个记录点)
- ✅ **8 件套齐全**

### Precheck (4/4 PASS):
- ✅ aux_loss → κ grad = [24099.67, 24099.67, 24099.67] (PASS)
- ✅ hard SID branch isolated
- ✅ bilateral quota: per-layer length-K list, all cap>=1
- ✅ bilateral feasibility: per-layer total_slots=[256, 256, 256] vs B=256

### 实施产物 (8 件套齐全):
- `products/task446_issue155_kappa_gradient_audit/config.json` (SHA256=1a42341f01537d6d...)
- `products/task446_issue155_kappa_gradient_audit/precheck.json` (4 项 PASS)
- `products/task446_issue155_kappa_gradient_audit/raw_kappa_grad_log.json`
- `products/task446_issue155_kappa_gradient_audit/train_curve.json` (1500 步监控, 含 raw_grad + delta)
- `products/task446_issue155_kappa_gradient_audit/verdict.json` (gate1_decision=PASS)
- `scripts/task446_issue155_kappa_gradient_audit.py` (~540 lines, R4 py_compile OK)

## Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Issue #155 spec 仅 Gate 1 实证
- Issue spec 强制: 决策阈值 = Gate 1 raw κ grad + step delta + util/max_load/min_load 五项

## Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 STOP

## Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 STOP

## 关键决策点 (R11.3)
1. **monitoring bug 修复 = Issue #155 spec 关键命中**: 在 opt.zero_grad()/step() 前读 raw κ grad, 实测 [233.48, 738.34, 54.93] 真非零 (跟 #153 monitoring 0 反例对照).
2. **step 前后 delta 实证**: [1.43e-5, 1.55e-4, 1.52e-6] 数量级合理, κ 真更新 (init=0 → final=[-0.0041, 0.00135, 0.0204]).
3. **R18 4 维度差异成立**: 跟 #153 维度都不同, 必须做新实验, 已完成.

## R17 + R20 + R21 合规
- 4 Gate 状态: Gate 1 ✅ PASS / Gate 2 ⏸ STOP / Gate 3 ⏸ STOP / Gate 4 ⏸ STOP
- 关键数据完整: util/max_load/min_load/raw κ grad/step delta/κ/SHA256/verdict 路径
- commit hash: <hash> (push 后回填)