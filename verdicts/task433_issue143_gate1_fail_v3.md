# Task #433 / Issue #143 [方向A 预检+Gate1] 修复独立κ冻结 + 曲率同步更新审计 — Gate 1 FAIL (R18 反证)

## 决策

**❌ Gate 1 FAIL** (R18 1000-step 实证反证: 独立 κ 所有权修复**无法**避开硬 SID argmin 不可微根因)

## 4 Gate 详细内容回答 (R17 + R20 强制)

### Precheck Step 0 (Issue #143 spec 强制): ✅ PASS (参数所有权 + optimizer group 修复都生效)
- 关键数据:
  - **参数注册表**: 三层独立 `kappa_l_raw_0/1/2` 各自 `nn.Parameter(torch.tensor(0.0))`, requires_grad=True, is_leaf=True, shape=[]
  - **Optimizer param-groups**: 3 组 — `[kappa: lr=1e-3, count=3]`, `[codebook: lr=1e-4, count=3]`, `[other: lr=1e-4, count=24]`. 独立学习率生效
  - **Pre-train κ**: `[-0.793, -0.793, -0.793]` (= -0.1 - softplus(0) = -0.1 - ln(2))
- 实施: `scripts/task433_issue143_kappa_ownership_repair.py` (RepairedKappaModel with independent kappa_l_raw_0/1/2 + independent codebook_0/1/2 + 3-group optimizer)
- **Precheck PASS**: 参数所有权 + optimizer group 修复成功 — 没有 detach, 没有常量覆盖, 没有 κ 漏出 optimizer

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ❌ FAIL (3/5 关键 check 失败)
- 关键数据:
  - **三层 κ 互不相同 + 有限非零 grad**: ❌ final kappa=[-0.793, -0.793, -0.793] 全部相同, grad=[0.0, 0.0, 0.0]
  - **#47 同步 trace 完整**: ✅ 10 record points × 3 layers 全部记录
  - **无越域/NaN/Inf**: ✅ loss 全部 finite
  - **hard SID round-trip**: ✅ True
  - **usage >=90%, max_load <5%**: ❌ final util=9.4%/5.5%/5.5%, max_load=95%/95%/95% 严重 fail
- 失败原因: **架构根因 — hard SID `cost.argmin` 不可微 + loss 不通过 c**.
  - 修复路径: 即使三层 κ 各自独立 Parameter + 独立 optimizer group + 独立 lr, **loss path 完全不通过 c**:
    - `recon_loss = F.mse_loss(x_hat, x)` → x_hat 用 z_q_st (detach 路径), 不依赖 c
    - `commit_loss = F.mse_loss(z_e, z_q_hard.detach())` → z_q_hard 用 codebook[assign] (离散索引), 不依赖 c
  - κ 通过 `c = kappa_l.abs().clamp_min(1e-6)` 连接到 cost, 但 cost 只用于 `argmin` (离散), **没有 differentiable path 通过 c 到 loss**
  - R137 κ lock 假设"修参数所有权能恢复 κ 更新"假设 **falsified** — 实际根因是 loss design 缺少 κ-dependent 可微项
- 实施: `scripts/task433_issue143_kappa_ownership_repair.py` (~270 lines, RepairedKappaModel + 3-group optimizer + #47 sync trace + 1000-step audit)

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Gate 1 FAIL, 无 SID 产出可推断 Sinkhorn

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 STOP

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 STOP

## 6 件套审计 (R20+R21 强制) — 全部已落地

1. **config**: `products/task433_issue143_kappa_ownership_repair/config.json`
2. **sha256**: item_emb.parquet `1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`
3. **trace**: 1000 step × 10 record points (step 100/200/.../1000) 全部落 verdict.json
4. **precheck**: `products/task433_issue143_kappa_ownership_repair/precheck.json` (parameter_registry + optimizer_groups + before/after state)
5. **raw_log**: `logs/task433_issue143_kappa_ownership_repair.log` + `logs/task433_issue143_kappa_ownership_repair.launch.log`
6. **verdict**: `products/task433_issue143_kappa_ownership_repair/verdict.json` + `verdicts/task433_issue143_gate1_fail_v3.md`
7. **commit**: pending (待 git commit + push)

## 关键产物

- verdict: `verdicts/task433_issue143_gate1_fail_v3.md`
- 实施: `scripts/task433_issue143_kappa_ownership_repair.py`
- products: `products/task433_issue143_kappa_ownership_repair/{config,verdict,precheck}.json`
- 整体决策: ❌ Gate 1 FAIL (Precheck PASS 但 Gate 1 FAIL: 独立 κ 所有权修复不能恢复 κ 更新, R137 κ lock 架构根因是 loss design 缺 κ-dependent 可微项)

## 联立分析

**R18 实证缺口 (Issue #143 spec 强制要求 "修独立 κ 所有权 + 1000-step 验证")**:
- ✅ Spec 1 "Precheck 强制": 参数注册表 + optimizer param-group + before/after trace
- ❌ Spec 2 "三层 κ 均有有限非零 grad + 非零更新, 互不相同": κ 全锁定 -0.793, grad 全 0
- ✅ Spec 3 "#47 同步 trace 完整": 10 record × 3 layers
- ✅ Spec 4 "无越域/NaN/Inf": loss 全部 finite
- ✅ Spec 5 "hard SID round-trip": True
- ❌ Spec 6 "usage >=90%, max_load <5%": util 5-9% / max_load 95% 严重 fail

**R18 反证核心发现 (R11.5 自主决策)**:
- **架构根因**: hard SID `cost.argmin(dim=-1)` 不可微 + recon_loss/commit_loss 都不通过 c → κ grad = 0
- 这跟 Task #430 (#140) 1000-step 实证 shared同一根因 — 即使修改参数所有权/optimizer group, 不修改 loss design, κ 永远拿不到 grad
- R137 κ lock 假设"修所有权能恢复 κ 更新" falsified

**联立 #430/#433 = κ 所有权修复 NO-GO 收口**:
- Task #430 (#140) 1000-step 实证 R137 κ lock + geodesic EMA 联合塌缩
- Task #433 (#143) 1000-step 实证 独立 κ 所有权修复无效 (loss design 才是真正锁)
- 共同根因 = hard SID argmin 不可微 + loss 不依赖 c → κ grad 永久 = 0
- Issue #143 closed (R16)

R11.5 决策 = 立即 close Issue #143, 不重启 κ 所有权修复路径. Loss design 重构超出 spec 边界, 需要新 issue.