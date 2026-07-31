## Issue #143 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### 7 件套审计 (Issue #143 spec 强制) — 全部已落地
1. **config**: `products/task433_issue143_kappa_ownership_repair/config.json`
2. **sha256**: item_emb.parquet `1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`
3. **trace**: 1000 step × 10 record points (step 100/200/.../1000) 全落 verdict.json
4. **precheck**: `products/task433_issue143_kappa_ownership_repair/precheck.json` (parameter_registry + optimizer_groups + before/after state trace)
5. **raw_log**: `logs/task433_issue143_kappa_ownership_repair.log` + `logs/task433_issue143_kappa_ownership_repair.launch.log`
6. **verdict**: `products/task433_issue143_kappa_ownership_repair/verdict.json` + `verdicts/task433_issue143_gate1_fail_v3.md`
7. **commit**: 07efb07 (R21 v2: 不允许 "pending" 占位, commit hash 在 commit + push 后立即填入)

### Precheck Step 0: ✅ PASS (Issue #143 spec 强制)
- 关键数据:
  - **参数注册表**: 3 个独立 `kappa_l_raw_0/1/2` 各自 `nn.Parameter(torch.tensor(0.0))`, requires_grad=True, is_leaf=True
  - **Optimizer param-groups**: 3 组 — `[kappa: lr=1e-3, count=3]`, `[codebook: lr=1e-4, count=3]`, `[other: lr=1e-4, count=24]`
  - **Pre-train κ**: `[-0.793, -0.793, -0.793]` (= -0.1 - softplus(0) = -0.1 - ln(2))
- Precheck PASS: 参数所有权 + optimizer group 修复都生效 (没有 detach, 没有常量覆盖, 没有 κ 漏出 optimizer)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ❌ FAIL (3/5 关键 check 失败)
- 关键数据:
  - **三层 κ 互不相同 + 有限非零 grad**: ❌ final kappa=[-0.793, -0.793, -0.793] 全部相同, grad=[0.0, 0.0, 0.0]
  - **#47 同步 trace 完整**: ✅ 10 record points × 3 layers
  - **无越域/NaN/Inf**: ✅ loss 全部 finite
  - **hard SID round-trip**: ✅ True
  - **usage >=90%, max_load <5%**: ❌ final util=9.4%/5.5%/5.5%, max_load=95%/95%/95% 严重 fail
- 失败原因: **架构根因 — hard SID `cost.argmin` 不可微 + loss 不通过 c**.
  - 修复路径: 即使三层 κ 各自独立 Parameter + 独立 optimizer group + 独立 lr, **loss path 完全不通过 c**:
    - `recon_loss = F.mse_loss(x_hat, x)` → x_hat 用 z_q_st (detach 路径), 不依赖 c
    - `commit_loss = F.mse_loss(z_e, z_q_hard.detach())` → z_q_hard 用 codebook[assign] (离散索引), 不依赖 c
  - κ 通过 `c = kappa_l.abs().clamp_min(1e-6)` 连接到 cost, 但 cost 只用于 `argmin` (离散), **没有 differentiable path 通过 c 到 loss**
  - R137 κ lock 假设"修参数所有权能恢复 κ 更新"假设 **falsified** — 实际根因是 loss design 缺少 κ-dependent 可微项
- 实施: `scripts/task433_issue143_kappa_ownership_repair.py` (RepairedKappaModel + 3-group optimizer + #47 sync + 1000-step audit)

### Gate 2/3/4: ⏸ STOP per spec

### 关键产物
- commit hash: 07efb07
- verdict: verdicts/task433_issue143_gate1_fail_v3.md
- 实施: scripts/task433_issue143_kappa_ownership_repair.py
- 整体决策: ❌ Gate 1 FAIL (Precheck PASS 但 Gate 1 FAIL: 独立 κ 所有权修复不能恢复 κ 更新, R137 κ lock 架构根因是 loss design 缺 κ-dependent 可微项)

### 联立 #430/#433 = κ 所有权修复 NO-GO 收口
Task #430 (#140) 1000-step 实证 R137 κ lock + geodesic EMA 联合塌缩 + Task #433 (#143) 1000-step 实证 独立 κ 所有权修复无效 (loss design 才是真正锁). 共同根因 = hard SID argmin 不可微 + loss 不依赖 c → κ grad 永久 = 0. Issue #143 closed (R16).