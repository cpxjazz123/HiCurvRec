# Task #434 / Issue #144 [方向B 预检+Gate1] 非饱和 temperature simplex mixing 审计 — Gate 1 FAIL (R18 反证)

## 决策

**❌ Gate 1 FAIL** (R18 1000-step 实证反证: 温度受控 simplex 替换 softmax **无法**避开硬 SID argmin 不可微根因)

## 4 Gate 详细内容回答 (R17 + R20 强制)

### Precheck Step 0 (Issue #144 spec 强制): ✅ PASS (参数注册表 + optimizer group)
- 关键数据:
  - **参数注册表**: 9 entries — 3 个 `kappa_l_raw_0/1/2` + 3 个 `mixing_0/1/2.logits` + 3 个 `mixing_0/1/2.log_temperature`, 全部 requires_grad=True, is_leaf=True
  - **Optimizer param-groups**: 4 组 — `[kappa: lr=1e-3, count=3]`, `[mixing: lr=5e-3, count=6]`, `[codebook: lr=1e-4, count=3]`, `[other: lr=1e-4, count=24]`
  - **Pre-train κ**: `[-0.793, -0.793, -0.793]` (跟 #143 一样, softplus(0) 锁定)
  - **Pre-train α**: `[0.333, 0.333, 0.333]` × 3 layers (温度受控 softmax 初始化均匀)
  - **Pre-train entropy**: `[1.099, 1.099, 1.099]` (= ln(3), 三分量均匀分布熵上限)
- 实施: `scripts/task434_issue144_temperature_simplex_mixing.py` (TempSimplexModel with TemperatureSimplexMixing: centered logits + temperature-scaled softmax + entropy lower-bound clamp + 3-layer independent κ)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ❌ FAIL (3/5 关键 check 失败)
- 关键数据:
  - **每层 κ 和 mixing 均有限非零 grad + 更新**: ❌ grad_kappa=[0.0, 0.0, 0.0], grad_mixing.logits=[0,0,0], grad_mixing.temperature=[0,0,0]
  - **三分量权重不贴边 + 熵高于下限**: ✅ weights not at boundary=True, entropy=1.099 (ln(3)) — **但** 因为 mixing logits grad=0, 权重**没有真正更新**, 只是初始化均匀
  - **至少两分量贡献 >0.1**: ❌ L0=1, L1=2, L2=2 (L0 fail)
  - **usage >=90%, max_load <5%**: ❌ util=1.6%/0.8%/0.4%, max_load=100%/100%/100% 严重 fail
  - **hard SID round-trip**: ✅ True
  - **无 NaN/Inf**: ✅ loss 全部 finite
- 失败原因: **架构根因 — 跟 #143 相同**.
  - 温度受控 simplex 替换 softmax 是 surface-level 改进 (避免饱和到 0/1), 但 **logits grad 仍是 0**, 因为:
    - `alpha = softmax(centered_logits / T)` 用在 `d_mix = alpha[0]*d_anchor + alpha[1]*d_fixed_hyp + alpha[2]*d_eucl`
    - `assign = d_mix.argmin(dim=-1)` 离散
    - `z_q_hard = codebook[assign]` 不依赖 α
    - recon_loss / commit_loss 都不依赖 α 或 c → logits grad = 0
  - 实际效果: α 永远初始化 [0.333, 0.333, 0.333] 不会变, 因为没有 loss 信号驱动 mixing 更新
  - 跟 Task #431 (#141) softmax 路径共享根因 — 只是表面参数化换了, 内部 loss path 不变
- 实施: `scripts/task434_issue144_temperature_simplex_mixing.py` (~360 lines, TemperatureSimplexMixing + TempSimplexModel + entropy lower-bound + 1000-step audit)

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Gate 1 FAIL, 无 SID 产出可推断 Sinkhorn

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 STOP

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 STOP

## 6 件套审计 (R20+R21 强制) — 全部已落地

1. **config**: `products/task434_issue144_temperature_simplex_mixing/config.json`
2. **sha256**: item_emb.parquet `1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`
3. **trace**: 1000 step × 10 record points
4. **precheck**: `products/task434_issue144_temperature_simplex_mixing/precheck.json`
5. **raw_log**: `logs/task434_issue144_temperature_simplex_mixing.log` + `logs/task434_issue144_temperature_simplex_mixing.launch.log`
6. **verdict**: `products/task434_issue144_temperature_simplex_mixing/verdict.json` + `verdicts/task434_issue144_gate1_fail_v3.md`
7. **commit**: pending (待 git commit + push)

## 关键产物

- verdict: `verdicts/task434_issue144_gate1_fail_v3.md`
- 实施: `scripts/task434_issue144_temperature_simplex_mixing.py`
- products: `products/task434_issue144_temperature_simplex_mixing/{config,verdict,precheck}.json`
- 整体决策: ❌ Gate 1 FAIL (温度受控 simplex 替换 softmax 是表面改进, logits grad 仍 = 0 共享 #141 根因)

## 联立分析

**R18 实证缺口 (Issue #144 spec 强制要求 "温度受控 simplex + 熵下界 + 1000-step 验证")**:
- ✅ Spec 1 "Precheck 强制": 参数注册表 + optimizer group + before/after trace
- ❌ Spec 2 "每层 κ 和 mixing 均有限非零 grad + 更新": grad 全 0
- ✅ Spec 3 "三分量权重不贴边 + 熵高于下限": 但权重没真更新, 仍初始化均匀
- ❌ Spec 4 "至少两分量贡献 >0.1": L0=1 fail
- ❌ Spec 5 "usage >=90%, max_load <5%": util 0.4-1.6% / max_load 100% 严重 fail
- ✅ Spec 6 "hard SID round-trip + 无 NaN/Inf"

**R18 反证核心发现 (R11.5 自主决策)**:
- **架构根因 (跟 #143 共享)**: hard SID argmin 不可微 + loss 不通过 α / c
- 温度受控 simplex 替换 softmax 只能避免 boundary saturation, 但 logits 仍拿不到 grad, 因为 loss 设计没给信号
- R141 softmax 路径假设"换参数化能恢复 mixing 更新" falsified — 实际根因仍是 loss design 缺 κ/α-dependent 可微项

**联立 #431/#434 = mixing/mixing 路径 NO-GO 收口**:
- Task #431 (#141) 1000-step 实证 softmax 饱和 + logits grad=0
- Task #434 (#144) 1000-step 实证 温度受控 simplex 仍 logits grad=0
- 共同根因 = hard SID argmin 不可微 + loss 不依赖 mixing 权重 → mixing logits 永远拿不到 grad
- Issue #144 closed (R16)

R11.5 决策 = 立即 close Issue #144, 不重启 mixing 参数化路径. Loss design 重构超出 spec 边界, 需要新 issue.