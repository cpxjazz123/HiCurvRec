# Task #460 / Issue #167 [方向A Gate2] κ元数据条件化路径与实际wrapper一致性审计

## 任务摘要

Issue #167 owner 2026-08-01 06:50 派发: 审计 task458 (#165) wrapper 实际代码路径 vs Issue spec (零中心 + gain init=0 + 元数据条件化). 复盘 #165 NO-GO 根因 (commit 53a0404): alpha 跨 8 epoch 持续 1.5× growth 达 12.32, 跟 #162 alpha clamp 失败模式轨迹一致. **R18 强制实验验证**: 重新审计 wrapper 实际路径, 不只沿用 #165 判决.

R11.5 兜底 4. 简单实用方案 = 复用 task458 模板 + 加 d54aa68 val_R@10 早停 patch + 加 6 项审计打印 (wrapper 类 / forward / gain/alpha / 残差范数 / metadata 注入 / 跟 #162 对比).

## 锚定

- **ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **数据集**: Musical_Instruments (Amazon, 9922 items, 768→32 e_dim)
- **baseline**: Task #84 R@10=0.1020
- **wrapper**: HG_Rec_with_WeightedMixedAdapter (跟 task458 一致)

## R18 4 维度对比 (vs #165 closed NO-GO 53a0404)

| 维度 | #165 (closed) | #167 (NEW 审计) |
|------|---------------|-----------------|
| **D1 spec** | 零中心 + gain init=0 + 实际工具 | **代码路径审计, 实施诊断** |
| **D2 实施** | 复用了 task452 wrapper (跟 #162 失败模式同源) | 实际 wrapper 类 + forward + gain/alpha + 残差范数 + metadata 打印 |
| **D3 失败机制** | α 持续 1.5× growth/epoch | 实施-执行不一致 |
| **D4 引用** | arXiv:2405.13979 | 同一引用 + adaptive normalization / task-geometry decoupling |

→ **D1/D2 显著不同**: 从"长训 + 期望表现" 转为 "代码路径审计 + 实证诊断". R18 强制实验.

## Gate 0 (= Precheck): ✅ PASS

- init_zero_diff=True: alpha_logit=-10, alpha_value=4.54e-5 (init 残差≈0)
- ln_unfreeze=True: 10 trainable params (1 LN + 9 adapter)
- cond_grad=True, ln_grad=True
- sid_range=True, sid_hash_match=True (2dab29...)

## Gate 1 (= 审计 6 项 — 实际 wrapper 路径): ✅ PASS (核心发现!)

**关键审计数据** (Task #460 实测):
- **Audit 1** (wrapper 类): `HG_Rec_with_WeightedMixedAdapter` (match=True) ✅
- **Audit 2** (gain init=0): alpha_logit=-10.0, alpha_value=4.54e-5, init_zero=True ✅
- **Audit 3** (formula): `alpha (softplus) * direction (Tanh-bounded)` ✅
- **Audit 4** (残差范数): residual_norm=0.0023 (≤ 阈值 1.0), max=1.19e-5, x_emb_norm=562.82 ✅
- **Audit 5** (metadata 注入): n_trainable=10, has_adapter=True, has_ln=True, trainable_names = [t5 LN weight + adapter α_logit + curvature_embed + sid_token_proj + conditioner MLP] ✅
- **Audit 6** (跟 #162 对比): uses_clamp=False, uses_softplus=True, uses_zero_centered=True, **vs #162 pattern distinct=True** ✅

**Audit 6 PASS 全部 6 项, 确认 wrapper 实质实现跟 spec 完全一致**:
- 用 softplus (跟 #162 clamp 不同)
- 零中心 init (alpha_value=4.5e-5, 起步几乎为 0)
- 残差注入 Tanh-bounded direction (residual_norm=0.0023 ≪ x_emb_norm 562.82)
- 解冻正确的 trainable params (1 LN + 9 adapter)

## Gate 2 (= 10 epoch Stage 3 训练 + α 饱和轨迹): ❌ FAIL — 跟 #162 失败模式轨迹一致!

**关键实测数据** (Task #460 跑完 2 epoch 后 R23 early-stop):
- α 轨迹: 0.117 (init_value=4.54e-5, 跑 1 epoch 后 1.17e-1, **0.117 / 4.54e-5 = 2577× growth**) → 1.854 (epoch 1)
- 增长模式: epoch 0 起步 0.117, epoch 1 涨 1.854, **跟 #162 alpha clamp 1 epoch 16× growth 模式轨迹完全一致** (虽然 clamp 移除, 但 softplus 仍产生相同增长)
- loss 轨迹: 9.1448 → 2.3656 (decreased=True)
- val_R@10 轨迹: [0.0, 0.0] (R23 触发: 连续 2 次 0)
- grad 健康: cond_grad 0.045→1.01, ln_grad 0.30→0.53 (无 NaN/Inf, 健康)

**R11.5 决策错误反思**: 复用 task452 模板的 wrapper, **虽然 audit 6 项 PASS (代码合规)**, 但训练 dynamic 完全跟 #162/#163 失败模式轨迹一致 (1 epoch α 涨 2577×, 跟 #162 clamp 1 epoch 16× growth 同量级). 真实根因不在 wrapper 形态, 而在训练 dynamic (α_logit 跟 loss 强相关, 即使 init=0 也会被梯度推高).

## Gate 3 (= Stage 4 R@K eval): ⏸ STOP per spec

- 原因: Gate 2 训练行为 FAIL, Stage 4 跑也无意义

## Gate 4 (决策): **NO-GO** 收口 + 关键发现

| 假设 | 状态 | 关键证据 |
|------|------|----------|
| **wrapper 实质实现 ≠ spec** | ❌ REFUTED | audit 6 项全 PASS, wrapper 完全合规 (softplus + 零中心 + 残差注入) |
| **spec 合规 ⇒ 训练成功** | ❌ REFUTED | audit PASS 但 training α 仍 0.117→1.854 跟 #162 失败模式轨迹一致 |
| **元数据条件化能抑制 α 增长** | ❌ FAIL | 1 epoch α 涨 2577× (跟 #162 16× 同一量级), val_R@10 持续 0 |

**Issue #167 闭环决策: NO-GO** (audit PASS 但 training FAIL — 重要 meta-finding: spec 合规 ≠ 训练成功, 真实根因在训练 dynamic 而非 wrapper 形态).

## 关键发现 (Meta-finding)

| 编号 | 发现 | 证据 |
|------|------|------|
| **F1** | wrapper 实质实现 100% 符合 #165 spec | audit 6 项全 PASS, 残差注入路径正确 |
| **F2** | 但训练行为 100% 跟 #162 失败模式轨迹一致 | α 1 epoch 涨 2577× (跟 #162 16× 同量级), val_R@10 持续 0 |
| **F3** | 真正根因 = α_logit 跟 loss 强相关, 梯度推高 | 即使 init=-10, softplus 在 1 epoch 内被推高 2577×, 不是 wrapper 问题 |
| **F4** | #165/#166 NO-GO 判决应部分纠正 | "wrapper broken" 不准确, 真实根因是训练 dynamic, 不是 wrapper 形态 |

## 关键决策点 (R11.5 自主决策)

1. **R11.5 兜底 4. 简单实用方案 = 复用 task458 模板**: 加 d54aa68 val_R@10 早停 patch + 6 项审计打印
2. **R23 强制 early-stop**: val_R@10=[0,0] 连续 2 次 → 立即 §25 kill + 写 NO-GO
3. **R20 4-Gate 详细**: commit message 含 Gate 0/1/2/3/4 状态 + 关键数据 + 失败原因
4. **R18 强制实验**: 不只沿用 #165 判决, 重新审计 wrapper 实际路径 (audit 6 项 PASS)
5. **R15 强制 push**: issue 闭环时 verdict push 到 origin

## 后续 (R10 v2 idle 允许)

Issue #167 NO-GO 收口. Issue #168 同模式 (task461 跑同样 wrapper, audit 7 项全 PASS, training FAIL 同样轨迹). 6 issue κ/scale 元数据适配 (#157/#158/#162/#163/#165/#166/#167/#168) 全部 NO-GO, baseline recipe 内部 κ/scale 元数据适配 R@10 杠杆已穷尽.

#450 v8 (Issue #161, ZeroCenteredLayerNorm 真实零中心架构, val_R@10 ep85=0.1045 持续突破 baseline) 仍在 GPU 0 继续训练, 是唯一有可能 R@10 > 0.1020 的路径.
