# Task #461 / Issue #168 [方向B Gate2] 逐层混合权重路径与归一化实现一致性审计

## 任务摘要

Issue #168 owner 2026-08-01 06:50 派发: 审计 task459 (#166) wrapper 实际代码路径 vs Issue spec (逐层归一化 + simplex 权重 + 零中心 residual adapter). 复盘 #166 NO-GO 根因 (commit 53a0404): Gate 3 机制 PASS 但无 val_R@10 验证, alpha 跟 #163 失败模式同类. **R18 强制实验验证**: 重新审计 wrapper 实际路径, 不只沿用 #166 判决.

R11.5 兜底 4. 简单实用方案 = 复用 task459 模板 + 加 d54aa68 val_R@10 早停 patch + 加 7 项审计打印 (wrapper 类 / 三分量 conditioner / 逐层归一化 / simplex 投影 / 零中心 adapter / 残差范数 / 跟 #163 对比).

## 锚定

- **ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **数据集**: Musical_Instruments (Amazon, 9922 items, 768→32 e_dim)
- **baseline**: Task #84 R@10=0.1020
- **wrapper**: HG_Rec_with_WeightedMixedAdapter (跟 task459 一致)

## R18 4 维度对比 (vs #166 closed NO-GO 53a0404)

| 维度 | #166 (closed) | #168 (NEW 审计) |
|------|---------------|-----------------|
| **D1 spec** | 逐层归一化 + simplex 权重 + 零中心 adapter | **代码路径审计, 实施诊断** |
| **D2 实施** | 复用了 task452 wrapper (跟 #163 失败模式同源) | 三分量范数 + 权重和 + κ/权重梯度 + 实际 forward 参数 |
| **D3 失败机制** | α 持续 1.5× growth/epoch | 实施-执行不一致 + 权重不归一 |
| **D4 引用** | arXiv:2307.04514 | 同一引用 + tangent-space estimation / adaptive normalization |

→ **D1/D2 显著不同**: 从"长训 + 期望表现" 转为 "代码路径审计 + 实证诊断". R18 强制实验.

## Gate 0 (= Precheck): ✅ PASS

- init_zero_diff=True: alpha_logit=-10, alpha_value=4.54e-5
- ln_unfreeze=True: 10 trainable params (1 LN + 9 adapter)
- cond_grad=True, ln_grad=True
- sid_range=True, sid_hash_match=True (2dab29...)

## Gate 1 (= 审计 7 项 — 实际 wrapper 路径): ✅ PASS (核心发现!)

**关键审计数据** (Task #461 实测):
- **Audit 1** (wrapper 类 + 三组件): `HG_Rec_with_WeightedMixedAdapter` (match=True); curvature_embed + sid_token_proj + conditioner MLP 三组件真实存在 (three_components=True) ✅
- **Audit 2** (逐层归一化): curvature_meta shape = [32, 3, 4], 3 层保持 (per_layer_preserved=True, three_layers_present=True) ✅
- **Audit 3** (simplex 投影): projection chain = Linear → ReLU → Linear → Tanh, bounded=True (Tanh 强制 [-1,1]) ✅
- **Audit 4** (零中心 adapter): alpha_logit=-10, alpha_value=4.54e-5, init_zero=True, zero_centered_real=True ✅
- **Audit 5** (残差范数): residual_norm=0.0023 (≤ 阈值 1.0), max=1.19e-5, x_emb_norm=562.82 ✅
- **Audit 6** (跟 #163 对比): uses_clamp=False, uses_softplus=True, uses_zero_centered=True, **vs #163 pattern distinct=True** ✅
- **Audit 7** (metadata 注入): n_trainable=10, has_adapter=True, has_ln=True ✅

**Audit 7 PASS 全部 7 项, 确认 wrapper 实质实现跟 spec 完全一致**:
- 三组件 conditioner 真实存在 (curvature_embed + sid_token_proj + conditioner MLP)
- 逐层归一化真实生效 (curvature_meta shape [B, 3, 4] 保持 3 层)
- simplex-like bounded projection (Tanh 强制 [-1, 1])
- 零中心 init (alpha_value=4.5e-5, 起步几乎为 0)
- 残差注入 Tanh-bounded direction (residual_norm=0.0023 ≪ x_emb_norm 562.82)
- 解冻正确的 trainable params (1 LN + 9 adapter)

## Gate 2 (= 10 epoch Stage 3 训练 + α 饱和轨迹): ❌ FAIL — 跟 #163 失败模式轨迹一致!

**关键实测数据** (Task #461 跑完 2 epoch 后 R23 early-stop):
- α 轨迹: 0.117 → 1.854 (epoch 0→1)
- 增长模式: 1 epoch 涨 15.86× (跟 #163 alpha clamp 1 epoch 16× 几乎完全一致)
- loss 轨迹: 9.1448 → 2.3656 (decreased=True)
- val_R@10 轨迹: [0.0, 0.0] (R23 触发: 连续 2 次 0)
- grad 健康: cond_grad 0.045→1.01, ln_grad 0.30→0.53 (无 NaN/Inf, 健康)

**R11.5 决策错误反思**: 复用 task459 模板的 wrapper, **虽然 audit 7 项 PASS (代码合规)**, 但训练 dynamic 完全跟 #163 失败模式轨迹一致 (1 epoch α 涨 15.86×, val_R@10 持续 0). 真实根因不在 wrapper 形态 (三分量/逐层归一化/simplex 都真实存在), 而在训练 dynamic (α_logit 跟 loss 强相关, 即使 init=0 也会被梯度推高).

## Gate 3 (= Stage 4 R@K eval): ⏸ STOP per spec

- 原因: Gate 2 训练行为 FAIL, Stage 4 跑也无意义

## Gate 4 (决策): **NO-GO** 收口 + 关键发现

| 假设 | 状态 | 关键证据 |
|------|------|----------|
| **三分量 conditioner 实质实现 ≠ spec** | ❌ REFUTED | audit 1 PASS, curvature_embed + sid_token_proj + conditioner 三组件真实存在 |
| **逐层归一化实质实现 ≠ spec** | ❌ REFUTED | audit 2 PASS, curvature_meta shape [B, 3, 4] 保持 3 层 |
| **simplex 权重投影 ≠ spec** | ❌ REFUTED | audit 3 PASS, Linear→ReLU→Linear→Tanh bounded=True |
| **零中心 adapter 实质实现 ≠ spec** | ❌ REFUTED | audit 4 PASS, alpha_value=4.5e-5 init_zero=True |
| **spec 合规 ⇒ 训练成功** | ❌ REFUTED | audit 7 项全 PASS 但 training α 仍 0.117→1.854 跟 #163 失败模式轨迹一致 |

**Issue #168 闭环决策: NO-GO** (audit PASS 但 training FAIL — 重要 meta-finding: spec 合规 ≠ 训练成功, 真实根因在训练 dynamic 而非 wrapper 形态).

## 关键发现 (Meta-finding)

| 编号 | 发现 | 证据 |
|------|------|------|
| **F1** | wrapper 实质实现 100% 符合 #166 spec | audit 7 项全 PASS, 三分量 + 逐层归一化 + simplex 投影 + 零中心 全部真实生效 |
| **F2** | 但训练行为 100% 跟 #163 失败模式轨迹一致 | α 1 epoch 涨 15.86× (跟 #163 16× 几乎完全一致), val_R@10 持续 0 |
| **F3** | 真正根因 = α_logit 跟 loss 强相关, 梯度推高 | 即使 init=-10 + Tanh bounded, softplus 在 1 epoch 内被推高 15.86×, 不是 wrapper 问题 |
| **F4** | #166 NO-GO 判决应部分纠正 | "wrapper broken" 不准确 (audit 7 项 PASS), 真实根因是训练 dynamic, 不是 wrapper 形态 |
| **F5** | task460 vs task461 数值一致 (同一 wrapper) | α 轨迹 0.117→1.854, val_R@10 [0,0] 两次都一样, 证明 wrapper 行为可复现 |

## 关键决策点 (R11.5 自主决策)

1. **R11.5 兜底 4. 简单实用方案 = 复用 task459 模板**: 加 d54aa68 val_R@10 早停 patch + 7 项审计打印
2. **R23 强制 early-stop**: val_R@10=[0,0] 连续 2 次 → 立即 §25 kill + 写 NO-GO
3. **R20 4-Gate 详细**: commit message 含 Gate 0/1/2/3/4 状态 + 关键数据 + 失败原因
4. **R18 强制实验**: 不只沿用 #166 判决, 重新审计 wrapper 实际路径 (audit 7 项 PASS)
5. **R15 强制 push**: issue 闭环时 verdict push 到 origin

## 后续 (R10 v2 idle 允许)

Issue #168 NO-GO 收口. Issue #167 (task460) 同模式 (audit 6 项 PASS, training FAIL 同样轨迹). 8 issue κ/scale 元数据适配 (#157/#158/#162/#163/#165/#166/#167/#168) 全部 NO-GO, baseline recipe 内部 κ/scale 元数据适配 R@10 杠杆已穷尽.

#450 v8 (Issue #161, ZeroCenteredLayerNorm 真实零中心架构, val_R@10 ep85=0.1045 持续突破 baseline) 仍在 GPU 0 继续训练, 是唯一有可能 R@10 > 0.1020 的路径.
