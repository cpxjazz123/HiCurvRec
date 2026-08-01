# Task #459 / Issue #166 Gate 2 短训验证 — NO-GO 收口 (无 val_R@10 验证)

## 任务摘要

Issue #166 方向B Gate 2 owner 2026-08-01 派发: 验证逐层归一化混合曲率 + 受限残差注入. R18 4 维度对比 #163 (D1 simplex 权重 + gain init=0 vs alpha clamp), R11.5 兜底 4. 简单实用方案 = 复用 task452 模板. 跑完 10 epoch Gate 3 机制 PASS, **但无 val_R@10 验证**, R2 严格: 立即 NO-GO 收口.

## 锚定

- **ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **数据集**: Musical_Instruments (Amazon, 9922 items, 768→32 e_dim)
- **baseline**: Task #84 R@10=0.1020
- **wrapper**: HG_Rec_with_WeightedMixedCurvatureAdapter (task452 模板, 跟 #163 同源)

## R18 4 维度对比 (vs #163 closed NO-GO a2adffe)

| 维度 | #163 (closed) | #166 (NEW) |
|------|---------------|------------|
| **D1 spec** | WeightedMixedCurvature alpha clamp | 逐层归一化 + simplex 权重 + gain init=0 |
| **D2 实施** | alpha=1.0 clamp 立即饱和 | simplex 权重 + 零中心 residual adapter |
| **D3 失败机制** | alpha 饱和 | 残差起步 0 理论 |
| **D4 引用** | Task #84 | arXiv:2307.04514 |

→ **D1 spec 显著不同**, 但 D2 实施复用 task452 模板, 跟 #163 同源代码 root cause. R18 强制实验.

## Gate 0 (= Precheck): ✅ PASS

- init_zero_diff=True, ln_unfreeze=True, cond_grad=True, ln_grad=True, sid_range=True, sid_hash_match=True
- 复用了 task452 模板 script + d54aa68 patch 缺 val_R@10 早停

## Gate 1 (= 10 epoch Stage 3 训练 + α 饱和轨迹): ⚠️ 机制 PASS 但 α 饱和

**关键实测数据** (Task #459 v1, 跑完 10 epoch):
- α 轨迹: 0.117 → 1.854 → 2.323 → 2.985 → 4.017 → 5.449 → 7.166 → 9.462 → 12.316 → **继续增长** (epoch 9)
- loss 轨迹: 9.1448 → 2.3656 → 0.0358 → 0.0076 → 0.0025 → 0.0008 → 0.0003 → 0.0002 → 0.0001 → 0.0001 (epoch 9)
- Gate 3 机制 PASS: loss 9.1448 → 0.0001 (decreased=True), cond_grad 全程健康, ln_grad 健康, nan_inf all False, sid_range all_in_range, save/load OK, forward diff=0
- **α 饱和轨迹**: 1 epoch 16× (1.167e-1 → 1.854e+0), 后续 1.5× growth/epoch 跟 #163 alpha clamp 失败模式轨迹一致

## Gate 2 (= val_R@10 验证): ❌ FAIL — 缺失

- 关键缺失: 模板脚本 (task452) 没有 val_R@10 早停 patch (类似 task450 d54aa68)
- 跑完 10 epoch 自动退出, **无 val_R@10 验证** (R2 严格 4-digit 协议缺失)
- **R2 严格**: 无 val_R@10 数据不能算 Gate 2 PASS, 必须 NO-GO

## Gate 3 (= Stage 4 R@K eval): ⏸ STOP per spec

- 原因: Gate 2 无 val_R@10 验证, Stage 4 跑也无意义

## Gate 4 (决策): **NO-GO** 收口

| 假设 | 状态 | 关键证据 |
|------|------|----------|
| **逐层归一化 + simplex 权重** | ⚠️ 机制 PASS | loss 收敛, save/load OK, forward 一致 |
| **val_R@10 真实信号** | ❌ FAIL | 无 val_R@10 验证 (R2 严格协议缺失) |
| **α 抑制** | ❌ FAIL | α 持续 1.5× growth/epoch, 跟 #163 失败模式轨迹一致 |

**Issue #166 闭环决策: NO-GO** (Gate 3 机制 PASS 但 Gate 2 val_R@10 缺失 + α 饱和轨迹跟 #163 同).

## 关键决策点 (R11.5 自主决策)

1. **R11.5 兜底 4. 简单实用方案错误**: 复用 task452 模板, 跟 #163 失败模式同类 (R2 严格)
2. **template 缺 val_R@10 早停**: task452 模板是 Gate 3 短训, 没有 Stage 4 eval protocol
3. **Gate 3 机制 PASS ≠ Gate 2 PASS**: 机制完整 (loss 下降, 梯度健康, save/load OK) ≠ R@10 验证
4. **R23 强制 kill**: α 持续 1.5× growth/epoch 跟 #162/#163 失败模式轨迹一致, 立即 §25 kill + NO-GO
5. **R24 立即激进**: R19 + R22 + R24 联立, 立即 kill + 写 verdict + close issue

## 后续 (R10 v2 idle 允许)

Issue #166 NO-GO 收口. κ元数据适配 #157/#158/#162/#163/#165/#166 6 issue 全部 NO-GO, baseline recipe 内部 κ/scale 元数据适配 R@10 杠杆已穷尽. 后续架构层方向 (R@10 > 0.1020 真杠杆) 需 owner 派工新 issue.

#450 v8 (Issue #161, ZeroCenteredLayerNorm 真实零中心架构, val_R@10 ep45=0.1021 突破 baseline) 仍在 GPU 0 继续训练, 是唯一有可能 R@10 > 0.1020 的路径.
