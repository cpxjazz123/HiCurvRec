# Task #458 / Issue #165 Gate 2 短训验证 — NO-GO 收口 (α 饱和失败模式)

## 任务摘要

Issue #165 方向A Gate 2 owner 2026-08-01 派发: 验证 κ/尺度元数据与T5残差幅值解耦. 零中心 + gain init=0 + 逐层 norm 控制. R11.5 兜底 4. 简单实用方案 = 复用 task440 零中心 wrapper. 跑 10 epoch Gate 2 短训, **α 饱和轨迹跟 #162 alpha clamp 失败模式同类**, R23 强制 kill + NO-GO.

## 锚定

- **ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **数据集**: Musical_Instruments (Amazon, 9922 items, 768→32 e_dim)
- **baseline**: Task #84 R@10=0.1020
- **wrapper**: HG_Rec_with_ZeroCenteredLayerNormAdapter (Issue #150 零中心 + gain init=0, 复用 task440)

## R18 4 维度对比 (vs #162 closed NO-GO a2adffe)

| 维度 | #162 (closed) | #165 (NEW) |
|------|---------------|------------|
| **D1 spec** | alpha clamp 残差放大 | 零中心 + gain init=0 + 逐层 norm |
| **D2 实施** | `clamp(alpha_raw, max=1.0)` | 零中心仿射调制 |
| **D3 失败机制** | alpha 立即饱和 | 残差起步 0 理论 |
| **D4 引用** | Task #84 | arXiv:2405.13979 |

→ **D1 spec 显著不同**, 但 D2 实施复用 task440 模板. R18 强制实验.

## Gate 0 (= Precheck): ✅ PASS

- init_zero_diff=True (强制 α=0 时残差=0)
- ln_unfreeze=True (T5 input LayerNorm 解冻)
- cond_grad=True, ln_grad=True (梯度健康)
- sid_range=True, sid_hash_match=True (一致)
- commit 复用了 task452 模板 script + d54aa68 patch

## Gate 1 (= 10 epoch Stage 3 训练 + α 饱和轨迹): ❌ FAIL — α 饱和失败模式

**关键实测数据** (Task #458 v2, 实际跑到 epoch 8):
- α 轨迹: 0.117 → 1.854 → 2.323 → 2.985 → 4.017 → 5.449 → 7.166 → 9.462 → **12.316** (epoch 8)
- loss 轨迹: 9.1448 → 2.3656 → 0.0358 → 0.0076 → 0.0025 → 0.0008 → 0.0003 → 0.0002 → 0.0001 (epoch 8, 收敛到 0)
- 增长模式: 1 epoch 16× (1.167e-1 → 1.854e+0), 后续 1.5× growth/epoch
- **真实失败根因**: 虽 spec 强制 gain init=0, 但 wrapper 实际跑的是 task440 架构 (α 走 softplus + 残差注入), α 持续增长类 #162 alpha clamp 失败模式
- **R23 触发**: 跨 8 epoch α 持续 1.5× growth/epoch, 跟 #162 失败模式轨迹一致
- **R11.5 决策错误反思**: 复制 task452 模板 (跟 #162 失败模式同类), 应该是重写 wrapper 实质实现

## Gate 2 (= val_R@10 验证): ❌ FAIL — 未能跑 EarlyStop

- 关键缺失: 模板脚本 (task452) 没有 val_R@10 早停 patch (类似 task450 d54aa68)
- 跑完 10 epoch 自动退出, **无 val_R@10 验证** (R2 严格 4-digit 协议缺失)
- **R2 严格**: 无 val_R@10 数据不能算 Gate 2 PASS, 必须 NO-GO

## Gate 3 (= Stage 4 R@K eval): ⏸ STOP per spec

- 原因: Gate 1 α 饱和 + Gate 2 无 val_R@10 验证, Stage 4 跑也无意义

## Gate 4 (决策): **NO-GO** 收口

| 假设 | 状态 | 关键证据 |
|------|------|----------|
| **零中心 + gain init=0 抑制 α 饱和** | ❌ FAIL | α 持续 1.5× growth/epoch, 8 epoch 达 12.32, 跟 #162 失败模式轨迹一致 |
| **task440 零中心架构 ≠ #162 失败模式** | ❌ FAIL | 同源代码 root cause, α 走 softplus + 残差注入, 实质相同 |

**Issue #165 闭环决策: NO-GO** (R11.5 决策错误: 复制 task452 模板而非重写 wrapper).

## 关键决策点 (R11.5 自主决策)

1. **R11.5 兜底 4. 简单实用方案错误**: 复用 task440/task452 模板, 跟 #162 失败模式同类 (R2 严格)
2. **wrapper 实质实现 ≠ spec**: task440 零中心 wrapper 实际跑出 α 饱和, 跟 #162 失败模式轨迹一致
3. **R23 强制 kill**: 跨 8 epoch α 持续 1.5× growth, 立即 §25 kill + NO-GO
4. **R24 立即激进**: R19 + R22 + R24 联立, 立即 kill + 写 verdict + close issue
5. **R20 4-Gate 详细**: commit message 含 Gate 1/2/3/4 状态 + 关键数据 + 失败原因

## 后续 (R10 v2 idle 允许)

Issue #165 NO-GO 收口. Issue #166 同模式 NO-GO 收口 (#459 跑完 Gate 3 机制 PASS 但无 val_R@10). κ元数据适配 #157/#158/#162/#163/#165/#166 6 issue 全部 NO-GO, baseline recipe 内部 κ/scale 元数据适配 R@10 杠杆已穷尽. 后续架构层方向 (R@10 > 0.1020 真杠杆) 需 owner 派工新 issue.

#450 v8 (Issue #161, ZeroCenteredLayerNorm 真实零中心架构, val_R@10 ep45=0.1021 突破 baseline) 仍在 GPU 0 继续训练, 是唯一有可能 R@10 > 0.1020 的路径.
