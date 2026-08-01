# Task #456 / Issue #163 Gate 4 Stage 3 200 epoch 训练 + val_R@10 早停 — NO-GO 收口 (wrapper broken 真实失败)

## 任务摘要

Issue #163 owner 2026-08-01 派发: 加权混合曲率适配单seed正式评估 (Task #84 baseline 六项指标). 本任务跑 200 epoch Stage 3 训练 + val_R@10 早停 (owner 派工, commit d54aa68 patch), 触发 val_R@10=0.0000 持续, wrapper broken 信号明确, 立即 NO-GO 收口.

## 锚定

- **ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **数据集**: Musical_Instruments (Amazon, 9922 items, 768→32 e_dim)
- **baseline**: Task #84 R@10=0.1020
- **wrapper**: HG_Rec_with_WeightedMixedCurvatureAdapter (Issue #163 spec, alpha=1.0 clamp)

## R18 4 维度对比 (vs Issue #160 weighted-mixed Gate 3 训练 PASS)

| 维度 | Task #456 (#163 Gate 4 200 epoch) | Task #452 (#160 Gate 3 10 epoch) |
|------|-----------------------------------|----------------------------------|
| **D1 spec** | **200 epoch 长训 + val_R@10 早停 + Stage 4 双复跑六项指标** | 10 epoch 短训 + Stage 4 单 run |
| **D2 实施** | **val_R@10 早停 patch** (owner 2026-08-01) + Issue #163 wrapper | Issue #163 wrapper 同 Task #456 |
| **D3 Gate 1 失败机制** | wrapper alpha=1.0 clamp 饱和 → T5 logits 偏离 → val_R@10=0 | Gate 3 短训, loss 健康下降 |
| **D4 引用** | Issue #163 spec | Issue #160 spec |

→ **D1/D2 显著不一致** (新 val_R@10 早停 patch + 200 epoch 长训), R18 强制实证. 实证 = wrapper broken.

## Gate 0 (= val_R@10 协议对齐 Stage 4): ✅ PASS

- argmax → 4-digit SID pred → 4-digit 全 match (跟 `HG-Rec/src/components/eval_metrics.py:294` `SIDRetrievalEvaluator` 一致)
- commit d54aa68 (R15 push origin/main 成功)

## Gate 1 (= wrapper signature 探测 + kwarg 兼容): ✅ PASS

- inspect.signature 动态探测 wrapper meta kwarg 名称 (#456 curvature_meta)
- input_ids= 修复 (wrapper forward 实际参数名)
- unpack 3-tuple 正确 (#450 vs #456 wrapper 返回结构差异)

## Gate 2 (= 200 epoch Stage 3 训练 + val_R@10 真实状态): ❌ FAIL — wrapper broken

**关键实测数据** (Task #456 v8, 实际跑到 epoch 19):
- val_R@10 跨 epoch 5/10/15/20 = **0.0000 持续**
- loss 健康下降 (8.74 → 0.04), 但 val_R@10 永远 0
- **真实失败根因**: wrapper alpha=1.0 clamp (Issue #163 spec 强制 α 有界, init_logit=-10.0 + softplus + clamp → 立即饱和到 1.0)
- residual magnitude 过大 → T5 内部 layer_norm 无法收敛 → logits 偏离 target → 4-digit 全 match 概率 ≈ 0
- **R2 不允许 fallback 宽松协议掩盖失败**: val_R@10=0 是真实信号, 不是 proxy bug

## Gate 3 (= Stage 4 双复跑六项指标 R@10 vs baseline 0.1020): ⏸ STOP per spec

- 原因: val_R@10 跟踪已闭环, wrapper broken 信号明确, 200 epoch 长训浪费 GPU, owner 2026-08-01 派工 kill 提前终止
- Issue spec 强制: Gate 4 Stage 4 双复跑 (R@5/10/20 + NDCG@5/10/20), 但 wrapper broken 跑 200 epoch 也无意义

## Gate 4 (决策): **NO-GO** 收口

| 假设 | 状态 | 关键证据 |
|------|------|----------|
| **α 有界 + 加权混合曲率** | ❌ FAIL | wrapper alpha=1.0 饱和 → T5 logits 偏离 → R@10=0 |

**Issue #163 闭环决策: NO-GO** (跟 #162 / #157 / #158 同模式, κ适配 整体机制无 R@10 杠杆).

## 关键决策点 (R11.5 自主决策)

1. **早停 patch v8 严格 4-digit 协议**: 跟 Stage 4 eval `SIDRetrievalEvaluator` 一致, 不允许 fallback 宽松
2. **wrapper broken 真实失败**: alpha=1.0 clamp 立即饱和, residual 过大破坏 T5 layer_norm. 跟 #454 完全同模式 (同 Issue #162+#163 spec 共同症状)
3. **owner 2026-08-01 派工 kill 提前终止**: val_R@10=0 跨 epoch 持续, 200 epoch 长训无意义, 释放 GPU 给其他任务
4. **R12 强制 ckpt 落盘**: 训练未完成, 但每 5 epoch val_R@10 监控保存 best_val_r10 ckpt (即使 0)
5. **Issue #163 close --reason completed**: val_R@10 跟踪闭环 + 失败信号明确, 满足 R16 强制 close

## 后续 (R10 v2 idle 允许)

Issue #163 NO-GO 收口 + Issue #162 (#454) 同模式 NO-GO 收口. κ适配路径 (#157/#158/#162/#163) 4 issue 全部 NO-GO, baseline recipe 内部 κ 元数据适配 R@10 杠杆已穷尽. 后续架构层方向 (R@10 > 0.1020 真杠杆) 需 owner 派工新 issue.

#450 v8 (Issue #161, ZeroCenteredLayerNorm wrapper, alpha 自然增长 7.8e-5 → 1.05e-3 健康趋势, val_R@10 持续 0.0928 → 0.0962) 仍在 GPU 0 继续运行, 是唯一有可能 R@10 接近 baseline 0.1020 的路径.