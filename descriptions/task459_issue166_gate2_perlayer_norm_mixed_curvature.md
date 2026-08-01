# Task #459 / Issue #166 方向B Gate 2 逐层归一化混合曲率受限残差注入验证

## 背景

Issue #166 owner 2026-08-01 派发: 验证"逐层归一化混合曲率 + 受限残差注入"是否能替代 #163 alpha clamp 失败方案. 方向B 上一 issue #163 已 wrap up (a2adffe NO-GO, 根因 WeightedMixedCurvature alpha 1.0 饱和). 新 spec 用 simplex 权重 + 零中心 residual adapter (gain init=0) 替代.

## R18 4 维度对比 (vs #163 closed NO-GO)

| 维度 | #163 (closed NO-GO, a2adffe) | #166 NEW |
|------|------------------------------|----------|
| **D1 spec** | WeightedMixedCurvature alpha clamp | **逐层归一化 + 受限残差注入 (gain init=0)** |
| **D2 实施** | alpha=1.0 clamp 立即饱和 | simplex 权重 + 零中心 residual adapter |
| **D3 失败机制** | alpha 饱和 → 残差过大 → T5 LN 崩溃 | norm-preserving + gain init=0 → 残差起步 0 |
| **D4 引用** | Task #84 HG-Rec baseline | arXiv:2307.04514 weighted mixed-curvature product manifold |

→ **D1/D2/D3 显著不一致** (R18 强制实验).

## 任务内容 (Gate 2 短训 sanity)

- **Gate 1 预检**: 三层独立 κ + 每层三分量 (固定双曲 + 固定欧氏 + learnable κ) + 逐层混合权重导出
- **Gate 2 短训**: 冻结 #158 同一 SID, 跑 5-10 epoch sanity
- **关键报告**: 三分量范数 + simplex 权重 + κ 梯度 + metadata/SID 数据流 + loss 下降 + val_R@10 协议
- **关键指标**: 连续两次 val_R@10 ≠ 0 且无残差爆炸
- **失败立即停**: 残差爆炸 / 单次 val_R@10=0 → 立即 R23 kill + NO-GO verdict + close issue

## 关键决策点 (R11.5 自主决策)

1. **R7 分配 GPU 2** (R7 + R19 并行 + #450 占 GPU 0 + #458 占 GPU 1)
2. **Stage 3 短训 epoch = 5** (R22 立即开工)
3. **R12 强制 ckpt 落盘** (每 epoch 末)
4. **R23 监控**: 跨 2 checkpoint val_R@10=0 或残差爆炸 → 立即 kill

## 验收

- 6 项 Stage 4 指标齐全 + R@10 > 0.1020 → Target reached
- 任一 Gate FAIL → 立即 NO-GO + close issue (R16)
